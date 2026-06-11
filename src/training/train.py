import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
    CosineAnnealingWarmRestarts,
)
from torchmetrics.classification import (
    BinaryAUROC,
    MulticlassAUROC,
    BinaryRecall,
    BinarySpecificity,
    MulticlassRecall,
)
from tqdm import tqdm
from pathlib import Path
import argparse
import random
import sys
import os
import gc
import warnings
import numpy as np

# Reduce la fragmentación del arena de memoria de CUDA durante runs largos
# (especialmente en Colab), permitiendo segmentos expandibles.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Añadir el raíz al path para importaciones absolutas
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.data.dataset import get_dataloaders
from src.models.architectures import SkinCancerModel
from src.utils.losses import (
    FocalLoss,
    MulticlassFocalLoss,
    SupConLoss,
    SeesawLoss,
    BalancedSoftmaxLoss,
)
from src.utils.early_stopping import EarlyStopping
from src.utils.safe_io import save_checkpoint_safe as _save_checkpoint_safe
from src.utils.seeding import seed_everything as set_seed
from src.training.sam import SAM
from src.training.self_distillation import SelfDistillationScheduler, self_distillation_loss
from src.data.constants import (
    MULTICLASS_INDEX_MAP,
    MALIGNANT_INDICES,
    PRECANCEROUS_INDICES,
    REFERRAL_INDICES,
    CLINICAL_THRESHOLD_BINARY,
)
from src.training.distillation import (
    load_teacher,
    compute_distillation_loss,
    precompute_teacher_prob,
)


# Reproducibilidad: la fuente única de siembra de semillas es
# `src/utils/seeding.py` (fija PYTHONHASHSEED, las semillas de torch/cuda/numpy
# y deja cudnn en modo determinista), importada arriba como alias `set_seed`.


class ModelEMA:
    """
    Exponential Moving Average de los pesos del modelo.

    Mantiene un promedio exponencial durante todo el entrenamiento (no solo el último 25% como SWA).
    Con decay=0.999, cada step mezcla 0.1% de los nuevos pesos con 99.9% del histórico.
    Produce pesos más suaves y generalizables. Estándar en el estado del arte de
    imagen médica (nnU-Net, soluciones ganadoras de ISIC).

    A diferencia de SWA:
    - No necesita recalibración de BatchNorm (los buffers se copian directamente)
    - Promedia desde el inicio del entrenamiento, no solo desde el 75%
    - Coste: ~50-76MB extra de VRAM (una copia del modelo)
    """
    def __init__(self, model, decay=0.999):
        import copy
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        # Copiar buffers directamente (BN running_mean/running_var ya son promedios móviles)
        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


@torch.no_grad()
def _apply_gradient_centralization(model, embedding_ids: frozenset = frozenset()):
    """Gradient Centralization (Yong et al., ECCV 2020).

    Resta la media del gradiente a lo largo de los ejes no-canal para tensores
    con `dim() > 1` (pesos de conv y linear). Overhead ~0%, +0.1-0.3% AUC típico,
    ortogonal al weight decay. Se aplica DESPUÉS de `unscale_`/`clip_grad_norm_`
    y ANTES de `scaler.step`, para operar sobre gradientes ya en fp32 y clampeados.

    No toca biases ni parámetros 1D (BN/LN): la centralización no tiene sentido
    en esos casos y degradaría la capacidad del modelo.

    También excluye explícitamente `nn.Embedding.weight` mediante el conjunto
    `embedding_ids` que precomputa el caller. Las embeddings categóricas de este
    proyecto (`sex_emb [3,4]`, `site_emb [8,8]`) tienen `dim()==2` y caerían bajo
    la regla genérica. Centralizar una fila de 4 valores elimina una de las 4
    direcciones efectivas del embedding (reduce sus grados de libertad de D a
    D-1) y degrada la señal demográfica que alimenta FiLM (p. ej. head/neck →
    lentigo maligno). Yong et al. validan GC sobre conv+linear, nunca sobre
    embeddings.
    """
    for p in model.parameters():
        if p.grad is None or p.grad.dim() <= 1:
            continue
        if id(p) in embedding_ids:
            continue
        g = p.grad
        dims = tuple(range(1, g.dim()))
        g.sub_(g.mean(dim=dims, keepdim=True))


@torch.no_grad()
def _tta_forward_views(model, images, meta, num_views: int, amp_dtype=torch.float16):
    """Test-Time Augmentation durante validación para un early stopping más estable.

    Sin esto, el val loop evaluaría con single-crop mientras la inferencia de
    producción (predict.py) usa TTA-6, de modo que el early stopping
    seleccionaría el checkpoint óptimo bajo un régimen distinto al del
    despliegue (un mismo modelo puede tener pAUC≈0.85 con single-crop y ≈0.88
    con TTA-6, y el early stopping no lo vería). Aplicar TTA determinista en
    validación alinea ambos regímenes (+0.3-0.8% AUC típico). Referencia:
    Simonyan & Zisserman 2015; práctica estándar en los retos ISIC 2023-2024.

    Argumentos:
        model: modelo en eval mode.
        images: [B, C, H, W] tensor ya en device.
        meta: [B, meta_dim] tensor de metadata clínica ya en device.
        num_views: número de vistas TTA. 1 = single-crop (no-TTA, preserva
            comportamiento baseline cuando el flag está apagado). 4 = original
            + hflip + vflip + rot180 (coste 4× val). 6 = + rot90 + rot270
            (matching predict.py TTA-6, coste 6×).
        amp_dtype: dtype del autocast (bf16 en H100, fp16 en T4).

    Retorna:
        Logits promediados en float32, con la misma shape que un forward único.
        El promedio se hace en ESPACIO LOGIT (no de probabilidad) — sigmoid o
        softmax se aplican UNA sola vez tras el promedio, lo que preserva la
        calibración del ensemble.
    """
    if num_views <= 1:
        # Shortcut backward-compat: sin TTA, comportamiento idéntico al val original.
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            return model(images, meta)

    # Transformaciones deterministas sobre ejes H/W. `torch.flip` y `torch.rot90`
    # son ortogonales → batches idénticos en runs distintos (determinismo crítico
    # para reproducibilidad de early stopping).
    _transforms = [
        lambda x: x,                                     # 1: original
        lambda x: torch.flip(x, dims=[-1]),              # 2: hflip
        lambda x: torch.flip(x, dims=[-2]),              # 3: vflip
        lambda x: torch.rot90(x, k=2, dims=(-2, -1)),    # 4: rot180
    ]
    if num_views >= 6:
        _transforms.append(lambda x: torch.rot90(x, k=1, dims=(-2, -1)))  # 5: rot90
        _transforms.append(lambda x: torch.rot90(x, k=3, dims=(-2, -1)))  # 6: rot270

    _transforms = _transforms[:num_views]

    logits_sum = None
    valid_count = 0
    for tfm in _transforms:
        x_view = tfm(images)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            out = model(x_view, meta)
        # Forzar float32 antes de acumular: bajo bf16 el rango es similar a fp32,
        # pero bajo fp16 el acumulador podría saturar con muchas vistas. Float32
        # es defensivo y coherente con el TTA de inferencia.
        out_f32 = out.float()
        # Comprobación de finitud por vista (igual que el TTA de predict.py): si
        # una vista produce NaN (overflow en GeM/AMP), se descarta y se promedia
        # el resto.
        if torch.isfinite(out_f32).all():
            logits_sum = out_f32 if logits_sum is None else logits_sum + out_f32
            valid_count += 1

    if valid_count == 0:
        # Todas las vistas produjeron NaN. Retornar el logit original (single
        # forward) con warning — el NaN guard del val loop lo capturará y
        # saltará el batch. Mantenemos el contrato de retorno (tensor, no excepción).
        import warnings as _w
        _w.warn(
            "Todas las vistas TTA de validación produjeron NaN. Se devuelve un "
            "forward único como fallback; el val loop saltará el batch.",
            RuntimeWarning,
            stacklevel=2,
        )
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            return model(images, meta)

    return logits_sum / valid_count


_PAUC_SELECTION_METRICS = ("pauc_spec95", "pauc_sens80")


def _selection_metric_label(selection_metric):
    """Etiqueta corta de la métrica de selección para los logs."""
    if selection_metric == "pauc_spec95":
        return "pAUC@spec95"
    if selection_metric == "pauc_sens80":
        return "pAUC@sens80"
    return "AUC"


def _compute_pauc_selection(
    y_true, y_probs, selection_metric, pauc_spec_target, pauc_sens_target
):
    """Dispatch centralizado de la métrica pAUC de selección de checkpoint.

    - ``pauc_spec95`` → ``partial_auc(max_fpr=1-spec_target)``: región de ALTA
      ESPECIFICIDAD (FPR≤0.05).
    - ``pauc_sens80`` → ``partial_auc_above_sensitivity(min_sens=sens_target)``:
      región de ALTA SENSIBILIDAD (estándar ISIC 2020/2024). Alinea la selección
      de checkpoint con el umbral clínico 0.20 del triaje binario
      (nunca perder un melanoma; los falsos positivos los filtra el
      especialista). Es la métrica recomendada para el triaje: optimizar la
      especificidad premiaría una deriva conservadora que degrada recall@0.20.

    Ambas variantes devuelven el valor McClish-normalizado en [0.5, 1.0] — la
    MISMA escala — de modo que no corrompen ``EarlyStopping.best_saved_score``
    (mezclar un pAUC ~0.5-1.0 con un AUC 0-1 sería un error). Devuelve float o
    None (sklearn ausente o conjunto monoclase); el caller trata None como fatal.

    Está centralizado para que las rutas de val / SWA / EMA / Model Soup no
    divergan entre sí.
    """
    if selection_metric == "pauc_spec95":
        from src.utils.evaluation import partial_auc as _fn
        return _fn(y_true, y_probs, max_fpr=1.0 - pauc_spec_target)
    if selection_metric == "pauc_sens80":
        from src.utils.evaluation import partial_auc_above_sensitivity as _fn
        return _fn(y_true, y_probs, min_sens=pauc_sens_target)
    raise ValueError(
        f"_compute_pauc_selection recibió selection_metric="
        f"{selection_metric!r} inesperado (debe ser uno de "
        f"{_PAUC_SELECTION_METRICS})."
    )


def _binary_recall_floor_block(mode, clin_recall, recall_floor_binary):
    """Guardrail clínico de sensibilidad para el triaje binario.

    Bloquea el guardado de un checkpoint binario cuyo recall@0.20 (sensibilidad
    en el umbral clínico real) cae por debajo del suelo. Es una red
    de seguridad complementaria a la métrica de selección pAUC@sens: aunque ésta
    optimice la región de alta sensibilidad, el suelo impide persistir un
    checkpoint que perdería más de `(1-floor)` de los melanomas en el punto
    operativo real de despliegue.

    Reutiliza la maquinaria ``save_blocked`` de ``EarlyStopping`` sin estado
    nuevo, y se aplica también a SWA/EMA/SOUP, ya que el operador puede desplegar
    cualquiera de ellos.

    Contratos:
    - ``mode != "binary"`` → no aplica (multiclase usa class_health).
    - ``recall_floor_binary <= 0.0`` → desactivado (baseline bit-exact).
    - ``clin_recall is None`` (no computado) → no bloquea (no penalizar
      ausencia de dato; el caller garantiza el dato en binario).
    """
    if mode != "binary" or recall_floor_binary <= 0.0:
        return False
    if clin_recall is None:
        return False
    return clin_recall < recall_floor_binary


def _evaluate_state_dict_on_val(
    state_dict, raw_model, val_loader, mode, device,
    amp_dtype=torch.float16, tta_val_views=1,
    selection_metric: str = "auc",
    pauc_spec_target: float = 0.95,
    pauc_sens_target: float = 0.80,
):
    """Evalúa un state_dict sobre el val set para el Model Soup.

    Carga un state_dict en `raw_model` y computa la métrica de selección sobre
    el val loop respetando la comprobación de finitud estándar. Devuelve
    (metric_value, num_valid_samples). `raw_model` se muta durante la
    evaluación; el caller es responsable de restaurar pesos si lo necesita.

    Incluye un fallback defensivo si `torchmetrics` no está instalado: cae a
    `sklearn.metrics.roc_auc_score` (disponible incluso en entornos mínimos), y
    si AMBOS faltan emite un warning y devuelve metric=NaN para que
    `_model_soup_greedy` salte el candidato.

    La métrica de selección se propaga desde el caller: cuando se entrena con
    `--selection_metric pauc_spec95`/`pauc_sens80`, el early stopping y los
    comparadores SWA/EMA seleccionan por pAUC (el régimen clínico real de
    despliegue), no por AUC global. Este helper debe usar la MISMA métrica para
    que el Model Soup no elija candidatos por un criterio distinto al del resto
    del pipeline. Para pAUC en binario se fuerza la rama de acumulación en CPU
    (igual que el fallback sklearn) y se computa via
    `src.utils.evaluation.partial_auc`.

    Args:
        selection_metric: "auc" (default), "pauc_spec95" (alta especificidad) o
            "pauc_sens80" (alta sensibilidad, estándar ISIC 2020). En multiclase
            el flag se ignora (se usa macro AUC).
        pauc_spec_target: target de specificity para pauc_spec95 (0.95).
        pauc_sens_target: target de sensibilidad para pauc_sens80 (0.80).
    """
    raw_model.load_state_dict(state_dict, strict=True)
    raw_model.eval()

    # El pAUC necesita y_true/y_probs completos, así que se fuerza la rama de
    # acumulación en CPU SOLO para binario+pAUC (multiclase ignora el flag).
    # `pauc_spec95` (alta especificidad) y `pauc_sens80` (alta sensibilidad)
    # comparten el mismo flujo de acumulación.
    use_pauc = (selection_metric in _PAUC_SELECTION_METRICS and mode == "binary")

    # Se intenta torchmetrics primero (dtype/device coherentes con el val loop)
    # y se cae a sklearn si falta. Las preds y labels se acumulan en CPU para
    # ese fallback (sin overhead si torchmetrics está disponible: la lista queda
    # vacía y se descarta). Para pAUC no se usa torchmetrics (no expone un pAUC
    # con max_fpr re-normalizado tipo McClish) → se fuerza la acumulación CPU.
    _use_tm = not use_pauc
    metric = None
    if _use_tm:
        try:
            from torchmetrics.classification import BinaryAUROC, MulticlassAUROC
            if mode == "binary":
                metric = BinaryAUROC().to(device)
            else:
                from src.data.constants import MULTICLASS_LABEL_MAP
                metric = MulticlassAUROC(num_classes=len(MULTICLASS_LABEL_MAP)).to(device)
        except ImportError:
            _use_tm = False
            warnings.warn(
                "torchmetrics no disponible — Model Soup usando fallback "
                "sklearn.roc_auc_score. Instalar torchmetrics para mejor rendimiento.",
                RuntimeWarning,
                stacklevel=2,
            )

    _all_preds = [] if not _use_tm else None
    _all_labels = [] if not _use_tm else None

    n_valid = 0
    with torch.no_grad():
        for images, labels, meta in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            outputs = _tta_forward_views(raw_model, images, meta, tta_val_views, amp_dtype)
            if mode == "binary":
                outputs = outputs.squeeze(1)
                preds = torch.sigmoid(outputs.float())
            else:
                preds = torch.softmax(outputs.float(), dim=1)
            # Guard NaN: si una fila es NaN, saltar todo el batch.
            if torch.isfinite(preds).all():
                if _use_tm:
                    metric.update(preds, labels)
                else:
                    _all_preds.append(preds.detach().cpu().numpy())
                    _all_labels.append(labels.detach().cpu().numpy())
                n_valid += images.size(0)

    if n_valid == 0:
        return float("nan"), 0

    if _use_tm:
        return float(metric.compute().item()), n_valid

    # Fallback con sklearn / cálculo de pAUC
    try:
        import numpy as _np
        from sklearn.metrics import roc_auc_score
        y_pred = _np.concatenate(_all_preds, axis=0)
        y_true = _np.concatenate(_all_labels, axis=0)
        if use_pauc:
            # pAUC en binario (spec≥target o sens≥target según selection_metric).
            # El helper centralizado devuelve el valor McClish-normalizado en
            # [0.5, 1.0], la misma escala que el resto del pipeline.
            pauc_val = _compute_pauc_selection(
                y_true, y_pred, selection_metric,
                pauc_spec_target, pauc_sens_target,
            )
            if pauc_val is None:
                # `partial_auc` retorna None si sklearn no está o si todas las
                # labels son una sola clase. Caer a NaN para que el caller
                # salte el candidato (mismo contrato que el path normal).
                return float("nan"), n_valid
            return float(pauc_val), n_valid
        if mode == "binary":
            auc = roc_auc_score(y_true, y_pred)
        else:
            auc = roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")
        return float(auc), n_valid
    except Exception as exc:
        warnings.warn(
            f"El fallback sklearn/pAUC también falló ({exc}). Se devuelve "
            "metric=NaN — Model Soup saltará este candidato.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan"), n_valid


def _average_state_dicts(state_dicts):
    """Promedio uniforme de una lista de state_dicts en CPU float32.

    Requiere que todos compartan el mismo conjunto de claves. Valida las shapes
    ANTES de promediar para evitar un broadcast silencioso ante un mismatch.
    Devuelve un nuevo OrderedDict con los promedios.
    """
    from collections import OrderedDict
    if not state_dicts:
        raise ValueError("_average_state_dicts requiere al menos un state_dict")
    if len(state_dicts) == 1:
        # Trivial: copia defensiva.
        return OrderedDict((k, v.detach().clone()) for k, v in state_dicts[0].items())

    ref_keys = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != ref_keys:
            raise ValueError(
                f"state_dict[{i}] tiene claves distintas a la referencia. "
                "Los candidatos del Model Soup deben compartir arquitectura exacta."
            )
        for k in ref_keys:
            if sd[k].shape != state_dicts[0][k].shape:
                raise ValueError(
                    f"shape mismatch en la clave '{k}' entre candidatos "
                    f"({state_dicts[0][k].shape} vs {sd[k].shape}). Misma "
                    "arquitectura pero distintos tamaños de capa — el Model Soup "
                    "requiere checkpoints bit-compatibles."
                )

    averaged = OrderedDict()
    n = len(state_dicts)
    for k in state_dicts[0].keys():
        tensors = [sd[k].detach().to("cpu", dtype=torch.float32) for sd in state_dicts]
        stacked = torch.stack(tensors, dim=0)
        # Mantener el dtype original del primer state_dict al reconstituir
        # (BN running_mean/var son float32, pero pesos pueden ser float16 en
        # checkpoints con AMP — preservar para load_state_dict compat).
        orig_dtype = state_dicts[0][k].dtype
        averaged[k] = stacked.mean(dim=0).to(orig_dtype)
    return averaged


# `_save_checkpoint_safe` vive en `src/utils/safe_io.py` (importado arriba como
# alias). Está allí para que `EarlyStopping.save_checkpoint` pueda reutilizarlo:
# así el best_checkpoint queda protegido frente a una desconexión a mitad del
# guardado (escritura atómica), igual que ya lo estaban SWA/EMA/SOUP.


def _model_soup_greedy(
    candidate_paths, raw_model, val_loader, mode, device,
    save_dir="models/checkpoints", amp_dtype=torch.float16, tta_val_views=1,
    selection_metric: str = "auc",
    pauc_spec_target: float = 0.95,
    pauc_sens_target: float = 0.80,
):
    """Model Soup greedy (Wortsman et al., ICML 2022).

    Promediado SELECTIVO de checkpoints: parte del mejor candidato individual y
    añade los demás uno a uno solo si el promedio mejora la métrica de selección.
    A diferencia del "uniform soup" (promediar todo a ciegas), el greedy soup
    evita que un checkpoint malo degrade el ensemble.

    Referencia: Wortsman et al., "Model soups: averaging weights of multiple
    fine-tuned models improves accuracy without increasing inference time",
    ICML 2022, arXiv:2203.05482.

    Beneficios sobre SWA/EMA:
    - Zero coste de inferencia (un solo forward, igual que cualquier
      checkpoint individual).
    - Ortogonal a SWA (promediado uniforme denso) y EMA (decay exponencial);
      el soup selecciona CANDIDATOS, no pesos.
    - Robusto a candidatos dañados (class_health bloqueó el save pero el
      archivo persiste del run anterior): si no mejora el soup, se descarta.

    Args:
        candidate_paths: lista de (label, path) a los state_dicts candidatos.
            Típico: [("best", "best_X.pth"), ("swa", "best_X_SWA.pth"),
            ("ema", "best_X_EMA.pth")].
        raw_model: modelo mutado por load_state_dict; el caller debe pasar
            uno "descartable" (se sobrescribe al final con el soup).
        val_loader, mode, device, amp_dtype, tta_val_views: flags de eval.
        selection_metric: "auc" (default), "pauc_spec95" o "pauc_sens80". El
            Model Soup propaga la MISMA métrica que el resto del pipeline (early
            stopping, comparadores SWA/EMA) a su evaluador interno; sin esto, el
            soup elegiría candidatos por AUC global y el checkpoint final
            quedaría desalineado con el régimen clínico.
        pauc_spec_target: target de specificity para pauc_spec95 (0.95).
        pauc_sens_target: target de sensibilidad para pauc_sens80 (0.80).

    Returns:
        (soup_state_dict, soup_metric, members): el soup final, el VALOR de la
        métrica de selección sobre val (AUC o pAUC según `selection_metric`, con
        TTA si está activo), y la lista de labels usados. `None` si no hubo
        candidatos válidos.

    El fairness report del soup lo genera el caller vía `_run_fairness_report`
    con `checkpoint_source="soup"`, igual que para los demás checkpoints.
    """
    if not candidate_paths:
        return None, None, []

    # Etiqueta de la métrica para que los logs sean claros.
    _metric_label = (
        _selection_metric_label(selection_metric)
        if (selection_metric in _PAUC_SELECTION_METRICS and mode == "binary")
        else "AUC"
    )
    _ = save_dir  # reservado para usos futuros (logging de path); silencia hint.

    # 1. Cargar todos los candidatos (solo los que existen en disco) y evaluar.
    loaded = []  # [(label, state_dict, individual_metric)]
    for label, path in candidate_paths:
        p = Path(path)
        if not p.exists():
            # Se emite con warnings.warn (no print) para que un backend con
            # logging estructurado pueda capturar el evento como auditable.
            warnings.warn(
                f"[Soup] Candidato '{label}' no existe en {path} — se omite.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        try:
            sd = torch.load(str(p), map_location="cpu", weights_only=True)
        except Exception as exc:
            warnings.warn(
                f"No se pudo cargar el candidato '{label}' ({exc}). Se omite.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        metric_val, n = _evaluate_state_dict_on_val(
            sd, raw_model, val_loader, mode, device,
            amp_dtype=amp_dtype, tta_val_views=tta_val_views,
            selection_metric=selection_metric,
            pauc_spec_target=pauc_spec_target,
            pauc_sens_target=pauc_sens_target,
        )
        if not math.isfinite(metric_val) or n == 0:
            # warnings.warn (no print) — ver justificación en el callsite previo.
            warnings.warn(
                f"[Soup] Candidato '{label}' no evaluable ({_metric_label}={metric_val}, n={n}) — se omite.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        print(f"   [Soup] Candidato '{label}' — {_metric_label} val: {metric_val:.4f} (n={n})")
        loaded.append((label, sd, metric_val))

    if not loaded:
        print("   [Soup] 0 candidatos válidos — abortando Model Soup.")
        return None, None, []

    # 2. Ordenar por la métrica de selección descendente.
    loaded.sort(key=lambda t: t[2], reverse=True)

    # 3. Greedy: partir del mejor y añadir uno a uno si mejora.
    current_labels = [loaded[0][0]]
    current_sds = [loaded[0][1]]
    current_metric = loaded[0][2]
    print(f"   [Soup] Seed: '{loaded[0][0]}' ({_metric_label}={current_metric:.4f})")

    for label, sd, _ind_metric in loaded[1:]:
        candidate_sds = current_sds + [sd]
        try:
            avg_sd = _average_state_dicts(candidate_sds)
        except ValueError as exc:
            warnings.warn(
                f"No se pudo promediar el candidato '{label}' ({exc}). Se omite "
                "— probable mismatch de arquitectura entre candidatos.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        new_metric, _ = _evaluate_state_dict_on_val(
            avg_sd, raw_model, val_loader, mode, device,
            amp_dtype=amp_dtype, tta_val_views=tta_val_views,
            selection_metric=selection_metric,
            pauc_spec_target=pauc_spec_target,
            pauc_sens_target=pauc_sens_target,
        )
        if math.isfinite(new_metric) and new_metric > current_metric:
            print(f"   [Soup] +'{label}' mejora {_metric_label}: {current_metric:.4f} → {new_metric:.4f}")
            current_labels.append(label)
            current_sds.append(sd)
            current_metric = new_metric
        else:
            print(f"   [Soup] +'{label}' NO mejora ({current_metric:.4f} vs {new_metric:.4f}) — descartado.")

    # 4. Construir el state_dict final.
    final_sd = _average_state_dicts(current_sds)
    return final_sd, current_metric, current_labels


def _run_fairness_report(model, val_loader, mode, device, save_dir="models/checkpoints", amp_dtype=torch.float16, checkpoint_source="unknown", min_positive: int = 10):
    """
    Ejecuta fairness_report() sobre el val set y persiste el JSON junto al checkpoint.
    Emite RuntimeWarning si algun gap AUC > 0.05 entre subgrupos.

    El path de salida incluye un sufijo derivado de `checkpoint_source` cuando el
    checkpoint no es el "best" estándar (p. ej. SWA/EMA): `fairness_report_{mode}.json`
    sigue correspondiendo al best_checkpoint (o al last_epoch_fallback cuando el
    class_health bloqueó todos los guardados), mientras que SWA y EMA viven en
    `fairness_report_{mode}_swa.json` y `_ema.json`. Así cada checkpoint tiene su
    propio registro; de lo contrario, el modelo realmente desplegado (EMA es
    lo habitual) certificaría la equidad de otro distinto (el best), con el
    consiguiente riesgo clínico y legal.
    """
    import json
    import warnings as _w
    try:
        from src.utils.evaluation import fairness_report
    except ImportError:
        # Se emite con warnings.warn (no print) para que un servicio con logging
        # estructurado pueda capturarlo: un fallback que degrada la trazabilidad
        # clínica no debe perderse en stdout.
        warnings.warn(
            "sklearn no disponible — fairness report omitido.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, meta in val_loader:
            images = images.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                outputs = model(images, meta)
            if mode == "binary":
                probs = torch.sigmoid(outputs.squeeze(1).float()).cpu().numpy()
            else:
                probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
            # Comprobación de finitud, igual que en el val loop normal y en SWA.
            # Sin ella, unos probs NaN corromperían el fairness report sin avisar.
            if np.any(np.isnan(probs)):
                continue
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    if len(all_probs) == 0:
        # Se usa warnings.warn (no print): en producción el print se perdería en
        # stdout mezclado con la barra de progreso, mientras que el warning sí lo
        # captura el handler de logging del servicio.
        warnings.warn(
            "Fairness report omitido: todos los batches produjeron NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    y_probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    val_df = val_loader.dataset.df
    metadata_df = val_df[["age", "sex", "anatom_site"]].copy()
    metadata_df = metadata_df.rename(columns={"age": "age_approx"})

    # En binario se usa el umbral clínico 0.20 para detectar sesgo en el punto
    # operativo real (0.20), no el 0.50 que solo sirve para las métricas
    # de training. `min_positive` excluye del chequeo de sesgo a los subgrupos
    # con menos de ese número de positivos (su intervalo de confianza de Wilson
    # sería inutilizable). En multiclase no aplica este umbral binario porque
    # cada clase tiene su propio recall.
    _threshold = CLINICAL_THRESHOLD_BINARY if mode == "binary" else 0.5
    report = fairness_report(
        y_true, y_probs, metadata_df, mode=mode, threshold=_threshold,
        min_positive=int(min_positive),
    )
    # Trazabilidad: qué checkpoint se midió.
    report["checkpoint_source"] = checkpoint_source

    # Sufijo según el checkpoint. `best_checkpoint` / `last_epoch_fallback`
    # mantienen el path sin sufijo (retrocompatibilidad); SWA y EMA reciben el
    # suyo propio para no sobrescribirse.
    _suffix_map = {
        "best_checkpoint": "",
        "last_epoch_fallback": "",
    }
    suffix = _suffix_map.get(checkpoint_source, f"_{checkpoint_source}")
    out_path = Path(save_dir) / f"fairness_report_{mode}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Fairness report guardado en: {out_path}")

    for flag in report.get("bias_flags", []):
        _w.warn(
            f"Fairness: {flag}",
            RuntimeWarning,
            stacklevel=2,
        )


def _run_selective_prediction_curve(
    model, val_loader, mode, device, save_dir="models/checkpoints",
    amp_dtype=torch.float16, checkpoint_source="unknown",
):
    """Selective Prediction Curve (Geifman & El-Yaniv, NeurIPS 2017).

    Persiste la curva como JSON con cobertura simétrica al fairness report para
    los 4 checkpoints que produce el pipeline (best, SWA, EMA, SOUP). Un operador
    que despliega EMA (práctica robusta habitual) lee tanto
    `fairness_report_{mode}_ema.json` como `selective_prediction_curve_{mode}_ema.json`,
    de modo que el registro clínico queda completo para el checkpoint
    realmente desplegado, no solo para el best estándar.

    El path del JSON sigue el mismo esquema de sufijos que el fairness report
    (`best_checkpoint`/`last_epoch_fallback` → sin sufijo, para
    retrocompatibilidad con scripts externos; `swa`/`ema`/`soup` → sufijo propio).
    Coste de inferencia nulo; se calcula post-hoc sobre el val set (~1-2 min por
    checkpoint).

    Degradación graciosa: cualquier fallo (todos los batches NaN, sklearn no
    importable, etc.) emite `RuntimeWarning` capturable por el handler de
    logging sin tumbar el pipeline.
    """
    try:
        import json as _json
        from src.utils.evaluation import selective_prediction_curve as _spc
    except ImportError:
        warnings.warn(
            f"Selective Prediction Curve ({checkpoint_source}) omitida — "
            "`selective_prediction_curve` no importable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    _probs, _labels = [], []
    model.eval()
    try:
        with torch.no_grad():
            for images, labels, meta in val_loader:
                images = images.to(device, non_blocking=True)
                meta = meta.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(images, meta)
                if mode == "binary":
                    probs = torch.sigmoid(outputs.squeeze(1).float()).cpu().numpy()
                else:
                    probs = torch.softmax(outputs.float(), dim=1).cpu().numpy()
                if np.any(np.isnan(probs)):
                    continue
                _probs.append(probs)
                _labels.append(labels.numpy())
    except Exception as _exc:
        warnings.warn(
            f"Selective Prediction Curve ({checkpoint_source}) falló en el "
            f"forward loop ({_exc}). Audit trail clínico incompleto — revisar "
            "antes del despliegue.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if not _probs:
        warnings.warn(
            f"Selective Prediction Curve ({checkpoint_source}) omitida — todos "
            "los batches del val set produjeron NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    try:
        _probs = np.concatenate(_probs, axis=0)
        _labels = np.concatenate(_labels, axis=0)
        _result = _spc(_labels, _probs, mode=mode, n_coverage_levels=21)
        _result["checkpoint_source"] = checkpoint_source
        # Mismo esquema de sufijos que `_run_fairness_report`: best y
        # last_epoch_fallback preservan el path sin sufijo.
        _suffix_map = {
            "best_checkpoint": "",
            "last_epoch_fallback": "",
        }
        _suffix = _suffix_map.get(checkpoint_source, f"_{checkpoint_source}")
        _path = Path(save_dir) / f"selective_prediction_curve_{mode}{_suffix}.json"
        _path.parent.mkdir(parents=True, exist_ok=True)
        with open(_path, "w") as _fh:
            _json.dump(_result, _fh, indent=2, default=str)
        print(f"Selective Prediction Curve guardada en: {_path}")
    except Exception as _exc:
        warnings.warn(
            f"Selective Prediction Curve ({checkpoint_source}) falló en el "
            f"cálculo/persistencia ({_exc}). Audit trail clínico incompleto.",
            RuntimeWarning,
            stacklevel=2,
        )


def _audit_rare_classes_support(
    val_loader, class_names, mode, min_support=30, test_csv_path=None
):
    """
    Audita el tamaño (support) por clase en val y test ANTES del training loop.
    Emite un warning para las clases con menos de `min_support` muestras: sus
    métricas per-class (recall, AUC) tendrán una varianza muy alta y pueden
    falsear el juicio del operador al comparar checkpoints.

    Motivación clínica: el subconjunto multiclase puede tener muy pocas muestras
    de clases raras (p. ej. ~5 de Dermatofibroma en test). Un checkpoint que
    clasifique mal una sola de ellas cae a recall=80% sin que eso refleje la
    calidad real del modelo. Además, el bloqueo de guardado por AUC baja solo
    cubre las clases malignas (MEL/BCC/SCC); las benignas raras como
    Dermatofibroma no tienen guard estructural. Esta función es informativa
    (warning de una sola vez), complementaria a ese bloqueo: el operador debe
    tener en cuenta el support al interpretar las métricas per-class.

    Args:
        val_loader: DataLoader de validación (con `dataset.df` y
            `dataset.label_col`, `dataset.label_map`).
        class_names: lista de nombres de clase.
        mode: "binary" o "multiclass".
        min_support: umbral bajo el cual la clase se considera low-support.
        test_csv_path: path opcional al CSV de test para auditar también.
    """
    try:
        ds = val_loader.dataset
        val_df = ds.df
        label_col = ds.label_col
        label_map = ds.label_map
    except AttributeError:
        # Si el loader no tiene la estructura esperada, omitir el audit
        # silenciosamente (no bloqueante — solo ayuda).
        return

    val_counts = val_df[label_col].map(label_map).value_counts().to_dict()

    test_counts = None
    if test_csv_path is not None and os.path.exists(test_csv_path):
        try:
            import pandas as _pd
            test_df = _pd.read_csv(test_csv_path)
            # Test puede usar columna distinta (07_*_multiclass tiene target_idx)
            if label_col in test_df.columns:
                test_counts = (
                    test_df[label_col].map(label_map).value_counts().to_dict()
                )
            elif "target_idx" in test_df.columns:
                test_counts = test_df["target_idx"].value_counts().to_dict()
        except Exception:
            test_counts = None

    print(f"\n[AUDIT CLASES] Support por clase en val (mode={mode}):")
    low_support = []
    for idx, name in enumerate(class_names):
        nv = int(val_counts.get(idx, 0))
        nt = int(test_counts.get(idx, 0)) if test_counts is not None else None
        tag = " [SUPPORT BAJO]" if (nv < min_support or (nt is not None and nt < min_support)) else ""
        support_str = f"val={nv}"
        if nt is not None:
            support_str += f", test={nt}"
        print(f"   [{idx}] {name:28s} {support_str}{tag}")
        if nv < min_support or (nt is not None and nt < min_support):
            low_support.append((idx, name, nv, nt))

    if low_support:
        for idx, name, nv, nt in low_support:
            test_str = f", test={nt}" if nt is not None else ""
            warnings.warn(
                f"Clase '{name}' (idx={idx}) tiene support bajo "
                f"(val={nv}{test_str}, umbral={min_support}). Las métricas "
                f"per-class (recall, AUC) tendrán varianza alta y pueden "
                f"falsear la comparación entre checkpoints. Considerar "
                f"cross-validation o intervalo de Wilson para reportes clínicos.",
                RuntimeWarning,
                stacklevel=2,
            )


def _check_class_health(class_recalls, class_aucs, class_names, context=""):
    """
    Detecta colapso de clase (recall bajo) y AUC bajo en clases malignas.
    Emite RuntimeWarning capturable por el handler del servicio.

    Devuelve False si alguna clase maligna tiene AUC < 0.80 o recall < 0.30, lo
    que permite al caller bloquear la persistencia del checkpoint. Los umbrales
    de BLOQUEO (0.80/0.30) son deliberadamente más bajos que los de WARNING
    (0.90/0.30) para no impedir el guardado en épocas tempranas, donde el AUC aún
    no ha convergido.

    Args:
        class_recalls: tensor [num_classes] con recall por clase.
        class_aucs: tensor [num_classes] con AUC por clase, o None.
        class_names: lista de nombres de clase alineada con los tensores.
        context: prefijo para los mensajes (ej. "Época 5: ", "SWA: ").

    Returns:
        bool: True si todas las clases malignas pasan health checks (checkpoint seguro),
              False si alguna clase maligna tiene AUC < 0.80 o recall < 0.30.
    """
    is_healthy = True

    # La detección de colapso usa REFERRAL_INDICES (= MALIGNANT ∪ PRECANCEROUS),
    # no solo MALIGNANT: la queratosis actínica (AK) exige una detección tan
    # fiable como el melanoma, ya que sin tratar progresa a SCC en años. Umbrales
    # de recall: < 0.30 para las clases REFERRAL (Mel/BCC/SCC/AK) y < 0.10 para
    # el resto (benignas). Dermatofibroma (84 imgs) puede colapsar a recall=0
    # afectando al macro recall solo ~0.016 — invisible en las métricas globales
    # pero catastrófico para los pacientes de esa clase.
    for i, (name, recall_val) in enumerate(zip(class_names, class_recalls)):
        r = recall_val.item() if hasattr(recall_val, 'item') else float(recall_val)
        is_referral = i in REFERRAL_INDICES
        # Etiqueta clínica para mensajes — diferencia entre cáncer invasivo y precanceroso
        clinical_kind = (
            "MALIGNA" if i in MALIGNANT_INDICES
            else "PRECANCEROSA" if i in PRECANCEROUS_INDICES
            else "BENIGNA"
        )
        if math.isnan(r):
            warnings.warn(
                f"[CRÍTICO] {context}{name} ({clinical_kind}) recall=NaN "
                f"(sin muestras en val). Verificar split de validación.",
                RuntimeWarning,
                stacklevel=2,
            )
            if is_referral:
                is_healthy = False
            continue
        threshold = 0.30 if is_referral else 0.10
        if r < threshold:
            severity = "CRÍTICO" if is_referral else "AVISO"
            warnings.warn(
                f"[{severity}] {context}Class collapse: {name} ({clinical_kind}) "
                f"recall={r:.3f} < {threshold}",
                RuntimeWarning,
                stacklevel=2,
            )
            if is_referral:
                is_healthy = False

    # AUC bajo en clases REFERRAL (< 0.90): el modelo puede alcanzar 0.98 de
    # macro AUC mientras el AUC de melanoma o de AK baja a 0.85, invisible porque
    # Nevus (clase dominante) infla la media. Para AK en concreto, un AUC bajo
    # significa que el triaje binario (que ahora aprende AK como maligna) podría
    # no discriminarla de los nevus benignos → falsos negativos clínicamente
    # peligrosos.
    if class_aucs is not None:
        for idx in REFERRAL_INDICES:
            clinical_kind = (
                "MALIGNA" if idx in MALIGNANT_INDICES else "PRECANCEROSA"
            )
            a = class_aucs[idx].item() if hasattr(class_aucs[idx], 'item') else float(class_aucs[idx])
            if math.isnan(a):
                warnings.warn(
                    f"[CRÍTICO] {context}{class_names[idx]} ({clinical_kind}) "
                    f"AUC=NaN. Verificar representación de la clase en val.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                is_healthy = False
            elif a < 0.90:
                warnings.warn(
                    f"[CRÍTICO] {context}{class_names[idx]} ({clinical_kind}) "
                    f"AUC={a:.4f} < 0.90. El modelo puede fallar en discriminar "
                    "esta lesión que requiere derivación dermatológica.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                # Umbral de BLOQUEO de checkpoint: 0.80 (más bajo que el warning 0.90).
                # Las primeras épocas naturalmente tienen AUC bajo — solo bloqueamos
                # cuando el rendimiento es críticamente insuficiente.
                if a < 0.80:
                    is_healthy = False

    return is_healthy


def _compute_grad_norms(model):
    """
    Calcula la norma L2 de gradientes para backbone y head por separado.

    Permite diagnosticar:
    - Backbone grad norm < 1e-7 → vanishing gradients, backbone efectivamente frozen
    - Head grad norm >> backbone grad norm → desbalance excesivo (esperado al inicio
      por LR diferencial, preocupante si persiste tras warmup)
    - Ambas normas NaN → gradientes corruptos (debería ser capturado antes por NaN guard)

    Returns:
        (backbone_norm, head_norm): floats con las normas L2.
        Retorna (0.0, 0.0) si no hay gradientes (ej. tras zero_grad).
    """
    backbone_grads = []
    head_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if "backbone" in name:
                backbone_grads.append(param.grad.detach().float().norm())
            else:
                head_grads.append(param.grad.detach().float().norm())

    backbone_norm = torch.stack(backbone_grads).norm().item() if backbone_grads else 0.0
    head_norm = torch.stack(head_grads).norm().item() if head_grads else 0.0
    return backbone_norm, head_norm


def train_model(
    mode="binary",
    batch_size=32,
    epochs=20,
    lr=1e-4,
    patience=5,
    accum_steps=2,
    *,
    backbone_name=None,
    fusion_mode="film",
    drop_path_rate=None,
    use_sam=False,
    sam_rho=0.05,
    sam_adaptive=False,
    use_seesaw=False,
    seesaw_p=0.8,
    seesaw_q=2.0,
    balanced_softmax_weight=0.0,
    use_self_distill=False,
    self_distill_weight=0.1,
    self_distill_T=4.0,
    self_distill_warmup=5,
    distill_from=None,
    distill_alpha=0.3,
    scheduler_type="sequential",
    lr_head_multi=None,
    wd_head_multi=None,
    use_gc=True,
    selection_metric="auc",
    pauc_spec_target=0.95,
    pauc_sens_target=0.80,
    recall_floor_binary=0.0,
    focal_alpha=0.7,
    optimizer_name="adamw",
    seed=42,
    use_fitnet=False,
    fitnet_weight=0.05,
    fitnet_teacher_dim=1536,
    grad_checkpointing=False,
    use_attention_pool=False,
    use_hybrid_pool=False,
    use_multiscale_vit=False,
    cbam_adaptive_kernel=False,
    drop_path_schedule="uniform",
    tta_val_views=1,
    use_model_soup=False,
    poly_epsilon=0.0,
    group_balance_dims=(),
    group_balance_smooth=0.7,
    min_positive=10,
):
    """
    Loop de entrenamiento principal.
    Maneja tanto el Triaje (binary) como el Especialista (multiclass).

    Todos los argumentos keyword-only son opcionales: sus valores por defecto
    preservan exactamente el comportamiento de referencia y respetan los
    invariantes clínicos del proyecto.
    """
    print(f"Iniciando Entrenamiento - Modo: {mode.upper()}")
    print(f"   Gradient Accumulation: {accum_steps} steps (effective batch = {batch_size * accum_steps})")

    # Validación de un número mínimo de épocas. El scheduler compone
    # `LinearLR(warmup) → CosineAnnealingLR(T_max=swa_start-warmup)` con
    # `swa_start = int(epochs * 0.60)`. Con `epochs` pequeños, cualquiera de:
    #   - `warmup_epochs >= swa_start` → `T_max = max(1, neg) = 1` degenerado
    #   - `swa_start >= epochs` → SWA nunca arranca
    #   - `warmup_epochs >= epochs` → `SequentialLR(milestones=[warmup])`
    #     jamás alcanza el milestone; main_scheduler nunca se activa.
    # El fallo sería silencioso: con `--epochs 4` se obtendría un modelo con
    # `lr ≈ 0.01 * lr_nominal` sin avisar, certificado como "válido" por los
    # checks posteriores. Se valida de forma explícita con mínimos recomendados.
    _warmup_epochs_check = 3 if mode == "multiclass" else 2
    _swa_start_check = int(epochs * 0.60)
    if _warmup_epochs_check >= _swa_start_check or _swa_start_check >= epochs or _warmup_epochs_check >= epochs:
        _min_epochs = 10 if mode == "multiclass" else 8
        raise ValueError(
            f"epochs={epochs} insuficiente para modo '{mode}'. "
            f"warmup={_warmup_epochs_check}, swa_start={_swa_start_check}. "
            f"Requisitos: warmup < swa_start < epochs. "
            f"Mínimo recomendado: epochs >= {_min_epochs}. "
            f"Con valores menores, el scheduler queda degenerado silenciosamente y "
            f"el modelo nunca alcanza el LR nominal — riesgo clínico inaceptable."
        )

    # Métrica de selección de checkpoint (mode-aware). El umbral clínico binario
    # es 0.20, que corresponde al lado de alta sensibilidad de la
    # ROC. El macro AUC integra sobre toda la curva, incluyendo regímenes de
    # spec<0.5 irrelevantes en producción; una pAUC restringida (McClish 1989,
    # normalizada a [0,1]) alinea la selección con el régimen real de despliegue.
    # `selection_metric="auc"` (default) preserva el comportamiento previo. En
    # multiclase la pAUC macro-OVR no está implementada: el flag se ignora con un
    # warning explícito.
    if selection_metric not in ("auc",) + _PAUC_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric debe ser 'auc', 'pauc_spec95' o 'pauc_sens80', "
            f"recibido: {selection_metric!r}"
        )
    # Validación de rango de los hiperparámetros clínicos. Una configuración
    # inválida en uno de ellos nunca debe degradar el entrenamiento de forma
    # silenciosa. Sin estos guards, un typo (`--focal_alpha 1.75` en vez de
    # 0.75) haría `alpha_t = 1-1.75 = -0.5` para los benignos → loss invertida
    # silenciosa; `--recall_floor_binary 9.0` (typo de 0.90) bloquearía TODOS
    # los guardados. Los defaults (alpha=0.7, floor=0.0, sens=0.80, spec=0.95)
    # pasan todos los checks.
    if not (0.0 < focal_alpha < 1.0):
        raise ValueError(
            f"focal_alpha debe estar en (0, 1), recibido {focal_alpha}. "
            f"alpha=0.5 neutral; >0.5 up-weight melanoma (rango clínico típico "
            f"0.70-0.80); fuera de (0,1) → `alpha_t` de clase negativo → loss "
            f"corrupta silenciosa. Revisa si hubo un typo (ej. 1.75 → 0.75)."
        )
    if not (0.0 <= recall_floor_binary <= 1.0):
        raise ValueError(
            f"recall_floor_binary debe estar en [0, 1], recibido "
            f"{recall_floor_binary}. 0.0 = guardrail desactivado (baseline); "
            f"clínico 0.90. >1.0 bloquearía TODOS los checkpoints → solo "
            f"forced-baseline silencioso. ¿Typo de un valor en [0,1]?"
        )
    if not (0.0 < pauc_sens_target < 1.0):
        raise ValueError(
            f"pauc_sens_target debe estar en (0, 1), recibido "
            f"{pauc_sens_target}. Estándar ISIC 2020 = 0.80 (TPR≥0.80). "
            f"1.0 → región nula (max_fpr=0); 0.0 → AUC completo (no es pAUC)."
        )
    if not (0.0 < pauc_spec_target < 1.0):
        raise ValueError(
            f"pauc_spec_target debe estar en (0, 1), recibido "
            f"{pauc_spec_target}. Estándar dermatológico = 0.95 (FPR≤0.05). "
            f"1.0 → max_fpr=0 (sklearn ValueError); 0.0 → AUC completo."
        )
    if selection_metric in _PAUC_SELECTION_METRICS and mode != "binary":
        warnings.warn(
            f"selection_metric='{selection_metric}' solo está soportado en modo "
            f"binario. Modo actual '{mode}' → se usa 'auc' (macro AUC). "
            "La pAUC macro-OVR para multiclase no está implementada.",
            RuntimeWarning,
            stacklevel=2,
        )
        selection_metric = "auc"
    _use_pauc_selection = (selection_metric in _PAUC_SELECTION_METRICS)
    if _use_pauc_selection:
        if selection_metric == "pauc_sens80":
            # Régimen de ALTA SENSIBILIDAD (estándar ISIC 2020/2024), alineado
            # con el umbral clínico 0.20 del triaje (nunca perder
            # un melanoma; los falsos positivos los filtra el especialista).
            print(
                f"Checkpoint selection: Partial AUC @ sens>="
                f"{pauc_sens_target:.2f} (ISIC 2020 — alta sensibilidad, "
                f"alineado con el triaje clínico)."
            )
        else:
            # Régimen de alta especificidad (McClish 1989).
            print(
                f"Checkpoint selection: Partial AUC @ spec>="
                f"{pauc_spec_target:.2f} (McClish 1989 — alta especificidad)."
            )
    if recall_floor_binary > 0.0 and mode == "binary":
        # Guardrail de sensibilidad: bloquea el guardado de cualquier checkpoint
        # (best/SWA/EMA/SOUP) con recall@0.20 por debajo del suelo.
        print(
            f"Guardrail clínico: recall@{CLINICAL_THRESHOLD_BINARY:.2f} "
            f">= {recall_floor_binary:.2f} obligatorio para guardar checkpoint "
            f"binario."
        )

    # 0. Sellar el entorno para reproducibilidad y velocidad máxima.
    # `seed` es un parámetro keyword (default 42) para que un wrapper de deep
    # ensemble pueda propagar semillas distintas (42/101/202) vía `--seed` y
    # producir checkpoints genuinamente diferentes. Con un seed fijo, el "deep
    # ensemble" sería un triple clon del mismo modelo (sin ganancia).
    set_seed(seed)
    print(f"   Seed de reproducibilidad: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo principal: {device}")

    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(
            f"Detectadas {n_gpus} GPUs. Activando DataParallel."
        )

    # 1. Cargar Datos
    # Determinar tipo de backbone para seleccionar resolución nativa (IMAGE_SIZES).
    # CNN (EfficientNet): 300/380px. ViT (DINOv2): 294/378px (múltiplos de patch_size=14).
    # Sin esto, ViT recibe 300px y hace resize interno 300→294, degradando features finas.
    _backbone_kind = "cnn"
    if backbone_name and ("vit" in backbone_name or "dinov2" in backbone_name):
        _backbone_kind = "vit"
    print("Cargando Dataloaders...")
    # Se propaga el seed a get_dataloaders: sin él, los workers y el
    # WeightedRandomSampler verían secuencias idénticas entre runs con --seed
    # distintos, eliminando la diversidad real de los deep ensembles. También se
    # propaga group_balance_* para el rebalanceo de equidad por subgrupo (site,
    # age) sobre el sampler; con `group_balance_dims=()` el comportamiento es el
    # de referencia.
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size, mode=mode, backbone_kind=_backbone_kind, seed=seed,
        group_balance_dims=tuple(group_balance_dims),
        group_balance_smooth=float(group_balance_smooth),
    )

    # Recalcular class_counts inline (Seesaw/BalancedSoftmax los necesitan crudos).
    # `get_dataloaders` los usa internamente para el WeightedRandomSampler pero no los
    # devuelve — recalcular aquí evita romper el contrato de dataset.py (solo 2 callers).
    _train_ds = train_loader.dataset
    _train_targets = (
        _train_ds.df[_train_ds.label_col].map(_train_ds.label_map).values.astype(int)
    )
    class_counts_np = np.bincount(_train_targets)

    # Auditoría (una sola vez) del support por clase en val/test. Emite warning
    # si alguna clase tiene menos de 30 muestras en val o test — sus métricas
    # per-class (recall, AUC) tendrían una varianza muy alta (con ~5 muestras de
    # una clase, un solo error son 20 puntos de recall). El bloqueo de guardado
    # por AUC baja solo cubre MEL/BCC/SCC, no las clases benignas raras (DF,
    # VASC), así que esta auditoría complementaria es informativa.
    _audit_class_names = (
        [MULTICLASS_INDEX_MAP[i] for i in range(len(MULTICLASS_INDEX_MAP))]
        if mode == "multiclass"
        else ["benign", "malignant"]
    )
    _audit_test_csv = (
        "data/processed/07_test_multiclass.csv"
        if mode == "multiclass"
        else "data/processed/04_test.csv"
    )
    _audit_rare_classes_support(
        val_loader,
        _audit_class_names,
        mode=mode,
        min_support=30,
        test_csv_path=_audit_test_csv,
    )

    # 2. Configurar Arquitectura según el Nivel Clínico
    aux_criterion = None
    if mode == "binary":
        num_classes = 1
        model_name = "tf_efficientnet_b3.ns_jft_in1k"  # B3 para Triaje
        # alpha compensa el desbalance de diversidad. El WeightedRandomSampler
        # equilibra los batches a 50/50, pero el modelo dispone de ~331K imágenes
        # benignas ÚNICAS frente a solo ~6K malignas ÚNICAS (cada melanoma se
        # repite ~28x). alpha=0.7 penaliza fallar un melanoma 2.3x más que fallar
        # un benigno; alpha=0.75 → ~3x. Es configurable vía `--focal_alpha`
        # (default 0.7); FocalLoss.__init__ exige alpha explícito (nunca None).
        # `poly_epsilon > 0` activa el término Poly-1 (Leng et al., ICCV 2022).
        criterion = FocalLoss(alpha=focal_alpha, gamma=2.0, label_smoothing=0.1, poly_epsilon=poly_epsilon)
        print(
            f"   Loss: FocalLoss(alpha={focal_alpha}, gamma=2.0, "
            f"poly_epsilon={poly_epsilon})"
            + (" — Poly-1-Focal activo" if poly_epsilon > 0.0 else "")
        )
    else:
        num_classes = 8
        model_name = (
            "tf_efficientnet_b4.ns_jft_in1k"  # B4 para Especialista (más profundo)
        )
        # INVARIANTE 7: NO usar class_weights en el loss.
        # WeightedRandomSampler maneja el equilibrio de clases. Tanto Focal como Seesaw/
        # BalancedSoftmax respetan este invariante (Seesaw usa ratios n_j/n_i sobre logits,
        # no class_weights globales; BalancedSoftmax inyecta log-prior en los logits).
        if use_seesaw:
            counts_tensor = torch.from_numpy(class_counts_np).float()
            criterion = SeesawLoss(
                class_counts=counts_tensor,
                p=seesaw_p,
                q=seesaw_q,
                label_smoothing=0.1,
            )
            print(
                f"   Loss: SeesawLoss(p={seesaw_p}, q={seesaw_q}, LS=0.1) "
                f"— counts={class_counts_np.tolist()}"
            )
            if balanced_softmax_weight > 0:
                aux_criterion = BalancedSoftmaxLoss(
                    class_counts=counts_tensor,
                    label_smoothing=0.1,
                )
                print(
                    f"   Aux loss: BalancedSoftmax (weight={balanced_softmax_weight})"
                )
        else:
            # `poly_epsilon > 0` activa el término Poly-1-Focal multiclase.
            criterion = MulticlassFocalLoss(gamma=2.0, label_smoothing=0.1, poly_epsilon=poly_epsilon)
            if poly_epsilon > 0.0:
                print(f"   Loss: MulticlassFocalLoss(gamma=2.0, LS=0.1, poly_epsilon={poly_epsilon}) — Poly-1-Focal activo")

    # SupCon projection head solo para multiclass (mejora discriminacion AK<->BKL, MEL<->NV).
    # Binary no lo necesita (ya tiene FocalLoss bien calibrado y solo 2 clases).
    use_supcon = (mode == "multiclass")
    effective_drop_path = drop_path_rate if drop_path_rate is not None else 0.2

    # Kwargs de las mejoras opcionales del modelo (CBAM adaptativo, attention
    # pool, proyección FitNets, etc.). Sus defaults preservan el comportamiento
    # de referencia.
    _extra_model_kwargs = dict(
        cbam_adaptive_kernel=cbam_adaptive_kernel,
        use_attention_pool=use_attention_pool,
        use_hybrid_pool=use_hybrid_pool,
        use_multiscale_vit=use_multiscale_vit,
        use_fitnet=use_fitnet,
        fitnet_teacher_dim=fitnet_teacher_dim,
        use_grad_checkpointing=grad_checkpointing,
    )

    if backbone_name is not None:
        # Path con factory + wrapper. State_dict = `backbone.backbone.<layer>`.
        print(
            f"Inicializando Modelo (factory): {backbone_name} "
            f"| fusion={fusion_mode} | Clases: {num_classes} | drop_path={effective_drop_path}"
        )
        from src.models.backbones import build_backbone as _bb  # noqa: F401 (ya importado transitivamente)
        model = SkinCancerModel(
            backbone_name=backbone_name,
            fusion_mode=fusion_mode,
            mode=mode,
            num_classes=num_classes,
            use_metadata=True,
            use_projection_head=use_supcon,
            drop_path_rate=effective_drop_path,
            **_extra_model_kwargs,
        )
        # Stochastic Depth con schedule lineal aplicado tras la construcción
        # sobre el backbone envuelto. `SkinCancerModel.__init__` usa
        # `build_backbone` sin pasar el schedule, así que se aplica aquí para no
        # cambiar la firma del modelo.
        if drop_path_schedule == "linear":
            from src.models.backbones import _apply_linear_drop_path_schedule
            inner = getattr(model.backbone, "backbone", model.backbone)
            _applied = _apply_linear_drop_path_schedule(inner, effective_drop_path)
            print(f"   Stochastic Depth linear aplicado a {_applied} capas (post-factory).")
    else:
        # Path legacy: timm directo. State_dict = `backbone.<layer>`.
        # Retrocompatible con todos los checkpoints existentes.
        print(f"Inicializando Modelo: {model_name} (Clases: {num_classes})")
        model = SkinCancerModel(
            model_name=model_name,
            num_classes=num_classes,
            use_metadata=True,
            use_projection_head=use_supcon,
            fusion_mode=fusion_mode,
            drop_path_rate=effective_drop_path,
            **_extra_model_kwargs,
        )
        # En el path legacy el backbone es el módulo de timm sin wrapper, así
        # que el schedule se aplica directamente sobre `model.backbone`.
        if drop_path_schedule == "linear":
            from src.models.backbones import _apply_linear_drop_path_schedule
            _applied = _apply_linear_drop_path_schedule(model.backbone, effective_drop_path)
            print(f"   Stochastic Depth linear aplicado a {_applied} capas (legacy).")

    # SupConLoss complementa a MulticlassFocalLoss mejorando el espacio de
    # features para clases visualmente similares. Solo se aplica en batches SIN
    # MixUp (los labels mezclados no tienen pares positivos claros). Como con
    # mixup_prob=0.5 en multiclase SupCon solo ve ~50% de los batches, el peso
    # nominal 0.1 daría un peso efectivo ~0.05; se sube a 0.2 para compensar ese
    # duty-cycle y que el peso efectivo vuelva a ser ~0.1.
    supcon_criterion = SupConLoss(temperature=0.07).to(device) if use_supcon else None
    supcon_weight = 0.2

    if n_gpus > 1:
        model = nn.DataParallel(model)  # Divide el batch_size entre todas las GPUs.

    model.to(device)
    # Mover criterion a device una sola vez. SeesawLoss y BalancedSoftmaxLoss tienen
    # buffers registrados (class_counts, log_prior) que deben estar en GPU. Sin este
    # .to(device) explícito, los buffers quedarían en CPU y el forward fallaría con
    # device mismatch. FocalLoss/MulticlassFocalLoss no tienen estado — .to() es no-op.
    criterion = criterion.to(device)
    if aux_criterion is not None:
        aux_criterion = aux_criterion.to(device)

    # Se extrae `raw_model` ANTES de torch.compile. torch.compile reemplaza
    # `model` por un OptimizedModule cuyos named_parameters() pueden no incluir
    # el prefijo "backbone" → los param groups pondrían TODOS los params en la
    # head (10x LR en el backbone, destruyendo las features de ImageNet).
    # Además, isinstance(model, DataParallel) devuelve False tras compilar → EMA
    # y SWA recibirían el wrapper compilado → state_dict con artefactos de
    # compilación → checkpoints corruptos. `raw_model` es el nn.Module limpio: se
    # usa para EMA, SWA, los param groups y los saves; `model` (potencialmente
    # compilado) solo para forward/backward y operaciones sobre gradientes.
    raw_model = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model

    # Se precomputan una sola vez los IDs de los pesos de nn.Embedding para
    # excluirlos de la Gradient Centralization: los embeddings categóricos
    # (sex/site) perderían grados de libertad efectivos si GC les restara la
    # media por fila. El conjunto es estable durante todo el training (la
    # topología del modelo no cambia).
    _embedding_param_ids = frozenset(
        id(m.weight) for m in raw_model.modules() if isinstance(m, nn.Embedding)
    )

    # torch.compile: optimización de grafos para H100/A100 (PyTorch 2.x).
    # mode="reduce-overhead" minimiza el overhead de kernel launch vía CUDA
    # Graphs, ideal en Ampere+ (sm_80+, H100/A100). Speedup 15-25%. Si falla, el
    # training continúa sin compilar (best-effort, no fatal). No se compilan
    # swa_model ni ema (sus wrappers no son compatibles con compile).
    #
    # CUDA Graphs en `reduce-overhead` reciclan memoria entre invocaciones del
    # modelo, y eso es incompatible con dos escenarios:
    #   1. Turing (sm_75): capturan buffers de fusión temprana (FiLM /
    #      MetaCrossAttention / CBAM) y fallan con `RuntimeError: accessing
    #      tensor output of CUDAGraphs that has been overwritten by a subsequent
    #      run`, dejando el contexto CUDA inutilizable hasta reiniciar la sesión.
    #   2. Gradient accumulation (accum_steps>1, en cualquier GPU): el segundo
    #      backward del ciclo necesita activations cuya memoria fue reciclada por
    #      el forward siguiente → mismo RuntimeError. Es un problema conocido de
    #      PyTorch (pytorch/pytorch#111327, #108333), no específico a este
    #      pipeline; la recomendación oficial es usar `mode="default"` con
    #      gradient accumulation.
    #
    # Política adaptativa resultante:
    # - cap[0] >= 8 (Ampere/Ada/Hopper) AND accum_steps == 1 → reduce-overhead
    #   (CUDA Graphs habilitado, +15-25% speedup, seguro sin gradient accum).
    # - cap[0] >= 8 AND accum_steps > 1 → default (sin CUDA Graphs, +10-15%
    #   speedup vs sin compile, estable bajo gradient accumulation).
    # - cap[0] < 8 (Turing y previos) → default.
    # La precisión clínica se preserva al 100%: `default` produce los mismos
    # gradientes y pesos (módulo aritmética float) que `reduce-overhead`.
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        _cap_compile = torch.cuda.get_device_capability(0)
        if _cap_compile[0] >= 8 and accum_steps == 1:
            _compile_mode = "reduce-overhead"
        else:
            _compile_mode = "default"
        try:
            model = torch.compile(model, mode=_compile_mode)
            if _compile_mode == "reduce-overhead":
                print(
                    f"torch.compile activado (mode=reduce-overhead, "
                    f"sm_{_cap_compile[0]}{_cap_compile[1]} Ampere+, accum_steps=1)"
                )
            elif _cap_compile[0] >= 8:
                # Ampere+ con gradient accumulation → default (CUDA Graphs off)
                print(
                    f"torch.compile activado (mode=default, "
                    f"sm_{_cap_compile[0]}{_cap_compile[1]} Ampere+, "
                    f"accum_steps={accum_steps} → CUDA Graphs desactivado por gradient accumulation)"
                )
            else:
                warnings.warn(
                    f"torch.compile activado (mode=default, sm_{_cap_compile[0]}{_cap_compile[1]} "
                    f"Turing — reduce-overhead haría fallar CUDA Graphs en <sm_80)",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except Exception as e:
            warnings.warn(
                f"torch.compile no disponible, continuando sin compilar: {e}",
                RuntimeWarning,
                stacklevel=2,
            )

    # EMA: Exponential Moving Average (complementario a SWA)
    # Promedia pesos durante todo el entrenamiento, no solo el último 25%.
    # Se inicializa con raw_model (sin DataParallel ni compile wrapper) para pesos limpios.
    ema = ModelEMA(raw_model, decay=0.999)
    save_path_ema = f"models/checkpoints/best_{mode}_model_EMA.pth"

    # 3. Optimizador y schedulers
    # Learning Rate diferencial + Weight Decay decoupling. El backbone
    # (EfficientNet preentrenado en ImageNet) ya sabe extraer texturas y bordes,
    # así que recibe un LR 10x menor para preservar esas representaciones; la
    # head (clasificador nuevo) parte de pesos aleatorios y necesita un LR alto.
    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []
    # Se itera sobre `raw_model.named_parameters()` para que el prefijo
    # "backbone" siempre esté presente, independientemente de torch.compile.
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "backbone" in name
        is_no_decay = len(param.shape) == 1 or name.endswith(".bias")
        if is_backbone:
            (backbone_no_decay if is_no_decay else backbone_decay).append(param)
        else:
            (head_no_decay if is_no_decay else head_decay).append(param)

    param_groups = [
        {"params": backbone_decay,    "lr": lr * 0.1, "weight_decay": 1e-2},  # Backbone: LR bajo
        {"params": backbone_no_decay, "lr": lr * 0.1, "weight_decay": 0.0},
        {"params": head_decay,        "lr": lr,       "weight_decay": 1e-2},  # Head: LR estándar
        {"params": head_no_decay,     "lr": lr,       "weight_decay": 0.0},
    ]
    # Overrides opcionales para la head del especialista multiclase (lr/wd).
    if mode == "multiclass" and lr_head_multi is not None:
        param_groups[2]["lr"] = lr_head_multi
        param_groups[3]["lr"] = lr_head_multi
        print(f"   Head LR multiclase override: {lr_head_multi}")
    if mode == "multiclass" and wd_head_multi is not None:
        param_groups[2]["weight_decay"] = wd_head_multi
        print(f"   Head WD multiclase override: {wd_head_multi}")

    # Lion optimizer como alternativa drop-in a AdamW, activable con
    # `--optimizer lion`. Requiere LR ~10x menor y WD ~10x mayor que el AdamW
    # equivalente; si se pasa el LR estándar (1e-4) con Lion, se emite un warning
    # para evitar una divergencia silenciosa.
    if optimizer_name == "lion":
        from src.training.lion import Lion
        # No se ajusta el weight_decay automáticamente: hacerlo pisaría un valor
        # que el usuario pudo fijar a propósito. En su lugar se emite un warning
        # explícito para que lo ajuste a mano si quiere el scaling del paper.
        if lr > 1e-5 * 1.5:
            warnings.warn(
                f"Lion optimizer con lr={lr} es probablemente demasiado alto. "
                f"El paper recomienda lr ~10x menor que AdamW equivalente. "
                f"Sugerencia: lr=1e-5 para este pipeline.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Warning explícito sobre weight_decay, sin auto-ajuste.
        _current_max_wd = max((pg.get("weight_decay", 0.0) for pg in param_groups), default=0.0)
        if 0.0 < _current_max_wd < 5e-2:
            warnings.warn(
                f"Lion optimizer con weight_decay max={_current_max_wd:.0e}. "
                f"El paper recomienda wd ~10x mayor que AdamW equivalente. "
                f"Sugerencia: 1e-1 en vez de 1e-2. "
                f"No ajustamos automáticamente para respetar tu configuración — "
                f"si quieres el scaling del paper, edita train.py param_groups.",
                RuntimeWarning,
                stacklevel=2,
            )
        base_optimizer = Lion(param_groups)
        print(
            f"   Lion optimizer activo (lr_base={lr}, -15% VRAM, signed "
            f"momentum update — Chen et al. NeurIPS 2023)."
        )
    else:
        base_optimizer = optim.AdamW(param_groups)
    if use_sam:
        optimizer = SAM(base_optimizer, rho=sam_rho, adaptive=sam_adaptive)
        # `step_target` es el optimizador que el GradScaler y el scheduler manejan.
        # Con SAM, los slots internos de AdamW viven en `base_optimizer`, no en el wrapper.
        step_target = base_optimizer
        print(
            f"   SAM activo (rho={sam_rho}, adaptive={sam_adaptive}) — "
            "dos forwards por step, ~+80% tiempo."
        )
    else:
        optimizer = base_optimizer
        step_target = base_optimizer

    # WARMUP + COSINE DECAY: Calentamiento lineal (0.01x → 1x LR)
    # seguido de Cosine Annealing suave hasta el final. Evita sobrepasarse en las primeras iteraciones.
    # Multiclase necesita más warmup (3 épocas) por tener 8 clases con distribución compleja.
    warmup_epochs = 3 if mode == "multiclass" else 2
    warmup_scheduler = LinearLR(step_target, start_factor=0.01, total_iters=warmup_epochs)
    if scheduler_type == "warm_restarts":
        restart_period = max(1, (epochs - warmup_epochs) // 3)
        main_scheduler = CosineAnnealingWarmRestarts(
            step_target, T_0=restart_period, T_mult=1
        )
        print(
            f"   Scheduler: warm_restarts (T_0={restart_period}, tras warmup {warmup_epochs} épocas)"
        )
    else:
        # T_max = swa_start - warmup_epochs (no epochs - warmup_epochs). El cosine
        # debe completar su ciclo ANTES de SWA: así llega a LR≈0 justo cuando SWA
        # arranca y la transición al LR de SWA no produce un salto discontinuo.
        # Con el valor anterior el cosine seguía descendiendo durante toda la fase
        # SWA, y el hand-off a un SWALR uniforme introducía un spike/drop por
        # grupo (backbone 2.4x, head 4x).
        swa_start_tmp = int(epochs * 0.60)
        cosine_T_max = max(1, swa_start_tmp - warmup_epochs)
        main_scheduler = CosineAnnealingLR(step_target, T_max=cosine_T_max)
    scheduler = SequentialLR(
        step_target, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
    )

    # Stochastic Weight Averaging (SWA). Se construye sobre `raw_model` para que
    # SWA no envuelva un wrapper compilado.
    swa_model = AveragedModel(raw_model)
    # SWA empieza en el último 40% del entrenamiento. Más épocas de averaging →
    # más diversidad de pesos promediados → mejor generalización. Con warmup +
    # cosine el modelo converge bien antes del 60%, así que el trade-off es
    # aceptable.
    swa_start = int(epochs * 0.60)
    # SWA LR diferencial por grupo. Un SWALR genérico aplicaría swa_lr=1e-5 a
    # TODOS los param groups, destruyendo el ratio 0.1x (backbone vs head). En su
    # lugar, al entrar en SWA se fija manualmente cada grupo preservando el ratio
    # (backbone=1e-6, head=1e-5). No se usa torch.optim.swa_utils.SWALR — solo se
    # necesita un LR estable durante SWA.
    swa_lr_backbone = 1e-6
    swa_lr_head = 1e-5
    _swa_lr_applied = False

    # Precisión mixta: bfloat16 en GPUs que lo soporten NATIVAMENTE (H100/A100, sm_80+),
    # fp16 en el resto (incluyendo Turing sm_75 / RTX 2070 Super).
    # bfloat16 tiene el mismo rango dinámico que fp32 (exponente 8 bits vs fp16 5 bits),
    # eliminando riesgo de overflow en GeM y otros ops. GradScaler se desactiva con
    # enabled=False (todas las llamadas scaler.scale/step/update se vuelven no-ops,
    # sin necesidad de cambiar el training loop).
    #
    # `torch.cuda.is_bf16_supported()` puede devolver True en Turing (sm_75) por
    # emulación parcial, pero `torch._inductor` no soporta bf16 nativamente y la
    # combinación bf16 emulado + torch.compile + Model Soup/SWA llega a dejar el
    # contexto CUDA inutilizable hasta reiniciar la sesión. Por eso se exige
    # Ampere+ (sm_80) para usar bf16 nativo; en Turing se fuerza fp16 + GradScaler.
    if torch.cuda.is_available():
        _cap = torch.cuda.get_device_capability(0)
        _bf16_native = torch.cuda.is_bf16_supported() and _cap[0] >= 8
    else:
        _cap = None
        _bf16_native = False
    _use_bf16 = _bf16_native
    _amp_dtype = torch.bfloat16 if _use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=not _use_bf16)
    if _use_bf16:
        print(f"AMP: bfloat16 (sm_{_cap[0]}{_cap[1]} nativo, GradScaler desactivado)")
    elif torch.cuda.is_available():
        _bf16_flag = torch.cuda.is_bf16_supported()
        print(f"AMP: float16 + GradScaler activo (sm_{_cap[0]}{_cap[1]}, "
              f"is_bf16_supported={_bf16_flag} pero <sm_80 → bf16 emulado descartado)")
    else:
        print("AMP: float16 + GradScaler activo (sin CUDA)")

    # Umbral de detección de vanishing gradient adaptado al dtype AMP activo. Un
    # literal `1e-7` sería inservible: bajo bf16 (min normal ~1.2e-38) jamás
    # dispararía salvo con grad == 0, y bajo fp16 cae en territorio subnormal
    # (min normal ~6.1e-5). Se calibra para detectar un vanishing real sin falsos
    # positivos en la cola natural de las grad norms de backbones preentrenados.
    _vanishing_grad_threshold = 1e-5 if _use_bf16 else 1e-4

    # Se aborta cuando --use_fitnet se activa sin teacher (--distill_from) o sin
    # --mode multiclass. Sin esto, `fitnet_projection` se construye en
    # SkinCancerModel.__init__ pero nunca se invoca (`_return_inter` queda False
    # porque depende de `teacher is not None`) → no-op silencioso: los parámetros
    # del Conv2d 1×1 acumularían gradientes nulos vía concat sin uso real, y quien
    # esperaba el KD de features no obtendría nada sin diagnóstico. Es preferible
    # romper al inicio que degradar silenciosamente.
    if use_fitnet and (mode != "multiclass" or not distill_from):
        raise ValueError(
            "--use_fitnet requiere --mode multiclass Y --distill_from <teacher.pth>. "
            f"Recibido: mode={mode!r}, distill_from={distill_from!r}, use_fitnet=True. "
            "FitNets feature distillation necesita features del teacher como target del MSE; "
            "sin teacher, fitnet_projection se construye pero nunca se invoca (silent no-op). "
            "Soluciones: (a) añadir --distill_from <path al teacher binario>, o "
            "(b) eliminar --use_fitnet del comando si solo querías el teacher KD logit-only."
        )

    # Teacher para la destilación binario→multiclase.
    teacher = None
    teacher_input_size = None
    if mode == "multiclass" and distill_from:
        print(f"   Cargando teacher binario desde {distill_from} (frozen, eval)")
        teacher, teacher_input_size = load_teacher(distill_from, device)
        # Validación cross-modelo de FitNets. El Conv2d 1×1 del student
        # (`self.fitnet_projection`) proyecta student_channels → teacher_channels.
        # Si el teacher real tiene otro num_features (p. ej. ViT-L=1024 vs el
        # default 1536 de EffNet-B3), el MSE entre proyecciones de shapes
        # distintas propagaría gradientes engañosos o fallaría con un error
        # críptico. La validación estructural de architectures.py cubre tipo y
        # rango; ésta cubre el contrato cross-modelo, que solo puede verificarse
        # con ambos modelos ya cargados.
        if use_fitnet:
            _teacher_dim = getattr(teacher, "feature_dim", None)
            _student_fitnet_out = getattr(
                getattr(raw_model, "fitnet_projection", None),
                "out_channels",
                None,
            )
            if _teacher_dim is None:
                raise RuntimeError(
                    "use_fitnet=True pero el teacher cargado no expone "
                    "`feature_dim`. El teacher debe ser un `SkinCancerModel` "
                    "que exponga ese atributo. Re-generar el checkpoint del "
                    "teacher con el código actual."
                )
            if _student_fitnet_out is None:
                raise RuntimeError(
                    "use_fitnet=True pero el student no tiene "
                    "`fitnet_projection` instanciado. Revisar que el "
                    "constructor de SkinCancerModel recibió use_fitnet=True."
                )
            if _teacher_dim != _student_fitnet_out:
                raise ValueError(
                    f"dim del teacher ({_teacher_dim}) != "
                    f"fitnet_teacher_dim del student ({_student_fitnet_out}). "
                    f"El Conv2d 1×1 de FitNets proyecta student→teacher; unas "
                    f"dims inconsistentes producen un MSE entre shapes distintas. "
                    f"Pasar `--fitnet_teacher_dim {_teacher_dim}` al invocar "
                    f"train.py, o usar un teacher con feature_dim="
                    f"{_student_fitnet_out}."
                )
            print(
                f"   FitNets validado: teacher.feature_dim="
                f"{_teacher_dim} == student.fitnet_projection.out_channels"
                f"={_student_fitnet_out}"
            )

    # Self-distillation (born-again KD).
    kd_scheduler = None
    if mode == "multiclass" and use_self_distill:
        # label_smoothing=0.0 en el scheduler KD: MulticlassFocalLoss ya aplica
        # label_smoothing=0.1 al término supervisado. Si el soft target del KD lo
        # duplicara, el efecto combinado sobrepasaría el 0.1 pretendido y
        # difuminaría la señal de las clases raras. Así el soft target queda como
        # one-hot puro (tras T) y el suavizado se mantiene solo en el criterion focal.
        kd_scheduler = SelfDistillationScheduler(
            warmup_epochs=self_distill_warmup,
            kd_weight=self_distill_weight,
            temperature=self_distill_T,
            label_smoothing=0.0,
        )
        print(
            f"   Self-distillation: warmup={self_distill_warmup} ep, "
            f"weight={self_distill_weight}, T={self_distill_T}"
        )
        if self_distill_warmup >= epochs:
            raise ValueError(
                f"self_distill_warmup ({self_distill_warmup}) debe ser < epochs ({epochs}). "
                f"Con warmup >= epochs el scheduler KD nunca se activa."
            )

    # Validación de la suma de pesos del KD antes del training loop. El check en
    # tiempo de ejecución dentro de `_compute_loss` (rama `w_sup <= 0`) se
    # conserva como defensa en profundidad por-batch, pero validar aquí evita
    # desperdiciar minutos de setup + warmup del dataloader + primer forward del
    # teacher antes de fallar.
    if mode == "multiclass":
        _w_self = self_distill_weight if use_self_distill else 0.0
        _w_teacher = distill_alpha if distill_from else 0.0
        _w_fitnet = fitnet_weight if (use_fitnet and distill_from) else 0.0
        _w_total_kd = _w_self + _w_teacher + _w_fitnet
        if _w_total_kd >= 1.0:
            raise ValueError(
                f"Σ pesos KD = {_w_total_kd:.4f} >= 1.0 → "
                f"w_sup = 1 - {_w_total_kd:.4f} <= 0 (loss supervisado se "
                f"anularía o invertiría). Componentes: w_self="
                f"{_w_self:.3f} (self-distill), w_teacher={_w_teacher:.3f} "
                f"(teacher-KD), w_fitnet={_w_fitnet:.3f} (FitNets). Reducir "
                f"al menos uno hasta que la suma sea < 1.0. Recomendación "
                f"clínica: mantener Σ <= 0.5 para que el supervisado focal "
                f"siga dominando el aprendizaje."
            )

    # 4. Métricas Críticas Clínicas
    if mode == "binary":
        auc_metric = BinaryAUROC().to(device)
        # MONITORIZACIÓN con umbral estándar (0.50) para ver progreso REAL durante el entrenamiento.
        # El umbral clínico de triaje (0.20) se aplica SOLO en predict.py durante la inferencia.
        # Usar 0.20 aquí producía Recall=1.0 trivial (todo se marca positivo) → Métrica inútil.
        recall_metric = BinaryRecall(threshold=0.50).to(device)
        spec_metric = BinarySpecificity(threshold=0.50).to(device)
        # Métricas informativas @0.20: muestran el rendimiento en el umbral
        # clínico real. No afectan al early stopping ni al checkpointing — solo
        # se loguean para visibilidad. El umbral se deriva de la constante
        # canónica, no de un literal, para evitar drift si cambia.
        recall_clinical = BinaryRecall(threshold=CLINICAL_THRESHOLD_BINARY).to(device)
        spec_clinical = BinarySpecificity(threshold=CLINICAL_THRESHOLD_BINARY).to(device)
        per_class_recall = None
        per_class_auc = None
    else:
        auc_metric = MulticlassAUROC(num_classes=num_classes).to(device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average="macro").to(
            device
        )
        # Recall por clase para detectar si una clase individual colapsa: con
        # macro recall, si Dermatofibroma cae a recall=0 el macro solo baja ~12.5%.
        per_class_recall = MulticlassRecall(num_classes=num_classes, average="none").to(device)
        # Per-class AUC: detecta si una malignidad pierde discriminación mientras macro
        # AUC se mantiene alto (inflado por Nevus, clase dominante ~60% del dataset).
        # one-vs-rest AUC por clase — complementa per_class_recall con un umbral-free metric.
        per_class_auc = MulticlassAUROC(num_classes=num_classes, average="none").to(device)
        spec_metric = None
        recall_clinical = None
        spec_clinical = None

    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    save_path = f"models/checkpoints/best_{mode}_model.pth"
    save_path_swa = f"models/checkpoints/best_{mode}_model_SWA.pth"
    # El checkpoint se selecciona por AUC (gold standard clínico), no por loss:
    # se guarda el modelo con mejor ROC AUC, no el de menor entropía.
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path, mode="max")

    # Configuración de MixUp mode-aware. El modelo multiclase necesita ver batches
    # limpios (sin mezclar) para aprender las clases raras (Dermatofibroma 84 imgs,
    # Vascular Lesion ~200): con MixUp al 100% nunca vería un ejemplo puro y el
    # recall macro se estancaría (~0.53). El binario tolera MixUp continuo porque
    # dispone de 343K muestras únicas.
    if mode == "binary":
        mixup_alpha = 0.4   # Beta(0.4, 0.4): mezcla variada, dataset grande lo absorbe
        mixup_prob = 1.0    # 100%: siempre aplicar (comportamiento original, AUC=0.991)
        mixup_warmup_epochs = 0  # Siempre activo para binario
    else:
        mixup_alpha = 0.2   # Beta(0.2, 0.2): mezcla más suave, preserva señal de clases raras
        mixup_prob = 0.5    # 50%: alternar batches limpios y mixtos para aprender patrones puros
        # MixUp retardado para multiclase: con solo 84 imgs de Dermatofibroma, el
        # modelo necesita ver ejemplos puros en las primeras épocas para aprender
        # los patrones base de cada clase antes de empezar a mezclar.
        mixup_warmup_epochs = 5

    # Aviso de VRAM: la combinación SAM + teacher KD + MixUp en multiclase
    # dispara el peor caso de 4 forwards de B4@380 por step efectivo (SAM hace 2
    # forwards y cada uno dispara un `out_clean` extra cuando hay MixUp para
    # mantener la coherencia de la destilación). El pico de VRAM es ≈1.7-1.9× el
    # base porque las activations del primer forward siguen vivas hasta el
    # backward del segundo forward de SAM. No es un error: la combinación está
    # soportada (en A100 40GB es trivial); el aviso da visibilidad para Colab T4
    # (16GB), donde el margen es ajustado incluso sin SAM.
    if use_sam and teacher is not None and mode == "multiclass" and mixup_alpha > 0:
        try:
            total_mem_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if torch.cuda.is_available() else 0.0
            )
            print(
                f"[AVISO VRAM] SAM + teacher KD + MixUp activos en multiclase. "
                f"Pico estimado ≈1.9x forward base (hasta 4 fwds/step en el peor caso). "
                f"GPU detectada: {total_mem_gb:.1f} GB. "
                f"Con B4@380 y batch_size>{batch_size if batch_size <= 32 else 32}, "
                f"monitorizar `nvidia-smi` en epocas 15+ (fragmentacion progresiva)."
            )
        except Exception:
            # Nunca bloquear init del loop por un aviso informativo.
            pass

    # 5. LOOP DE ÉPOCAS
    for epoch in range(epochs):
        print(f"\n--- Época {epoch+1}/{epochs} ---")

        # === FASE ENTRENAMIENTO ===
        model.train()
        train_loss = 0.0
        train_samples = 0

        # Flag para trackear si algún batch del ciclo de gradient accumulation
        # produjo un backward válido. Sin esto, un ciclo donde TODOS los batches son NaN
        # dispararía scaler.unscale_/step sin gradientes registrados → AssertionError.
        # Además, EMA.update() corría sobre step saltado, induciendo drift mínimo.
        valid_backward_in_cycle = False

        pbar = tqdm(train_loader, desc="Entrenando")
        for batch_idx, (images, labels, meta) in enumerate(pbar):
            batch_size_current = images.size(0)
            # Transferencia Asíncrona (Acelera +10% overlaping CPU-GPU I/O con pin_memory=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)

            # Gradient Accumulation: zero_grad solo al inicio de cada ciclo de acumulación.
            # Esto simula un batch_size efectivo = batch_size * accum_steps sin VRAM extra.
            if batch_idx % accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            # ═══════════════════════════════════════════════════════════════
            # MixUp / CutMix
            # Mezcla pares de imágenes con un factor λ de la distribución Beta.
            # Aplicación probabilística mode-aware (ver configuración arriba).
            # CutMix preserva las texturas locales mejor que MixUp para dermoscopia.
            # ═══════════════════════════════════════════════════════════════
            use_mixup = (epoch >= mixup_warmup_epochs) and (random.random() < mixup_prob)

            # El estado de MixUp se calcula UNA sola vez por batch. Con SAM
            # activo, el segundo forward reutiliza exactamente este mismo estado
            # (invariante del paper de SAM: no resamplear lambda/index entre forwards).
            mixup_state = None
            if use_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                # Estrategia "dominant-instance" para la multimodalidad: se fuerza
                # lam >= 0.5 para que la imagen original domine la mezcla, de modo
                # que los metadatos categóricos (sexo, sitio) sigan siendo coherentes.
                if lam < 0.5:
                    lam = 1.0 - lam
                index = torch.randperm(batch_size_current, device=device)

                # 50/50 MixUp vs CutMix: CutMix preserva las texturas locales de la
                # lesión y MixUp suaviza las fronteras de decisión; son complementarios.
                if random.random() < 0.5:
                    # MixUp: mezcla global de píxeles
                    mixed_images = lam * images + (1 - lam) * images[index]
                else:
                    # CutMix: pega un parche rectangular de otra imagen
                    mixed_images = images.clone()
                    _, _, H, W = images.shape
                    cut_ratio = np.sqrt(1.0 - lam)
                    cut_h = max(1, int(H * cut_ratio))
                    cut_w = max(1, int(W * cut_ratio))
                    cy = np.random.randint(H)
                    cx = np.random.randint(W)
                    y1 = max(0, cy - cut_h // 2)
                    y2 = min(H, y1 + cut_h)
                    x1 = max(0, cx - cut_w // 2)
                    x2 = min(W, x1 + cut_w)
                    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
                    # Ajustar lam al área real del parche cortado
                    lam = 1 - ((y2 - y1) * (x2 - x1)) / (H * W)

                if mode == "binary":
                    labels_a = labels.float()
                    labels_b = labels[index].float()
                else:
                    labels_a = labels
                    labels_b = labels[index]
                mixup_state = {
                    "mixed_images": mixed_images,
                    "labels_a": labels_a,
                    "labels_b": labels_b,
                    "lam": lam,
                }

            # Teacher forward cacheado: el teacher está frozen y su salida solo
            # depende de los inputs, así que SAM (dos forwards del estudiante)
            # puede compartir el mismo teacher_prob sin diferencias numéricas, lo
            # que ahorra un forward del teacher por step. Si FitNets está activo se
            # usa el helper que además captura los feature maps del teacher en el
            # mismo forward.
            teacher_prob_cache = None
            teacher_feats_cache = None
            if teacher is not None and mode == "multiclass":
                if use_fitnet:
                    from src.training.distillation import precompute_teacher_prob_and_features
                    teacher_prob_cache, teacher_feats_cache = precompute_teacher_prob_and_features(
                        images, meta, teacher, teacher_input_size=teacher_input_size
                    )
                else:
                    teacher_prob_cache = precompute_teacher_prob(
                        images, meta, teacher, teacher_input_size=teacher_input_size
                    )

            def _compute_loss():
                """
                Calcula el loss completo (criterion supervisado + KD self + KD teacher + aux).

                Composición ADITIVA:
                    final = w_sup * loss_sup + w_self * kd_self + w_teacher * kd_teacher

                con `w_sup = 1 - w_self - w_teacher`. Si self-KD y teacher-KD se
                anidaran (`(1-a_t)*((1-w_s)*sup + w_s*kd_s) + a_t*kd_t`), el peso
                efectivo del supervisado bajaría a ~0.63 con ambos activos; la
                composición aditiva lo evita.

                La self-distillation también funciona con MixUp: el scheduler acepta
                targets_a/targets_b/lam y reconstruye el soft target mezclado.
                """
                # FitNets necesita las features del student sincronizadas con el
                # forward, así que se usa `return_intermediate_features` cuando está
                # activo; en caso contrario, comportamiento de referencia.
                _return_inter = use_fitnet and teacher is not None and mode == "multiclass"
                student_feats_for_fitnet = None

                if mixup_state is not None:
                    mi = mixup_state["mixed_images"]
                    la = mixup_state["labels_a"]
                    lb = mixup_state["labels_b"]
                    lm = mixup_state["lam"]
                    # SupCon NO aplica con MixUp: labels mezclados no tienen pares
                    # positivos limpios. Omitimos la rama return_features en MixUp.
                    if _return_inter:
                        out, student_feats_for_fitnet = model(
                            mi, meta, return_intermediate_features=True
                        )
                    else:
                        out = model(mi, meta)
                    if mode == "binary":
                        out = out.squeeze(1)
                    loss_sup = lm * criterion(out, la) + (1 - lm) * criterion(out, lb)
                    mixup_targets_a = la
                    mixup_targets_b = lb
                    mixup_lam = lm
                    clean_labels_for_kd = None
                else:
                    if supcon_criterion is not None:
                        if _return_inter:
                            out, projected, student_feats_for_fitnet = model(
                                images, meta,
                                return_features=True,
                                return_intermediate_features=True,
                            )
                        else:
                            out, projected = model(images, meta, return_features=True)
                        focal_term = criterion(out, labels)
                        supcon_term = supcon_criterion(projected.float(), labels)
                        loss_sup = focal_term + supcon_weight * supcon_term
                    else:
                        if _return_inter:
                            out, student_feats_for_fitnet = model(
                                images, meta, return_intermediate_features=True
                            )
                        else:
                            out = model(images, meta)
                        if mode == "binary":
                            out = out.squeeze(1)
                            loss_sup = criterion(out, labels.float())
                        else:
                            loss_sup = criterion(out, labels)
                    mixup_targets_a = None
                    mixup_targets_b = None
                    mixup_lam = 1.0
                    clean_labels_for_kd = labels

                # === Composición aditiva de pesos ===
                w_self = 0.0
                w_teacher = 0.0
                kd_self_term = None
                kd_teacher_term = None

                # Self-distillation (born-again KD) — funciona con y sin MixUp.
                if (
                    kd_scheduler is not None
                    and mode == "multiclass"
                    and kd_scheduler.is_active(epoch)
                ):
                    if mixup_state is not None:
                        kd_self_term = self_distillation_loss(
                            logits=out,
                            targets_a=mixup_targets_a,
                            targets_b=mixup_targets_b,
                            lam=mixup_lam,
                            temperature=kd_scheduler.temperature,
                            label_smoothing=kd_scheduler.label_smoothing,
                        )
                    else:
                        kd_self_term = self_distillation_loss(
                            logits=out,
                            targets_a=clean_labels_for_kd,
                            targets_b=None,
                            lam=1.0,
                            temperature=kd_scheduler.temperature,
                            label_smoothing=kd_scheduler.label_smoothing,
                        )
                    w_self = kd_scheduler.kd_weight

                # Distillation binario→multiclase con teacher frozen. Con MixUp se
                # hace un re-forward sobre las imágenes limpias para mantener la
                # fidelidad de la destilación. Esto también resuelve un problema con
                # FitNets: si `student_feats_for_fitnet` se capturara sobre las
                # imágenes mezcladas mientras `teacher_feats_cache` vive en el espacio
                # limpio (precompute sobre `images`), el MSE compararía dominios
                # distintos e inyectaría gradientes engañosos. El re-forward limpio
                # captura las features del student en el mismo dominio que el teacher,
                # sobrescribiendo el valor mezclado anterior.
                if teacher is not None and mode == "multiclass":
                    if mixup_state is not None:
                        if _return_inter:
                            out_clean, student_feats_for_fitnet = model(
                                images, meta, return_intermediate_features=True
                            )
                        else:
                            out_clean = model(images, meta)
                        kd_teacher_term = compute_distillation_loss(
                            out_clean, images, meta, teacher,
                            teacher_input_size=teacher_input_size,
                            teacher_prob_cached=teacher_prob_cache,
                        )
                    else:
                        kd_teacher_term = compute_distillation_loss(
                            out, images, meta, teacher,
                            teacher_input_size=teacher_input_size,
                            teacher_prob_cached=teacher_prob_cache,
                        )
                    w_teacher = distill_alpha

                # FitNets feature distillation: término aditivo
                # `w_fitnet * MSE(proj(student_feats), teacher_feats)`. Requiere las
                # features del student (return_intermediate_features) y las del
                # teacher (cacheadas en teacher_feats_cache).
                w_fitnet = 0.0
                fitnet_term = None
                if (
                    use_fitnet
                    and teacher is not None
                    and mode == "multiclass"
                    and student_feats_for_fitnet is not None
                    and teacher_feats_cache is not None
                ):
                    from src.training.distillation import fitnet_feature_loss
                    # La proyección vive dentro del modelo.
                    _proj = getattr(
                        model.module if hasattr(model, "module") else model,
                        "fitnet_projection",
                        None,
                    )
                    if _proj is None:
                        # Si raw_model está fuera del DataParallel, usarlo directamente.
                        _proj = getattr(raw_model, "fitnet_projection", None)
                    if _proj is None:
                        raise RuntimeError(
                            "use_fitnet=True pero `fitnet_projection` no está "
                            "instanciada en el modelo. Asegurarse de construir "
                            "SkinCancerModel con use_fitnet=True."
                        )
                    fitnet_term = fitnet_feature_loss(
                        student_feats_for_fitnet, teacher_feats_cache, _proj
                    )
                    w_fitnet = fitnet_weight

                # Sanity: asegurar w_sup > 0. Si w_self + w_teacher + w_fitnet >= 1
                # el supervisado desaparece — se aborta ante una configuración inválida.
                w_sup = 1.0 - w_self - w_teacher - w_fitnet
                if w_sup <= 0.0:
                    raise ValueError(
                        f"Composición de loss inválida: w_sup={w_sup:.3f} "
                        f"(w_self={w_self}, w_teacher={w_teacher}, w_fitnet={w_fitnet}). "
                        f"Reducir kd_scheduler.kd_weight, distill_alpha o fitnet_weight."
                    )
                loss_ = w_sup * loss_sup
                if kd_self_term is not None:
                    loss_ = loss_ + w_self * kd_self_term
                if kd_teacher_term is not None:
                    loss_ = loss_ + w_teacher * kd_teacher_term
                if fitnet_term is not None:
                    loss_ = loss_ + w_fitnet * fitnet_term

                # Balanced Softmax auxiliary (solo sin MixUp — targets duros)
                if (
                    aux_criterion is not None
                    and mode == "multiclass"
                    and mixup_state is None
                ):
                    loss_ = loss_ + balanced_softmax_weight * aux_criterion(out, labels)

                return loss_

            # === Forward 1 (normal, scaled backward) ===
            with torch.amp.autocast('cuda', dtype=_amp_dtype):
                loss = _compute_loss()

            # GUARD NaN: verificar ANTES del backward. Si el loss es NaN (overflow
            # en AMP/GeM), ejecutar el backward propagaría gradientes NaN que
            # contaminarían los pesos del modelo. No se usa `continue` (saltaría el
            # step check y perdería gradientes válidos acumulados en batches
            # anteriores del ciclo): el NaN solo salta el backward, pero el step
            # check SIEMPRE se evalúa.
            train_loss_item = loss.item()
            if not math.isnan(train_loss_item):
                # Gradient Accumulation: dividir loss por accum_steps para que el gradiente
                # acumulado sea equivalente a un batch grande.
                scaled_loss = loss / accum_steps
                scaler.scale(scaled_loss).backward()
                valid_backward_in_cycle = True

                train_loss += train_loss_item * batch_size_current
                train_samples += batch_size_current
                pbar.set_postfix({"Loss": train_loss_item})
            else:
                pbar.set_postfix({"Loss": "NaN(skip)"})

            # El step check SIEMPRE se ejecuta (fuera del guard NaN) para no perder
            # gradientes válidos acumulados en batches anteriores del ciclo. PERO,
            # si NINGÚN batch del ciclo produjo un backward válido, se salta el step
            # entero: `scaler.unscale_` sin inf checks registrados fallaría, y EMA
            # no debe actualizarse cuando los pesos no han cambiado.
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if valid_backward_in_cycle:
                    scaler.unscale_(step_target)
                    # Se usa `raw_model.parameters()` (no `model`), en paralelo a
                    # `_apply_gradient_centralization(raw_model, ...)` y a la regla
                    # general "tras compile, operar siempre sobre raw_model". El
                    # OptimizedModule delega hoy `.parameters()` a `_orig_mod`, pero
                    # un refactor interno de PyTorch podría cambiarlo sin avisar.
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
                    if use_gc:
                        _apply_gradient_centralization(raw_model, _embedding_param_ids)

                    # Monitoreo de gradientes cada 100 steps efectivos:
                    # diagnostica vanishing gradients en el backbone y desbalances
                    # backbone/head persistentes tras el warmup.
                    _effective_step = (batch_idx + 1) // accum_steps
                    if _effective_step % 100 == 0:
                        _bn, _hn = _compute_grad_norms(raw_model)
                        if _bn < _vanishing_grad_threshold and epoch >= warmup_epochs:
                            warnings.warn(
                                f"[AVISO] Época {epoch+1}, step {_effective_step}: "
                                f"backbone grad norm={_bn:.2e} < {_vanishing_grad_threshold:.0e} "
                                f"(umbral adaptado al dtype AMP). "
                                "Posible vanishing gradient — backbone efectivamente frozen.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                        # Log informativo cada 100 steps (solo en tqdm postfix, no warning)
                        pbar.set_postfix({
                            "Loss": train_loss_item,
                            "gn_bb": f"{_bn:.1e}",
                            "gn_hd": f"{_hn:.1e}",
                        })

                    # Flag de validez del step de SAM: True por defecto, pasa a
                    # False solo si hay NaN en el segundo forward. Con SAM
                    # desactivado nunca cambia, así que el path es idéntico.
                    sam_step_valid = True

                    if use_sam:
                        # === SAM first step: w ← w + ε·∇L(w)/‖∇L(w)‖ ===
                        # Los gradientes ya están en fp32 (unscaled), norma fiable.
                        # `first_step` retorna bool: si `grad_norm` no es finita
                        # (NaN/Inf), aborta SIN perturbar los pesos — `NaN - NaN =
                        # NaN` haría el rollback matemáticamente imposible. En ese
                        # caso se salta el segundo forward y se marca
                        # `sam_step_valid=False` para omitir scaler.step/ema.update.
                        first_applied = optimizer.first_step(zero_grad=True)
                        if not first_applied:
                            sam_step_valid = False
                            warnings.warn(
                                "grad_norm NaN/Inf antes del SAM first_step. "
                                "Saltando el ciclo completo para evitar corromper los pesos.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            pbar.set_postfix({"Loss": "SAM_nan_grad(skip)"})

                        # Reset del per-optimizer state del scaler: PyTorch prohíbe
                        # llamar `unscale_` dos veces en el mismo step sin un step()
                        # entre medias. Al borrar la entrada, el defaultdict crea un
                        # estado fresco READY en el próximo acceso (técnica estándar
                        # en el ecosistema SAM+AMP, p. ej. timm, fastai).
                        # `_per_optimizer_states` es un atributo privado de
                        # `GradScaler`; un refactor interno de PyTorch podría
                        # renombrarlo o eliminarlo, así que el acceso se envuelve en
                        # try/except AttributeError: si la API desaparece, el scaler
                        # conservará el estado stale del primer unscale y, en el peor
                        # caso, `scaler.step` del segundo forward saltará con "already
                        # unscaled" — no corrompe pesos, solo aborta el step SAM.
                        try:
                            if id(step_target) in scaler._per_optimizer_states:
                                del scaler._per_optimizer_states[id(step_target)]
                        except AttributeError:
                            pass

                        # === Forward 2 (pesos perturbados, MISMO mixup_state) ===
                        # Solo si first_step aplicó la perturbación. Si grad_norm era
                        # NaN/Inf, los pesos NO se mutaron → no hay punto adversarial
                        # que evaluar y second_step es un no-op automático
                        # (self.state vacío).
                        if first_applied:
                            with torch.amp.autocast('cuda', dtype=_amp_dtype):
                                loss_adv = _compute_loss()
                            loss_adv_item = loss_adv.item()
                            if not math.isnan(loss_adv_item):
                                # NO dividir por accum_steps: el segundo forward de
                                # SAM opera sobre un solo batch tras
                                # first_step(zero_grad=True) — no hay acumulación.
                                scaler.scale(loss_adv).backward()
                                scaler.unscale_(step_target)
                                # `raw_model.parameters()` (no `model`).
                                torch.nn.utils.clip_grad_norm_(
                                    raw_model.parameters(), max_norm=1.0
                                )
                                if use_gc:
                                    _apply_gradient_centralization(raw_model, _embedding_param_ids)
                            else:
                                # loss_adv NaN → se hace el rollback (second_step
                                # abajo) pero se omiten scaler.step/update y ema.update.
                                sam_step_valid = False
                                pbar.set_postfix({"Loss": "SAM_NaN(skip)"})

                            # Rollback SIEMPRE tras un first_step exitoso, incluso si
                            # loss_adv fue NaN: no se puede dejar el modelo en w_adv
                            # entre ciclos. Si first_step falló, no hubo perturbación
                            # que revertir y second_step es un no-op.
                            optimizer.second_step(zero_grad=False)

                    if sam_step_valid:
                        scaler.step(step_target)
                        scaler.update()
                        # Actualizar EMA tras cada step efectivo del optimizador.
                        # Con SAM se hace DESPUÉS del second_step y del scaler.step
                        # final — nunca entre first_step y second_step (pesos
                        # inestables). Se usa raw_model directamente (extraído antes
                        # de compile).
                        ema.update(raw_model)
                    else:
                        # No scaler.step/update, no ema.update. Limpiar grads
                        # residuales defensivamente para mantener la invariante
                        # "entre ciclos, grads=None" (first_step ya hizo
                        # zero_grad pero second_step no toca los grads).
                        optimizer.zero_grad(set_to_none=True)
                valid_backward_in_cycle = False

        # Se aborta si toda la época fue NaN. Un `max(..., 1)` en el denominador
        # daría train_loss=0.0 y enmascararía un colapso total del modelo. Este
        # caso requiere atención manual inmediata (reducir LR, revisar gradientes,
        # relanzar con otro seed).
        if train_samples == 0:
            raise RuntimeError(
                f"Época {epoch+1}: todos los batches produjeron NaN "
                f"(train_samples=0). Colapso total del modelo — revisar "
                f"LR, scaler, gradient clipping antes de relanzar."
            )
        train_loss = train_loss / train_samples

        # GeM p tracking: el parámetro aprendible de GeM (inicializado en 3.0, clamp [1,6])
        # indica cómo pondera el pooling entre average (p=1) y max (p→∞).
        # Un drift excesivo (< 1.5 o > 5.5) sugiere inestabilidad en el entrenamiento.
        if hasattr(raw_model, 'gem_pool') and hasattr(raw_model.gem_pool, 'p'):
            gem_p = raw_model.gem_pool.p.item()
            print(f"   GeM p = {gem_p:.4f}")
            if gem_p < 1.5 or gem_p > 5.5:
                warnings.warn(
                    f"[AVISO] GeM p={gem_p:.4f} fuera del rango típico [1.5, 5.5]. "
                    "Posible inestabilidad — verificar LR del backbone.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # === FASE VALIDACIÓN ===
        # Se fuerza la limpieza de VRAM antes de la validación para reducir la
        # fragmentación de memoria, que provoca crashes de CUDA Out of Memory en
        # las épocas tardías (15+) en Colab.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        model.eval()
        val_loss = 0.0
        val_samples = 0
        # Se acumulan probs/labels solo si la métrica de selección es pAUC (ahorra memoria).
        _pauc_probs = [] if _use_pauc_selection else None
        _pauc_labels = [] if _use_pauc_selection else None

        # Resetear métricas
        auc_metric.reset()
        recall_metric.reset()
        if spec_metric is not None:
            spec_metric.reset()
        if recall_clinical is not None:
            recall_clinical.reset()
            spec_clinical.reset()
        if per_class_recall is not None:
            per_class_recall.reset()
        if per_class_auc is not None:
            per_class_auc.reset()

        with torch.no_grad():
            for images, labels, meta in tqdm(val_loader, desc="Validando"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                meta = meta.to(device, non_blocking=True)

                # TTA determinista cuando tta_val_views > 1. Con views==1 el helper
                # hace autocast + forward único (comportamiento de referencia); con
                # views>=4 promedia logits en float32, alineando el early stopping
                # con el régimen de inferencia (predict.py usa TTA-6). Los criterios
                # (FocalLoss, MulticlassFocalLoss, SeesawLoss...) tienen su propio
                # guard de autocast, así que funcionan con `outputs` en amp_dtype
                # (views=1) o en float32 (views>=4).
                outputs = _tta_forward_views(model, images, meta, tta_val_views, _amp_dtype)
                if mode == "binary":
                    outputs = outputs.squeeze(1)
                    labels_float = labels.float()
                else:
                    labels_float = labels

                loss = criterion(outputs, labels_float)

                val_loss_item = loss.item()
                # GUARD: un solo batch con NaN contaminaría toda la época
                # (NaN + x = NaN); puede ocurrir por overflow en GeM/AMP. Se salta
                # el batch corrupto.
                if not math.isnan(val_loss_item):
                    val_loss += val_loss_item * images.size(0)
                    val_samples += images.size(0)

                    # Actualizar métricas. Se fuerza float32 antes de sigmoid/softmax:
                    # bajo autocast, `outputs` puede ser fp16 (max=65504), y un
                    # softmax fp16 de 8 clases pierde resolución en distribuciones
                    # suaves, contaminando el AUC y el Recall acumulados durante la
                    # validación. El guard NaN anterior garantiza que sigmoid/softmax
                    # no reciban NaN (que envenenaría el acumulador de AUC).
                    if mode == "binary":
                        preds = torch.sigmoid(outputs.float())
                    else:
                        preds = torch.softmax(outputs.float(), dim=1)

                    auc_metric.update(preds, labels)
                    recall_metric.update(preds, labels)
                    if spec_metric is not None:
                        spec_metric.update(preds, labels)
                    if recall_clinical is not None:
                        recall_clinical.update(preds, labels)
                        spec_clinical.update(preds, labels)
                    if per_class_recall is not None:
                        per_class_recall.update(preds, labels)
                    if per_class_auc is not None:
                        per_class_auc.update(preds, labels)
                    # Acumular para la partial AUC clínica. Se acumulan tensors en
                    # GPU (no numpy por-batch) para evitar N sincronizaciones
                    # GPU→CPU en el hot path de validación; al final del loop se hace
                    # una sola transferencia con `torch.cat(...).cpu().numpy()`
                    # (~5-10% menos de tiempo de validación con 8K-40K samples).
                    if _pauc_probs is not None:
                        _pauc_probs.append(preds.detach())
                        _pauc_labels.append(labels.detach())

        # Se aborta si TODOS los batches de validación fueron NaN (mismo principio
        # que en el training). Sin esto, un val_loss=0.0 silencioso y un val_auc
        # NaN/0 engañarían al early stopping y a la selección de checkpoint; es
        # preferible abortar que certificar un modelo clínico sobre métricas vacías.
        if val_samples == 0:
            raise RuntimeError(
                f"Validación sin samples válidos en época {epoch+1}. "
                "TODOS los batches produjeron NaN — revisar LR, scaler, gradient "
                "clipping, o data loader. Abortando training para evitar "
                "checkpoint/métricas clínicas corruptas."
            )
        val_loss = val_loss / val_samples
        val_auc = auc_metric.compute().item()
        val_recall = recall_metric.compute().item()
        # Init defensivo de `clin_recall` (recall@0.20): se puebla en binario
        # dentro de la rama `spec_metric is not None`. Inicializarlo a None
        # garantiza que el guardrail no lance NameError si esa rama no corre (en
        # multiclase, donde el guardrail está desactivado).
        clin_recall = None

        # Calcular la pAUC y elegir la métrica de selección. Por defecto se usa
        # val_auc (macro AUC de torchmetrics); con pAUC activa y modo binario se
        # usa `partial_auc(...)` de `src.utils.evaluation` (wrapper de sklearn con
        # normalización McClish).
        val_selection = val_auc
        val_pauc = None
        if _use_pauc_selection:
            # Se aborta si la métrica de selección falla. Caer silenciosamente a
            # macro AUC mezclaría escalas en `EarlyStopping.best_saved_score` (pAUC
            # ~0.5-1.0 vs AUC 0-1) y corrompería la selección de checkpoint. El
            # flag es una elección consciente: si se activa y falla, es preferible
            # fallar de forma dura a degradar.
            if not _pauc_probs or len(_pauc_probs) == 0:
                raise RuntimeError(
                    f"selection_metric='{selection_metric}' activo pero sin probs "
                    f"acumuladas en época {epoch+1}. val_samples={val_samples}. "
                    "Revisar val_loader / NaN guard / sklearn disponible."
                )
            # Una sola sincronización GPU→CPU al final (los tensors se acumularon
            # en GPU durante el loop), con ravel en numpy tras el cat.
            _probs_cat = torch.cat(_pauc_probs, dim=0).cpu().numpy().ravel()
            _labels_cat = torch.cat(_pauc_labels, dim=0).cpu().numpy().ravel()
            # Helper centralizado (spec≥target o sens≥target según
            # selection_metric). Valor McClish-normalizado [0.5,1.0] — misma escala.
            val_pauc = _compute_pauc_selection(
                _labels_cat, _probs_cat, selection_metric,
                pauc_spec_target, pauc_sens_target,
            )
            if val_pauc is None or not math.isfinite(val_pauc):
                raise RuntimeError(
                    f"pAUC ('{selection_metric}') retornó {val_pauc!r} en época "
                    f"{epoch+1}. Posibles causas: sklearn no instalado, val set sin "
                    f"positivos, target demasiado extremo para "
                    f"n_samples={len(_labels_cat)}. Un fallback silencioso a macro "
                    "AUC corrompería best_saved_score mezclando escalas (~0.5-1.0 "
                    "pAUC vs 0-1 AUC). Abortando el training."
                )
            val_selection = val_pauc
            print(
                f"   {_selection_metric_label(selection_metric)} = "
                f"{val_pauc:.4f} (métrica de selección)"
            )
        if spec_metric is not None:
            val_spec = spec_metric.compute().item()
            print(
                f"Estadísticas -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Recall: {val_recall:.4f} | Val Spec: {val_spec:.4f}"
            )
            # Métricas informacionales en umbral clínico real (0.20)
            if recall_clinical is not None:
                clin_recall = recall_clinical.compute().item()
                clin_spec = spec_clinical.compute().item()
                print(
                    f"   @Umbral Clínico 0.20 -> Recall: {clin_recall:.4f} | Spec: {clin_spec:.4f}"
                )
        else:
            print(
                f"Estadísticas -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val Recall(macro): {val_recall:.4f}"
            )
            # default True — pasa a False si alguna clase maligna no supera el
            # health check. En binario (rama if de arriba), class_healthy no se usa
            # (block_save hace short-circuit con mode != "multiclass").
            class_healthy = True
            if per_class_recall is not None:
                class_recalls = per_class_recall.compute()
                class_names = [MULTICLASS_INDEX_MAP[i] for i in range(len(MULTICLASS_INDEX_MAP))]
                recall_str = " | ".join([f"{n}:{r:.2f}" for n, r in zip(class_names, class_recalls)])
                print(f"   Per-class Recall: {recall_str}")
                # Per-class AUC: detecta si una malignidad pierde discriminación
                # mientras macro AUC se mantiene alto (inflado por Nevus, clase dominante).
                _class_aucs = None
                if per_class_auc is not None:
                    _class_aucs = per_class_auc.compute()
                    auc_str = " | ".join([f"{n}:{a:.3f}" for n, a in zip(class_names, _class_aucs)])
                    print(f"   Per-class AUC:    {auc_str}")
                # Detección de colapso de clase y AUC bajo en malignas
                class_healthy = _check_class_health(class_recalls, _class_aucs, class_names, context=f"Época {epoch+1}: ")

        # Actualización de Learning Rate
        if epoch >= swa_start:
            # Se usa raw_model para que SWA promedie pesos limpios.
            swa_model.update_parameters(raw_model)
            # Fijar el SWA LR por grupo una sola vez al entrar en SWA.
            # param_groups: [backbone_decay, backbone_no_decay, head_decay, head_no_decay]
            if not _swa_lr_applied:
                for _pg_idx, _pg in enumerate(step_target.param_groups):
                    _pg["lr"] = swa_lr_backbone if _pg_idx < 2 else swa_lr_head
                _swa_lr_applied = True
                print(
                    f"   SWA LR diferencial aplicado: backbone={swa_lr_backbone}, head={swa_lr_head}"
                )
        else:
            scheduler.step()

        # Early Stopping basado en la métrica de selección (AUC/pAUC). Solo se
        # activa ANTES de SWA: durante SWA el modelo promedia pesos activamente e
        # interrumpirlo perdería el beneficio; además, llamar a early_stopping()
        # durante SWA sobreescribiría el mejor checkpoint pre-SWA si la métrica
        # mejorase ligeramente con el lr bajo de SWA.
        if epoch < swa_start:
            # En multiclase se bloquea el guardado del checkpoint si alguna clase
            # maligna (Melanoma, BCC, SCC) tiene AUC < 0.80 o recall < 0.30 (el
            # macro AUC puede ser 0.98, inflado por Nevus, con melanoma-AUC en
            # 0.85). En binario se bloquea si recall@0.20 < suelo clínico (red de
            # seguridad de sensibilidad); `clin_recall` se computó en
            # la rama `spec_metric is not None`.
            _recall_block = _binary_recall_floor_block(
                mode, clin_recall, recall_floor_binary
            )
            if _recall_block:
                print(
                    f"   recall@{CLINICAL_THRESHOLD_BINARY:.2f}="
                    f"{clin_recall:.4f} < suelo {recall_floor_binary:.2f} → "
                    f"save del checkpoint BLOQUEADO (sensibilidad insuficiente "
                    f"para despliegue clínico)."
                )
            block_save = ((mode == "multiclass") and not class_healthy) or _recall_block
            # Se pasa raw_model (no `model`): si torch.compile envolvió el modelo,
            # su state_dict() incluiría el prefijo `_orig_mod.` y dejaría
            # `best_{mode}_model.pth` con claves incompatibles con el engine de
            # inferencia (early_stopping.py añade además su propia defensa). Se pasa
            # `val_selection` (pAUC o AUC según el flag).
            early_stopping(val_selection, raw_model, save_blocked=block_save)
            if early_stopping.early_stop:
                print(
                    f"Early Stopping activado en la época {epoch+1}. Modelo guardado en: {save_path}"
                )
                break

    # 6. Finalización SWA (recalibración de Batch Norm)
    # Solo se guarda SWA si realmente fue actualizado: si el Early Stopping paró
    # antes de swa_start, swa_model contendría pesos del epoch 0 (deep copy sin
    # entrenar) y guardarlos sería un error grave.
    swa_was_active = swa_model.n_averaged.item() > 0

    if swa_was_active:
        print("Actualizando Batch Normalization para el modelo promediado (SWA)...")
        # Fijar el device antes de la actualización de SWA.
        swa_model.to(device)
        # Recalibración manual de la BN de SWA (compatible con la entrada
        # multimodal). torch.optim.swa_utils.update_bn asume un único input y
        # fallaría con un "Shape Mismatch" al final del entrenamiento; aquí la
        # calibración estocástica de BatchNorm se hace a mano inyectando también
        # los metadatos.
        momenta = {}
        for module in swa_model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
        
        if momenta:
            swa_model.train()
            # Desactivar Dropout durante la recalibración de la BN de SWA.
            # swa_model.train() activa el modo training de BatchNorm (necesario)
            # pero también el Dropout (no deseado): con Dropout activo,
            # Var(X * mask/p) > Var(X), inflando systematicamente running_var.
            # Poner los módulos Dropout en eval() desactiva el masking estocástico
            # sin afectar al modo training de BN, y no necesita restauración porque
            # se llama a swa_model.eval() al terminar la recalibración.
            #
            # El isinstance se extiende a `timm.layers.DropPath`, que hereda
            # directamente de nn.Module (no de nn.Dropout/Dropout2d ni de
            # _DropoutNd): EffNet-B3 con drop_path_rate=0.2 tiene 25 instancias de
            # DropPath que un check restringido a nn.Dropout no capturaría, dejando
            # el masking estocástico activo durante la recalibración e inflando
            # running_var. Se detecta por nombre (`type(m).__name__ == "DropPath"`)
            # porque DropPath no expone una clase base importable canónica desde
            # timm; el nombre es estable en las versiones soportadas (>=0.9, <2.0).
            # Afecta a EffNet-B3/B4 y DINOv2 ViT-L/B (drop_path_rate=0.2 por defecto).
            for module in swa_model.modules():
                if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)) \
                        or type(module).__name__ == "DropPath":
                    module.eval()
            for module in momenta.keys():
                module.momentum = None
                module.num_batches_tracked *= 0
                
            # Guard NaN en la recalibración de la BN de SWA. Con momentum=None
            # (media acumulativa), un solo batch NaN corrompería running_mean y
            # running_var de forma permanente. Se guarda un snapshot de las stats
            # de BN antes de cada batch y se restauran si el output tiene NaN.
            with torch.no_grad():
                for images, _, meta in tqdm(train_loader, desc="Calibrando SWA BN"):
                    images = images.to(device, non_blocking=True)
                    meta = meta.to(device, non_blocking=True)

                    # Snapshot BN stats antes del forward (NaN guard)
                    bn_snapshots = {}
                    for name, module in swa_model.named_modules():
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                            bn_snapshots[name] = (
                                module.running_mean.clone(),
                                module.running_var.clone(),
                                module.num_batches_tracked.clone(),
                            )

                    with torch.amp.autocast('cuda', dtype=_amp_dtype):
                        output = swa_model(images, meta)

                    # Si output tiene NaN, este batch corrompió BN stats → restaurar
                    if torch.isnan(output).any():
                        for name, module in swa_model.named_modules():
                            if name in bn_snapshots:
                                mean, var, tracked = bn_snapshots[name]
                                module.running_mean.copy_(mean)
                                module.running_var.copy_(var)
                                module.num_batches_tracked.copy_(tracked)
                        # Se emite con warnings.warn (capturable por el logging
                        # estructurado del servicio): un batch NaN durante la
                        # recalibración de BN es un evento de auditoría relevante,
                        # señal de inestabilidad del modelo.
                        warnings.warn(
                            "SWA BN: Batch con NaN detectado y descartado (stats restaurados).",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        
            for bn_module in momenta.keys():
                bn_module.momentum = momenta[bn_module]
        
        # --- Validación SWA frente al mejor modelo estándar ---
        # Se compara el AUC del modelo SWA con el del mejor checkpoint para decidir
        # cuál usar en producción.
        print("Evaluando SWA en validación para comparar con el mejor modelo estándar...")
        swa_model.eval()
        auc_metric.reset()
        recall_metric.reset()
        if spec_metric is not None:
            spec_metric.reset()
        if per_class_recall is not None:
            per_class_recall.reset()
        if per_class_auc is not None:
            per_class_auc.reset()
        # Resetear las métricas clínicas @0.20 para la validación SWA: dan
        # visibilidad del rendimiento en el umbral de despliegue real (sin ellas,
        # no se sabría si SWA mantiene el recall clínico).
        if recall_clinical is not None:
            recall_clinical.reset()
            spec_clinical.reset()

        swa_val_samples = 0
        # Acumular para la pAUC del SWA (solo si la métrica de selección es pAUC).
        _swa_pauc_probs = [] if _use_pauc_selection else None
        _swa_pauc_labels = [] if _use_pauc_selection else None
        with torch.no_grad():
            for images, labels, meta in tqdm(val_loader, desc="Validando SWA"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                meta = meta.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=_amp_dtype):
                    outputs = swa_model(images, meta)
                    if mode == "binary":
                        outputs = outputs.squeeze(1)
                        loss = criterion(outputs, labels.float())
                    else:
                        loss = criterion(outputs, labels)

                # Guard NaN: misma protección que en la validación normal. Sin él,
                # un batch NaN contaminaría el acumulador de AUC/Recall del SWA.
                val_loss_item = loss.item()
                if not math.isnan(val_loss_item):
                    swa_val_samples += images.size(0)
                    if mode == "binary":
                        preds = torch.sigmoid(outputs.float())
                    else:
                        preds = torch.softmax(outputs.float(), dim=1)
                    auc_metric.update(preds, labels)
                    recall_metric.update(preds, labels)
                    if spec_metric is not None:
                        spec_metric.update(preds, labels)
                    if per_class_recall is not None:
                        per_class_recall.update(preds, labels)
                    if per_class_auc is not None:
                        per_class_auc.update(preds, labels)
                    if recall_clinical is not None:
                        recall_clinical.update(preds, labels)
                        spec_clinical.update(preds, labels)
                    # Acumular tensors en GPU, con una sola sincronización al final.
                    if _swa_pauc_probs is not None:
                        _swa_pauc_probs.append(preds.detach())
                        _swa_pauc_labels.append(labels.detach())

        # Se aborta si TODOS los batches de la evaluación SWA fueron NaN.
        if swa_val_samples == 0:
            raise RuntimeError(
                "Evaluación SWA sin samples válidos. TODOS los batches "
                "produjeron NaN — el modelo SWA está corrupto. Abortando para "
                "evitar guardar un checkpoint SWA degenerado."
            )

        swa_auc = auc_metric.compute().item()
        swa_recall = recall_metric.compute().item()
        # pAUC del SWA si la métrica de selección es pAUC (se aborta si falla).
        swa_pauc = None
        if _use_pauc_selection:
            if not _swa_pauc_probs or len(_swa_pauc_probs) == 0:
                raise RuntimeError(
                    f"selection_metric='{selection_metric}' pero sin probs SWA "
                    "acumuladas. Revisar el flujo de _swa_pauc_probs."
                )
            # Una sola sincronización post-loop (tensors GPU acumulados).
            _sp = torch.cat(_swa_pauc_probs, dim=0).cpu().numpy().ravel()
            _sl = torch.cat(_swa_pauc_labels, dim=0).cpu().numpy().ravel()
            # Helper centralizado (spec o sens según selection_metric).
            swa_pauc = _compute_pauc_selection(
                _sl, _sp, selection_metric, pauc_spec_target, pauc_sens_target,
            )
            if swa_pauc is None or not math.isfinite(swa_pauc):
                raise RuntimeError(
                    f"pAUC SWA ('{selection_metric}') retornó {swa_pauc!r}. "
                    "Se aborta para evitar mezclar escalas en el comparador."
                )
        swa_selection = swa_pauc if swa_pauc is not None else swa_auc
        swa_log = f"   SWA Model     -> AUC: {swa_auc:.4f} | Recall: {swa_recall:.4f}"
        if swa_pauc is not None:
            swa_log += f" | pAUC: {swa_pauc:.4f}"
        if spec_metric is not None:
            swa_spec = spec_metric.compute().item()
            swa_log += f" | Spec@0.50: {swa_spec:.4f}"
        print(swa_log)
        # `best_for_comparison` refleja la métrica del checkpoint REAL en disco,
        # que puede diferir del "mejor visto" si hubo epochs bloqueadas por
        # class_health. Comparar contra el mejor visto podría rechazar un SWA que
        # en realidad supera al modelo desplegado.
        _best_saved = early_stopping.best_for_comparison
        print(f"   Best Standard -> AUC: {_best_saved:.4f}")
        swa_clin_recall = None  # init defensivo (binario lo puebla abajo)
        if recall_clinical is not None:
            swa_clin_recall = recall_clinical.compute().item()
            swa_clin_spec = spec_clinical.compute().item()
            print(f"   SWA @Umbral Clinico 0.20 -> Recall: {swa_clin_recall:.4f} | Spec: {swa_clin_spec:.4f}")
        swa_class_healthy = True
        if per_class_recall is not None:
            class_recalls = per_class_recall.compute()
            class_names = [MULTICLASS_INDEX_MAP[i] for i in range(len(MULTICLASS_INDEX_MAP))]
            recall_str = " | ".join([f"{n}:{r:.2f}" for n, r in zip(class_names, class_recalls)])
            print(f"   SWA Per-class Recall: {recall_str}")
            _class_aucs = None
            if per_class_auc is not None:
                _class_aucs = per_class_auc.compute()
                auc_str = " | ".join([f"{n}:{a:.3f}" for n, a in zip(class_names, _class_aucs)])
                print(f"   SWA Per-class AUC:    {auc_str}")
            swa_class_healthy = _check_class_health(class_recalls, _class_aucs, class_names, context="SWA: ")
        # Comparación contra `best_for_comparison` (checkpoint en disco) usando la
        # métrica de selección (pAUC si está activa).
        if _best_saved is None or swa_selection > _best_saved:
            print("   SWA GANA: el modelo SWA supera al mejor checkpoint estándar.")
        else:
            print("   ESTÁNDAR GANA: el mejor checkpoint estándar supera al SWA.")

        # Desenvolver AveragedModel para guardar pesos limpios. swa_model.module
        # es raw_model (sin DataParallel ni compile), así que swa_to_save es el
        # modelo core directamente.
        core_swa = swa_model.module
        swa_to_save = core_swa.module if isinstance(core_swa, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else core_swa
        # Defensa adicional: desenvolver el OptimizedModule de torch.compile.
        swa_to_save = getattr(swa_to_save, "_orig_mod", swa_to_save)

        # El checkpoint SWA se bloquea por class_health igual que el `best_*`: sin
        # este guard, el checkpoint de "producción robusta" podría guardarse con
        # clases malignas colapsadas. Se aplica el mismo criterio que en
        # early_stopping, conservando un baseline forzado si el check nunca pasó.
        # En binario se bloquea además por recall@0.20 (suelo clínico): el operador
        # puede desplegar SWA, así que aplica el mismo guardrail de sensibilidad
        # que el best checkpoint.
        _swa_recall_block = _binary_recall_floor_block(
            mode, swa_clin_recall, recall_floor_binary
        )
        _swa_block = ((mode == "multiclass") and not swa_class_healthy) or _swa_recall_block
        if (not _swa_block) or (not Path(save_path_swa).exists()):
            if _swa_block:
                _swa_reason = (
                    f"recall@{CLINICAL_THRESHOLD_BINARY:.2f}="
                    f"{swa_clin_recall:.4f} < suelo {recall_floor_binary:.2f}"
                    if _swa_recall_block
                    else "las clases malignas no pasan el health check"
                )
                warnings.warn(
                    f"FORCED BASELINE SWA: {_swa_reason}, "
                    "pero no hay SWA previo → guardando baseline forzado.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            # Guardado atómico + comprobación de finitud (anti-corrupción en Colab).
            _save_checkpoint_safe(swa_to_save.state_dict(), save_path_swa, "SWA")
            print(f"Modelo SWA guardado en: {save_path_swa}")
            # Fairness report específico del SWA (no del best): es el registro
            # del checkpoint realmente guardado, el JSON a leer si se despliega SWA
            # (no `fairness_report_{mode}.json`, que corresponde al best estándar).
            try:
                _run_fairness_report(
                    swa_to_save, val_loader, mode, device,
                    amp_dtype=_amp_dtype,
                    checkpoint_source="swa",
                    min_positive=int(min_positive),
                )
            except Exception as _fr_exc:
                warnings.warn(
                    f"Fairness report SWA falló: {_fr_exc}. "
                    "Audit trail SWA incompleto; revisar antes del despliegue.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            # SPC específico del SWA, en paralelo al fairness report del mismo
            # checkpoint. El helper maneja los fallos con degradación graciosa.
            _run_selective_prediction_curve(
                swa_to_save, val_loader, mode, device,
                save_dir=str(Path(save_path).parent),
                amp_dtype=_amp_dtype, checkpoint_source="swa",
            )
        else:
            _swa_blk_reason = (
                f"recall@{CLINICAL_THRESHOLD_BINARY:.2f}={swa_clin_recall:.4f} "
                f"< suelo {recall_floor_binary:.2f}"
                if _swa_recall_block
                else "clases malignas con AUC/recall crítico"
            )
            warnings.warn(
                f"SWA CHECKPOINT BLOCKED: {_swa_blk_reason}. "
                f"SWA AUC={swa_auc:.4f} NO guardado (preservado el SWA previo).",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        # SWA nunca se activó (Early Stopping antes de swa_start). Se copia el
        # mejor modelo estándar como SWA para evitar confusión.
        import shutil
        if Path(save_path).exists():
            shutil.copy2(save_path, save_path_swa)
            # Se avisa con warnings.warn (capturable): el operador necesita saber,
            # al decidir el despliegue, que el checkpoint SWA es una copia del best
            # estándar y no un SWA realmente promediado.
            warnings.warn(
                f"SWA no se activó (Early Stopping antes de la época {swa_start+1}). "
                f"Copiando el mejor modelo estándar como SWA → {save_path_swa}",
                RuntimeWarning,
                stacklevel=2,
            )
        print(f"Modelo SWA guardado en: {save_path_swa}")

    # --- EVALUACIÓN EMA (Exponential Moving Average) ---
    # EMA no necesita recalibración de BN (los buffers se copian del modelo de training en cada step).
    print("Evaluando EMA en validación para comparar con el mejor modelo estándar y con SWA...")
    ema.module.to(device)
    ema.module.eval()
    auc_metric.reset()
    recall_metric.reset()
    if spec_metric is not None:
        spec_metric.reset()
    if per_class_recall is not None:
        per_class_recall.reset()
    if per_class_auc is not None:
        per_class_auc.reset()
    if recall_clinical is not None:
        recall_clinical.reset()
        spec_clinical.reset()

    ema_val_samples = 0
    # Acumular para la pAUC del EMA (solo si la métrica de selección es pAUC).
    _ema_pauc_probs = [] if _use_pauc_selection else None
    _ema_pauc_labels = [] if _use_pauc_selection else None
    with torch.no_grad():
        for images, labels, meta in tqdm(val_loader, desc="Validando EMA"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=_amp_dtype):
                outputs = ema.module(images, meta)
                if mode == "binary":
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, labels.float())
                else:
                    loss = criterion(outputs, labels)

            # Guard NaN: misma protección que en la validación normal y en SWA.
            val_loss_item = loss.item()
            if not math.isnan(val_loss_item):
                ema_val_samples += images.size(0)
                if mode == "binary":
                    preds = torch.sigmoid(outputs.float())
                else:
                    preds = torch.softmax(outputs.float(), dim=1)
                auc_metric.update(preds, labels)
                recall_metric.update(preds, labels)
                if spec_metric is not None:
                    spec_metric.update(preds, labels)
                if per_class_recall is not None:
                    per_class_recall.update(preds, labels)
                if per_class_auc is not None:
                    per_class_auc.update(preds, labels)
                if recall_clinical is not None:
                    recall_clinical.update(preds, labels)
                    spec_clinical.update(preds, labels)
                # Acumular tensors en GPU, con una sola sincronización al final.
                if _ema_pauc_probs is not None:
                    _ema_pauc_probs.append(preds.detach())
                    _ema_pauc_labels.append(labels.detach())

    # Se aborta si TODOS los batches de la evaluación EMA fueron NaN.
    if ema_val_samples == 0:
        raise RuntimeError(
            "Evaluación EMA sin samples válidos. TODOS los batches "
            "produjeron NaN — el modelo EMA está corrupto. Abortando para "
            "evitar guardar un checkpoint EMA degenerado."
        )

    ema_auc = auc_metric.compute().item()
    ema_recall = recall_metric.compute().item()
    # pAUC del EMA si la métrica de selección es pAUC (se aborta si falla).
    ema_pauc = None
    if _use_pauc_selection:
        if not _ema_pauc_probs or len(_ema_pauc_probs) == 0:
            raise RuntimeError(
                f"selection_metric='{selection_metric}' pero sin probs EMA "
                "acumuladas. Revisar el flujo de _ema_pauc_probs."
            )
        # Una sola sincronización GPU→CPU al final (tensors acumulados en GPU
        # durante el loop EMA), igual que en los loops de val y SWA.
        _ep = torch.cat(_ema_pauc_probs, dim=0).cpu().numpy().ravel()
        _el = torch.cat(_ema_pauc_labels, dim=0).cpu().numpy().ravel()
        # Helper centralizado (spec o sens según selection_metric).
        ema_pauc = _compute_pauc_selection(
            _el, _ep, selection_metric, pauc_spec_target, pauc_sens_target,
        )
        if ema_pauc is None or not math.isfinite(ema_pauc):
            raise RuntimeError(
                f"pAUC EMA ('{selection_metric}') retornó {ema_pauc!r}. "
                "Se aborta para evitar mezclar escalas en el comparador."
            )
    ema_selection = ema_pauc if ema_pauc is not None else ema_auc
    ema_log = f"   EMA Model     -> AUC: {ema_auc:.4f} | Recall: {ema_recall:.4f}"
    if ema_pauc is not None:
        ema_log += f" | pAUC: {ema_pauc:.4f}"
    if spec_metric is not None:
        ema_spec = spec_metric.compute().item()
        ema_log += f" | Spec@0.50: {ema_spec:.4f}"
    print(ema_log)
    # Usar `best_for_comparison` (la métrica del checkpoint en disco, no la del
    # mejor visto).
    _best_saved_ema = early_stopping.best_for_comparison
    print(f"   Best Standard -> AUC: {_best_saved_ema:.4f}")
    ema_clin_recall = None  # init defensivo (binario lo puebla abajo)
    if recall_clinical is not None:
        ema_clin_recall = recall_clinical.compute().item()
        ema_clin_spec = spec_clinical.compute().item()
        print(f"   EMA @Umbral Clínico 0.20 -> Recall: {ema_clin_recall:.4f} | Spec: {ema_clin_spec:.4f}")
    ema_class_healthy = True
    if per_class_recall is not None:
        class_recalls = per_class_recall.compute()
        class_names = [MULTICLASS_INDEX_MAP[i] for i in range(len(MULTICLASS_INDEX_MAP))]
        recall_str = " | ".join([f"{n}:{r:.2f}" for n, r in zip(class_names, class_recalls)])
        print(f"   EMA Per-class Recall: {recall_str}")
        _class_aucs = None
        if per_class_auc is not None:
            _class_aucs = per_class_auc.compute()
            auc_str = " | ".join([f"{n}:{a:.3f}" for n, a in zip(class_names, _class_aucs)])
            print(f"   EMA Per-class AUC:    {auc_str}")
        ema_class_healthy = _check_class_health(class_recalls, _class_aucs, class_names, context="EMA: ")
    # Comparación contra el checkpoint en disco (no contra el "mejor visto")
    # usando la métrica de selección (pAUC si está activa).
    if _best_saved_ema is None or ema_selection > _best_saved_ema:
        print("   EMA GANA: el modelo EMA supera al mejor checkpoint estándar.")
    else:
        print("   ESTÁNDAR GANA: el mejor checkpoint estándar supera al EMA.")

    # El checkpoint EMA se bloquea por class_health igual que SWA: baseline
    # forzado si no existe un EMA previo, bloqueo con warning si existe. En
    # binario se bloquea además por recall@0.20 (suelo clínico): EMA es el
    # checkpoint típico de producción robusta y el operador puede desplegarlo.
    _ema_to_save = getattr(ema.module, "_orig_mod", ema.module)
    _ema_recall_block = _binary_recall_floor_block(
        mode, ema_clin_recall, recall_floor_binary
    )
    _ema_block = ((mode == "multiclass") and not ema_class_healthy) or _ema_recall_block
    if (not _ema_block) or (not Path(save_path_ema).exists()):
        if _ema_block:
            _ema_reason = (
                f"recall@{CLINICAL_THRESHOLD_BINARY:.2f}={ema_clin_recall:.4f} "
                f"< suelo {recall_floor_binary:.2f}"
                if _ema_recall_block
                else "las clases malignas no pasan el health check"
            )
            warnings.warn(
                f"FORCED BASELINE EMA: {_ema_reason}, "
                "pero no hay EMA previo → guardando baseline forzado.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Guardado atómico + comprobación de finitud (anti-corrupción en Colab).
        _save_checkpoint_safe(_ema_to_save.state_dict(), save_path_ema, "EMA")
        print(f"Modelo EMA guardado en: {save_path_ema}")
        # Fairness report específico del EMA (checkpoint típico de producción
        # robusta — la trazabilidad debe corresponder al modelo desplegable).
        try:
            _run_fairness_report(
                _ema_to_save, val_loader, mode, device,
                amp_dtype=_amp_dtype,
                checkpoint_source="ema",
                min_positive=int(min_positive),
            )
        except Exception as _fr_exc:
            warnings.warn(
                f"Fairness report EMA falló: {_fr_exc}. "
                "Audit trail EMA incompleto; revisar antes del despliegue.",
                RuntimeWarning,
                stacklevel=2,
            )
        # SPC específico del EMA, en paralelo al fairness report — el checkpoint
        # típicamente desplegado en producción robusta.
        _run_selective_prediction_curve(
            _ema_to_save, val_loader, mode, device,
            save_dir=str(Path(save_path).parent),
            amp_dtype=_amp_dtype, checkpoint_source="ema",
        )
    else:
        _ema_blk_reason = (
            f"recall@{CLINICAL_THRESHOLD_BINARY:.2f}={ema_clin_recall:.4f} "
            f"< suelo {recall_floor_binary:.2f}"
            if _ema_recall_block
            else "clases malignas con AUC/recall crítico"
        )
        warnings.warn(
            f"EMA CHECKPOINT BLOCKED: {_ema_blk_reason}. "
            f"EMA AUC={ema_auc:.4f} NO guardado (preservado el EMA previo).",
            RuntimeWarning,
            stacklevel=2,
        )

    # Fairness report sobre el MEJOR checkpoint, no sobre los pesos de la última
    # época: `raw_model` al cierre contiene los pesos de la última época, y si el
    # early stopping seleccionó un checkpoint anterior, el report certificaría la
    # equidad de un modelo que NO es el desplegado (trazabilidad incorrecta).
    fairness_source = "best_checkpoint"
    if Path(save_path).exists():
        try:
            _best_state = torch.load(save_path, map_location=device, weights_only=True)
            raw_model.load_state_dict(_best_state, strict=True)
        except Exception as exc:
            # Se imprime en stdout además del warning: en Colab, warnings.warn
            # puede perderse entre la salida de tqdm, y el operador debe ver el
            # fallback en el scroll crudo para auditar correctamente.
            print(
                f"\n[FAIRNESS] Fallback a última época: {save_path} no "
                f"cargable ({exc}). El JSON registrará checkpoint_source="
                f"'last_epoch_fallback' — revisar antes del despliegue.\n"
            )
            warnings.warn(
                f"Fairness report: no se pudo cargar {save_path} ({exc}). "
                "Usando pesos de última época como fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            fairness_source = "last_epoch_fallback"
    else:
        # Mismo print en stdout: si class_health bloqueó todos los saves, el audit
        # trail del fairness report corresponde a un modelo distinto al que se
        # desplegaría. Es señal de un training roto y el operador debe verla de
        # inmediato en la consola.
        print(
            f"\n[FAIRNESS] Fallback a última época: {save_path} NO EXISTE "
            f"(class_health bloqueó todos los saves). El JSON registrará "
            f"checkpoint_source='last_epoch_fallback'. Training probablemente "
            f"degradado — revisar class_health logs antes del despliegue.\n"
        )
        warnings.warn(
            f"Fairness report: {save_path} no existe (class_health bloqueó todo). "
            "Usando pesos de última época como fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        fairness_source = "last_epoch_fallback"
    _run_fairness_report(
        raw_model, val_loader, mode, device,
        amp_dtype=_amp_dtype, checkpoint_source=fairness_source,
        min_positive=int(min_positive),
    )

    # Selective Prediction Curve (Geifman & El-Yaniv, NeurIPS 2017) del
    # best_checkpoint. En paralelo al fairness report, la SPC también se ejecuta
    # para SWA/EMA/SOUP en sus respectivos callsites. Permite calibrar el threshold
    # operativo según el contexto clínico (alta cobertura vs alta accuracy).
    _run_selective_prediction_curve(
        raw_model, val_loader, mode, device,
        save_dir=str(Path(save_path).parent),
        amp_dtype=_amp_dtype, checkpoint_source=fairness_source,
    )

    # Model Soup greedy (Wortsman et al., ICML 2022), opcional. Tras los 3 saves
    # (best, SWA, EMA) y sus fairness reports, se computa un "soup" que parte del
    # mejor individual y añade los otros candidatos UNO A UNO solo si mejoran la
    # métrica de selección con TTA (si está activo). Coste: 3-5 evaluaciones extra
    # sobre val (~1-3 min). Output: `best_{mode}_model_SOUP.pth` +
    # `fairness_report_{mode}_soup.json`. Coste de inferencia nulo — se despliega
    # como cualquier checkpoint individual.
    if use_model_soup:
        print("\nModel Soup Greedy (Wortsman 2022) — computando...")
        _candidates = [
            ("best", save_path),
            ("swa", save_path_swa),
            ("ema", save_path_ema),
        ]
        save_path_soup = save_path.replace(".pth", "_SOUP.pth")
        # Se propaga selection_metric/pauc_*_target al Soup para que elija
        # candidatos por el MISMO criterio que el resto del pipeline (early
        # stopping, comparadores SWA/EMA). Sin esto, el SOUP elegiría por AUC
        # global aunque se entrenara con `--selection_metric pauc_*`, dejando el
        # checkpoint final desalineado con el régimen clínico de despliegue.
        soup_sd, soup_auc, soup_members = _model_soup_greedy(
            _candidates, raw_model, val_loader, mode, device,
            save_dir=str(Path(save_path).parent),
            amp_dtype=_amp_dtype, tta_val_views=tta_val_views,
            selection_metric=selection_metric,
            pauc_spec_target=pauc_spec_target,
            pauc_sens_target=pauc_sens_target,
        )
        if soup_sd is not None:
            # class_health gate para el Model Soup, igual que SWA/EMA. El soup
            # combina los checkpoints {best, SWA, EMA}, cuyo macro AUC puede estar
            # inflado por Nevus (~60% del val set multiclase): un soup con
            # val_auc=0.97 podría tener melanoma_AUC=0.75 silenciosamente. Se
            # cargan los pesos del soup, se evalúan las métricas per-class con TTA
            # (si está activo) y se aplica el mismo umbral de `_check_class_health`
            # que en SWA/EMA. La rama de forced-baseline (guardar la primera vez
            # aunque falle) evita dejar al operador sin checkpoint SOUP cuando
            # todos están unhealthy.
            raw_model.load_state_dict(soup_sd, strict=True)
            soup_class_healthy = True
            if mode == "multiclass" and per_class_recall is not None:
                raw_model.eval()
                per_class_recall.reset()
                if per_class_auc is not None:
                    per_class_auc.reset()
                _soup_eval_samples = 0
                with torch.no_grad():
                    for images, labels, meta in tqdm(val_loader, desc="Eval SOUP class_health"):
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        meta = meta.to(device, non_blocking=True)
                        # TTA-aware: usa `_tta_forward_views` con el mismo
                        # tta_val_views activo en el resto del val loop, para que
                        # las decisiones de bloqueo se alineen con el régimen de
                        # despliegue.
                        outputs = _tta_forward_views(
                            raw_model, images, meta, tta_val_views, _amp_dtype
                        )
                        preds = torch.softmax(outputs.float(), dim=1)
                        if torch.isfinite(preds).all():
                            per_class_recall.update(preds, labels)
                            if per_class_auc is not None:
                                per_class_auc.update(preds, labels)
                            _soup_eval_samples += images.size(0)
                if _soup_eval_samples > 0:
                    _soup_class_recalls = per_class_recall.compute()
                    _soup_class_aucs = per_class_auc.compute() if per_class_auc is not None else None
                    _soup_class_names = [
                        MULTICLASS_INDEX_MAP[i] for i in range(len(MULTICLASS_INDEX_MAP))
                    ]
                    _soup_recall_str = " | ".join(
                        [f"{n}:{r:.2f}" for n, r in zip(_soup_class_names, _soup_class_recalls)]
                    )
                    print(f"   SOUP Per-class Recall: {_soup_recall_str}")
                    if _soup_class_aucs is not None:
                        _soup_auc_str = " | ".join(
                            [f"{n}:{a:.3f}" for n, a in zip(_soup_class_names, _soup_class_aucs)]
                        )
                        print(f"   SOUP Per-class AUC:    {_soup_auc_str}")
                    soup_class_healthy = _check_class_health(
                        _soup_class_recalls, _soup_class_aucs, _soup_class_names,
                        context="SOUP: ",
                    )
                else:
                    # 0 samples válidos durante la evaluación → se trata como fallo.
                    warnings.warn(
                        "Evaluación SOUP class_health sin samples válidos "
                        "(todos los batches NaN). Tratando como UNHEALTHY para "
                        "preservar el checkpoint SOUP previo si existe.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    soup_class_healthy = False

            # En binario se bloquea el SOUP por recall@0.20 (suelo clínico). El
            # soup es un checkpoint desplegable y, aunque promedie candidatos con
            # recall aceptable, el promedio de pesos NO es lineal en recall, así
            # que el soup puede caer por debajo del suelo. Se evalúa recall@0.20
            # sobre val con TTA (mismo régimen que el resto) usando una instancia
            # fresca de BinaryRecall, sin contaminar `recall_clinical`.
            _soup_clin_recall = None
            if mode == "binary" and recall_floor_binary > 0.0:
                raw_model.eval()
                _soup_rec_metric = BinaryRecall(
                    threshold=CLINICAL_THRESHOLD_BINARY
                ).to(device)
                _soup_bin_samples = 0
                with torch.no_grad():
                    for images, labels, meta in tqdm(
                        val_loader, desc="Eval SOUP recall@0.20"
                    ):
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        meta = meta.to(device, non_blocking=True)
                        outputs = _tta_forward_views(
                            raw_model, images, meta, tta_val_views, _amp_dtype
                        )
                        preds = torch.sigmoid(outputs.float().squeeze(1))
                        if torch.isfinite(preds).all():
                            _soup_rec_metric.update(preds, labels)
                            _soup_bin_samples += images.size(0)
                if _soup_bin_samples > 0:
                    _soup_clin_recall = _soup_rec_metric.compute().item()
                    print(
                        f"   SOUP @Umbral Clínico {CLINICAL_THRESHOLD_BINARY:.2f}"
                        f" -> Recall: {_soup_clin_recall:.4f}"
                    )
                else:
                    # 0 samples válidos → se aborta, igual que en los demás loops.
                    warnings.warn(
                        "Evaluación SOUP recall@0.20 sin samples válidos (todos "
                        "los batches NaN). Tratando como UNHEALTHY para preservar "
                        "el SOUP previo si existe.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _soup_clin_recall = 0.0

            _soup_recall_block = _binary_recall_floor_block(
                mode, _soup_clin_recall, recall_floor_binary
            )
            _soup_block = (
                ((mode == "multiclass") and not soup_class_healthy)
                or _soup_recall_block
            )
            if (not _soup_block) or (not Path(save_path_soup).exists()):
                if _soup_block:
                    # Baseline forzado, igual que en SWA/EMA: evita dejar al
                    # operador sin checkpoint SOUP la primera vez aunque el
                    # health/recall falle. Los re-runs posteriores NO sobrescriben.
                    _soup_fb_reason = (
                        f"recall@{CLINICAL_THRESHOLD_BINARY:.2f}="
                        f"{_soup_clin_recall:.4f} < suelo "
                        f"{recall_floor_binary:.2f}"
                        if _soup_recall_block
                        else "las clases malignas no pasan el health check"
                    )
                    warnings.warn(
                        f"FORCED BASELINE SOUP: {_soup_fb_reason}, "
                        "pero no hay SOUP previo → guardando baseline forzado.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                # Guardado atómico + comprobación de finitud (anti-corrupción Colab).
                _save_checkpoint_safe(soup_sd, save_path_soup, "Model Soup")
                # Label dinámico (AUC / pAUC@spec95 / pAUC@sens80) según
                # `selection_metric`.
                _soup_metric_label = (
                    _selection_metric_label(selection_metric)
                    if (selection_metric in _PAUC_SELECTION_METRICS and mode == "binary")
                    else "AUC"
                )
                print(
                    f"   Model Soup guardado en {save_path_soup} — "
                    f"{_soup_metric_label}={soup_auc:.4f} | miembros={soup_members}"
                )
                try:
                    _run_fairness_report(
                        raw_model, val_loader, mode, device,
                        amp_dtype=_amp_dtype,
                        checkpoint_source="soup",
                        min_positive=int(min_positive),
                    )
                except Exception as _fr_exc:
                    warnings.warn(
                        f"Fairness report SOUP falló: {_fr_exc}. "
                        "Audit trail del soup incompleto.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                # SPC específico del SOUP, en paralelo al fairness report.
                # `raw_model` ya tiene `soup_sd` cargado.
                _run_selective_prediction_curve(
                    raw_model, val_loader, mode, device,
                    save_dir=str(Path(save_path).parent),
                    amp_dtype=_amp_dtype, checkpoint_source="soup",
                )
            else:
                # Label dinámico (paridad con el log de guardado).
                _blocked_label = (
                    _selection_metric_label(selection_metric)
                    if (selection_metric in _PAUC_SELECTION_METRICS and mode == "binary")
                    else "AUC"
                )
                _soup_blk_reason = (
                    f"recall@{CLINICAL_THRESHOLD_BINARY:.2f}="
                    f"{_soup_clin_recall:.4f} < suelo {recall_floor_binary:.2f}"
                    if _soup_recall_block
                    else "clases malignas con AUC/recall crítico"
                )
                warnings.warn(
                    f"SOUP CHECKPOINT BLOCKED: {_soup_blk_reason}. "
                    f"SOUP {_blocked_label}={soup_auc:.4f} NO guardado "
                    f"(preservado el SOUP previo en {save_path_soup}). Revisar "
                    f"fairness_report_{mode}_soup.json del run anterior antes "
                    f"del despliegue.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Model Soup no produjo un ensemble válido "
                "(0 candidatos evaluables). Revisar class_health "
                "(puede haber bloqueado todos los saves) o el estado de "
                "los checkpoints individuales.",
                RuntimeWarning,
                stacklevel=2,
            )

    print(f"Entrenamiento '{mode}' finalizado. Mejor modelo estándar -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Skin Cancer Trainer")
    parser.add_argument(
        "--mode",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
        help="Nivel de la cascada a entrenar.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5, help="Épocas sin mejora antes de parar.")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * accum_steps).")

    # --- Flags opcionales (sus valores por defecto reproducen el baseline) ---
    parser.add_argument("--backbone_name", type=str, default=None,
                        help="Factory backbone (dinov2_vitl14, dinov2_vitb14, effnet_b3, effnet_b4). None → legacy timm.")
    parser.add_argument("--fusion_mode", type=str, default="film", choices=["film", "cross_attention"])
    parser.add_argument("--drop_path_rate", type=float, default=None, help="Override drop_path_rate (default 0.2).")
    parser.add_argument("--sam", action="store_true", help="Activar Sharpness-Aware Minimization.")
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--sam_adaptive", action="store_true", help="ASAM (adaptive SAM).")
    parser.add_argument("--use_seesaw", action="store_true",
                        help="Reemplaza MulticlassFocalLoss por SeesawLoss (multiclase).")
    parser.add_argument("--seesaw_p", type=float, default=0.8)
    parser.add_argument("--seesaw_q", type=float, default=2.0)
    parser.add_argument("--balanced_softmax_weight", type=float, default=0.0,
                        help="Peso auxiliar BalancedSoftmaxLoss (0 → off).")
    parser.add_argument("--use_self_distill", action="store_true",
                        help="Born-again self-distillation KD (multiclase).")
    parser.add_argument("--self_distill_weight", type=float, default=0.1)
    parser.add_argument("--self_distill_T", type=float, default=4.0)
    parser.add_argument("--self_distill_warmup", type=int, default=5)
    parser.add_argument("--distill_from", type=str, default=None,
                        help="Path a checkpoint binario para destilación teacher→student.")
    parser.add_argument("--distill_alpha", type=float, default=0.3)
    parser.add_argument("--scheduler", type=str, default="sequential",
                        choices=["sequential", "warm_restarts"])
    parser.add_argument("--lr_head_multi", type=float, default=None,
                        help="Override del LR del head en multiclase (p. ej. 1e-4).")
    parser.add_argument("--wd_head_multi", type=float, default=None,
                        help="Override del weight_decay del head en multiclase (p. ej. 5e-3).")
    parser.add_argument("--no_gc", action="store_true",
                        help="Desactivar Gradient Centralization (activa por defecto).")
    # Métrica de selección de checkpoint
    parser.add_argument("--selection_metric", type=str, default="auc",
                        choices=["auc", "pauc_spec95", "pauc_sens80"],
                        help="Métrica para early stopping / SWA / EMA / Soup. "
                             "'auc' (default, preserva el baseline). 'pauc_sens80' "
                             "(RECOMENDADO para el triaje binario): partial AUC "
                             "en alta SENSIBILIDAD (TPR≥0.80, estándar ISIC 2020) "
                             "— alineado con el umbral clínico 0.20. "
                             "'pauc_spec95': partial AUC en alta especificidad "
                             "(FPR≤0.05, McClish 1989). pAUC solo en binario.")
    parser.add_argument("--pauc_spec_target", type=float, default=0.95,
                        help="Specificity target para pauc_spec95 (default 0.95 → FPR≤0.05).")
    parser.add_argument("--pauc_sens_target", type=float, default=0.80,
                        help="Sensitivity target para pauc_sens80 "
                             "(default 0.80 = estándar ISIC 2020 → TPR≥0.80).")
    parser.add_argument("--recall_floor_binary", type=float, default=0.0,
                        help="Suelo clínico de recall@0.20 (sensibilidad en el "
                             "umbral de despliegue). Bloquea el "
                             "guardado de cualquier checkpoint binario (best/SWA/"
                             "EMA/SOUP) con recall@0.20 < suelo. Default 0.0 = "
                             "desactivado (baseline). Recomendado clínico: 0.90. "
                             "Solo binario.")
    parser.add_argument("--focal_alpha", type=float, default=0.7,
                        help="Alpha de la FocalLoss binaria (default 0.7 = "
                             "baseline; penaliza fallar un melanoma 2.3x más que "
                             "un benigno). 0.75 → ~3x. Nunca puede ser None.")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "lion"],
                        help="Optimizer: 'adamw' (default, baseline) o 'lion' "
                             "(Chen 2023, -15%% VRAM, usa lr ~10x menor).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed de reproducibilidad (default 42). Se propaga a "
                             "train_model para permitir deep ensembles con seeds "
                             "distintas.")
    # CLI flags para las mejoras opcionales del modelo
    parser.add_argument("--use_fitnet", action="store_true",
                        help="Feature distillation FitNets "
                             "(Romero 2015, Heo 2019). Requiere --distill_from. "
                             "Añade un término MSE aditivo sobre los feature maps "
                             "intermedios student↔teacher. +0.3-0.7%% AUC esperado.")
    parser.add_argument("--fitnet_weight", type=float, default=0.05,
                        help="Peso del término FitNets en la composición "
                             "aditiva (default 0.05 del paper Heo 2019).")
    parser.add_argument("--fitnet_teacher_dim", type=int, default=1536,
                        help="Dim de features del teacher para el Conv2d 1×1 de "
                             "FitNets (proyecta student→teacher). NO se "
                             "auto-detecta. Default 1536 (EffNet-B3). Teacher "
                             "DINOv2 ViT-L → pasar 1024; ViT-B → 768. DEBE ser "
                             "igual a teacher.feature_dim o el check cross-modelo "
                             "se detiene con error antes de entrenar.")
    parser.add_argument("--grad_checkpointing", action="store_true",
                        help="Gradient checkpointing (recompute de activaciones "
                             "en el backward) en el backbone de timm vía "
                             "set_grad_checkpointing(True). Reduce VRAM ~30-40%% "
                             "del backbone a cambio de ~+25-35%% de tiempo/época. "
                             "Default OFF = idéntico a todo checkpoint/run "
                             "existente (binario incluido). Necesario para el "
                             "multiclase ViT-L + teacher KD + FitNets: a partir "
                             "de la época 6, MixUp dispara un doble forward del "
                             "ViT-L que sin esto provoca OOM en A100 40GB. Seguro: "
                             "con accum_steps>1, torch.compile cae a mode=default "
                             "(CUDA Graphs OFF), sin el reuse-bug de CUDA Graphs.")
    parser.add_argument("--use_attention_pool", action="store_true",
                        help="AttentionPoolViT en vez de GeM para backbones ViT "
                             "(CLS token aprendible + MHA). Solo válido con "
                             "backbone ViT; se detiene con error si se usa con CNN. MUTUAMENTE "
                             "EXCLUSIVO con --use_hybrid_pool.")
    parser.add_argument("--use_hybrid_pool", action="store_true",
                        help="Pooling híbrido GeM + AttentionPool concatenados "
                             "[B, 2*C]. PRESERVA FiLM/CBAM (a diferencia de "
                             "--use_attention_pool, que los desactiva). Compatible "
                             "con CNN y ViT, y con FitNets. +0.5-1.0%% AUC "
                             "esperado. MUTUAMENTE EXCLUSIVO con "
                             "--use_attention_pool y --use_multiscale_vit.")
    parser.add_argument("--use_multiscale_vit", action="store_true",
                        help="Pooling multi-escala ViT: AttentionPool "
                             "independientes sobre 3 capas del encoder "
                             "([N//3, 2N//3, N-1]) → [B, 3*C]. Captura texturas "
                             "de bajo/medio/alto nivel del ViT. SOLO para "
                             "backbones ViT (ViT-L, ViT-B). DESACTIVA FiLM/CBAM "
                             "(asimetría con los tokens intermedios). +0.5-1.0%% "
                             "AUC esperado en lesiones con texturas multi-escala. "
                             "INCOMPATIBLE con FitNets. MUTUAMENTE EXCLUSIVO con "
                             "--use_attention_pool y --use_hybrid_pool.")
    parser.add_argument("--cbam_adaptive_kernel", action="store_true",
                        help="Kernel adaptativo para SpatialAttention: calcula k "
                             "proporcional al grid size.")
    parser.add_argument("--drop_path_schedule", type=str, default="uniform",
                        choices=["uniform", "linear"],
                        help="Schedule de Stochastic Depth: 'uniform' (default, "
                             "como timm) o 'linear' (default del paper de EfficientNet).")
    # TTA durante validación para un early stopping más estable.
    parser.add_argument("--tta_val_views", type=int, default=1,
                        choices=[1, 4, 6],
                        help="Número de vistas TTA en el val loop. "
                             "1 (default) = single-crop, preserva el baseline. "
                             "4 = original + hflip + vflip + rot180 (coste 4×). "
                             "6 = + rot90 + rot270 (igual que el TTA-6 de predict.py). "
                             "Alinea el early stopping con el régimen de despliegue "
                             "clínico; +0.3-0.8%% AUC esperado (ISIC 2023-2024).")
    # Model Soup greedy.
    parser.add_argument("--use_model_soup", action="store_true",
                        help="Promediar selectivamente {best, SWA, EMA} vía greedy "
                             "soup (Wortsman ICML 2022). Parte del mejor y añade "
                             "los otros solo si mejoran la métrica de selección. "
                             "Output: best_{mode}_model_SOUP.pth. Coste de "
                             "inferencia nulo; +0.2-0.5%% AUC esperado.")
    # Polyloss (Leng ICCV 2022).
    parser.add_argument("--poly_epsilon", type=float, default=0.0,
                        help="Término Poly-1 aditivo `epsilon*(1-pt)` sobre la "
                             "Focal (binario y multiclase). Default 0.0 preserva el "
                             "comportamiento baseline exacto. El paper recomienda 1.0 "
                             "para Poly-1-Focal (Leng ICCV 2022). Mejora el gradiente "
                             "cerca de pt≈1, donde (1-pt)^gamma se estanca. +0.2-0.5%% "
                             "AUC esperado en clases raras (Dermatofibroma 84 imgs).")
    # Sampler equilibrado por subgrupo para remediar sesgos demográficos.
    parser.add_argument("--group_balance_dims", type=str, nargs="*", default=[],
                        choices=["site", "age"],
                        help="Dimensiones para el rebalanceo del WeightedRandomSampler. "
                             "Subset de {site, age}. Default [] preserva el baseline "
                             "(sample_weights solo por clase). `site` rebalancea por "
                             "(clase × site_code) para resolver la specificity=0%% en "
                             "palms/soles (el modelo aprende el site como atajo). `age` "
                             "rebalancea por (clase × age_bucket) para mejorar el recall "
                             "pediátrico. Aplicar AMBAS (`--group_balance_dims site age`) "
                             "es la configuración recomendada.")
    parser.add_argument("--group_balance_smooth", type=float, default=0.7,
                        help="Factor smooth ∈ [0, 1] para moderar el rebalanceo. "
                             "0 = sin balance (idéntico al baseline). 1 = balance total "
                             "(prevalencia post-balance ≈ target). Default 0.7, validado "
                             "matemáticamente con palms/soles (reduce el ratio "
                             "maligno/benigno de 56× a 3.3× sin memorización). Si el run "
                             "degrada el pAUC global >0.01, bajar a 0.5 (corrección parcial).")
    # min_positive del fairness_report.
    parser.add_argument("--min_positive", type=int, default=10,
                        help="Mínimo de muestras POSITIVAS por subgrupo en el "
                             "fairness_report binario. El intervalo de Wilson 95%% para "
                             "recall=0.5 con n_pos=10 es [0.24, 0.76] (rango 52pp); con "
                             "n_pos=4 es [0.10, 0.90] (rango 80pp). Excluye del chequeo "
                             "de sesgo a los subgrupos cuyo recall no es estadísticamente "
                             "fiable (p. ej. sex=unknown con n=372 pero solo 4 malignos). "
                             "Default 10.")

    args = parser.parse_args()

    train_model(
        mode=args.mode, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        patience=args.patience, accum_steps=args.accum_steps,
        backbone_name=args.backbone_name,
        fusion_mode=args.fusion_mode,
        drop_path_rate=args.drop_path_rate,
        use_sam=args.sam,
        sam_rho=args.sam_rho,
        sam_adaptive=args.sam_adaptive,
        use_seesaw=args.use_seesaw,
        seesaw_p=args.seesaw_p,
        seesaw_q=args.seesaw_q,
        balanced_softmax_weight=args.balanced_softmax_weight,
        use_self_distill=args.use_self_distill,
        self_distill_weight=args.self_distill_weight,
        self_distill_T=args.self_distill_T,
        self_distill_warmup=args.self_distill_warmup,
        distill_from=args.distill_from,
        distill_alpha=args.distill_alpha,
        scheduler_type=args.scheduler,
        lr_head_multi=args.lr_head_multi,
        wd_head_multi=args.wd_head_multi,
        use_gc=not args.no_gc,
        selection_metric=args.selection_metric,
        pauc_spec_target=args.pauc_spec_target,
        pauc_sens_target=args.pauc_sens_target,
        recall_floor_binary=args.recall_floor_binary,
        focal_alpha=args.focal_alpha,
        optimizer_name=args.optimizer,
        seed=args.seed,
        use_fitnet=args.use_fitnet,
        fitnet_weight=args.fitnet_weight,
        fitnet_teacher_dim=args.fitnet_teacher_dim,
        grad_checkpointing=args.grad_checkpointing,
        use_attention_pool=args.use_attention_pool,
        use_hybrid_pool=args.use_hybrid_pool,
        use_multiscale_vit=args.use_multiscale_vit,
        cbam_adaptive_kernel=args.cbam_adaptive_kernel,
        drop_path_schedule=args.drop_path_schedule,
        tta_val_views=args.tta_val_views,
        use_model_soup=args.use_model_soup,
        poly_epsilon=args.poly_epsilon,
        group_balance_dims=tuple(args.group_balance_dims),
        group_balance_smooth=args.group_balance_smooth,
        min_positive=args.min_positive,
    )
