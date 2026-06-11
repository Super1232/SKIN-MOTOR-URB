"""
Explicabilidad clínica para SkinCancerModel.

Los clínicos necesitan ver POR QUÉ el modelo predice melanoma — no basta con un
número. Este módulo expone UN único método de explicabilidad, agnóstico al backbone
(CNN y ViT):

- Occlusion Sensitivity (Zeiler & Fergus, ECCV 2014): tapa una región de la imagen
  con un parche deslizante y mide cuánto CAE la probabilidad de la clase objetivo.
  Donde más cae = región más importante para la decisión -> contribución CAUSAL
  directa en el espacio de PÍXELES. No usa activaciones ni atención, así que trata el
  modelo como una caja negra `imagen -> logits` y funciona idéntico para CNN y ViT.

  Por qué NO la familia CAM (GradCAM, Score-CAM, GradCAM++, LayerCAM): todas
  construyen el heatmap como combinación lineal de los mapas de activación del
  backbone `Sum_k w_k * A_k`; solo cambian los pesos `w_k`. Si la lesión es
  oscura/lisa NO enciende los filtros de la CNN (que responden a bordes y textura)
  -> casi todos los `A_k` están fríos sobre el lunar -> NINGUNA combinación lineal lo
  ilumina (limitación compartida por TODA la familia CAM). El rollout de atención del
  ViT sufre un análogo (outlier tokens de DINOv2-sin-registers). Occlusion rompe esa
  dependencia perturbando píxeles directamente.

El dispatcher `build_explainer(model)` devuelve siempre `OcclusionSensitivityExplainer`
(un único método para todos los backbones).

Referencia:
  - Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks",
    ECCV 2014. https://arxiv.org/abs/1311.2901
"""
import math

import torch
import torch.nn.functional as F
import numpy as np


def _calibrate_logits(logits_raw, temperature, explainer_label):
    """Aplica Temperature/Vector Scaling a los logits para reportar
    probabilidades calibradas, consistentes con el engine clínico.

    Fuente única usada por el explainer. La temperatura se acota al mismo rango
    clínico [0.3, 5.0] que `calibration.py::TemperatureScaler.calibrate()`. El
    heatmap NO depende de T; solo cambian las probabilidades
    reportadas. Se valida la finitud ANTES de acotar porque `nan.clamp(0.3, 5.0)` y
    `min(max(nan, ...))` devuelven NaN (el clamp preserva NaN) — sin esto, un
    checkpoint de temperatura corrupto cargado por una vía que omite
    `_load_temperature_safe` produciría `prob = nan` silenciosamente en el dict
    `prediction`.

    Args:
        logits_raw: Tensor `[1, C]` de logits crudos.
        temperature: float, lista/tupla/np.ndarray (vector scaling) o None.
        explainer_label: nombre de la clase para el mensaje de error.
    Returns:
        Tensor `[1, C]` de logits calibrados (= logits_raw si temperature is None).
    """
    if temperature is None:
        return logits_raw
    if isinstance(temperature, (list, tuple, np.ndarray)):
        _t_arr = np.asarray(temperature, dtype=np.float32)
        if not np.all(np.isfinite(_t_arr)):
            raise ValueError(
                f"{explainer_label}.explain recibió un vector de temperatura con "
                f"valores no finitos: {_t_arr.tolist()}. Causa probable: checkpoint "
                "de temperatura corrupto cargado sin `_load_temperature_safe`. "
                "Verificar el origen del valor."
            )
        t_vec = (
            torch.tensor(_t_arr, device=logits_raw.device)
            .clamp(min=0.3, max=5.0)
            .view(1, -1)
        )
        return logits_raw / t_vec
    _t_val = float(temperature)
    if not math.isfinite(_t_val):
        raise ValueError(
            f"{explainer_label}.explain recibió una temperatura escalar no finita: "
            f"{_t_val}. Causa probable: checkpoint de temperatura corrupto cargado "
            "sin `_load_temperature_safe`. Verificar el origen del valor."
        )
    t_scalar = min(max(_t_val, 0.3), 5.0)
    return logits_raw / t_scalar


def _build_prediction_dict(logits_raw, logits_cal, target_class, method):
    """Construye el dict `prediction` del explainer.

    Expone `logits` crudos, `logits_calibrated`, `target_class` y `method`
    (consumido por la trazabilidad clínica). Binario añade `prob`;
    multiclase añade `probs`, `pred_class` y `confidence`.
    """
    num_classes = logits_raw.shape[-1]
    prediction = {
        "logits": logits_raw.cpu().numpy()[0],
        "logits_calibrated": logits_cal.cpu().numpy()[0],
        "target_class": target_class,
        "method": method,
    }
    if num_classes == 1:
        prediction["prob"] = torch.sigmoid(logits_cal).item()
    else:
        probs = F.softmax(logits_cal, dim=1).cpu().numpy()[0]
        prediction["probs"] = probs
        prediction["pred_class"] = int(np.argmax(probs))
        prediction["confidence"] = float(probs[prediction["pred_class"]])
    return prediction


class OcclusionSensitivityExplainer:
    """Mapas de calor por oclusión (Occlusion Sensitivity, Zeiler & Fergus 2014)
    para SkinCancerModel — agnóstico al backbone (CNN y ViT).

    Referencia: Zeiler & Fergus, "Visualizing and Understanding Convolutional
    Networks", ECCV 2014. https://arxiv.org/abs/1311.2901

    **Motivación clínica:** la familia CAM (GradCAM, Score-CAM, GradCAM++, LayerCAM)
    construye el heatmap como combinación lineal de los mapas de activación del
    backbone `Σ_k w_k·A_k`; solo cambian los pesos `w_k` entre métodos. Si la lesión
    es oscura/lisa NO enciende los filtros de la CNN (responden a bordes y textura) →
    casi todos los `A_k` están fríos sobre el lunar → NINGUNA combinación lineal lo
    ilumina (limitación compartida por toda la familia CAM). El rollout de atención
    del ViT sufre un análogo (outlier tokens de DINOv2-sin-registers).

    Occlusion Sensitivity rompe esa dependencia: NO usa activaciones ni atención.
    Desliza un parche que tapa una región de la imagen y mide cuánto CAE la
    probabilidad de la clase objetivo. Donde más cae = región más importante para la
    decisión → contribución causal directa en el espacio de PÍXELES. Es el método más
    fiel ("¿qué pasa si tapo esto?") y EXACTAMENTE el mismo para CNN y ViT (trata el
    modelo como una caja negra `imagen → logits`).

    **Algoritmo (Zeiler & Fergus 2014):**

    1. Forward limpio → logits base y `base_score = P(clase objetivo)`.
    2. Deslizar un parche `patch_size × patch_size` con paso `stride`. Para cada
       posición, tapar esa región con `occlusion_value` (gris-medio ImageNet = 0
       sobre el tensor ya normalizado) y medir `score_ocluido`. La caída
       `base_score − score_ocluido` se acumula (con solape promediado) sobre el área
       del parche.
    3. `heatmap = ReLU(importancia)`, normalizar a [0, 1], devolver a la resolución X.

    **Coste:** ≈ (nº de posiciones del parche) forwards por imagen — típicamente
    ~150-400 con los defaults, agrupados en lotes de `batch_size`. Path one-shot que
    solo corre con `explain=True`; conviene GPU (en CPU son decenas de segundos). No
    usa gradientes (todo bajo `torch.no_grad()`).

    **Temperature Scaling:** `temperature` opcional calibra las probabilidades
    reportadas en `prediction`; el heatmap espacial NO depende de T.
    """

    def __init__(self, model, patch_size=None, stride=None, batch_size=32,
                 occlusion_value=0.0):
        """
        Args:
            model: SkinCancerModel (caja negra `model(image, metadata) -> logits`).
            patch_size: lado del parche en píxeles. None → auto `max(8, H // 8)`.
            stride: paso del deslizamiento en píxeles. None → auto `max(1, patch//2)`
                (50% de solape → heatmap más suave).
            batch_size: nº de imágenes ocluidas por forward (control de memoria).
            occlusion_value: valor con el que se rellena el parche. 0.0 = gris-medio
                ImageNet sobre el tensor ya normalizado (baseline estándar Zeiler).
        """
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = int(batch_size)
        self.occlusion_value = float(occlusion_value)

    def explain(self, image, metadata=None, target_class=None, temperature=None):
        """
        Genera heatmap de Occlusion Sensitivity para una imagen.

        Args:
            image: Tensor [1, C, H, W] — imagen preprocesada (normalizada ImageNet).
            metadata: Tensor [1, 3] o None — metadatos clínicos.
            target_class: int o None — clase objetivo. None = clase predicha
                (binario: logit único; multiclase: argmax).
            temperature: float, lista o None — temperatura(s) de calibración para
                reportar probabilidades calibradas en `prediction`. El heatmap NO
                depende de T.
        Returns:
            heatmap: np.array [H, W] normalizado [0, 1] (la importancia de cada píxel
                = caída de P(clase) al taparlo; ReLU → solo evidencia A FAVOR).
            prediction: dict con logits y clase predicha.
        """
        self.model.eval()

        # Occlusion es gradient-free: todo bajo no_grad (inocuo dentro del
        # enable_grad del engine cuando explain=True).
        with torch.no_grad():
            # 1. Forward limpio: logits base + score de la clase objetivo.
            logits_baseline = self.model(image, metadata)
            num_classes = logits_baseline.shape[-1]
            if target_class is None:
                target_class = (
                    0 if num_classes == 1 else int(logits_baseline.argmax(dim=1).item())
                )
            if num_classes == 1:
                base_score = torch.sigmoid(logits_baseline[0, 0]).item()
            else:
                base_score = F.softmax(logits_baseline, dim=1)[0, target_class].item()

            _, _, H, W = image.shape

            # Tamaño de parche / stride: auto-derivados de la resolución si None.
            patch = self.patch_size if self.patch_size is not None else max(8, H // 8)
            patch = max(1, min(patch, H, W))
            stride = self.stride if self.stride is not None else max(1, patch // 2)

            # 2. Posiciones del parche (rejilla con solape). Se ancla un parche al
            #    borde derecho/inferior para no dejar franjas sin cubrir.
            ys = list(range(0, H - patch + 1, stride))
            xs = list(range(0, W - patch + 1, stride))
            if not ys:
                ys = [0]
            if not xs:
                xs = [0]
            if ys[-1] != H - patch:
                ys.append(H - patch)
            if xs[-1] != W - patch:
                xs.append(W - patch)
            positions = [(y, x) for y in ys for x in xs]

            heat = torch.zeros(H, W, device=image.device, dtype=image.dtype)
            count = torch.zeros(H, W, device=image.device, dtype=image.dtype)

            # 3. Por lote de posiciones: tapar, forward, medir la caída de score.
            #    metadata se expande al tamaño del lote (el forward exige batch
            #    consistente con la imagen; meta_net lleva BatchNorm1d).
            for start in range(0, len(positions), self.batch_size):
                batch_positions = positions[start:start + self.batch_size]
                b = len(batch_positions)
                occluded = image.repeat(b, 1, 1, 1)  # [b, 3, H, W]
                for i, (y, x) in enumerate(batch_positions):
                    occluded[i, :, y:y + patch, x:x + patch] = self.occlusion_value
                meta_batch = None
                if metadata is not None:
                    meta_batch = metadata.expand(b, *metadata.shape[1:])
                out = self.model(occluded, meta_batch)
                if num_classes == 1:
                    scores = torch.sigmoid(out[:, 0])
                else:
                    scores = F.softmax(out, dim=1)[:, target_class]
                drop = base_score - scores  # [b] — caída de prob al tapar
                for i, (y, x) in enumerate(batch_positions):
                    heat[y:y + patch, x:x + patch] += drop[i]
                    count[y:y + patch, x:x + patch] += 1.0

            # 4. Promedio por solape → ReLU (solo evidencia a favor) → normalizar.
            heat = heat / count.clamp(min=1.0)
            heat = F.relu(heat)
            if heat.max() > 0:
                heat = heat / heat.max()
            heatmap = heat.cpu().numpy()

            # Dict de predicción — calibración (temperatura escalar/vector) y
            # construcción centralizadas en `_calibrate_logits` /
            # `_build_prediction_dict`. El heatmap no depende de T.
            logits_cal = _calibrate_logits(
                logits_baseline, temperature, "OcclusionSensitivityExplainer"
            )
            prediction = _build_prediction_dict(
                logits_baseline, logits_cal, target_class, "occlusion"
            )

        return heatmap, prediction

    def remove_hooks(self):
        """No-op: Occlusion Sensitivity no registra hooks (gradient-free, caja
        negra). Se mantiene por compatibilidad con la interfaz que `predict.py`
        espera de cualquier explainer (`build_explainer(...).explain(...);
        .remove_hooks()`)."""
        pass

    def __del__(self):
        pass


def build_explainer(model, **kwargs):
    """Dispatcher canónico — un único método de explicabilidad para todos los
    backbones: `OcclusionSensitivityExplainer` (Zeiler & Fergus 2014).

    Occlusion es agnóstico al backbone (trata el modelo como caja negra
    `imagen → logits`), así que el mismo método sirve para CNN (EfficientNet) y ViT
    (DINOv2). Reemplazó a la familia CAM (Score-CAM/GradCAM, que coloreaban el fondo
    por magnitud de activación) y al grad-weighted attention rollout del ViT (que
    marcaba los outlier tokens), porque ninguno de los dos rompía la dependencia de
    los mapas de activación/atención del backbone.

    `**kwargs` (p.ej. `patch_size`, `stride`, `batch_size`, `occlusion_value`) se
    pasan al constructor. Cualquier nuevo explainer debe añadirse aquí y NUNCA
    instanciarse directamente desde `predict.py`, para preservar el contrato "el
    dispatcher elige el método correcto según el modelo".
    """
    return OcclusionSensitivityExplainer(model, **kwargs)


def overlay_heatmap(image_rgb, heatmap, alpha=0.4):
    """
    Superpone un heatmap de explicabilidad sobre la imagen original.

    Args:
        image_rgb: np.array [H, W, 3] uint8 — imagen original RGB
        heatmap: np.array [H, W] float [0, 1] — mapa de calor
        alpha: float — transparencia del heatmap (0=invisible, 1=opaco)
    Returns:
        overlay: np.array [H, W, 3] uint8 — imagen con heatmap superpuesto
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV requerido para overlay_heatmap")

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.float32(image_rgb) * (1 - alpha) + np.float32(heatmap_colored) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay
