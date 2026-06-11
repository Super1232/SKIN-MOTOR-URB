import math
import warnings
from collections import deque
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
# Deshabilita el threadpool interno de OpenCV, igual que en dataset.py. Bajo un
# backend multi-worker, cada request ejecuta cv2.imread/cvtColor más
# augmentations de Albumentations que usan cv2 internamente. Sin setNumThreads(0),
# los workers compiten por el threadpool global de OpenCV y pueden producir
# deadlock bajo carga concurrente.
cv2.setNumThreads(0)
import numpy as np
from pathlib import Path
import sys
import os

# Añadir el raíz al path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.models.architectures import SkinCancerModel
from src.features.preprocessing import HairRemoval, build_stain_transform
from src.data.constants import (
    CLINICAL_THRESHOLD_BINARY,
    DEFAULT_BINARY_BACKBONE,
    DEFAULT_MULTICLASS_BACKBONE,
    IMAGE_SIZES,
    MALIGNANT_INDICES,
    PRECANCEROUS_INDICES,
    MULTICLASS_INDEX_MAP,
    SEX_MAP,
    SITE_MAP,
    encode_clinical_metadata_scalar,
)


from src.inference.calibration import TemperatureScaler, PerSiteThresholdCalibrator
from src.inference.conformal import (
    APSConformal,
    BinaryConformal,
    ConformalRiskControl,
)


def _compute_defer_to_human(report: dict) -> "tuple[bool, list[str]]":
    """Abstención explícita "DEFER_TO_HUMAN".

    Decide si un report clínico debe marcar `defer_to_human=True`. La lista
    de razones es acumulativa (un report puede disparar más de una causa).

    Causas actuales:
    - `low_confidence_tta`: <3 de 6 vistas TTA válidas.
    - `high_uncertainty`: MC Dropout detectó incertidumbre epistémica >0.15.
    - `conformal_binary_ambiguous`: set conformal binario = {benigno, maligno}
      → el modelo no puede distinguir.
    - `conformal_multiclass_too_wide`: set conformal multiclase tiene ≥4
      clases → 50% del espacio, informativamente inútil.
    - `risk_control_disagrees`: el triaje estándar (threshold clínico 0.20)
      discrepa con el predictor ConformalRiskControl calibrado para FNR acotado.
      Típicamente: triaje="benigno" pero risk_control="maligno" (el predictor
      FNR-calibrado detecta un melanoma que el threshold estándar perdería).
      Caso raro pero clínicamente crítico — el FNR es la métrica que regula la
      autoridad sanitaria.

    Formaliza el contrato "el sistema sabe cuándo NO sabe": cuando la
    incertidumbre o la ambigüedad son altas, deriva la decisión a un
    especialista humano en lugar de emitir un diagnóstico automático.
    """
    reasons: list[str] = []
    if report.get("low_confidence_tta", False):
        reasons.append("low_confidence_tta")
    if report.get("high_uncertainty", False):
        reasons.append("high_uncertainty")
    cb = report.get("conformal_set_binary")
    if cb is not None:
        cb_set = set(cb) if not isinstance(cb, set) else cb
        if "benigno" in cb_set and "maligno" in cb_set:
            reasons.append("conformal_binary_ambiguous")
    cm = report.get("conformal_set_multiclass")
    if cm is not None and len(cm) >= 4:
        reasons.append("conformal_multiclass_too_wide")
    # Discrepancia entre el triaje estándar y el risk_control calibrado para FNR.
    # Solo se marca si ambos están presentes en el report (si no se activó
    # `risk_control=True` en infer(), no se emite risk_control_binary).
    rc = report.get("risk_control_binary")
    triage_status = report.get("status")
    if rc is not None and triage_status in ("GREEN", "RED"):
        triage_class = "benigno" if triage_status == "GREEN" else "maligno"
        if rc != triage_class:
            reasons.append("risk_control_disagrees")
    return (len(reasons) > 0, reasons)


def _detect_backbone_config(state_dict):
    """
    Infiere (backbone_name, fusion_mode) a partir de las claves del state_dict.

    Retrocompatibilidad: checkpoints legacy tienen keys `backbone.<layer>` (timm
    directo) → devolvemos `(None, "film")` para que `SkinCancerModel` siga el
    path legacy. Checkpoints nuevos usan la factory `build_backbone`, que
    anida un wrapper (`backbone.backbone.<layer>`); la presencia de
    `meta_cross_attn.*` identifica la fusión cross-attention.

    Firma de 2 elementos para mantener la compatibilidad con `export_onnx.py`
    y otros callers. La presencia de CBAM se detecta por separado con
    `_detect_cbam` — añadir aquí una cuarta dimensión rompería el contrato
    público de esta función.

    Discriminación ViT-L vs ViT-B por el `embed_dim` real del state_dict. Si se
    asumiera siempre `dinov2_vitl14`, un checkpoint entrenado con
    `--backbone_name dinov2_vitb14` (también soportado por el factory, ~85M
    params frente a ~300M de la variante L) se reconstruiría como ViT-L (1024
    dim) y `load_state_dict(strict=True)` fallaría con un shape mismatch poco
    informativo. Para evitarlo se lee `embed_dim` del state_dict: ViT-L = 1024,
    ViT-B = 768. Las claves canónicas son `cls_token` (shape [1, 1, embed_dim])
    y `patch_embed.proj.weight` (shape [embed_dim, 3, 14, 14]). Se prueban los
    prefijos comunes (`backbone.backbone.*` del wrapper de la factory, además de
    `module.*` de DataParallel y `_orig_mod.*` de torch.compile por defensa,
    aunque el caller suele hacer el strip antes). Variantes no registradas
    (ViT-S=384, ViT-G=1536) se detienen con error para forzar su registro
    explícito antes de usarlas.
    """
    keys = list(state_dict.keys())
    is_wrapped = any(".backbone.backbone." in k or k.startswith("backbone.backbone.") for k in keys)
    is_vit = any(("dinov2" in k.lower()) or ("blocks." in k and ".attn." in k) for k in keys)
    is_cross_attn = any("meta_cross_attn." in k for k in keys)

    if is_wrapped and is_vit:
        # Discriminar por embed_dim del state_dict. Se buscan las claves canónicas
        # en orden de fiabilidad: cls_token expone embed_dim como última dimensión
        # del shape ([1, 1, D]); patch_embed.proj.weight como primera dimensión
        # ([D, 3, 14, 14]). Se prueban los prefijos comunes para cubrir el wrapper
        # anidado y cualquier prefijo residual que el caller no haya eliminado
        # (DataParallel, torch.compile).
        embed_dim = None
        candidate_keys = (
            ("backbone.backbone.cls_token", "cls"),
            ("module.backbone.backbone.cls_token", "cls"),
            ("_orig_mod.backbone.backbone.cls_token", "cls"),
            ("backbone.backbone.patch_embed.proj.weight", "patch"),
            ("module.backbone.backbone.patch_embed.proj.weight", "patch"),
            ("_orig_mod.backbone.backbone.patch_embed.proj.weight", "patch"),
        )
        for key, kind in candidate_keys:
            if key in state_dict:
                tensor = state_dict[key]
                if kind == "cls":
                    # cls_token: [1, 1, embed_dim] — última dimensión.
                    embed_dim = int(tensor.shape[-1])
                else:
                    # patch_embed.proj.weight: [embed_dim, 3, 14, 14] — primera.
                    embed_dim = int(tensor.shape[0])
                break

        if embed_dim == 768:
            backbone_name = "dinov2_vitb14"
        elif embed_dim == 1024:
            backbone_name = "dinov2_vitl14"
        else:
            # Variantes desconocidas (ViT-S=384, ViT-G=1536, custom) o claves
            # ausentes: se detiene con error para forzar su registro explícito.
            raise ValueError(
                f"ViT detectado en el checkpoint pero no se pudo "
                f"identificar el backbone exacto. embed_dim={embed_dim!r} "
                f"(esperado 768 para dinov2_vitb14 o 1024 para dinov2_vitl14). "
                f"Si el checkpoint usa una variante ViT no soportada, añadir "
                f"entrada al factory `_SUPPORTED_BACKBONES` en "
                f"`src/models/backbones.py` y ampliar este discriminator."
            )
    elif is_wrapped:
        # CNN envuelto por la factory build_backbone (BackboneWrapper) → el
        # state_dict tiene la anidación backbone.backbone.*. Reconstruir con
        # `model_name=` (path legacy, sin wrapper) produce claves backbone.* de un
        # solo nivel → load_state_dict(strict=True) falla con doble prefijo
        # ("Missing backbone.conv_stem / Unexpected backbone.backbone.conv_stem").
        # Discriminar effnet_b3 (feature_dim 1536) vs effnet_b4 (1792) por la
        # dimensión de salida de conv_head. Mismo patrón fail-loud que
        # la rama ViT: variante CNN no registrada → ValueError explícito. Antes
        # esta rama devolvía None (stub "futuras entradas del factory") y cualquier
        # checkpoint EffNet entrenado con --backbone_name no se podía reconstruir
        # (ni calibrar ni desplegar en predict.py); DINOv2 no se veía afectado
        # porque cae en la rama is_wrapped and is_vit.
        feat_dim = None
        cnn_candidate_keys = (
            "backbone.backbone.conv_head.weight",
            "module.backbone.backbone.conv_head.weight",
            "_orig_mod.backbone.backbone.conv_head.weight",
        )
        for key in cnn_candidate_keys:
            if key in state_dict:
                # conv_head.weight: [feature_dim, in, 1, 1] — primera dimensión.
                feat_dim = int(state_dict[key].shape[0])
                break
        if feat_dim == 1536:
            backbone_name = "effnet_b3"
        elif feat_dim == 1792:
            backbone_name = "effnet_b4"
        else:
            raise ValueError(
                f"CNN envuelto detectado en el checkpoint pero no se pudo "
                f"identificar el backbone exacto. conv_head out={feat_dim!r} "
                f"(esperado 1536 para effnet_b3 o 1792 para effnet_b4). Si el "
                f"checkpoint usa una variante CNN no soportada, añadir entrada al "
                f"factory `_SUPPORTED_BACKBONES` en `src/models/backbones.py` y "
                f"ampliar este discriminator."
            )
    else:
        backbone_name = None  # legacy timm directo
    fusion_mode = "cross_attention" if is_cross_attn else "film"
    return backbone_name, fusion_mode


def _detect_cbam(state_dict):
    """Detecta la presencia de CBAM (use_attention=True) en el checkpoint.

    Análogo a la detección de `projection_head`. Si un checkpoint se entrena con
    `use_attention=False` (una ablación experimental), el engine debe construir
    el modelo con ese kwarg explícito; de lo contrario `load_state_dict(strict=
    True)` falla con "Unexpected keys: cbam.*". Hoy ningún pipeline entrena sin
    CBAM (default True), pero cualquier ablación futura lo activaría.

    Detecta:
    - `cbam.*` (legacy direct)
    - `.cbam.*` (wrapper anidado por DataParallel o compile)
    - `backbone.cbam.*` (caso más común)
    """
    return any(
        k.startswith("cbam.") or ".cbam." in k or k.startswith("backbone.cbam.")
        for k in state_dict.keys()
    )


def _detect_pool(state_dict):
    """Detecta la variante de pooling del checkpoint.

    Igual que con backbone, projection_head y CBAM: el lado de
    inferencia/calibración debe reconstruir `SkinCancerModel` con la
    configuración de pooling con la que se entrenó, o `load_state_dict(strict=
    True)` falla con "Unexpected/Missing key(s)" y "size mismatch for classifier"
    (HybridPool concatena → `2*in_features`; MultiScale → `num_scales*in_features`).

    `SkinCancerModel.__init__` crea `self.hybrid_pool` / `self.multiscale_pool`
    / `self.attention_pool` según el flag activo. El pooling por defecto (GeM)
    no crea ninguno → devuelve None.

    Detección por substring (tolerante a los prefijos `module.`/`_orig_mod.`,
    igual que `_detect_cbam`). El ORDEN es crítico: comprobar `multiscale`/`hybrid`
    ANTES que `attention`, porque `hybrid_pool` contiene internamente `attn_pool`
    y `multiscale_pool` contiene `attn_pools`; el atributo del pooling de atención
    puro es exactamente `attention_pool.` en el nivel superior.

    Returns: "hybrid" | "multiscale" | "attention" | None (None = GeM por defecto).
    """
    keys = list(state_dict.keys())
    if any("multiscale_pool." in k for k in keys):
        return "multiscale"
    if any("hybrid_pool." in k for k in keys):
        return "hybrid"
    if any("attention_pool." in k for k in keys):
        return "attention"
    return None


def _pool_kwargs(state_dict):
    """Mapea `_detect_pool(...)` a los kwargs de `SkinCancerModel`.

    Función única reutilizada por los cinco puntos de reconstrucción (el triaje y
    el especialista de predict.py, sus dos variantes de ensemble, los tres
    scripts de calibración y export_onnx). El pooling por defecto (GeM) devuelve
    `{}`, lo que reconstruye exactamente el comportamiento previo para checkpoints
    sin pooling especial.
    """
    p = _detect_pool(state_dict)
    if p == "hybrid":
        return {"use_hybrid_pool": True}
    if p == "multiscale":
        return {"use_multiscale_vit": True}
    if p == "attention":
        return {"use_attention_pool": True}
    return {}


def _detect_fitnet(state_dict):
    """Detecta `use_fitnet`/`fitnet_teacher_dim` del checkpoint.

    Igual que con pooling, projection_head, CBAM y backbone: el lado de
    inferencia/calibración debe reconstruir `SkinCancerModel` con la
    configuración FitNets exacta del checkpoint, o `load_state_dict(strict=True)`
    falla con:

        Unexpected key(s) in state_dict:
            "fitnet_projection.weight", "fitnet_projection.bias".

    En el flujo de entrenamiento, `train.py` valida que
    `teacher.feature_dim == fitnet_teacher_dim` antes de entrenar; este helper
    aporta la auto-detección simétrica en inferencia y calibración.

    `SkinCancerModel.__init__` crea `self.fitnet_projection = nn.Conv2d(
    in_features, fitnet_teacher_dim, 1, bias=True)` solo si `use_fitnet=True`.
    El weight tiene shape `(teacher_dim, student_dim, 1, 1)`, de donde se lee
    `teacher_dim = shape[0]`.

    Tolerante a los prefijos `module.` (DataParallel), `_orig_mod.`
    (torch.compile) y `backbone.` (anidamiento), igual que `_detect_cbam` y
    `_detect_pool` — coincidencia por sufijo. Sin FitNets en el checkpoint
    devuelve `(False, None)` y la reconstrucción es idéntica a la previa.

    Returns:
        (use_fitnet: bool, fitnet_teacher_dim: int | None)
    """
    target_suffix = "fitnet_projection.weight"
    for k, v in state_dict.items():
        if k == target_suffix or k.endswith("." + target_suffix):
            # weight.shape = (teacher_dim, student_dim, 1, 1)
            return True, int(v.shape[0])
    return False, None


def _fitnet_kwargs(state_dict):
    """Mapea `_detect_fitnet(...)` a los kwargs de `SkinCancerModel`.

    Función única reutilizada por los cinco puntos de reconstrucción (el triaje y
    el especialista de predict.py, sus dos variantes de ensemble, los tres
    scripts de calibración y export_onnx). Sin FitNets en el checkpoint devuelve
    `{}` y la reconstrucción es idéntica a la previa.

    Paralelo a `_pool_kwargs`. Devuelve también `fitnet_teacher_dim` para que el
    constructor cree el `Conv2d` con el mismo shape que el checkpoint; sin esto,
    el valor por defecto (1536) chocaría con cualquier teacher_dim distinto
    (p. ej. 1024 para ViT-L o 768 para ViT-B).
    """
    use_fitnet, teacher_dim = _detect_fitnet(state_dict)
    if use_fitnet:
        return {"use_fitnet": True, "fitnet_teacher_dim": teacher_dim}
    return {}


class ClinicalInferenceEngine:
    """
    Motor de inferencia clínica.

    Reproduce el proceso de decisión de un entorno hospitalario mediante la
    arquitectura en cascada (triaje binario → especialista multiclase). Soporta
    TTA-6, MC Dropout (incertidumbre), Occlusion Sensitivity (explicabilidad)
    y calibración.
    """

    def __init__(
        self, binary_model_path, multiclass_model_path=None, binary_threshold=CLINICAL_THRESHOLD_BINARY,
        temperature_binary_path=None, temperature_multiclass_path=None,
        conformal_binary_path=None, conformal_multiclass_path=None,
        binary_ensemble_paths=None, multiclass_ensemble_paths=None,
        binary_ensemble_weights=None, multiclass_ensemble_weights=None,
        ensemble_min_auc: float = 0.5,
        risk_control_binary_path=None,
        confidence_weighted_tta: bool = False,
        per_site_threshold_path: str | None = None,
        per_site_val_csv_path: str | None = None,
        # TENT (Wang ICLR 2021): adaptación en tiempo de inferencia, opcional.
        # Con `tent_enabled=False` (por defecto) la inferencia es determinista.
        tent_enabled: bool = False,
        tent_lr: float = 1e-3,
        tent_n_steps: int = 1,
        tent_adapt_backbone: bool = False,
        tent_target_modules: list[str] | None = None,
    ):
        """
        Args:
            binary_model_path: Ruta a los pesos del Nivel 1 (Triaje).
            multiclass_model_path: Ruta a los pesos del Nivel 2 (Especialista). Opcional.
            binary_threshold: Umbral clínico. Muy bajo (20%) para asegurar que NINGÚN cáncer
                              se escape (Alta Sensibilidad/Recall). Si pasa del 20%, el especialista lo revisa.
            temperature_binary_path: Ruta a temperatura calibrada del binario. Opcional.
            temperature_multiclass_path: Ruta a temperatura calibrada del multiclase. Opcional.
            ensemble_min_auc: Umbral mínimo clínico para aceptar un modelo en el
                ensemble cuando los pesos se proveen como AUCs de validación.
                Default 0.5 (random guessing como suelo — cualquier modelo peor
                que el azar es inversamente predictivo y arrastra la predicción en
                la dirección incorrecta, haciendo el ensemble peor que el modelo
                principal solo). Si `binary_ensemble_weights` o
                `multiclass_ensemble_weights` contienen entradas
                `< ensemble_min_auc` (excluyendo el índice 0 = modelo principal,
                que SIEMPRE se preserva con warning), se emite `RuntimeWarning` y
                esos modelos se excluyen del ensemble antes de normalizar. Pasar
                `ensemble_min_auc=0.0` desactiva el filtro. Si `ensemble_weights`
                es None, el filtro no aplica (no hay AUC para evaluar).
        """
        # Validar que ensemble_min_auc está en [0, 1].
        if not (0.0 <= ensemble_min_auc <= 1.0):
            raise ValueError(
                f"ensemble_min_auc debe estar en [0, 1], "
                f"recibido {ensemble_min_auc!r}. Pasar 0.0 para desactivar el filtro."
            )
        self._ensemble_min_auc = float(ensemble_min_auc)
        # Confidence-weighted TTA (opcional). Cuando es True, las 6 vistas TTA se
        # promedian con pesos softmax sobre su propia confianza (|p - 0.5|*2 en
        # binario, max(softmax(logits)) en multiclase) en lugar de peso uniforme:
        # una vista confiada (logit=10) pesa más que una incierta (logit=0.1). El
        # resultado sigue siendo determinista (función determinista de los
        # logits). Con False (por defecto) el promedio es uniforme.
        self._confidence_weighted_tta = bool(confidence_weighted_tta)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = binary_threshold
        print(f"Inicializando motor clínico en {self.device}...")

        # 1. Cargar Nivel 1 (Triaje)
        print("Cargando Nivel 1: Triaje Binario (High Recall)...")
        peek_bin = torch.load(binary_model_path, map_location="cpu", weights_only=True)
        bin_backbone, bin_fusion = _detect_backbone_config(peek_bin)
        # Detectar CBAM antes de descartar el peek. El default True conserva el
        # comportamiento en todos los checkpoints entrenados por este proyecto.
        bin_has_cbam = _detect_cbam(peek_bin)
        # Detectar la variante de pooling (hybrid/multiscale/attention) antes de
        # descartar el peek. Sin esto, un checkpoint con hybrid pooling rompería
        # load_state_dict(strict=True).
        bin_pool_kwargs = _pool_kwargs(peek_bin)
        # Detectar FitNets (use_fitnet + fitnet_teacher_dim). Aunque el binario
        # rara vez se entrena con FitNets, se detecta por simetría con el
        # multiclase: si en el futuro se hiciera un experimento de destilación
        # binaria, el reconstructor debe usar la misma config que el checkpoint.
        bin_fitnet_kwargs = _fitnet_kwargs(peek_bin)
        del peek_bin
        # Seleccionar resolución nativa: ViT usa múltiplos de patch_size=14 (294/378px)
        # para evitar resize interno que degrada features dermatoscópicas finas.
        _bin_kind = "vit" if (bin_backbone and ("vit" in bin_backbone or "dinov2" in bin_backbone)) else "cnn"
        _multi_kind = "cnn"  # Default; actualizado si se carga multiclase con ViT
        triaje_kwargs = dict(
            num_classes=1,
            pretrained=False,
            use_metadata=True,
            use_attention=bin_has_cbam,
            fusion_mode=bin_fusion,
            mode="binary",
        )
        triaje_kwargs.update(bin_pool_kwargs)
        triaje_kwargs.update(bin_fitnet_kwargs)
        if bin_backbone is None:
            triaje_kwargs["model_name"] = DEFAULT_BINARY_BACKBONE
        else:
            triaje_kwargs["backbone_name"] = bin_backbone
        self.triaje = SkinCancerModel(**triaje_kwargs)
        # Se asume que DataParallel fue usado en el entrenamiento si hay "module." en los keys
        self._load_weights(self.triaje, binary_model_path)
        self.triaje.to(self.device).eval()

        # Ensemble binario: modelos adicionales (SWA, EMA) para promediar logits.
        # Ganancia esperada: +0.5-1.5% AUC sin coste de entrenamiento adicional.
        # Coste: latencia Nx (N = 1 + len(ensemble_paths)). Cada miembro del
        # ensemble re-detecta su propio backbone/fusion: si tuviera una config
        # distinta (improbable pero posible), se construye correctamente.
        self._triaje_ensemble = []
        if binary_ensemble_paths:
            for ep in binary_ensemble_paths:
                if Path(ep).exists():
                    peek_e = torch.load(ep, map_location="cpu", weights_only=True)
                    e_backbone, e_fusion = _detect_backbone_config(peek_e)
                    # Detección de CBAM por cada miembro del ensemble, simétrica al
                    # re-detect de backbone/fusion.
                    e_has_cbam = _detect_cbam(peek_e)
                    e_pool_kwargs = _pool_kwargs(peek_e)
                    e_fitnet_kwargs = _fitnet_kwargs(peek_e)
                    del peek_e
                    triaje_kwargs_e = dict(triaje_kwargs)
                    triaje_kwargs_e["fusion_mode"] = e_fusion
                    triaje_kwargs_e["use_attention"] = e_has_cbam
                    # El pooling del miembro puede diferir del principal. Se limpian
                    # los heredados y se aplica el propio.
                    for _pk in ("use_hybrid_pool", "use_multiscale_vit", "use_attention_pool"):
                        triaje_kwargs_e.pop(_pk, None)
                    triaje_kwargs_e.update(e_pool_kwargs)
                    # La config FitNets del miembro también puede diferir del
                    # principal (defensa por simetría — en la práctica todos los
                    # miembros comparten arquitectura, pero un fork que combine
                    # checkpoints heterogéneos lo necesitaría).
                    for _fk in ("use_fitnet", "fitnet_teacher_dim"):
                        triaje_kwargs_e.pop(_fk, None)
                    triaje_kwargs_e.update(e_fitnet_kwargs)
                    if e_backbone is not None:
                        triaje_kwargs_e.pop("model_name", None)
                        triaje_kwargs_e["backbone_name"] = e_backbone
                    elif "backbone_name" in triaje_kwargs_e and e_backbone is None:
                        triaje_kwargs_e.pop("backbone_name")
                        triaje_kwargs_e["model_name"] = DEFAULT_BINARY_BACKBONE
                    extra = SkinCancerModel(**triaje_kwargs_e)
                    self._load_weights(extra, ep)
                    extra.to(self.device).eval()
                    self._triaje_ensemble.append(extra)
                    # Se usa warnings.warn (no print) para que el evento sea
                    # capturable por el handler de logging del proceso.
                    warnings.warn(
                        f"Ensemble binario: +1 modelo desde {Path(ep).name}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if self._triaje_ensemble:
                warnings.warn(
                    f"Ensemble binario total: {1 + len(self._triaje_ensemble)} modelos",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Ensemble ponderado por AUC de validación.
        # `binary_ensemble_weights` es una lista [w_main, w_extra_1, w_extra_2, ...]
        # donde cada peso es la métrica de validación (típicamente AUC) del
        # checkpoint correspondiente. Se normalizan a suma 1 y se usan como pesos
        # de promedio en `_forward_ensemble_binary`. Si es None, se usa promedio
        # uniforme.
        #
        # Antes de normalizar, se filtran los modelos con AUC < ensemble_min_auc
        # (default 0.5 = azar). Un modelo inversamente predictivo en el ensemble
        # arrastra las predicciones en la dirección incorrecta con mayor confianza
        # que el modelo principal solo, degradando el ensemble en silencio. El
        # modelo principal (índice 0) siempre se preserva, con warning si su propia
        # AUC es baja (no hay otro fallback).
        binary_ensemble_weights, self._triaje_ensemble = self._filter_ensemble_by_min_auc(
            binary_ensemble_weights,
            self._triaje_ensemble,
            min_auc=self._ensemble_min_auc,
            label="binario",
        )
        self._triaje_ensemble_weights = self._normalize_ensemble_weights(
            binary_ensemble_weights,
            expected_len=1 + len(self._triaje_ensemble),
            label="binario",
        )

        # Importados de src/data/constants.py (fuente de verdad unica)
        self.multiclass_map = MULTICLASS_INDEX_MAP
        self.sex_map = SEX_MAP
        self.site_map = SITE_MAP

        # num_classes se deriva del label map centralizado en vez de hardcodearse.
        # Si se añade una clase a MULTICLASS_LABEL_MAP en constants.py, el modelo
        # se construye con el tamaño correcto automáticamente.
        self._num_classes_multi = len(self.multiclass_map)

        # 2. Cargar Nivel 2 (Especialista) - Solo si se provee
        if multiclass_model_path and Path(multiclass_model_path).exists():
            print("Cargando Nivel 2: Especialista Multiclase (High Precision)...")
            # Se detecta automáticamente si el checkpoint contiene projection_head
            # (SupCon activo). Hardcodear use_projection_head=True rompería la carga
            # con strict=True si el multiclase se reentrena sin SupCon. El
            # projection_head no se usa en inferencia (return_features=False
            # siempre); solo se necesita para que las claves del state_dict
            # coincidan con la arquitectura instanciada.
            peek_state = torch.load(
                multiclass_model_path, map_location="cpu", weights_only=True
            )
            has_projection_head = any(
                k.startswith("projection_head.") or k.startswith("module.projection_head.")
                for k in peek_state.keys()
            )
            multi_backbone, multi_fusion = _detect_backbone_config(peek_state)
            # Detección de CBAM, paralela a has_projection_head.
            multi_has_cbam = _detect_cbam(peek_state)
            multi_pool_kwargs = _pool_kwargs(peek_state)
            multi_fitnet_kwargs = _fitnet_kwargs(peek_state)
            del peek_state
            _multi_kind = "vit" if (multi_backbone and ("vit" in multi_backbone or "dinov2" in multi_backbone)) else "cnn"
            multi_kwargs = dict(
                num_classes=self._num_classes_multi,
                pretrained=False,
                use_metadata=True,
                use_attention=multi_has_cbam,
                use_projection_head=has_projection_head,
                fusion_mode=multi_fusion,
                mode="multiclass",
            )
            multi_kwargs.update(multi_pool_kwargs)
            # La config FitNets es crítica en el multiclase: si el checkpoint se
            # entrenó con --use_fitnet, su state_dict contiene
            # fitnet_projection.{weight,bias}. Sin reconstruir con esa config, el
            # load_state_dict(strict=True) de abajo fallaría con "Unexpected keys:
            # fitnet_projection.*" y bloquearía la cascada de calibración,
            # inferencia y exportación a ONNX.
            multi_kwargs.update(multi_fitnet_kwargs)
            if multi_backbone is None:
                multi_kwargs["model_name"] = DEFAULT_MULTICLASS_BACKBONE
            else:
                multi_kwargs["backbone_name"] = multi_backbone
            self.especialista = SkinCancerModel(**multi_kwargs)
            self._load_weights(self.especialista, multiclass_model_path)
            self.especialista.to(self.device).eval()

            # Ensemble multiclase (SWA, EMA). Cada miembro re-detecta su
            # backbone/fusion/projection_head desde su propio checkpoint.
            self._especialista_ensemble = []
            if multiclass_ensemble_paths:
                for ep in multiclass_ensemble_paths:
                    if Path(ep).exists():
                        peek_e = torch.load(ep, map_location="cpu", weights_only=True)
                        has_ph_e = any(
                            k.startswith("projection_head.") or k.startswith("module.projection_head.")
                            for k in peek_e.keys()
                        )
                        e_backbone, e_fusion = _detect_backbone_config(peek_e)
                        # Detección de CBAM por cada miembro del ensemble.
                        e_has_cbam = _detect_cbam(peek_e)
                        e_pool_kwargs = _pool_kwargs(peek_e)
                        e_fitnet_kwargs = _fitnet_kwargs(peek_e)
                        del peek_e
                        multi_kwargs_e = dict(multi_kwargs)
                        multi_kwargs_e["use_projection_head"] = has_ph_e
                        multi_kwargs_e["use_attention"] = e_has_cbam
                        multi_kwargs_e["fusion_mode"] = e_fusion
                        # El pooling del miembro puede diferir del principal.
                        for _pk in ("use_hybrid_pool", "use_multiscale_vit", "use_attention_pool"):
                            multi_kwargs_e.pop(_pk, None)
                        multi_kwargs_e.update(e_pool_kwargs)
                        # La config FitNets del miembro puede diferir del principal
                        # (defensa por simetría con el re-detect del pooling).
                        for _fk in ("use_fitnet", "fitnet_teacher_dim"):
                            multi_kwargs_e.pop(_fk, None)
                        multi_kwargs_e.update(e_fitnet_kwargs)
                        if e_backbone is not None:
                            multi_kwargs_e.pop("model_name", None)
                            multi_kwargs_e["backbone_name"] = e_backbone
                        elif "backbone_name" in multi_kwargs_e and e_backbone is None:
                            multi_kwargs_e.pop("backbone_name")
                            multi_kwargs_e["model_name"] = DEFAULT_MULTICLASS_BACKBONE
                        extra = SkinCancerModel(**multi_kwargs_e)
                        self._load_weights(extra, ep)
                        extra.to(self.device).eval()
                        self._especialista_ensemble.append(extra)
                        # Igual que en el binario: warnings.warn para que el evento
                        # sea capturable por el handler de logging del proceso.
                        warnings.warn(
                            f"Ensemble multiclase: +1 modelo desde {Path(ep).name}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                if self._especialista_ensemble:
                    warnings.warn(
                        f"Ensemble multiclase total: {1 + len(self._especialista_ensemble)} modelos",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            # Mismo filtro de min_auc que en el binario.
            multiclass_ensemble_weights, self._especialista_ensemble = self._filter_ensemble_by_min_auc(
                multiclass_ensemble_weights,
                self._especialista_ensemble,
                min_auc=self._ensemble_min_auc,
                label="multiclase",
            )
            self._especialista_ensemble_weights = self._normalize_ensemble_weights(
                multiclass_ensemble_weights,
                expected_len=1 + len(self._especialista_ensemble),
                label="multiclase",
            )
        else:
            self.especialista = None
            self._especialista_ensemble = []
            self._especialista_ensemble_weights = None
            # Se usa warnings.warn en vez de print: un print se pierde en stdout
            # mezclado con tqdm y no es capturable por los handlers de logging del
            # proceso (el evento debe quedar registrado para auditoría).
            warnings.warn(
                "Nivel 2 no cargado (solo funcionará el triaje).",
                RuntimeWarning,
                stacklevel=2,
            )

        # Pipeline de Visión (DEBE ser idéntico al de validación en dataset.py)
        # Resolución nativa centralizada en IMAGE_SIZES (constants.py):
        # CNN (B3/B4): 300/380px. ViT (DINOv2): 294/378px (múltiplos de patch_size=14).
        def get_pipeline(size):
            return A.Compose([
                A.Resize(size, size, interpolation=cv2.INTER_CUBIC),
                HairRemoval(p=1.0),
                build_stain_transform(),  # fuente de verdad única en preprocessing.py
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        self.vision_pipeline_binary = get_pipeline(IMAGE_SIZES[_bin_kind]["binary"])
        self.vision_pipeline_multi = get_pipeline(IMAGE_SIZES[_multi_kind]["multiclass"])

        # Calibración de temperatura (opcional). La carga usa try/except más
        # validación de T para no detener el engine si el archivo está corrupto o
        # la temperatura aprendida es inválida; en ese caso, se opera sin calibración.
        self.temp_binary = self._load_temperature_safe(
            temperature_binary_path, label="binaria"
        )
        self.temp_multiclass = self._load_temperature_safe(
            temperature_multiclass_path, label="multiclase"
        )

        # Conformal Prediction (opcional). Mismo patrón defensivo que temperatura:
        # try/except + path-exists check → fallback a None con warning, nunca crashea el engine.
        self.binary_conformal = self._load_conformal_safe(
            conformal_binary_path, cls=BinaryConformal, label="binario"
        )
        self.multiclass_conformal = self._load_conformal_safe(
            conformal_multiclass_path, cls=APSConformal, label="multiclase"
        )
        # ConformalRiskControl (Angelopoulos 2022), opcional. Complementa a
        # `binary_conformal` con un threshold calibrado para acotar directamente
        # el FNR, más alineado con la métrica que regula la autoridad sanitaria
        # (fracción de melanomas perdidos, no la cobertura marginal 1-α). Si el
        # predictor se carga con éxito y se invoca `infer(risk_control=True)`, el
        # report incluye `risk_control_binary` y `risk_control_fnr_target`. Una
        # discrepancia con el triaje estándar dispara `risk_control_disagrees` en
        # `_compute_defer_to_human`.
        self.risk_control_binary = self._load_risk_control_safe(
            risk_control_binary_path, label="binario"
        )

        # Calibración del threshold por sitio anatómico, opcional. Si se carga, el
        # threshold de decisión RED/GREEN se selecciona dinámicamente por sitio en
        # `_get_effective_threshold`, con fallback al `binary_threshold` global
        # para los sitios no calibrados. `per_site_val_csv_path`
        # (opcional) habilita la verificación del `val_csv_hash`; si se omite, el
        # hash se verifica solo si el CSV de validación está accesible desde el
        # directorio de trabajo actual.
        self.per_site_threshold = self._load_per_site_threshold_safe(
            per_site_threshold_path, label="binario", val_csv_path=per_site_val_csv_path
        )

        # Inicializar class_names del APSConformal una sola vez (antes era lazy
        # en cada llamada a infer()).
        if (
            self.multiclass_conformal is not None
            and self.multiclass_conformal.class_names is None
        ):
            self.multiclass_conformal.class_names = [
                self.multiclass_map[i] for i in range(self._num_classes_multi)
            ]

        # Buffers de cobertura separados por tipo de predictor. Con un único
        # buffer compartido, registrar un outcome multiclase sin indicar el alpha
        # explícito haría que el monitor usara el alpha del binario (0.03) en lugar
        # del multiclase (0.10), dando falsas alarmas de domain shift o, peor, un
        # shift real invisible. Con dos deques dedicados y dos métodos separados,
        # el caller indica el tipo eligiendo el método y cada monitor usa su alpha
        # correcto sin ambigüedad.
        self._conformal_outcomes_binary = deque(maxlen=500)
        self._conformal_outcomes_multiclass = deque(maxlen=500)

        # Adapters TENT episódicos (Wang ICLR 2021). La construcción depende de
        # `tent_enabled=True`; con el valor por defecto la inferencia es
        # determinista. Cuando se habilita, se construye un adapter por nivel de la
        # cascada (binario y, si existe el especialista, multiclase). El flag de
        # runtime `infer(tent=True)` decide si se aplica la adaptación en cada
        # request; el adapter siempre está listo si `tent_enabled=True`.
        #
        # Ensemble + TENT: incoherencia matemática (cada modelo del ensemble se
        # adaptaría de forma distinta). Solo el modelo principal (`self.triaje` /
        # `self.especialista`) adapta; los extras del ensemble permanecen fijos.
        # Se emite `RuntimeWarning` si ambos están activos para avisar al operador.
        self._tent_enabled = bool(tent_enabled)
        self._tent_binary = None
        self._tent_multiclass = None
        if self._tent_enabled:
            from src.inference.tent import TENTAdapter
            self._tent_binary = TENTAdapter(
                self.triaje, mode="binary",
                target_modules=tent_target_modules,
                adapt_backbone=tent_adapt_backbone,
                lr=tent_lr, n_steps=tent_n_steps,
            )
            if self._triaje_ensemble:
                warnings.warn(
                    "tent_enabled=True con binary_ensemble_paths "
                    "cargados. TENT solo se aplica al modelo principal "
                    "(`self.triaje`); los extras del ensemble permanecen fijos. "
                    "El promedio ponderado del ensemble pierde coherencia "
                    "matemática porque el principal adapta y los extras no — "
                    "el resultado puede ser subóptimo frente a un único modelo con "
                    "TENT. Considerar desactivar el ensemble cuando tent_enabled=True.",
                    RuntimeWarning, stacklevel=2,
                )
            if self.especialista is not None:
                self._tent_multiclass = TENTAdapter(
                    self.especialista, mode="multiclass",
                    target_modules=tent_target_modules,
                    adapt_backbone=tent_adapt_backbone,
                    lr=tent_lr, n_steps=tent_n_steps,
                )
                if self._especialista_ensemble:
                    warnings.warn(
                        "tent_enabled=True con "
                        "multiclass_ensemble_paths cargados. Mismo caveat que en "
                        "el binario — el ensemble pierde coherencia con TENT.",
                        RuntimeWarning, stacklevel=2,
                    )

    @staticmethod
    def _to_numpy_safe(tensor):
        """Conversión tensor→numpy segura bajo cualquier contexto de gradiente.

        `torch.Tensor.numpy()` comprueba `requires_grad` y `grad_fn`; ambos pueden
        estar activos si el caller opera bajo `torch.enable_grad()` (que
        `infer(explain=True)` activa). Sin `.detach()` previo, el forward del
        modelo bajo enable_grad produce tensores con `requires_grad=True` y
        `.cpu().numpy()` falla con `RuntimeError: Can't call numpy() on Tensor
        that requires grad. Use tensor.detach().numpy() instead.`

        El problema es independiente del device: `tensor.numpy()` valida el
        grad-tracking, no el device, y `.cpu()` preserva `requires_grad` (no hace
        un detach implícito), de modo que se reproduce igual en CPU y GPU.

        Este helper aplica siempre `.detach().cpu().numpy()` — idempotente si el
        tensor ya está detached, pero robusto ante cualquier caller futuro que
        envuelva la inferencia en contextos con gradiente (explain=True, ataques
        adversariales, aprendizaje continuo, etc.).

        Contrato: todo uso de `.cpu().numpy()` sobre outputs del modelo dentro del
        engine debe pasar por este helper.
        """
        return tensor.detach().cpu().numpy()

    def _check_conformal_coverage(self, buffer, alpha, label):
        """Helper común para ambos monitores de cobertura (binary/multiclass)."""
        n = len(buffer)
        if n < 50:
            return
        coverage = sum(buffer) / n
        min_coverage = (1.0 - alpha) - 0.05
        if coverage < min_coverage:
            warnings.warn(
                f"Cobertura conformal {label} empirica {coverage:.2%} sobre ultimos "
                f"{n} casos esta por debajo del minimo esperado "
                f"{min_coverage:.2%} (1-alpha-0.05, alpha={alpha}). "
                "Posible domain shift. Recalibracion recomendada.",
                RuntimeWarning,
                stacklevel=3,
            )

    # Vocabularios canónicos para la validación defensiva de los monitores de
    # cobertura conformal. Si un caller envía `ground_truth` en mayúsculas
    # ("Benigno"), con case mismatch o con el vocabulario equivocado (p. ej. una
    # clase multiclase en el monitor binario), el operador `in` evaluaría a False
    # en silencio → el outcome se registraría como "no cubierto" → el monitor
    # inflaría su alerta de domain shift (cobertura empírica artificialmente baja)
    # → el operador recibiría un aviso falso o se acostumbraría a ignorarlo, de
    # modo que un domain shift real pasaría desapercibido. El vocabulario binario
    # es `{"benigno", "maligno"}` (en minúsculas). El multiclase se deriva en
    # tiempo de llamada de `self.multiclass_map` (fuente única de verdad).
    _CONFORMAL_VOCAB_BINARY = frozenset({"benigno", "maligno"})

    @staticmethod
    def _validate_conformal_log_args(pred_set, ground_truth, vocab, label):
        """Validación compartida por ambos monitores de cobertura conformal.

        Rechaza un pred_set no iterable, un ground_truth no string y los elementos
        fuera del vocabulario canónico. Se detiene con error en lugar de degradar
        en silencio: un monitor clínico con la cobertura empírica sesgada es peor
        que un ValueError inmediato que el operador puede diagnosticar.
        """
        if pred_set is None or not isinstance(pred_set, (list, tuple, set, frozenset)):
            raise ValueError(
                f"log_conformal_outcome_{label}.pred_set debe ser "
                f"list/tuple/set, recibido {type(pred_set).__name__}."
            )
        if not isinstance(ground_truth, str):
            raise ValueError(
                f"log_conformal_outcome_{label}.ground_truth debe "
                f"ser str, recibido {type(ground_truth).__name__}."
            )
        for item in pred_set:
            if not isinstance(item, str) or item not in vocab:
                raise ValueError(
                    f"log_conformal_outcome_{label}.pred_set "
                    f"contiene el elemento {item!r} fuera del vocabulario "
                    f"canónico {sorted(vocab)}. Verificar que el caller "
                    "pasa el set del predictor correcto (binario vs multiclase)."
                )
        if ground_truth not in vocab:
            raise ValueError(
                f"log_conformal_outcome_{label}.ground_truth "
                f"{ground_truth!r} fuera del vocabulario canónico "
                f"{sorted(vocab)}. Verificar el case (minúsculas en el "
                "binario) y la consistencia con el predictor usado."
            )

    def log_conformal_outcome_binary(self, pred_set, ground_truth):
        """
        Registra el outcome del set conformal BINARIO del triaje.

        Args:
            pred_set: lista de strings del set conformal binario (ej. `["maligno"]`
                o `["benigno", "maligno"]`).
            ground_truth: string con la clase real ("benigno" o "maligno"),
                típicamente tras confirmación por biopsia.

        Usa `self.binary_conformal.alpha` (sin fallback asimétrico). Si el
        predictor binario no está cargado, no registra nada. Valida tipos y
        vocabulario antes de registrar el outcome para proteger la integridad del
        monitor de cobertura empírica.
        """
        if self.binary_conformal is None:
            return
        self._validate_conformal_log_args(
            pred_set, ground_truth,
            vocab=self._CONFORMAL_VOCAB_BINARY,
            label="binary",
        )
        self._conformal_outcomes_binary.append(ground_truth in pred_set)
        self._check_conformal_coverage(
            self._conformal_outcomes_binary,
            self.binary_conformal.alpha,
            label="binaria",
        )

    def log_conformal_outcome_multiclass(self, pred_set, ground_truth):
        """
        Registra el outcome del set conformal APS MULTICLASE.

        Args:
            pred_set: lista de strings del APS set (nombres de clase del
                `MULTICLASS_INDEX_MAP`).
            ground_truth: string con la clase real confirmada.

        Usa `self.multiclass_conformal.alpha` — simétrico al binario pero con el
        alpha correcto del predictor multiclase (0.10 frente a 0.03 del binario).
        El vocabulario se deriva de `self.multiclass_map`.
        """
        if self.multiclass_conformal is None:
            return
        vocab_multiclass = frozenset(self.multiclass_map.values())
        self._validate_conformal_log_args(
            pred_set, ground_truth,
            vocab=vocab_multiclass,
            label="multiclass",
        )
        self._conformal_outcomes_multiclass.append(ground_truth in pred_set)
        self._check_conformal_coverage(
            self._conformal_outcomes_multiclass,
            self.multiclass_conformal.alpha,
            label="multiclase",
        )

    @staticmethod
    def _load_risk_control_safe(path, label):
        """Carga ConformalRiskControl con el mismo guard defensivo que el conformal.

        Paridad exacta con `_load_conformal_safe` pero para `ConformalRiskControl`.
        Este predictor no tiene atributo `.alpha`; su equivalente es `.target_fnr`.
        El hash del val CSV se verifica igual que en el conformal regular (mismo
        `_verify_conformal_val_hash` — el val CSV usado es el binario,
        label="binario").

        Fallback a None con warning si:
        - path=None o archivo inexistente (la opción no está activada)
        - checkpoint corrupto (ValueError en load por threshold no finito)
        - hash distinto del val CSV actual (warning capturable, no se detiene).
        """
        if not path or not Path(path).exists():
            return None
        try:
            obj = ConformalRiskControl.load(path)
            print(
                f"   RiskControl {label} cargado: target_fnr={obj.target_fnr}, "
                f"threshold={obj.threshold:.4f}, "
                f"n_calib_pos={obj.n_calib_pos} n_calib_neg={obj.n_calib_neg}"
            )
            # Mismo val CSV que BinaryConformal. El label "binario" hace que
            # `_verify_conformal_val_hash` compruebe `04_val.csv`.
            ClinicalInferenceEngine._verify_conformal_val_hash(obj, label)
            return obj
        except Exception as e:
            warnings.warn(
                f"RiskControl {label} no disponible ({e}). "
                "Inferencia continua sin control directo de FNR.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    @staticmethod
    def _load_conformal_safe(path, cls, label):
        """Carga un predictor conformal con guard contra archivos corruptos.

        Tras la carga, verifica que el hash SHA256 del val CSV que usó el script
        de calibración coincide con el val CSV actual en disco. Un hash distinto
        indica que el split se regeneró tras la calibración → la garantía de
        cobertura del Split Conformal asumía un conjunto que ya no existe. Emite
        `RuntimeWarning` explícito (no se detiene: en producción clínica, una
        degradación capturable por el handler del servicio es preferible a un
        crash del engine). La verificación es silenciosa en el caso normal y
        compatible con checkpoints antiguos (sin hash).
        """
        if not path or not Path(path).exists():
            return None
        try:
            obj = cls.load(path)
            print(f"   Conformal {label} cargado: alpha={obj.alpha}, n_calib={obj.n_calib}")
            # Verificación de hash del val CSV.
            ClinicalInferenceEngine._verify_conformal_val_hash(obj, label)
            return obj
        except Exception as e:
            warnings.warn(
                f"Conformal {label} no disponible ({e}). "
                "Inferencia continua sin prediccion conformal.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    @staticmethod
    def _verify_conformal_val_hash(obj, label):
        """Verifica que el val CSV usado en calibración sigue siendo el mismo que
        hay actualmente en disco. Si difiere → warning capturable (no se detiene).

        `label` es ``'binario'`` o ``'multiclase'`` — decide qué CSV comprobar
        (`04_val.csv` vs `07_val_multiclass.csv`).
        """
        stored_hash = getattr(obj, "val_csv_hash", "") or ""
        if not stored_hash:
            # Checkpoint legacy sin hash → no podemos verificar, silencioso.
            return
        csv_paths = {
            "binario": Path("data") / "processed" / "04_val.csv",
            "multiclase": Path("data") / "processed" / "07_val_multiclass.csv",
        }
        csv_path = csv_paths.get(label)
        if csv_path is None or not csv_path.exists():
            # No hay val CSV accesible desde el cwd → no podemos verificar.
            # Típico en entornos de producción donde solo se despliegan
            # checkpoints. Silencioso por diseño.
            return
        import hashlib as _hashlib
        sha = _hashlib.sha256()
        with open(csv_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        current_hash = sha.hexdigest()
        if current_hash != stored_hash:
            warnings.warn(
                f"Hash del val CSV ({csv_path}) divergente. "
                f"Calibración conformal {label} usó hash={stored_hash[:16]}..., "
                f"actual={current_hash[:16]}.... El val set se regeneró desde "
                f"la calibración → la garantía de cobertura del Split Conformal "
                f"puede estar rota. Re-ejecuta scripts/calibrate_conformal.py.",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _load_temperature_safe(path, label):
        """Carga un TemperatureScaler con guard contra archivos corruptos o T no finita.

        Soporta vector scaling (multiclase). Lee `num_classes` del
        checkpoint y reconstruye el scaler con el tamaño correcto. Backwards-
        compatible con checkpoints escalares legacy.
        """
        if not path or not Path(path).exists():
            return None
        try:
            # Peek del archivo para decidir num_classes antes de instanciar el scaler.
            peek = torch.load(path, map_location="cpu", weights_only=True)
            nc = int(peek.get("num_classes", 1)) if isinstance(peek, dict) else 1
            # Validar nc > 0. Un checkpoint corrupto con `num_classes=0` o
            # `num_classes=-1` construiría un TemperatureScaler con tensor
            # `torch.ones(0)` (vacío). Downstream, `all(... for v in [])` es True
            # (verdad vacua) → el guard `math.isfinite(v) and v > 0` pasaría
            # falsamente, el scaler quedaría como no-op silencioso y la inferencia
            # operaría con T=1.0 sin avisar al clínico. Se detiene con error.
            if nc < 1:
                raise ValueError(
                    f"num_classes debe ser >= 1, checkpoint reporta {nc}. "
                    "Causa probable: archivo corrupto o generado por script externo "
                    "sin validar. No se puede cargar T con dimensión no positiva."
                )
            scaler = TemperatureScaler(num_classes=nc)
            scaler.load(path)
            t_tensor = scaler.temperature.detach().cpu()
            t_vals = t_tensor.tolist()
            if not t_vals or not all(math.isfinite(v) and v > 0 for v in t_vals):
                raise ValueError(f"Temperatura invalida: T={t_vals}")
            if nc == 1:
                print(f"   Temperatura {label} cargada (scalar): T={t_vals[0]:.3f}")
            else:
                t_rounded = [round(v, 3) for v in t_vals]
                print(f"   Temperatura {label} cargada (vector, C={nc}): T={t_rounded}")
            return scaler
        except Exception as e:
            warnings.warn(
                f"Calibracion {label} no disponible ({e}). "
                "Inferencia continua sin calibracion (T=1.0).",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    @staticmethod
    def _load_per_site_threshold_safe(path, label, val_csv_path=None):
        """Carga un PerSiteThresholdCalibrator con guards.

        Paralelo a `_load_temperature_safe` y `_load_conformal_safe`:
          - Si `path` es None o el archivo no existe → retorna None (el engine usa
            el threshold global).
          - Si la carga falla por cualquier motivo, emite `RuntimeWarning` y
            retorna None (degradación controlada, nunca detiene el engine).
          - Si `val_csv_path` se provee y existe, valida `val_csv_hash`: un hash
            distinto emite warning pero permite continuar — el operador decide si
            recalibrar.

        Returns:
            PerSiteThresholdCalibrator | None
        """
        if not path or not Path(path).exists():
            return None
        try:
            calibrator = PerSiteThresholdCalibrator()
            calibrator.load(path)
            n_sites = sum(1 for v in calibrator.thresholds.values() if v is not None)
            n_total = len(calibrator.thresholds)
            # Detecta el caso patológico en el que todos los thresholds son None,
            # que dejaría el calibrador inservible. En ese caso el engine cargaría
            # el .pt sin alertar y caería silenciosamente al fallback global (0.20)
            # para todos los sitios, sin que el operador supiera que la calibración
            # por sitio está desactivada. Se emite un warning estructurado
            # (capturable por el handler de logging) y se retorna None para forzar
            # un fallback explícito con threshold_source="global_fallback" en el
            # report.
            if n_sites == 0:
                warnings.warn(
                    f"Per-site threshold {label} con TODOS los "
                    f"thresholds=None ({n_total} sitios). El calibrador queda "
                    f"desactivado — todos los sitios caen al "
                    f"fallback global ({calibrator.fallback_threshold}). "
                    f"Re-ejecutar scripts/calibrate_per_site_threshold.py para "
                    f"regenerar el .pt.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            # Se usa warnings.warn en vez de print para que este evento auditable
            # sea capturable por el handler de logging del proceso (un print se
            # perdería en stdout mezclado).
            warnings.warn(
                f"Per-site threshold {label}: {n_sites}/{n_total} sitios "
                f"calibrados (target_recall={calibrator.target_recall}, "
                f"fallback={calibrator.fallback_threshold}).",
                RuntimeWarning,
                stacklevel=2,
            )
            # Validar el hash del val CSV si se proporcionó.
            if val_csv_path and Path(val_csv_path).exists() and calibrator.val_csv_hash:
                try:
                    import hashlib
                    h = hashlib.sha256()
                    with open(val_csv_path, "rb") as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            h.update(chunk)
                    current_hash = h.hexdigest()
                    if current_hash != calibrator.val_csv_hash:
                        warnings.warn(
                            f"Hash del val CSV ({val_csv_path}) "
                            f"divergente. Calibracion per-site threshold {label} usa "
                            f"hash={calibrator.val_csv_hash[:16]}..., "
                            f"actual={current_hash[:16]}.... Re-ejecuta "
                            "scripts/calibrate_per_site_threshold.py.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                except OSError:
                    pass  # Lectura del val CSV fallo — no bloquea engine.
            return calibrator
        except Exception as e:
            warnings.warn(
                f"Per-site threshold {label} no disponible ({e}). "
                f"Inferencia continua con threshold global "
                f"(fallback global).",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    @staticmethod
    def _filter_ensemble_by_min_auc(raw_weights, ensemble_extras, min_auc, label):
        """Filtra del ensemble los modelos con AUC de validación < min_auc.

        Un modelo inversamente predictivo (AUC<0.5) en un ensemble promedia las
        predicciones en la dirección incorrecta con más confianza que el modelo
        principal solo → degradación silenciosa que en contexto clínico clasifica
        melanomas como benignos. `_normalize_ensemble_weights` solo valida
        finitud y no-negatividad — aceptaría AUC=0.47 sin alertar.

        Contrato: se aplica ANTES de la normalización a suma=1. El modelo
        principal (índice 0) SIEMPRE se preserva (no hay fallback si su propia AUC
        es baja); emite `RuntimeWarning` si su weight < min_auc. Los extras con
        weight < min_auc se excluyen de AMBAS listas (raw_weights y
        ensemble_extras) con `RuntimeWarning` explícito para trazabilidad clínica.

        Args:
            raw_weights: None | list — pesos crudos [w_main, w_extra_1, ...] o None.
            ensemble_extras: list — modelos extras cargados (sin main).
            min_auc: float — umbral mínimo. 0.0 desactiva el filtro.
            label: str — "binario"|"multiclase" para logging.

        Returns:
            (filtered_raw_weights, filtered_ensemble_extras):
                filtered_raw_weights: None si raw_weights era None; else lista
                  con los extras AUC<min_auc eliminados (main siempre conservado).
                filtered_ensemble_extras: lista de extras superviviente.
        """
        if raw_weights is None or min_auc <= 0.0:
            return raw_weights, ensemble_extras
        w = list(raw_weights)
        if len(w) != 1 + len(ensemble_extras):
            # Delega la validación de longitud a _normalize_ensemble_weights,
            # que se detiene con un mensaje descriptivo si no coincide.
            return raw_weights, ensemble_extras
        # Main (índice 0): siempre preservado, warning si < min_auc.
        if float(w[0]) < min_auc:
            warnings.warn(
                f"Ensemble {label}: el modelo principal tiene weight="
                f"{w[0]:.3f} < min_auc={min_auc}. Se preserva por ser el "
                f"principal (no hay fallback), pero la calidad del sistema está "
                f"comprometida — considerar reentrenar o bajar min_auc.",
                RuntimeWarning, stacklevel=3,
            )
        # Extras: filtrar índices con weight < min_auc.
        kept_weights = [w[0]]
        kept_extras = []
        excluded = []
        for idx, (weight, extra) in enumerate(zip(w[1:], ensemble_extras), start=1):
            if float(weight) < min_auc:
                excluded.append((idx, float(weight)))
            else:
                kept_weights.append(weight)
                kept_extras.append(extra)
        if excluded:
            warnings.warn(
                f"Ensemble {label}: {len(excluded)} modelo(s) "
                f"excluido(s) por AUC < {min_auc}: "
                f"{[f'idx{i}=auc{w:.3f}' for i, w in excluded]}. "
                f"El ensemble queda con {len(kept_extras)} extras + 1 principal.",
                RuntimeWarning, stacklevel=3,
            )
        return kept_weights, kept_extras

    @staticmethod
    def _normalize_ensemble_weights(raw_weights, expected_len, label):
        """Valida y normaliza pesos de ensemble a suma=1.

        Pesos negativos → ValueError. Longitud != expected_len → ValueError.
        Suma ≈ 0 → ValueError. None → retorna None (promedio uniforme en forward).
        """
        if raw_weights is None:
            return None
        w = np.asarray(raw_weights, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != expected_len:
            raise ValueError(
                f"ensemble_weights {label}: longitud {w.shape} ≠ {expected_len} "
                f"(= 1 modelo principal + {expected_len - 1} extras)."
            )
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError(
                f"ensemble_weights {label}: pesos deben ser finitos y ≥0, recibido {w}."
            )
        total = w.sum()
        if total < 1e-9:
            raise ValueError(
                f"ensemble_weights {label}: suma ≈ 0 ({total}), no normalizable."
            )
        normalized = (w / total).tolist()
        print(
            f"   Ensemble {label}: pesos AUC-ponderados normalizados = "
            f"{[round(v, 4) for v in normalized]}"
        )
        return normalized

    def _forward_ensemble_binary(self, view, meta):
        """Forward por todos los modelos binarios del ensemble, promediando logits.

        Verifica NaN por modelo individual: si 1 de N modelos produce NaN, descarta
        solo ese modelo y promedia los válidos. Si todos son NaN, retorna NaN para
        que el guard TTA lo maneje normalmente (vista descartada). Sin esta
        verificación, un solo modelo NaN envenenaría el promedio completo y el guard
        TTA descartaría la vista entera, perdiendo las contribuciones válidas.
        """
        logit = self.triaje(view, meta)
        if self._triaje_ensemble:
            # Ponderación por AUC de validación (si se proveyeron pesos). El
            # promedio sobre los modelos válidos es:
            #   avg = Σ(w_i * logit_i) / Σ(w_i)   — sobre los no-NaN
            # Si `weights` es None, equivale a promedio uniforme.
            weights = self._triaje_ensemble_weights
            # `_to_numpy_safe` aplica `.detach()` antes de `.numpy()`. Sin detach,
            # bajo `explain=True` (que activa enable_grad) fallaría con RuntimeError.
            logit_val = self._to_numpy_safe(logit)
            if not np.any(np.isnan(logit_val)):
                w0 = weights[0] if weights is not None else 1.0
                logit_sum = logit_val * w0
                weight_sum = w0
            else:
                logit_sum = np.zeros_like(logit_val)
                weight_sum = 0.0
            for idx, extra_model in enumerate(self._triaje_ensemble, start=1):
                extra_logit = extra_model(view, meta)
                extra_val = self._to_numpy_safe(extra_logit)
                if np.any(np.isnan(extra_val)):
                    warnings.warn(
                        "Ensemble binario: un modelo produjo NaN, descartado del promedio.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                w_i = weights[idx] if weights is not None else 1.0
                logit_sum = logit_sum + extra_val * w_i
                weight_sum += w_i
            if weight_sum <= 0.0:
                # Todos los modelos NaN → retornar NaN para que el guard TTA lo maneje.
                return torch.tensor(float("nan"), device=self.device, dtype=torch.float32)
            avg = logit_sum / weight_sum
            # Forzar float32: numpy puede promover a float64 con dtypes mixtos, y
            # una inconsistencia con el modelo (float32) podría afectar el cruce del
            # threshold 0.20 en valores limítrofes.
            return torch.tensor(avg, device=self.device, dtype=torch.float32)
        # Cast a float32 también en el camino single-model, por simetría con el
        # camino ensemble. Hoy `triaje.eval()` produce float32 nativo (sin
        # autocast), así que es idempotente; pero si un caller futuro envolviera la
        # inferencia en `torch.amp.autocast`, fp16 llegaría al threshold clínico
        # 0.20 con resolución degradada.
        return logit.float()

    def _forward_ensemble_multi(self, view, meta):
        """Forward por todos los modelos multiclase del ensemble, promediando logits.

        Mismo patrón que _forward_ensemble_binary, con verificación de NaN por modelo.
        """
        logit = self.especialista(view, meta)
        if self._especialista_ensemble:
            # Mismo esquema de ponderación por AUC que _forward_ensemble_binary.
            weights = self._especialista_ensemble_weights
            logit_val = self._to_numpy_safe(logit)
            if not np.any(np.isnan(logit_val)):
                w0 = weights[0] if weights is not None else 1.0
                logit_sum = logit_val * w0
                weight_sum = w0
            else:
                logit_sum = np.zeros_like(logit_val)
                weight_sum = 0.0
            for idx, extra_model in enumerate(self._especialista_ensemble, start=1):
                extra_logit = extra_model(view, meta)
                extra_val = self._to_numpy_safe(extra_logit)
                if np.any(np.isnan(extra_val)):
                    warnings.warn(
                        "Ensemble multiclase: un modelo produjo NaN, descartado del promedio.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                w_i = weights[idx] if weights is not None else 1.0
                logit_sum = logit_sum + extra_val * w_i
                weight_sum += w_i
            if weight_sum <= 0.0:
                return torch.tensor(
                    [float("nan")] * logit_val.shape[-1],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
            avg = logit_sum / weight_sum
            # Forzar float32 (ver _forward_ensemble_binary).
            return torch.tensor(avg, device=self.device, dtype=torch.float32)
        # Cast a float32 también en el camino single-model, por simetría con el
        # camino ensemble y como protección ante un autocast futuro.
        return logit.float()

    def _load_weights(self, model, path):
        # Maneja la incompatibilidad si el modelo fue entrenado con DataParallel pero probado sin él.
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name[len("module."):]
            # Elimina el prefijo _orig_mod. que inyecta torch.compile, igual que en
            # el resto de puntos de carga de checkpoints del proyecto.
            if name.startswith("_orig_mod."):
                name = name[len("_orig_mod."):]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    def preprocess(self, img_path, age, sex, anatom_site, is_multiclass=False):
        # 1. Imagen
        if not Path(img_path).exists():
            raise FileNotFoundError(f"Imagen {img_path} no encontrada.")
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(
                f"No se pudo leer la imagen: {img_path}. "
                "El archivo puede estar corrupto o en un formato no soportado por OpenCV."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pipeline = self.vision_pipeline_multi if is_multiclass else self.vision_pipeline_binary
        tensor_img = pipeline(image=image)["image"].unsqueeze(0)  # [1, C, H, W]

        # 2. Metadatos (normalizados igual que en dataset.py). La codificación usa
        # `encode_clinical_metadata_scalar` de constants.py como fuente de verdad
        # única, de modo que el preprocesado de inferencia y el de entrenamiento
        # coinciden exactamente: aplica `.strip()` defensivo (un valor como
        # `"male "` no cae silenciosamente a `unknown`), deriva los códigos de
        # SEX_UNKNOWN_CODE/SITE_UNKNOWN_CODE en lugar de literales, y maneja los NaN
        # de numpy (que `str(...).lower()` convertiría en `"nan"`). El warning
        # clínico de edad fuera de [0, 120] se queda aquí porque solo predict.py
        # tiene el contexto de la petición individual (dataset.py emite un warning
        # agregado en __init__).
        try:
            _age_for_warn = float(age) if age is not None else None
            if _age_for_warn is not None and math.isnan(_age_for_warn):
                _age_for_warn = None
        except (TypeError, ValueError):
            _age_for_warn = None
        if _age_for_warn is not None and (_age_for_warn < 0 or _age_for_warn > 120):
            warnings.warn(
                f"Edad {_age_for_warn} fuera del rango clinico esperado [0, 120]. "
                f"Se clampeara a {max(0.0, min(120.0, _age_for_warn))}. "
                "Verificar si es un error de entrada.",
                RuntimeWarning,
                stacklevel=2,
            )

        age_norm, sex_code, site_code = encode_clinical_metadata_scalar(
            sex=sex, anatom_site=anatom_site, age=age
        )
        tensor_meta = torch.tensor(
            [[age_norm, float(sex_code), float(site_code)]], dtype=torch.float32
        )

        return tensor_img.to(self.device), tensor_meta.to(self.device)

    @staticmethod
    def _enable_mc_dropout(model):
        """
        Activa Dropout en modo inferencia para MC Dropout (incertidumbre epistemica).
        BatchNorm permanece en eval (no queremos corromper running stats).
        """
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()

    @staticmethod
    def _disable_mc_dropout(model):
        """Restaura el modelo a eval completo."""
        model.eval()

    def _mc_inference_binary(self, model, views, meta, n_samples=10):
        """
        MC Dropout para TTA binario: N pases estocasticos por vista.

        Cada vista aporta peso uniforme al promedio final (media de medias),
        incluso si algunas vistas tienen pases NaN descartados. Sin esto, una
        vista con 10/10 muestras válidas pesaría 10x más que otra con 1/10.

        Std via ley de varianza total: Var(X) = E[Var(X|View)] + Var(E[X|View]).

        Limitación conocida: MC Dropout usa solo el modelo principal, no los
        modelos del ensemble (SWA, EMA). La predicción final sí usa el ensemble
        (promediando logits), pero la incertidumbre epistémica se estima solo del
        modelo principal, lo que puede subestimar la incertidumbre real. Una
        solución completa (MC sobre el ensemble) multiplicaría la latencia por
        mc_samples × ensemble_size, impracticable para uso clínico.

        Returns:
            mean_logit: float — logit promedio (ponderado uniforme por vista)
            std_logit: float — sqrt(Var total) sobre el ensemble MC+TTA
            valid_views: int — vistas con al menos 1 muestra no-NaN
        """
        per_view_samples = []  # lista de arrays [k_i]

        self._enable_mc_dropout(model)
        try:
            for view in views:
                samples = []
                for _ in range(n_samples):
                    logit = model(view, meta).item()
                    if not math.isnan(logit):
                        samples.append(logit)
                if samples:
                    per_view_samples.append(np.asarray(samples, dtype=np.float64))
        finally:
            self._disable_mc_dropout(model)

        if not per_view_samples:
            return float('nan'), float('nan'), 0

        view_means = np.array([v.mean() for v in per_view_samples])
        mean_logit = float(view_means.mean())
        intra_var = float(np.mean([v.var() for v in per_view_samples]))
        inter_var = float(view_means.var())
        std_logit = float(np.sqrt(intra_var + inter_var))
        return mean_logit, std_logit, len(per_view_samples)

    def _mc_inference_multiclass(self, model, views, meta, n_samples=10, temperature=None):
        """
        MC Dropout para TTA multiclase: N pases estocasticos por vista.

        Misma ponderacion uniforme por vista que el binario. Std por clase
        via ley de varianza total. Reporta tanto espacio logit (interno)
        como espacio probabilidad (clinicamente interpretable).

        Se reporta la std en espacio probabilidad además del espacio logit (que es
        interno). Un std_logit=2.0 cerca de la saturación del softmax puede
        corresponder a ~0.03 en espacio probabilidad, lo que daría al clínico una
        falsa sensación de baja incertidumbre. Por eso se softmaxea cada muestra MC
        *antes* de calcular la varianza, aplicando también la temperatura de
        calibración si está activa.

        Returns:
            mean_logits: np.array [8] — logits promedio (ponderado uniforme por vista)
            std_logits: np.array [8] — std por clase en espacio logit
            mean_probs: np.array [8] — probabilidades promedio (softmax + T)
            std_probs: np.array [8] — std por clase en espacio probabilidad
            valid_views: int — vistas con al menos 1 muestra no-NaN
        """
        per_view_samples = []  # lista de arrays [k_i, 8] (logits crudos)

        self._enable_mc_dropout(model)
        try:
            for view in views:
                samples = []
                for _ in range(n_samples):
                    # `_to_numpy_safe` aplica detach antes de numpy: bajo
                    # `infer(explain=True)` el contexto `torch.enable_grad()`
                    # propaga requires_grad al forward y `.numpy()` fallaría.
                    logits = self._to_numpy_safe(model(view, meta))[0]
                    if not np.any(np.isnan(logits)):
                        samples.append(logits)
                if samples:
                    per_view_samples.append(np.stack(samples).astype(np.float64))
        finally:
            self._disable_mc_dropout(model)

        n = self._num_classes_multi
        if not per_view_samples:
            nan_vec = np.full(n, float('nan'))
            return nan_vec, nan_vec, nan_vec, nan_vec, 0

        # Stats en espacio logit (ley de varianza total)
        view_means = np.stack([v.mean(axis=0) for v in per_view_samples])  # [V, 8]
        mean_logits = view_means.mean(axis=0)
        intra_var = np.stack([v.var(axis=0) for v in per_view_samples]).mean(axis=0)
        inter_var = view_means.var(axis=0)
        std_logits = np.sqrt(intra_var + inter_var)

        # Convertir cada muestra MC a espacio probabilidad aplicando temperatura
        # (si esta activa) antes del softmax. Softmax es no-lineal: hay que softmaxear
        # cada muestra por separado y luego calcular varianza.
        # `temperature` puede ser escalar O vector [C] (vector scaling).
        # Broadcast por el último eje en ambos casos.
        if temperature is None:
            t = 1.0
        elif isinstance(temperature, (list, tuple, np.ndarray)):
            t = np.asarray(temperature, dtype=np.float64).reshape(1, -1)
            # Defensa en profundidad: `_load_temperature_safe` filtra `isfinite`
            # al cargar, pero una mutación programática posterior podría introducir
            # Inf (`+inf > 0 = True` pasaría el filtro `np.where`) →
            # `view_samples / Inf = 0` → varianza por clase cero → falsa certeza.
            # Se detiene con error.
            if not np.all(np.isfinite(t)):
                raise RuntimeError(
                    f"temperature vector contiene NaN/Inf "
                    f"(t={t.flatten().tolist()}). Probable mutación posterior a la carga."
                )
            t = np.where(t > 0, t, 1.0)
        else:
            t = temperature if temperature > 0 else 1.0
        per_view_probs = []
        for view_samples in per_view_samples:  # [k_i, 8] logits
            scaled = view_samples / t
            # Softmax numericamente estable por muestra
            scaled -= np.max(scaled, axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
            per_view_probs.append(probs)

        view_means_p = np.stack([v.mean(axis=0) for v in per_view_probs])  # [V, 8]
        mean_probs = view_means_p.mean(axis=0)
        intra_var_p = np.stack([v.var(axis=0) for v in per_view_probs]).mean(axis=0)
        inter_var_p = view_means_p.var(axis=0)
        std_probs = np.sqrt(intra_var_p + inter_var_p)

        return mean_logits, std_logits, mean_probs, std_probs, len(per_view_samples)

    def _get_effective_threshold(self, meta_tensor):
        """
        Despacha el threshold de decisión RED/GREEN por sitio anatómico.

        Lee `site_code` del tensor de metadata (3er elemento, ya construido por
        `preprocess()` via `encode_clinical_metadata_scalar`) y consulta el
        `PerSiteThresholdCalibrator` cargado en el constructor. Si no hay
        calibrador o el sitio no esta en el dict, retorna el threshold global
        con source="global_fallback" para trazabilidad.

        Args:
            meta_tensor: Tensor [1, 3] con [age_norm, sex_code, site_code].

        Returns:
            (effective_threshold, threshold_source):
                effective_threshold: float — threshold a usar en este request.
                threshold_source: str ∈ {"per_site", "global_fallback"}.
        """
        if self.per_site_threshold is None:
            return self.threshold, "global_fallback"
        try:
            # `meta_tensor` esperado: torch.Tensor [1, 3] = [age_norm, sex_code, site_code].
            # Si meta_tensor es None o tiene shape distinta, fail-soft a global fallback
            # con warning. (TypeError, IndexError, RuntimeError, AttributeError) cubren
            # el espectro de errores razonables: None.subscript → TypeError,
            # tensor[0, 2] fuera de bounds → IndexError, .item() sobre dtype incompat → RuntimeError,
            # objeto sin .item() → AttributeError. ValueError reservado para validation upstream.
            site_code = int(meta_tensor[0, 2].item())
        except (IndexError, RuntimeError, TypeError, AttributeError) as e:
            warnings.warn(
                f"No se pudo extraer site_code de meta_tensor "
                f"({type(e).__name__}: {e}). Usando threshold global.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.threshold, "global_fallback"
        thresh, source = self.per_site_threshold.get_threshold(site_code)
        return float(thresh), source

    def infer(self, img_path, age=50, sex="unknown", anatom_site="unknown",
              uncertainty=False, mc_samples=10, explain=False, conformal=False,
              risk_control=False, tent=False):
        """
        Flujo de Decisión Clínica con Test-Time Augmentation (TTA).

        Args:
            img_path: Ruta a la imagen dermatoscopica.
            age: Edad del paciente (float o None).
            sex: Sexo ('male', 'female', 'unknown').
            anatom_site: Sitio anatomico ('torso', 'back', etc.).
            uncertainty: Si True, ejecuta MC Dropout para cuantificar incertidumbre.
            mc_samples: Numero de muestras MC Dropout por vista (default 10).
            explain: Si True, genera un mapa de calor por Occlusion Sensitivity.
            conformal: Si True, añade `conformal_set_binary`/`conformal_set_multiclass`
                al report (sets con cobertura formal `≥ 1-α`).
            risk_control: Si True, y se proporcionó `risk_control_binary_path` al
                constructor, añade `risk_control_binary` (str) y
                `risk_control_fnr_target` (float) al report. Una discrepancia con el
                triaje estándar dispara `risk_control_disagrees` en
                `defer_to_human`. Requiere que `risk_control_binary` esté cargado.
            tent: (TENT, Wang ICLR 2021). Si True y el constructor recibió
                `tent_enabled=True`, aplica TENT episódico ANTES del TTA-6 binario y
                (si la imagen pasa a Nivel 2) ANTES del TTA-6 multiclase. Adapta los
                parámetros affine de las BN del trunk minimizando la entropía sobre
                la imagen actual. Los pesos se restauran al final del request.
                Audit trail: `tent_applied`, `tent_steps_used`, `tent_loss_pre`,
                `tent_loss_post`, `tent_target` en el report. Con False (por defecto)
                la inferencia es determinista. Caveat: rompe la garantía formal del
                TemperatureScaler calibrado antes de TENT — la prob calibrada puede
                ser subóptima cuando `tent=True`. Caveat: ensemble + TENT pierde
                coherencia (warning en `__init__` si ambos están activos).
        Returns:
            report: dict con triaje, diagnostico, incertidumbre y heatmap.
        """
        # Por defecto la inferencia corre bajo no_grad. TENT necesita gradientes
        # activos (backward sobre la entropía → paso SGD en gamma/beta), así que
        # con tent=True se usa enable_grad. El flag explain también lo activa por
        # consistencia del contexto; la explicabilidad por oclusión es
        # gradient-free y gestiona su propio no_grad interno, por lo que es
        # inofensivo.
        needs_grad = bool(explain) or bool(tent)
        grad_ctx = torch.enable_grad() if needs_grad else torch.no_grad()

        with grad_ctx:
            return self._infer_impl(
                img_path, age, sex, anatom_site, uncertainty, mc_samples,
                explain, conformal, risk_control, tent=tent,
            )

    def _infer_impl(self, img_path, age, sex, anatom_site, uncertainty, mc_samples, explain, conformal=False, risk_control=False, tent=False):
        """Implementacion interna de infer (separada para manejar contexto de gradientes)."""
        img_bin, meta = self.preprocess(img_path, age, sex, anatom_site, is_multiclass=False)

        # --- NIVEL 1: TRIAJE con TTA-6 (Test-Time Augmentation 6-way) ---
        # TTA: Evaluamos la imagen en 6 orientaciones distintas.
        # Las lesiones de piel no tienen "arriba" ni "abajo". Promediar 6 vistas
        # destruye la varianza y exprime hasta la última gota de precisión del modelo.
        img_hflip = torch.flip(img_bin, dims=[3])  # H-Flip
        img_vflip = torch.flip(img_bin, dims=[2])  # V-Flip
        img_rot90 = torch.rot90(img_bin, k=1, dims=[2, 3]) # Rot 90
        img_rot180 = torch.rot90(img_bin, k=2, dims=[2, 3]) # Rot 180
        img_rot270 = torch.rot90(img_bin, k=3, dims=[2, 3]) # Rot 270

        # Inferencia en batch (simulada secuencialmente para claridad).
        # Guard NaN en TTA: si cualquier vista produce NaN (overflow en GeM/AMP),
        # el promedio se vuelve NaN → `NaN >= 0.20` es False → un melanoma pasaría
        # como benigno. Por eso se cuentan las vistas válidas, se saltan las NaN y
        # se aborta si todas son NaN.
        #
        # Se promedian LOGITS, no probabilidades. La sigmoide es no lineal:
        # promediar sigmoid(a) y sigmoid(b) ≠ sigmoid(avg(a,b)). Promediar logits
        # antes de activar preserva la calibración del ensemble.
        view_names = ["original", "hflip", "vflip", "rot90", "rot180", "rot270"]
        views = [img_bin, img_hflip, img_vflip, img_rot90, img_rot180, img_rot270]
        valid_logits = []  # acumulados por vista para el promedio ponderado
        valid_views = 0
        nan_views = []

        # Adaptación TENT episódica ANTES del bucle TTA binario. Cubre solo el
        # bucle TTA binario (no la incertidumbre ni la explicabilidad): TENT mejora
        # la predicción puntual (`triaje_prob`), mientras que MC Dropout y la
        # explicabilidad por oclusión operan tras el restore con los pesos
        # originales para mantener su interpretación clásica.
        # Se adapta UNA vez con `img_bin` (vista original, sin flip/rot) y los pesos
        # adaptados se aplican a las 6 vistas TTA; adaptar por vista (6×) sería 6×
        # más lento sin ganancia.
        _bin_tent_active = bool(tent) and self._tent_binary is not None
        _bin_tent_ctx = None
        if _bin_tent_active:
            _bin_tent_ctx = self._tent_binary.adapt_context(img_bin, meta)
            _bin_tent_ctx.__enter__()  # adapt pesos in-place

        try:
            for view, name in zip(views, view_names):
                logit_view = self._forward_ensemble_binary(view, meta)
                logit_val = logit_view.item()
                if not math.isnan(logit_val):
                    valid_logits.append(logit_val)
                    valid_views += 1
                else:
                    nan_views.append(name)
        finally:
            if _bin_tent_active:
                # Salir del adapt → pesos restaurados (episodic strict).
                # Audit fields se actualizan dentro de adapt_context.__exit__
                # (last_loss_pre/post/steps_used). Capturamos en `_bin_tent_audit`
                # para inyectar al report dict cuando se cree (más abajo).
                _bin_tent_ctx.__exit__(None, None, None)
        # Audit TENT: se capturan los campos binarios para inyectarlos al report
        # (variable local; el dict se crea más abajo).
        if _bin_tent_active:
            _bin_tent_audit = {
                "tent_applied_binary": True,
                "tent_loss_pre_binary": self._tent_binary.last_loss_pre,
                "tent_loss_post_binary": self._tent_binary.last_loss_post,
                "tent_steps_used_binary": self._tent_binary.last_steps_used,
                "tent_target_binary": (
                    "head_and_backbone" if self._tent_binary.adapt_backbone
                    else "head_only"
                ),
            }
        else:
            _bin_tent_audit = {"tent_applied_binary": False}

        if nan_views:
            # warnings.warn (no print) para que el evento sea capturable por el
            # handler de logging del proceso. Es un evento clínico operacional: el
            # operador necesita ver las vistas TTA con NaN en el registro.
            warnings.warn(
                f"TTA Binario: {len(nan_views)}/6 vistas NaN para {img_path}: {nan_views}",
                RuntimeWarning,
                stacklevel=2,
            )

        if valid_views == 0:
            raise RuntimeError(
                f"ALERTA CRITICA: Las 6 vistas TTA produjeron NaN para {img_path}. "
                "Imposible generar prediccion clinica segura."
            )

        # Warning si hay pocas vistas válidas — el promedio con N<3 es poco fiable.
        if valid_views < 3:
            warnings.warn(
                f"BAJA CONFIANZA TTA: Solo {valid_views}/6 vistas validas para {img_path}. "
                "La prediccion puede ser poco fiable.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Confidence-weighted TTA (opcional). Cada vista aporta según su propia
        # confianza `|sigmoid(logit) - 0.5| * 2` ∈ [0, 1]: las vistas en los
        # extremos (logit muy positivo o muy negativo) pesan más que las cercanas a
        # 0.5 (modelo indeciso para esa vista). El softmax sobre las confianzas
        # normaliza los pesos sin amplificar outliers — una sola vista confiada no
        # domina. Con el valor por defecto se promedia de forma uniforme.
        # La confianza se calcula con `torch.sigmoid`, numéricamente estable en todo
        # el dominio float32, en lugar de `1/(1+exp(-l))`, que desborda cuando
        # |l| ≥ ~700 (posible con un modelo mal calibrado o un batch adversarial).
        # El `math.exp(c - _max_c)` posterior es seguro porque `c - _max_c ∈ [-1, 0]`
        # (las confianzas están acotadas en [0, 1] por construcción).
        if self._confidence_weighted_tta:
            _logits_tensor = torch.tensor(valid_logits, dtype=torch.float32)
            _probs_tensor = torch.sigmoid(_logits_tensor)
            _conf = (torch.abs(_probs_tensor - 0.5) * 2.0).tolist()
            # Softmax sobre las confianzas para normalizar pesos sin dominancia.
            _max_c = max(_conf)
            _exp = [math.exp(c - _max_c) for c in _conf]
            _sum_exp = sum(_exp)
            _weights = [e / _sum_exp for e in _exp]
            avg_logit = sum(l * w for l, w in zip(valid_logits, _weights))
        else:
            avg_logit = sum(valid_logits) / valid_views

        # Calibración de temperatura: logit / T (si está disponible). Se comprueba
        # la finitud justo después: si una T corrupta produce inf/nan, se detecta
        # aquí con un mensaje diagnóstico claro en vez de propagarlo al guard final.
        if self.temp_binary is not None:
            calibrated = self.temp_binary.calibrate(torch.tensor([[avg_logit]])).item()
            if not math.isfinite(calibrated):
                raise RuntimeError(
                    f"ALERTA CRITICA: Calibracion binaria produjo valor no finito "
                    f"({calibrated}) para logit={avg_logit}, img={img_path}."
                )
            avg_logit = calibrated

        prob_malignant = torch.sigmoid(torch.tensor(avg_logit)).item()

        report = {
            "triaje_prob": prob_malignant,
            "status": "GREEN",
            "message": "Benigno. Monitoreo rutinario.",
            "diagnostico_especialista": None,
            "low_confidence_tta": valid_views < 3,
        }
        # Inyecta los audit fields TENT binarios (`_bin_tent_audit` se pobló justo
        # tras el bucle TTA). Si tent=True pero el adapter no estaba cargado
        # (constructor con tent_enabled=False), se emite warning para que el
        # operador no asuma TENT activo. tent_applied_multiclass arranca en False y
        # se sobrescribe en la rama RED si el especialista corre con TENT.
        report.update(_bin_tent_audit)
        report["tent_applied_multiclass"] = False
        if tent and not _bin_tent_active:
            warnings.warn(
                "tent=True solicitado en infer() pero el adapter "
                "binario NO está cargado (el constructor recibió tent_enabled=False). "
                "La inferencia continuó sin TENT. Reconstruir el engine con "
                "tent_enabled=True para activarlo.",
                RuntimeWarning, stacklevel=3,
            )

        # MC Dropout: cuantifica la incertidumbre epistémica del triaje.
        # La std se convierte de espacio logit a espacio probabilidad antes de
        # reportarla. Un std_logit=2.0 puede corresponder a std_prob~0.04 cerca de
        # la sigmoide saturada, dando falsa sensación de baja incertidumbre al
        # clínico. El umbral de "alta incertidumbre" vive en espacio probabilidad
        # (15 puntos porcentuales).
        if uncertainty:
            views = [img_bin, img_hflip, img_vflip, img_rot90, img_rot180, img_rot270]
            mc_mean, mc_std, mc_valid = self._mc_inference_binary(
                self.triaje, views, meta, n_samples=mc_samples
            )
            if math.isnan(mc_std):
                # MC Dropout fallo completamente (6/6 vistas NaN). Reportar None
                # consistente con el ramo multiclase y marcar flag explicito para
                # que el cliente distinga "incertidumbre no calculable" de "alta".
                report["uncertainty_prob_std"] = None
                report["uncertainty_prob_range"] = None
                report["uncertainty_computed"] = False
                report["high_uncertainty"] = True
            else:
                # Si hay calibración de temperatura activa, `avg_logit` ya está
                # escalado por 1/T pero `mc_std` viene del MC Dropout sobre logits
                # crudos (sin T). Mezclar una media calibrada con una std cruda
                # inflaría la incertidumbre por el factor T. Se escala la std al
                # mismo espacio para que `sigmoid(avg ± std)` sea coherente y el
                # umbral clínico 0.15 aplique correctamente.
                if self.temp_binary is not None:
                    # Extracción de la temperatura tolerante al tipo, con fallo
                    # explícito si llega un vector. El binario DEBE usar un escalar;
                    # recibir un vector indica que se cargó un checkpoint multiclase
                    # (Vector Scaling) como binario por error. En lugar de
                    # enmascararlo tomando `_t.max()`, se aborta para que el problema
                    # de carga en `_load_temperature_safe` se diagnostique.
                    _t_tensor = self.temp_binary.temperature.detach()
                    if _t_tensor.numel() > 1:
                        raise RuntimeError(
                            f"BinaryTemperatureScaler tiene "
                            f"{_t_tensor.numel()} elementos pero el binario "
                            "DEBE usar escalar. Causa probable: checkpoint "
                            "multiclase (Vector Scaling) cargado "
                            "como binario. Revisar `_load_temperature_safe` "
                            "y el path pasado en `--temperature_binary_path`."
                        )
                    t_binary = float(_t_tensor.item())
                    mc_std_calibrated = mc_std / max(t_binary, 1e-6)
                else:
                    mc_std_calibrated = mc_std
                lower_prob = torch.sigmoid(torch.tensor(avg_logit - mc_std_calibrated)).item()
                upper_prob = torch.sigmoid(torch.tensor(avg_logit + mc_std_calibrated)).item()
                prob_std = (upper_prob - lower_prob) / 2.0
                report["uncertainty_prob_std"] = prob_std
                report["uncertainty_prob_range"] = (lower_prob, upper_prob)
                report["uncertainty_computed"] = True
                report["high_uncertainty"] = prob_std > 0.15

        # Explicabilidad visual del triaje (Occlusion Sensitivity, agnóstica al backbone)
        if explain:
            # `build_explainer` devuelve el explainer canónico: Occlusion Sensitivity
            # (Zeiler & Fergus 2014), gradient-free y agnóstico al backbone, de modo
            # que la explicabilidad está disponible para todos los backbones.
            from src.inference.explainability import build_explainer
            explainer = build_explainer(self.triaje)
            # Se propaga la temperatura calibrada al explainer. Sin esto, su dict
            # `prediction` reportaría probabilidades sobre logits crudos mientras el
            # engine las reporta calibradas (post-Temperature Scaling), produciendo
            # dos valores divergentes para el mismo caso en la trazabilidad. El
            # heatmap espacial no cambia (el gradiente de `logit/T` solo escala por
            # 1/T sin alterar la geometría); solo cambian las probabilidades
            # reportadas. La extracción es tolerante a escalar o vector.
            temp_binary_val = None
            if self.temp_binary is not None:
                _t = self.temp_binary.temperature.detach().cpu()
                temp_binary_val = _t.tolist() if _t.numel() > 1 else float(_t.item())
            heatmap, _ = explainer.explain(img_bin, meta, temperature=temp_binary_val)
            report["heatmap_binary"] = heatmap
            explainer.remove_hooks()

        # --- DECISIÓN CLÍNICA ---
        # Guard NaN final (defensa en profundidad)
        if math.isnan(prob_malignant):
            raise RuntimeError(
                f"ALERTA CRITICA: prob_malignant es NaN para {img_path} tras TTA. "
                "No se puede tomar decision clinica segura."
            )

        # Conformal binario (opcional). Se calcula siempre que haya predictor
        # calibrado y el usuario lo active, con independencia del threshold. El set
        # predicho es informativo; solo el caso {benigno, maligno} dispara AMBER.
        conformal_amber = False
        if conformal and self.binary_conformal is not None:
            try:
                pred_set = self.binary_conformal.predict(prob_malignant)
                report["conformal_set_binary"] = pred_set
                # Se exporta alpha para auditabilidad clínica: sin él, el clínico
                # ve {benigno, maligno} pero no sabe si la garantía es del 95% u 80%.
                report["conformal_alpha_binary"] = self.binary_conformal.alpha
                if len(pred_set) == 2:
                    conformal_amber = True
            except Exception as e:
                warnings.warn(
                    f"Conformal binario fallo en inferencia ({e}). "
                    "Continuando sin prediccion conformal.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # ConformalRiskControl (opcional). Complementa el set conformal con un
        # threshold calibrado para acotar directamente el FNR. El predictor
        # devuelve una clase única ("maligno"/"benigno"), no un set. El flag clínico
        # `risk_control_disagrees` (en `_compute_defer_to_human`) se activa cuando
        # esta predicción discrepa del triaje estándar.
        if risk_control and self.risk_control_binary is not None:
            try:
                rc_pred = self.risk_control_binary.predict(prob_malignant)
                report["risk_control_binary"] = rc_pred
                report["risk_control_fnr_target"] = self.risk_control_binary.target_fnr
            except Exception as e:
                warnings.warn(
                    f"RiskControl binario fallo en inferencia ({e}). "
                    "Continuando sin control directo de FNR.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # El threshold de decisión RED/GREEN se selecciona dinámicamente por sitio
        # anatómico cuando hay un PerSiteThresholdCalibrator cargado. Para los
        # sitios no calibrados (n por debajo del mínimo, sex=unknown, etc.) se usa
        # el threshold global como fallback. `effective_threshold`
        # y `threshold_source` se exponen en el report para la trazabilidad
        # clínica.
        effective_threshold, threshold_source = self._get_effective_threshold(meta)
        report["effective_threshold"] = effective_threshold
        report["threshold_source"] = threshold_source

        if prob_malignant >= effective_threshold:
            report["status"] = "RED"
            report["message"] = (
                f"ALERTA: Probabilidad de malignidad {prob_malignant*100:.1f}% "
                f"(threshold={effective_threshold:.3f}, source={threshold_source}). "
                f"Derivación a especialista necesaria."
            )

            # --- NIVEL 2: ESPECIALISTA con TTA-6 ---
            if self.especialista:
                # Preprocesar imagen a la resolución nativa del especialista.
                # Se usa meta_multi (no el meta binario) por coherencia: hoy los
                # valores son idénticos, pero si el preprocess divergiera por modo,
                # esto lo protege.
                img_multi, meta_multi = self.preprocess(img_path, age, sex, anatom_site, is_multiclass=True)
                img_multi_hflip = torch.flip(img_multi, dims=[3])
                img_multi_vflip = torch.flip(img_multi, dims=[2])
                img_multi_rot90 = torch.rot90(img_multi, k=1, dims=[2, 3])
                img_multi_rot180 = torch.rot90(img_multi, k=2, dims=[2, 3])
                img_multi_rot270 = torch.rot90(img_multi, k=3, dims=[2, 3])

                views_multi = [img_multi, img_multi_hflip, img_multi_vflip, img_multi_rot90, img_multi_rot180, img_multi_rot270]

                # TTA Multiclase (6 vistas). Guard NaN igual que en el TTA binario:
                # np.argmax sobre un array con NaN devuelve resultados arbitrarios
                # → diagnóstico corrupto. Se promedian LOGITS, no softmax:
                # promediar distribuciones softmax pierde información de
                # incertidumbre; promediar logits y aplicar softmax una vez produce
                # un ensemble mejor calibrado.
                valid_logits_multi = []  # acumulados por vista para el promedio
                valid_views_multi = 0
                nan_views_multi = []

                # Adaptación TENT del especialista, mismo patrón que el binario:
                # cubre solo el bucle TTA multiclase. Se adapta UNA vez con
                # `img_multi` (sin flip/rot) y los pesos adaptados se aplican a las
                # 6 vistas; los pesos del especialista se restauran al salir del with.
                _multi_tent_active = bool(tent) and self._tent_multiclass is not None
                _multi_tent_ctx = None
                if _multi_tent_active:
                    _multi_tent_ctx = self._tent_multiclass.adapt_context(img_multi, meta_multi)
                    _multi_tent_ctx.__enter__()

                try:
                    for view, name in zip(views_multi, view_names):
                        logit_multi = self._forward_ensemble_multi(view, meta_multi)
                        # `_to_numpy_safe` aplica detach antes de numpy: bajo
                        # `infer(explain=True)`, cuando el triaje deriva al
                        # especialista bajo enable_grad, un `.numpy()` directo
                        # fallaría aquí.
                        logit_np = self._to_numpy_safe(logit_multi)[0]
                        if not np.any(np.isnan(logit_np)):
                            valid_logits_multi.append(logit_np)
                            valid_views_multi += 1
                        else:
                            nan_views_multi.append(name)
                finally:
                    if _multi_tent_active:
                        # Salir del adapt → pesos del especialista restaurados.
                        _multi_tent_ctx.__exit__(None, None, None)
                        # Audit fields multiclase: el último valor de
                        # last_loss_pre/post/steps_used queda fijo tras el exit.
                        report["tent_applied_multiclass"] = True
                        report["tent_loss_pre_multiclass"] = self._tent_multiclass.last_loss_pre
                        report["tent_loss_post_multiclass"] = self._tent_multiclass.last_loss_post
                        report["tent_steps_used_multiclass"] = self._tent_multiclass.last_steps_used
                        report["tent_target_multiclass"] = (
                            "head_and_backbone" if self._tent_multiclass.adapt_backbone
                            else "head_only"
                        )

                if nan_views_multi:
                    # warnings.warn (no print), simétrico al binario y capturable por
                    # el handler de logging del proceso. Es un evento clínico operacional.
                    warnings.warn(
                        f"TTA Multiclase: {len(nan_views_multi)}/6 vistas NaN para {img_path}: {nan_views_multi}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                if valid_views_multi == 0:
                    # Degradación controlada en vez de crash. El triaje ya marcó la
                    # imagen como sospechosa (prob >= threshold); perder esa
                    # información con un crash es peor que devolver un resultado
                    # parcial. Se mantiene el status RED para que el paciente sea
                    # derivado a revisión presencial.
                    warnings.warn(
                        f"ALERTA: Las 6 vistas TTA multiclase produjeron NaN para {img_path}. "
                        "Retornando resultado de triaje sin diagnostico especialista.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    report["diagnostico_especialista"] = {
                        "patologia": "Indeterminado (Error Especialista)",
                        "confianza": 0.0,
                    }
                    return report

                # Warning si hay pocas vistas válidas en multiclase.
                if valid_views_multi < 3:
                    warnings.warn(
                        f"BAJA CONFIANZA TTA Multiclase: Solo {valid_views_multi}/6 vistas validas para {img_path}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                # Confidence-weighted TTA multiclase. Pesos softmax sobre
                # `max(softmax(logits))` de cada vista: la vista donde el modelo es
                # más confiado (una clase domina) aporta más que una con softmax
                # plano (modelo indeciso entre clases). El resultado es determinista
                # (función determinista de los logits). Con el valor por defecto se
                # promedia de forma uniforme.
                if self._confidence_weighted_tta:
                    _logits_stack = np.stack(valid_logits_multi, axis=0)  # [V, C]
                    # Softmax por vista → max por vista = confianza de esa vista
                    _m = np.max(_logits_stack, axis=1, keepdims=True)  # [V, 1]
                    _exp_l = np.exp(_logits_stack - _m)
                    _softmax_v = _exp_l / np.sum(_exp_l, axis=1, keepdims=True)  # [V, C]
                    _conf_v = np.max(_softmax_v, axis=1)  # [V]
                    # Softmax sobre las V confianzas → pesos normalizados
                    _mc = np.max(_conf_v)
                    _ew = np.exp(_conf_v - _mc)
                    _weights = _ew / np.sum(_ew)  # [V]
                    # Weighted avg de logits (broadcast [V, 1] sobre [V, C])
                    avg_logits = np.sum(_logits_stack * _weights[:, None], axis=0)
                else:
                    # Promedio uniforme de logits → softmax única (mejor calibración que promediar softmax)
                    avg_logits = np.sum(np.stack(valid_logits_multi, axis=0), axis=0) / valid_views_multi

                # Guard NaN post-promedio: defensa en profundidad. Aunque cada vista
                # individual se filtra por NaN (L646), overflow numérico en la
                # acumulación float64 o un bug futuro podrían producir NaN parcial.
                # softmax(NaN) propaga NaN a TODAS las clases → argmax devuelve
                # índice arbitrario → diagnóstico INCORRECTO sin error.
                if np.any(np.isnan(avg_logits)) or not np.all(np.isfinite(avg_logits)):
                    warnings.warn(
                        f"ALERTA: Logits multiclase promediados contienen NaN/Inf "
                        f"tras {valid_views_multi} vistas validas para {img_path}. "
                        "Retornando resultado de triaje sin diagnostico especialista.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    report["diagnostico_especialista"] = {
                        "patologia": "Indeterminado (Logits NaN post-promedio)",
                        "confianza": 0.0,
                    }
                    return report

                avg_logits_tensor = torch.tensor(avg_logits).unsqueeze(0).float()

                # Calibración de temperatura multiclase (si está disponible). Se
                # comprueba la finitud justo después para detectar una T corrupta
                # aquí, en vez de propagar NaN a argmax (que devolvería un índice
                # arbitrario).
                if self.temp_multiclass is not None:
                    avg_logits_tensor = self.temp_multiclass.calibrate(avg_logits_tensor)
                    if not torch.isfinite(avg_logits_tensor).all():
                        raise RuntimeError(
                            f"ALERTA CRITICA: Calibracion multiclase produjo valores no finitos "
                            f"para img={img_path}. Revisar temperatura cargada."
                        )

                # `_to_numpy_safe` aplica detach antes de numpy: el tensor
                # post-calibración tiene grad porque `temp_multiclass.calibrate`
                # divide por un `nn.Parameter` (requires_grad=True). Bajo
                # `explain=True`, el forward del especialista propaga grad y un
                # `softmax(...).numpy()` directo fallaría.
                probs = self._to_numpy_safe(F.softmax(avg_logits_tensor, dim=1))[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]

                report["diagnostico_especialista"] = {
                    "patologia": self.multiclass_map[pred_idx],
                    "confianza": float(confidence),
                }

                # APS Conformal multiclase (opcional). Adaptive Prediction Sets
                # devuelve el conjunto mínimo de clases cuya masa acumulada cubre
                # 1-alpha.
                if conformal and self.multiclass_conformal is not None:
                    try:
                        aps_set = self.multiclass_conformal.predict_named(probs)
                        report["conformal_set_multiclass"] = aps_set
                        report["conformal_alpha_multiclass"] = self.multiclass_conformal.alpha
                        if len(aps_set) > 1:
                            # top_k debe contener EXACTAMENTE las clases del aps_set
                            # (APS acumula masa hasta el threshold y puede incluir una
                            # clase rara que no esté entre las top-N por probabilidad
                            # pura). Filtrar `order` por los índices presentes en
                            # aps_set garantiza la consistencia entre
                            # `conformal_set_multiclass` y `top_k`.
                            #
                            # La clave del sub-dict es `"confianza"`, igual que en el
                            # dict padre `diagnostico_especialista` — ambos niveles
                            # usan el mismo término (fuente única de verdad).
                            inv_map = {v: k for k, v in self.multiclass_map.items()}
                            aps_indices = {
                                inv_map[name] for name in aps_set if name in inv_map
                            }
                            order = np.argsort(-probs)
                            top_k = [
                                {"patologia": self.multiclass_map[int(i)], "confianza": float(probs[int(i)])}
                                for i in order if int(i) in aps_indices
                            ]
                            report["diagnostico_especialista"]["top_k"] = top_k
                    except Exception as e:
                        warnings.warn(
                            f"Conformal multiclase fallo en inferencia ({e}).",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                # MC Dropout multiclase — reporta incertidumbre en espacio PROBABILIDAD
                # (consistente con el binario; nunca en espacio logit).
                if uncertainty:
                    if self.temp_multiclass is not None:
                        _t_tensor = self.temp_multiclass.temperature.detach().cpu()
                        # Se usa el shape real del tensor (`.numel()`), no el campo
                        # de configuración `num_classes`. Ante un checkpoint con la
                        # metadata desincronizada (num_classes=8 pero tensor=[1]),
                        # ambos divergirían; el shape del tensor es la realidad.
                        t_multi = (
                            _t_tensor.tolist()
                            if _t_tensor.numel() > 1
                            else _t_tensor.item()
                        )
                    else:
                        t_multi = None
                    mc_mean, mc_std_logit, mc_mean_probs, mc_std_probs, mc_valid = (
                        self._mc_inference_multiclass(
                            self.especialista,
                            views_multi,
                            meta_multi,
                            n_samples=mc_samples,
                            temperature=t_multi,
                        )
                    )
                    if np.any(np.isnan(mc_std_probs)):
                        report["uncertainty_multiclass_prob_std"] = None
                        report["uncertainty_multiclass_prob_mean"] = None
                        report["uncertainty_multiclass_computed"] = False
                        report["high_uncertainty_multiclass"] = True
                    else:
                        report["uncertainty_multiclass_prob_std"] = mc_std_probs.tolist()
                        report["uncertainty_multiclass_prob_mean"] = mc_mean_probs.tolist()
                        report["uncertainty_multiclass_computed"] = True
                        # Flag clinico: alta incertidumbre si la clase predicha tiene
                        # prob_std > 0.15 (mismo umbral que el binario).
                        report["high_uncertainty_multiclass"] = bool(
                            mc_std_probs[int(pred_idx)] > 0.15
                        )
                        # Reportar clase con MAYOR incertidumbre (puede diferir de la predicha).
                        # Escenario crítico: modelo predice "Nevus" con baja std, pero
                        # "Melanoma" tiene alta std → el clínico debe saberlo para derivar.
                        max_unc_idx = int(np.argmax(mc_std_probs))
                        report["uncertainty_highest_class"] = self.multiclass_map[max_unc_idx]
                        report["uncertainty_highest_value"] = float(mc_std_probs[max_unc_idx])
                        # Flag adicional: cualquier clase con std > 0.15 merece atención
                        high_unc_classes = [
                            self.multiclass_map[i]
                            for i in range(len(mc_std_probs))
                            if mc_std_probs[i] > 0.15
                        ]
                        if high_unc_classes:
                            report["high_uncertainty_classes"] = high_unc_classes

                # Explicabilidad multiclase (Occlusion Sensitivity, agnóstica al backbone)
                if explain:
                    from src.inference.explainability import build_explainer
                    explainer = build_explainer(self.especialista)
                    # Se propaga la temperatura (escalar o vector). Igual que en el
                    # triaje binario, las probabilidades reportadas por el explainer
                    # deben coincidir con las del engine. La extracción usa
                    # `_t.numel() > 1` (shape real del tensor), no el campo de
                    # configuración `num_classes`: ante un checkpoint con la metadata
                    # desincronizada, ambos caminos divergirían y producirían un
                    # registro inconsistente entre el dict `prediction` del
                    # explainer y el report principal del engine.
                    temp_multi_val = None
                    if self.temp_multiclass is not None:
                        _t = self.temp_multiclass.temperature.detach().cpu()
                        temp_multi_val = _t.tolist() if _t.numel() > 1 else float(_t.item())
                    heatmap_multi, _ = explainer.explain(
                        img_multi, meta_multi,
                        target_class=int(pred_idx),
                        temperature=temp_multi_val,
                    )
                    report["heatmap_multiclass"] = heatmap_multi
                    explainer.remove_hooks()

                # Cascada de status post-especialista (precedencia clínica):
                #   CRITICAL > AMBER_PRECANCEROUS > RED
                #
                # CRITICAL = cáncer invasivo confirmado (Mel/BCC/SCC). Dispara
                # fast-track oncológico. Importado desde constants.py (fuente
                # única, alineado con distillation.py).
                #
                # AMBER_PRECANCEROUS: el especialista identifica una lesión
                # PRECANCEROSA (AK). No es cáncer invasivo, pero requiere
                # intervención dermatológica obligatoria. Riesgo lifetime de
                # progresión a SCC: 6-10% por AK no tratada (guías AAD/EADV; WHO
                # 2018: AK severa = "SCC in situ"). Este status da al operador la
                # guía de tratamiento específica (criocirugía/5-FU/imiquimod/PDT)
                # en vez de dejar la lesión en un RED genérico.
                if pred_idx in MALIGNANT_INDICES:
                    report["status"] = "CRITICAL"
                elif pred_idx in PRECANCEROUS_INDICES:
                    report["status"] = "AMBER_PRECANCEROUS"
                    report["message"] = (
                        f"LESIÓN PRECANCEROSA: {self.multiclass_map[int(pred_idx)]}. "
                        "Requiere intervención dermatológica obligatoria "
                        "(criocirugía, 5-FU tópico, imiquimod, o PDT). "
                        "Riesgo lifetime de progresión a SCC: 6-10%. "
                        "NO es cáncer invasivo, pero necesita tratamiento."
                    )
                    report["clinical_action"] = (
                        "Tratar la lesión precancerosa antes de progresión a "
                        "carcinoma invasivo. Opciones de primera línea: "
                        "criocirugía con nitrógeno líquido (lesiones aisladas), "
                        "5-fluorouracilo tópico al 5% (campo cancerizado), "
                        "imiquimod 5% (alternativa), terapia fotodinámica (PDT) "
                        "para lesiones múltiples. Seguimiento dermatológico cada "
                        "6-12 meses tras tratamiento."
                    )

        # AMBER override: si el set conformal binario es ambiguo y el especialista
        # NO confirmó malignidad (CRITICAL) ni precancerosidad (AMBER_PRECANCEROUS),
        # se degrada a AMBER (zona incierta → revisión humana). Nunca se sobrescribe
        # CRITICAL ni AMBER_PRECANCEROUS: si el multiclase dice Mel/BCC/SCC o AK, su
        # veredicto específico tiene precedencia sobre la ambigüedad del triaje.
        if conformal_amber and report["status"] not in ("CRITICAL", "AMBER_PRECANCEROUS"):
            report["status"] = "AMBER"
            report["message"] = (
                "Zona ambigua — set conformal binario contiene ambas clases. "
                "Revisión humana obligatoria."
            )

        # Abstención explícita. Añade al report dos claves (`defer_to_human: bool`,
        # `defer_reasons: list[str]`) para que el front-end pueda renderizar
        # "DERIVAR A DERMATÓLOGO" sin ambigüedad.
        defer, defer_reasons = _compute_defer_to_human(report)
        report["defer_to_human"] = defer
        report["defer_reasons"] = defer_reasons

        return report


# Bloque de prueba
if __name__ == "__main__":
    print("--------------------------------------------------")
    print("Motor de Inferencia compilado exitosamente.")
    print("Uso en producción esperado:")
    print("engine = ClinicalInferenceEngine(binary_model, multiclass_model)")
    print("resultado = engine.infer('foto_lunar.jpg', age=45, sex='male', anatom_site='back')")
    print("--------------------------------------------------")
