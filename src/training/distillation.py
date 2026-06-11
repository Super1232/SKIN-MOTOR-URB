"""
Destilación de Conocimiento: Binario → Multiclase.

El modelo binario (442K imágenes, incluyendo categorías "impuras") aprende una frontera
benigno/maligno robusta. El modelo multiclase (30K imágenes puras) se beneficia de
transferir este conocimiento como soft targets durante su entrenamiento.

Uso en train.py:
    teacher, teacher_input_size = load_teacher(args.distill_from, device)
    distill_loss = compute_distillation_loss(
        student_logits, images, meta, teacher,
        teacher_input_size=teacher_input_size,
    )
    total_loss = (1 - alpha) * focal_loss + alpha * distill_loss
"""
import torch
import torch.nn.functional as F
from collections import OrderedDict

# Fuente única: src/data/constants.py. Evita divergencias silenciosas con predict.py.
# El student agrega probabilidades sobre REFERRAL_INDICES (= MALIGNANT ∪
# PRECANCEROUS), no solo MALIGNANT_INDICES. El teacher binario se entrena con la
# queratosis actínica (AK) etiquetada como "malignant" (02_unify.py deriva
# benign_malignant desde REFERRAL_DIAGNOSIS_NAMES), de modo que el student
# multiclase debe reproducir ese contrato sumando la prob de AK en su
# agregación binaria.
from src.data.constants import (
    REFERRAL_INDICES,
    IMAGE_SIZES,
    DEFAULT_BINARY_BACKBONE,
)


def load_teacher(checkpoint_path, device, model_name=None):
    """
    Carga el modelo binario entrenado como profesor (frozen, eval).

    Auto-detección del backbone: el binario puede haberse entrenado con
    EfficientNet o con DINOv2 ViT (ambos soportados por `backbones.py`). Para
    que `load_state_dict(strict=True)` no falle, se lee primero el state_dict,
    se llama a `_detect_backbone_config` (la misma función que usa `predict.py`)
    y se construye con `backbone_name=` cuando aplica. `teacher_input_size` se
    devuelve junto al modelo para que el caller lo pase a
    `compute_distillation_loss`/`precompute_teacher_prob` sin tener que conocer
    el backbone.

    Args:
        checkpoint_path: Ruta al checkpoint del modelo binario
        device: torch.device
        model_name: Override opcional. Si None (recomendado), se deriva del checkpoint.
    Returns:
        teacher: SkinCancerModel en eval, sin gradientes
        teacher_input_size: int — resolución nativa del teacher (300 CNN / 294 ViT)
    """
    from src.models.architectures import SkinCancerModel
    from src.inference.predict import _detect_backbone_config, _detect_cbam, _pool_kwargs

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Se eliminan los prefijos que envuelven las claves del state_dict: `module.`
    # (DataParallel) y `_orig_mod.` (inyectado por `torch.compile`). Los
    # checkpoints propios no llevan `_orig_mod.`, pero un teacher entrenado en
    # otro entorno compilado sin desenvolver fallaría aquí con un error críptico
    # de "Missing keys / Unexpected keys".
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod."):]
        new_state_dict[name] = v

    backbone_name, fusion_mode = _detect_backbone_config(new_state_dict)
    # Detección de la cabeza de proyección SupCon en el state_dict (mismo
    # criterio que `predict.py`). El binario no entrena con SupCon hoy, pero un
    # experimento que lo active dejaría claves `projection_head.*` y, sin esta
    # detección, `load_state_dict(strict=True)` fallaría con "Unexpected keys".
    has_proj = any(k.startswith("projection_head.") for k in new_state_dict.keys())
    # Detección de CBAM en el state_dict, igual que los callsites de `predict.py`
    # y `export_onnx.load_and_export`. Si el teacher se entrenó sin CBAM,
    # reconstruirlo con `use_attention=True` por defecto fallaría con
    # "Missing keys: cbam.*". Se infiere del propio checkpoint.
    has_cbam = _detect_cbam(new_state_dict)
    is_vit = bool(backbone_name) and (
        "vit" in backbone_name or "dinov2" in backbone_name
    )
    kind = "vit" if is_vit else "cnn"
    teacher_input_size = IMAGE_SIZES[kind]["binary"]

    kwargs = dict(
        num_classes=1,
        pretrained=False,
        use_metadata=True,
        use_attention=has_cbam,
        fusion_mode=fusion_mode,
        use_projection_head=has_proj,
    )
    if model_name is not None:
        kwargs["model_name"] = model_name
    elif backbone_name is None:
        kwargs["model_name"] = DEFAULT_BINARY_BACKBONE
    else:
        kwargs["backbone_name"] = backbone_name

    # Auto-detección de la variante de pooling del checkpoint. El teacher
    # binario puede entrenarse con `--use_hybrid_pool` → su state_dict lleva
    # claves `hybrid_pool.*` y un `classifier` de ancho `2*in_features`. Sin
    # detectarlo, `SkinCancerModel(**kwargs)` se construiría con GeM por defecto
    # y `teacher.load_state_dict(strict=True)` fallaría con "Unexpected key(s)
    # hybrid_pool.* / Missing key(s) gem_pool.p / size mismatch for classifier",
    # abortando el training multiclase con `--distill_from` al cargar el teacher.
    # `_pool_kwargs` es la fuente única (compartida con `predict.py`); para
    # checkpoints sin pooling especial devuelve `{}` → construcción idéntica a la
    # de un teacher GeM. `feature_dim` no se ve afectado por hybrid_pool (es
    # `int(in_features)` crudo del backbone), de modo que la validación cruzada
    # de dimensiones FitNets sigue siendo correcta.
    kwargs.update(_pool_kwargs(new_state_dict))

    teacher = SkinCancerModel(**kwargs)
    teacher.load_state_dict(new_state_dict)

    teacher.to(device).eval()
    # Congelar completamente — no consume gradientes ni VRAM extra
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher, teacher_input_size


@torch.no_grad()
def precompute_teacher_prob(images, metadata, teacher, teacher_input_size: int = 300):
    """
    Ejecuta un único forward del teacher y devuelve P(maligno) [B].

    Pensado para usarse UNA vez por batch cuando SAM activa dos forwards del
    estudiante: el teacher está frozen, así que su salida es idéntica en ambos
    forwards; cachear evita un forward completo del teacher por step real.
    """
    if images.size(-1) != teacher_input_size or images.size(-2) != teacher_input_size:
        images_teacher = F.interpolate(
            images,
            size=(teacher_input_size, teacher_input_size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        images_teacher = images
    teacher_logit = teacher(images_teacher, metadata)  # [B, 1]
    # Se detiene de inmediato si el teacher produce logits no-finitos. Un
    # checkpoint corrupto o cargado parcialmente puede dejar pesos NaN; sin esta
    # comprobación, el NaN se propaga a `compute_distillation_loss` → BCE NaN →
    # el guard del training loop lo registra como "batch NaN skipped" y la causa
    # raíz queda oculta tras horas de entrenamiento degradado.
    if not torch.isfinite(teacher_logit).all():
        raise RuntimeError(
            "El teacher produjo logits no-finitos (NaN/Inf). "
            "Probable checkpoint corrupto o carga parcial. Verificar el "
            "checkpoint pasado a `load_teacher` y re-entrenar el binario si "
            "es necesario. No continuamos para evitar contaminar el student."
        )
    return torch.sigmoid(teacher_logit.squeeze(-1))  # [B]


@torch.no_grad()
def precompute_teacher_prob_and_features(
    images, metadata, teacher, teacher_input_size: int = 300
):
    """Forward único del teacher que devuelve prob + features.

    Variante de `precompute_teacher_prob` que TAMBIÉN captura los feature
    maps pre-pooling del teacher via `return_intermediate_features=True`
    (requiere que el teacher sea un `SkinCancerModel`). Usado por el loop
    de training cuando FitNets feature distillation está activo — una sola
    ejecución del teacher devuelve ambos outputs, minimizando overhead.

    Returns:
        (teacher_prob, teacher_features):
            - teacher_prob: Tensor [B] — P(maligno) como soft target.
            - teacher_features: Tensor [B, C_teacher, H, W] — feature maps
              pre-pooling del teacher. FitNets los usa como "hint".
    """
    if images.size(-1) != teacher_input_size or images.size(-2) != teacher_input_size:
        images_teacher = F.interpolate(
            images,
            size=(teacher_input_size, teacher_input_size),
            mode="bilinear",
            align_corners=False,
        )
    else:
        images_teacher = images

    out = teacher(images_teacher, metadata, return_intermediate_features=True)
    if not isinstance(out, tuple) or len(out) != 2:
        raise RuntimeError(
            "El teacher no soporta return_intermediate_features. "
            "FitNets requiere que el teacher sea un SkinCancerModel moderno "
            "con el kwarg. Re-cargar teacher con un checkpoint compatible."
        )
    teacher_logit, teacher_feats = out
    if not torch.isfinite(teacher_logit).all():
        raise RuntimeError(
            "El teacher produjo logits no-finitos. Ver docstring "
            "de `precompute_teacher_prob`."
        )
    # Se validan también los feature maps del teacher, no solo los logits.
    # FitNets consume estos feats en `fitnet_feature_loss` para el MSE; si están
    # corruptos (NaN/Inf por una BN "muerta" con `running_var=0` tras batches
    # NaN previos, o por un feature map corrupto cuyos logits la cabeza
    # consigue rescatar), el MSE produciría NaN sin avisar y el guard del
    # training loop lo registraría como "batch NaN skipped", ocultando que el
    # origen está en el teacher.
    if not torch.isfinite(teacher_feats).all():
        raise RuntimeError(
            "El teacher produjo feature maps no-finitos (NaN/Inf) aunque "
            "logits son finitos. Probable BN con `running_var` degenerado tras "
            "batches NaN previos en el training del teacher. Re-entrenar el "
            "teacher o verificar el checkpoint pasado a `load_teacher`."
        )
    return torch.sigmoid(teacher_logit.squeeze(-1)), teacher_feats


def compute_distillation_loss(
    student_logits,
    images,
    metadata,
    teacher,
    teacher_input_size: int = 300,
    teacher_prob_cached=None,
):
    """
    Calcula el loss de destilación binario→multiclase.

    El profesor produce P(maligno | imagen) como soft target.
    El estudiante agrupa sus 8 probabilidades en P(maligno) = sum(probs[malignas]).
    El loss es BCE entre ambas probabilidades binarias.

    Args:
        student_logits: Tensor [B, 8] — logits multiclase del estudiante
        images: Tensor [B, 3, H, W] — imágenes del batch (resolución del estudiante,
            típicamente 380px para el especialista multiclase)
        metadata: Tensor [B, 3] — metadatos clínicos
        teacher: SkinCancerModel binario (frozen, eval)
        teacher_input_size: resolución nativa del profesor (derivada por
            `load_teacher`: 300 EfficientNet-B3 / 294 DINOv2 ViT-L). Las
            imágenes se redimensionan antes del forward del profesor para
            eliminar el domain shift imagen-teacher.
        teacher_prob_cached: Tensor [B] opcional con P(maligno) precomputado por
            `precompute_teacher_prob()`. Cuando se provee, se omite el forward del
            teacher (caso SAM: ambos forwards del estudiante comparten teacher_prob).
    Returns:
        distill_loss: Tensor escalar
    """
    # Comprobación de finitud sobre los logits del student, simétrica a las de
    # los tensores del teacher (logits, feature maps y prob cacheada). Sin ella,
    # `softmax(NaN)=NaN`, `sum=NaN`, `clamp(NaN, eps, 1-eps)=NaN` (el clamp de
    # PyTorch preserva NaN) y `BCE(NaN, valid_target)` propaga el NaN; el guard
    # del training loop lo registra como "batch NaN skipped" apuntando al lado
    # equivocado del pipeline. Aquí se falla de inmediato con la causa probable.
    if not torch.isfinite(student_logits).all():
        raise RuntimeError(
            "student_logits contienen NaN/Inf antes de calcular el loss de "
            "destilación. Probable causa: GeM degenerado bajo AMP, "
            "BatchNorm con running_var corrompido tras un batch NaN "
            "previo, AMP overflow en el segundo forward de SAM, o pesos del "
            "student ligeramente NaN tras un step contaminado. Acción: revisar "
            "gradient norms (vanishing del backbone post-warmup), reducir lr/"
            "accum_steps, verificar BN running stats."
        )

    if teacher_prob_cached is not None:
        # El camino con cache evita `precompute_teacher_prob` y su comprobación
        # de finitud, así que se valida aquí explícitamente. Si el caller cacheó
        # un prob corrupto (p. ej. el primer forward de SAM produjo logits NaN
        # por AMP overflow), `F.binary_cross_entropy` lanzaría más abajo
        # `RuntimeError: all elements of target should be between 0 and 1` —
        # mensaje engañoso que no apunta al origen. Esta comprobación falla
        # antes con la causa probable identificada.
        if not torch.isfinite(teacher_prob_cached).all():
            raise RuntimeError(
                "`teacher_prob_cached` contiene NaN/Inf. "
                "Probable causa: el forward previo del teacher (típicamente el "
                "primer forward SAM cuyo prob se cachea para reutilizar en el "
                "segundo) produjo logits no-finitos. Revisar la estabilidad del "
                "teacher (comprobaciones de finitud upstream, AMP overflow, GeM "
                "degenerado) y considerar desactivar el cache si SAM+AMP+teacher "
                "son inestables juntos."
            )
        teacher_prob = teacher_prob_cached
    else:
        teacher_prob = precompute_teacher_prob(
            images, metadata, teacher, teacher_input_size=teacher_input_size
        )

    # Autocast desactivado para softmax + sum + BCE: bajo fp16, `F.softmax` de 8
    # clases puede dar exactamente 0.0 con logits extremos (diferencia > ~11), de
    # modo que la suma sobre REFERRAL_INDICES llegaría a 0.0 antes de que el clamp
    # la rescate, y el gradiente del clamp en el límite es 0. Mismo criterio que
    # GeM, SeesawLoss y SupConLoss: forzar float32.
    with torch.amp.autocast('cuda', enabled=False):
        student_logits_f32 = student_logits.float()
        teacher_prob_f32 = teacher_prob.float()

        # Se agrega sobre REFERRAL_INDICES (Mel/BCC/SCC + AK), no solo sobre
        # MALIGNANT_INDICES. El teacher binario fue entrenado con AK como
        # maligna; el student debe reproducir esa frontera de decisión binaria.
        # Sin esto, el KD empujaría al student a tratar AK como benigna (el
        # teacher diría "no maligno" para imágenes que el binario clasifica como
        # malignas) → confusión catastrófica. El estado CRITICAL de la cascada
        # (predict.py) sí usa MALIGNANT_INDICES puro.
        student_probs = F.softmax(student_logits_f32, dim=1)  # [B, 8]
        student_malignant_prob = student_probs[:, REFERRAL_INDICES].sum(dim=1)  # [B]

        # Clamp para estabilidad numérica: softmax puede dar probs ~0 o ~1, y
        # `F.binary_cross_entropy` calcula log(p) y log(1-p) directamente —
        # log(0) = -inf → NaN. El rango [1e-7, 1-1e-7] es suficiente en float32
        # (eps ~1.2e-7).
        _eps = 1e-7
        student_malignant_prob = student_malignant_prob.clamp(min=_eps, max=1.0 - _eps)
        teacher_prob_f32 = teacher_prob_f32.clamp(min=_eps, max=1.0 - _eps)

        # BCE entre la "opinión" del profesor y la agrupación del estudiante
        distill_loss = F.binary_cross_entropy(
            student_malignant_prob,
            teacher_prob_f32,
            reduction="mean",
        )

    return distill_loss


def fitnet_feature_loss(student_feat, teacher_feat, projection):
    """Feature Distillation canónica (FitNets, Romero 2015).

    Referencia: Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015.
    Refinamiento: Heo et al., "A Comprehensive Overhaul of Feature
    Distillation", ICCV 2019.

    Complementa `compute_distillation_loss` (que destila en el espacio de
    logits) con transferencia de representaciones intermedias. La intuición
    del paper es que el student no solo debe imitar la salida final del
    teacher, sino también su "forma de ver" las features — especialmente
    útil cuando el student tiene menor capacidad que el teacher (nuestro
    caso: EffNet-B4 student vs EffNet-B3 teacher o DINOv2 ViT-L teacher).

    **Algoritmo:**
    1. Extraer features intermedias del teacher (frozen, eval). Resolución y
       número de canales puede diferir del student.
    2. Proyectar las features del student al espacio del teacher usando una
       capa `Conv2d(1×1)` (o `nn.Identity` si las dimensiones coinciden).
    3. Loss = MSE entre features proyectadas y teacher. Reducción 'mean'
       sobre batch y spatial dims.

    **Nota clínica:** Este loss es ADITIVO al término supervisado y
    al logit-KD. Nunca reemplaza al focal loss — siempre peso pequeño
    (`fitnet_weight ∈ [0.01, 0.1]` según el paper).

    Args:
        student_feat: Tensor `[B, C_s, H_s, W_s]` — feature maps del student
            (capa intermedia, típicamente antes del GeM pooling).
        teacher_feat: Tensor `[B, C_t, H_t, W_t]` — feature maps del teacher.
            El teacher debe estar en `eval()` + `no_grad()` (el caller asegura).
        projection: `nn.Module` que proyecta `student_feat` al shape del
            teacher. Típicamente `nn.Conv2d(C_s, C_t, kernel_size=1)` con
            inicialización normal. Si `C_s == C_t` y las resoluciones
            coinciden, puede ser `nn.Identity()`.

    Returns:
        Tensor escalar con el loss MSE.

    **Notas:**
    - Si las resoluciones espaciales difieren, se hace `F.interpolate`
      bilineal del student al shape del teacher antes del MSE.
    - El loss se computa en float32 (autocast desactivado) por consistencia
      con el resto del módulo.
    """
    with torch.amp.autocast('cuda', enabled=False):
        s_f32 = student_feat.float()
        t_f32 = teacher_feat.float().detach()  # teacher frozen: sin grad

        # Comprobación de finitud de los feature maps del student, simétrica a
        # las de los logits y feats del teacher. El student también puede
        # producir feature maps NaN/Inf cuando sus buffers BN se corrompen por
        # batches NaN previos (running_var=0 tras una época inestable); el NaN
        # se propagaría al MSE y el guard del training loop lo marcaría como
        # "batch NaN skipped" sin señalar a FitNets como origen. Fallar aquí
        # convierte horas de depuración en un traceback inmediato con causa clara.
        if not torch.isfinite(s_f32).all():
            raise RuntimeError(
                "student feature maps contienen NaN/Inf. "
                "Causa probable: buffers BN del student con running_var=0 tras "
                "un batch NaN previo, o gradientes explosivos en el forward "
                "antes de `return_intermediate_features=True`. Revisar las "
                "comprobaciones de finitud en `train.py` y considerar reducir "
                "lr o accum_steps."
            )

        # Proyección de canales (student → teacher dim)
        s_proj = projection(s_f32)

        # Alinear resolución espacial si difiere
        if s_proj.shape[-2:] != t_f32.shape[-2:]:
            s_proj = F.interpolate(
                s_proj,
                size=t_f32.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # MSE reducción 'mean' sobre todos los elementos (batch + canal + spatial)
        loss = F.mse_loss(s_proj, t_f32, reduction="mean")

    return loss
