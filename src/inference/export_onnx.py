"""
Exportación de modelos a formato ONNX para despliegue.

ONNX (Open Neural Network Exchange) permite ejecutar los modelos en:
- Servidores de producción (ONNX Runtime, TensorRT)
- Dispositivos móviles (ONNX Runtime Mobile)
- Navegadores web (ONNX.js)
- Cualquier plataforma sin depender de PyTorch

Habilita el despliegue del backend de inferencia y de la interfaz de usuario
sin requerir PyTorch en el entorno de producción.
"""
import torch
from collections import OrderedDict
from pathlib import Path


def export_to_onnx(
    model,
    output_path,
    img_size=300,
    use_metadata=True,
    opset_version=14,
    dynamic_batch=True,
):
    """
    Exporta un SkinCancerModel a formato ONNX.

    Args:
        model: SkinCancerModel entrenado (en eval mode)
        output_path: Ruta de salida (.onnx)
        img_size: Resolución de entrada (300 para B3, 380 para B4)
        use_metadata: Si el modelo usa metadatos clínicos
        opset_version: Versión del opset ONNX (14 es compatible con la mayoría)
        dynamic_batch: Si True, permite batch size variable
    Returns:
        output_path: Path del archivo exportado
    """
    model.eval()
    device = next(model.parameters()).device

    # Los backbones DINOv2 (ViT) usan operadores que requieren opset >= 17
    # (LayerNorm nativa, scaled_dot_product_attention). Si el backbone es de la
    # familia "dinov2_", se fuerza opset 17 independientemente del parámetro del
    # caller; los backbones EfficientNet conservan el opset solicitado (14 por
    # defecto), suficiente para sus operadores.
    backbone_name = getattr(model, "backbone_name", None)
    if backbone_name is not None and str(backbone_name).startswith("dinov2_"):
        if opset_version < 17:
            print(
                f"[ONNX] Backbone {backbone_name} requiere opset >= 17; "
                f"forzando opset_version={opset_version} → 17."
            )
            opset_version = 17

    # Crear inputs dummy
    dummy_image = torch.randn(1, 3, img_size, img_size, device=device)
    input_names = ["image"]
    dynamic_axes = {}

    if dynamic_batch:
        dynamic_axes["image"] = {0: "batch_size"}
        dynamic_axes["logits"] = {0: "batch_size"}

    if use_metadata:
        dummy_meta = torch.tensor([[0.5, 0.0, 2.0]], device=device)
        dummy_inputs = (dummy_image, dummy_meta)
        input_names.append("metadata")
        if dynamic_batch:
            dynamic_axes["metadata"] = {0: "batch_size"}
    else:
        dummy_inputs = (dummy_image,)

    output_path = str(output_path)
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes if dynamic_batch else None,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    return output_path


def load_and_export(
    checkpoint_path,
    output_path,
    mode,
    use_metadata=True,
    model_name=None,
    num_classes=None,
    img_size=None,
):
    """
    Carga un checkpoint y exporta a ONNX con auto-detección de backbone.

    La arquitectura del modelo se infiere directamente del state_dict en lugar de
    exigir que el caller la describa: se lee el checkpoint, se detecta el backbone
    con `_detect_backbone_config` (la misma función que usa `predict.py`) y se
    construye con `backbone_name=` cuando aplica. Esto garantiza que el auto-upgrade
    a opset 17 para backbones DINOv2 (ViT) se active correctamente — la LayerNorm
    nativa requiere opset >= 17, y exportar un ViT con opset 14 produce un ONNX
    inválido. `img_size` se deriva de `IMAGE_SIZES` y `num_classes` de los mapas
    canónicos de `constants.py`. Los parámetros `model_name`, `num_classes` e
    `img_size` se mantienen como override opcional para compatibilidad.

    Args:
        checkpoint_path: Ruta al .pth
        output_path: Ruta de salida .onnx
        mode: "binary" o "multiclass" — determina num_classes e img_size canónicos
        use_metadata: Si usar metadatos
        model_name: Override opcional del backbone timm (legacy)
        num_classes: Override opcional (legacy)
        img_size: Override opcional (legacy)
    Returns:
        output_path: Path del archivo exportado
    """
    from src.models.architectures import SkinCancerModel
    from src.inference.predict import (
        _detect_backbone_config,
        _detect_cbam,
        _pool_kwargs,
        _fitnet_kwargs,
    )
    from src.data.constants import (
        IMAGE_SIZES,
        MULTICLASS_LABEL_MAP,
        DEFAULT_BINARY_BACKBONE,
        DEFAULT_MULTICLASS_BACKBONE,
    )

    if mode not in ("binary", "multiclass"):
        raise ValueError(f"mode debe ser 'binary' o 'multiclass', no {mode!r}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        # Elimina el prefijo `_orig_mod.` que inyecta torch.compile al envolver el
        # modelo. Mismo tratamiento que en el resto de puntos de carga del proyecto
        # (early_stopping, distillation, predict, calibración). Sin esto, un
        # checkpoint entrenado con torch.compile sin desenvolver fallaría al cargar
        # con "Unexpected keys: _orig_mod.*".
        if name.startswith("_orig_mod."):
            name = name[len("_orig_mod."):]
        new_state_dict[name] = v

    backbone_name, fusion_mode = _detect_backbone_config(new_state_dict)
    is_vit = bool(backbone_name) and (
        "vit" in backbone_name or "dinov2" in backbone_name
    )
    kind = "vit" if is_vit else "cnn"

    if num_classes is None:
        num_classes = 1 if mode == "binary" else len(MULTICLASS_LABEL_MAP)
    if img_size is None:
        img_size = IMAGE_SIZES[kind][mode]

    has_projection_head = any(
        k.startswith("projection_head.") for k in new_state_dict
    )
    # Auto-detecta CBAM también en la exportación. Si el checkpoint se entrenó con
    # `use_attention=False`, forzar True aquí haría que `load_state_dict(strict=True)`
    # fallara con "Unexpected keys: cbam.*".
    has_cbam = _detect_cbam(new_state_dict)

    kwargs = dict(
        num_classes=num_classes,
        pretrained=False,
        use_metadata=use_metadata,
        use_attention=has_cbam,
        fusion_mode=fusion_mode,
        use_projection_head=has_projection_head,
    )
    if model_name is not None:
        kwargs["model_name"] = model_name
    elif backbone_name is None:
        kwargs["model_name"] = (
            DEFAULT_BINARY_BACKBONE if mode == "binary" else DEFAULT_MULTICLASS_BACKBONE
        )
    else:
        kwargs["backbone_name"] = backbone_name

    # Variante de pooling (hybrid/multiscale/attention) auto-detectada del
    # checkpoint. Sin esto, un checkpoint con hybrid pooling se exportaría a un
    # ONNX roto silenciosamente. Mismo tratamiento que en predict.py y en los
    # scripts de calibración.
    kwargs.update(_pool_kwargs(new_state_dict))
    # Configuración FitNets (use_fitnet + fitnet_teacher_dim) auto-detectada del
    # checkpoint. Sin esto, un checkpoint entrenado con FitNets fallaría en
    # load_state_dict(strict=True) con "Unexpected keys: fitnet_projection.*"
    # antes de exportar.
    kwargs.update(_fitnet_kwargs(new_state_dict))

    model = SkinCancerModel(**kwargs)
    model.load_state_dict(new_state_dict)
    model.eval()

    return export_to_onnx(model, output_path, img_size=img_size, use_metadata=use_metadata)


def verify_onnx(onnx_path, pytorch_model, img_size=300, use_metadata=True, atol=None):
    """
    Verifica que la salida ONNX coincide con PyTorch.

    Args:
        onnx_path: Ruta al .onnx
        pytorch_model: Modelo PyTorch original
        img_size: Resolución
        use_metadata: Si usa metadatos
        atol: Tolerancia absoluta para comparación. Si `None`, se autodetecta
              según el tipo de backbone:
                - CNN (EfficientNet, opset 14): 1e-4 — operadores simples, sin
                  acumulación significativa de error.
                - ViT (DINOv2, opset 17): 1e-3 — la LayerNorm nativa de ONNX
                  Runtime difiere sutilmente del kernel de PyTorch. Con ~24 capas
                  del ViT-L la acumulación puede llegar a ~5e-4..2e-3 por dimensión
                  del output, lo que rechazaría ONNX válidos con atol=1e-4.
    Returns:
        matches: bool
        max_diff: float — diferencia máxima entre outputs
    """
    # Autodetecta la tolerancia según el tipo de backbone. El ViT acumula error
    # numérico a lo largo de LayerNorm en N capas bajo opset 17.
    if atol is None:
        backbone_name = getattr(pytorch_model, "backbone_name", None) or ""
        is_vit = any(
            tok in backbone_name.lower()
            for tok in ("vit", "dinov2", "swin", "beit", "deit")
        )
        atol = 1e-3 if is_vit else 1e-4
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime requerido para verificacion. "
            "Instalar con: pip install onnxruntime"
        )
    import numpy as np

    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device

    # Entrada común
    dummy_image = torch.randn(1, 3, img_size, img_size, device=device)

    # PyTorch inference
    with torch.no_grad():
        if use_metadata:
            dummy_meta = torch.tensor([[0.5, 0.0, 2.0]], device=device)
            pt_output = pytorch_model(dummy_image, dummy_meta).cpu().numpy()
        else:
            pt_output = pytorch_model(dummy_image).cpu().numpy()

    # ONNX inference
    sess = ort.InferenceSession(str(onnx_path))
    inputs = {"image": dummy_image.cpu().numpy()}
    if use_metadata:
        inputs["metadata"] = np.array([[0.5, 0.0, 2.0]], dtype=np.float32)

    ort_output = sess.run(None, inputs)[0]

    max_diff = np.abs(pt_output - ort_output).max()
    matches = max_diff < atol

    # Si la verificación falla, se imprime el max_diff real junto con atol para
    # facilitar el diagnóstico: distingue un problema de tolerancia numérica de
    # un export realmente roto.
    if not matches:
        print(f"[MISMATCH] max_diff={max_diff:.2e}, atol={atol:.2e}")
        print(
            "  Si max_diff está en ~1e-3 con backbone ViT, probar atol=1e-3 "
            "explícito (LayerNorm × N layers acumula error bajo opset 17)."
        )

    return matches, float(max_diff)
