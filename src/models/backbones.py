"""
Factory de backbones — unifica EfficientNet (CNN) y DINOv2 ViT bajo el mismo
contrato de salida: `[B, C, H, W]`.

Motivación
----------
El pipeline existente (GeM, CBAM, FiLM, SkinCancerModel) asume que el backbone
devuelve feature maps espaciales `[B, C, H, W]`. EfficientNet con
`global_pool=''` ya lo hace. DINOv2 ViT devuelve tokens `[B, N+1, D]` (CLS +
parches). Para introducir ViT sin reescribir toda la arquitectura downstream,
envolvemos el backbone en `BackboneWrapper` que:

- CNN (EffNet): pass-through, output `[B, C, H, W]` directo.
- ViT (DINOv2): extrae tokens de parche (dropea CLS), reordena a grilla
  `[B, D, √N, √N]` usando la geometría del parche (14×14) y el `img_size`
  efectivo. Los bloques posteriores (GeM, CBAM, FiLM) operan idénticamente.

Esto preserva el contrato del modelo: el pooling GeM sigue operando en
float32, la inicialización en identidad de FiLM no se ve alterada y la forma
del forward pass del `SkinCancerModel` no cambia.

El comportamiento por defecto mantiene compatibilidad exacta con los
checkpoints de EfficientNet existentes.

Resolución de entrada para ViT
------------------------------
DINOv2 ViT-L/14 requiere que `img_size` sea múltiplo del parche (14). La
pipeline actual usa 300px (binario) y 380px (multiclase). timm interpola
embeddings de posición si `img_size` difiere del pretrain (224), pero el
input DEBE ser múltiplo de 14 para que el reshape a grilla funcione sin
padding. Escogemos:
- binary  → 294 (21×14)  — cercano a 300, sin padding
- multi   → 378 (27×14)  — cercano a 380, sin padding

Los datasets siguen produciendo 300/380. El wrapper hace `F.interpolate` al
tamaño compatible ANTES de entrar al ViT. Este resize extra (~0.1ms/batch en
H100) es trivial comparado con el forward del ViT-L.
"""

from __future__ import annotations

import warnings
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


_SUPPORTED_BACKBONES = {
    "effnet_b3": {
        "timm_name": "tf_efficientnet_b3.ns_jft_in1k",
        "kind": "cnn",
        "native_size": 300,
    },
    "effnet_b4": {
        "timm_name": "tf_efficientnet_b4.ns_jft_in1k",
        "kind": "cnn",
        "native_size": 380,
    },
    "dinov2_vitl14": {
        "timm_name": "vit_large_patch14_dinov2.lvd142m",
        "kind": "vit",
        "patch_size": 14,
        # Tamaños múltiplo de 14 cercanos a 300/380 del pipeline actual.
        "native_size_binary": 294,
        "native_size_multiclass": 378,
    },
    "dinov2_vitb14": {
        # Variante más ligera (ViT-B) para cuando ViT-L no cabe en el
        # presupuesto de VRAM disponible.
        "timm_name": "vit_base_patch14_dinov2.lvd142m",
        "kind": "vit",
        "patch_size": 14,
        "native_size_binary": 294,
        "native_size_multiclass": 378,
    },
}


class BackboneWrapper(nn.Module):
    """
    Envoltorio que unifica la salida del backbone a `[B, C, H, W]`.

    Para CNN (EffNet) es pass-through. Para ViT (DINOv2) extrae tokens de
    parche, dropea el CLS y reordena a grilla espacial.

    Atributos públicos:
        num_features: dimensión de canal (C) del output. Análogo a
            `timm_model.num_features`, lo que permite al SkinCancerModel
            mantener la línea `in_features = self.backbone.num_features` sin
            cambios.
        kind: "cnn" o "vit".
        expected_input_size: tamaño espacial al que se redimensiona el input
            antes de entrar al backbone. Para ViT es múltiplo de patch_size.
    """

    # Aviso emitido una sola vez (variable de clase, compartida entre réplicas
    # de DataParallel) cuando el redimensionado defensivo del forward se activa.
    # Ese redimensionado absorbe inputs ligeramente fuera de especificación,
    # pero en uso normal el DataLoader ya entrega el tamaño correcto (ver
    # IMAGE_SIZES en constants.py). Si el aviso se dispara hay un desajuste de
    # configuración (p. ej. un backbone ViT con get_transforms(backbone_kind=
    # "cnn")) que degradaría la calidad con una doble interpolación silenciosa.
    _resize_warned: bool = False

    def __init__(
        self,
        backbone: nn.Module,
        kind: str,
        num_features: int,
        expected_input_size: int,
        patch_size: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.kind = kind
        self.num_features = num_features
        self.expected_input_size = expected_input_size
        self.patch_size = patch_size

        if kind == "vit":
            if patch_size is None:
                raise ValueError("patch_size requerido para backbones ViT")
            if expected_input_size % patch_size != 0:
                raise ValueError(
                    f"expected_input_size={expected_input_size} debe ser múltiplo de "
                    f"patch_size={patch_size} para reshape a grilla sin padding."
                )
            self._grid_side = expected_input_size // patch_size
        else:
            self._grid_side = None

        # Hooks para capturar tokens de capas intermedias del ViT (usados por
        # el pooling multi-escala). Solo se instalan cuando el caller lo pide
        # explícitamente vía `enable_intermediate_layers(...)`; desactivados por
        # defecto, el comportamiento es idéntico al baseline.
        self._intermediate_layer_indices: Optional[list[int]] = None
        self._intermediate_outputs: list[torch.Tensor] = []
        self._intermediate_hooks: list = []

    def enable_intermediate_layers(self, indices: list[int]) -> None:
        """Registra forward hooks en `backbone.blocks[i]` para los índices
        `indices` (típicamente 3 capas: tercio inferior, medio y último). Solo
        válido para `kind="vit"`. Se llama una vez en el `__init__` del
        SkinCancerModel cuando el pooling multi-escala está activo.

        Los hooks acumulan tokens `[B, N, D]` en `self._intermediate_outputs`
        durante cada forward del backbone. El caller los recupera con
        `forward(image, return_intermediate_layers=True)`.

        Args:
            indices: lista de índices 0-based de capas del backbone a capturar.
                Para DINOv2 ViT-L (24 blocks), típico [7, 15, 23].
                Para DINOv2 ViT-B (12 blocks), típico [3, 7, 11].
        """
        if self.kind != "vit":
            raise ValueError(
                "enable_intermediate_layers requiere kind='vit'. "
                f"Recibido kind={self.kind!r}. La fusión multi-escala solo "
                "está soportada en backbones ViT."
            )
        if self._intermediate_layer_indices is not None:
            # Ya se llamó antes: cleanup de hooks viejos antes de re-instalar
            for h in self._intermediate_hooks:
                h.remove()
            self._intermediate_hooks = []
        # Validar que los indices son válidos
        if not hasattr(self.backbone, "blocks"):
            raise RuntimeError(
                "El backbone ViT no expone el atributo `.blocks` esperado en "
                "los ViT/DINOv2 estándar de timm. Verificar la versión de timm."
            )
        n_blocks = len(self.backbone.blocks)
        for idx in indices:
            if not (0 <= idx < n_blocks):
                raise ValueError(
                    f"Índice {idx} fuera de rango [0, {n_blocks-1}] "
                    f"para un backbone con {n_blocks} bloques."
                )
        self._intermediate_layer_indices = list(indices)

        def _make_hook(slot_idx: int):
            def hook(_module, _inp, output):
                # output puede ser tensor [B, N, D] o tuple. timm ViT blocks
                # retornan tensor directamente.
                if isinstance(output, tuple):
                    output = output[0]
                # Asignar al slot de forma determinista según el orden de registro.
                # _intermediate_outputs se reinicia en cada forward (en nuestro forward).
                while len(self._intermediate_outputs) <= slot_idx:
                    self._intermediate_outputs.append(None)
                self._intermediate_outputs[slot_idx] = output
            return hook

        for slot, idx in enumerate(indices):
            h = self.backbone.blocks[idx].register_forward_hook(_make_hook(slot))
            self._intermediate_hooks.append(h)

    def forward(
        self,
        image: torch.Tensor,
        return_tokens: bool = False,
        return_intermediate_layers: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            image: [B, 3, H, W] — cualquier resolución; se ajusta a
                `expected_input_size` si es necesario.
            return_tokens: solo válido para `kind="vit"`. Si True, devuelve los
                tokens `[B, N_total, D]` SIN reshape a grilla espacial (sin
                dropear CLS/register tokens). Lo requiere `AttentionPoolViT`,
                que necesita la secuencia completa. Para `kind="cnn"` el
                argumento se ignora (las CNN no producen tokens). Por defecto
                False, equivalente al comportamiento original.
            return_intermediate_layers: solo válido cuando
                `enable_intermediate_layers(indices)` se llamó previamente
                (kind='vit'). Si True, retorna tupla
                `(feature_maps, [tokens_layer_0, tokens_layer_1, ...])`
                con los outputs `[B, N_total, D]` de las capas registradas en
                el orden de `indices`. Compatible con `return_tokens=False`
                (devuelve grid feature_maps + lista de tokens intermedios
                para AttentionPool). Default False preserva baseline.
        Returns:
            feature_maps: [B, num_features, H', W']
                - CNN: H'/W' definido por el stride del backbone.
                - ViT: H' = W' = expected_input_size // patch_size.
            OR (si `return_tokens=True` y kind="vit"):
            tokens: [B, N_total, D] secuencia completa del encoder ViT.
            OR (si `return_intermediate_layers=True`):
            (feature_maps, list[tokens_intermediate]) tupla.
        """
        # Limpiar los slots intermedios al inicio de cada forward para que las
        # salidas de una llamada no contaminen las posteriores. Sin esto, si el
        # caller reutiliza la instancia alternando llamadas con y sin
        # `return_intermediate_layers=True`, los slots quedarían obsoletos.
        if self._intermediate_layer_indices is not None:
            self._intermediate_outputs = [None] * len(self._intermediate_layer_indices)
        if image.size(-1) != self.expected_input_size or image.size(-2) != self.expected_input_size:
            # Advertir una sola vez cuando el redimensionado defensivo se
            # activa. En uso normal el DataLoader ya entrega el tamaño correcto
            # (IMAGE_SIZES[backbone_kind]); si no, hay un desajuste de
            # configuración que degrada la calidad con una doble interpolación
            # (Resize cv2 CUBIC seguido de F.interpolate bilinear).
            if not BackboneWrapper._resize_warned:
                warnings.warn(
                    f"BackboneWrapper ({self.kind}) redimensionando input de "
                    f"{tuple(image.shape[-2:])} → "
                    f"({self.expected_input_size}, {self.expected_input_size}). "
                    "Esto indica un mismatch entre la configuración del DataLoader "
                    "y el backbone — el DataLoader debería producir directamente "
                    "el tamaño esperado. Revisar IMAGE_SIZES en constants.py y "
                    "get_transforms() en dataset.py.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                BackboneWrapper._resize_warned = True
            # Resize a tamaño compatible con el backbone. Bilinear + align_corners=False
            # replica la convención de timm.data.transforms (interpolación suave sin
            # desalineación de bordes).
            image = F.interpolate(
                image,
                size=(self.expected_input_size, self.expected_input_size),
                mode="bilinear",
                align_corners=False,
            )

        if self.kind == "cnn":
            # timm CNN con global_pool='' devuelve [B, C, H, W] directamente.
            # `return_tokens` no aplica (CNN no tiene tokens); ignorado para
            # permitir que callers pasen el flag incondicionalmente.
            return self.backbone(image)

        # ViT: forward_features devuelve tokens [B, N+1, D] con CLS al inicio
        # (en la mayoría de vit_* de timm). Algunos modelos DINOv2 registran
        # tokens adicionales (register tokens), debemos manejar el caso general
        # dropeando cualquier prefijo de tokens no-espaciales.
        tokens = self.backbone.forward_features(image)  # [B, N_total, D]

        # Si return_tokens=True, devolver la secuencia completa SIN reshape ni
        # dropeo de prefijos. `AttentionPoolViT` gestiona el prefijo
        # internamente. Se mantiene la rama de dict por consistencia con el
        # manejo defensivo del output de forward_features.
        if return_tokens:
            if isinstance(tokens, dict):
                # Preferir patch tokens si ya están separados; sino x completo.
                if "x_norm_patchtokens" in tokens and "x_norm_clstoken" in tokens:
                    cls = tokens["x_norm_clstoken"]
                    if cls.ndim == 2:
                        cls = cls.unsqueeze(1)
                    return torch.cat([cls, tokens["x_norm_patchtokens"]], dim=1)
                if "x" in tokens:
                    return tokens["x"]
                raise RuntimeError(
                    f"ViT forward_features devolvió dict sin claves "
                    f"reconocibles para return_tokens: {list(tokens.keys())}"
                )
            if tokens.ndim != 3:
                raise RuntimeError(
                    f"ViT return_tokens: esperado ndim=3, recibido {tokens.ndim}"
                )
            return tokens

        # Algunas versiones de timm hacen que `forward_features` devuelva un
        # dict para DINOv2 (`x_norm_patchtokens`, `x_norm_clstoken`,
        # `x_norm_regtokens`, `x`) en lugar de un tensor. Lo manejamos
        # defensivamente: preferimos los patch tokens ya separados y, si no
        # están, caemos al tensor completo `x`. Si no se reconoce ninguna clave
        # se lanza un error explícito en vez de fallar silenciosamente.
        if isinstance(tokens, dict):
            if "x_norm_patchtokens" in tokens:
                patch_only = tokens["x_norm_patchtokens"]
                if patch_only.ndim != 3:
                    raise RuntimeError(
                        f"ViT backbone devolvió x_norm_patchtokens con "
                        f"ndim={patch_only.ndim}, esperado 3. "
                        f"Shape={tuple(patch_only.shape)}"
                    )
                b, n_patches, d = patch_only.shape
                expected_patches_d = self._grid_side * self._grid_side
                if n_patches != expected_patches_d:
                    raise RuntimeError(
                        f"ViT x_norm_patchtokens: {n_patches} parches, "
                        f"esperado {expected_patches_d} para grid "
                        f"{self._grid_side}×{self._grid_side}."
                    )
                return patch_only.transpose(1, 2).reshape(
                    b, d, self._grid_side, self._grid_side
                )
            if "x" in tokens:
                tokens = tokens["x"]
            else:
                raise RuntimeError(
                    f"ViT forward_features devolvió dict sin claves "
                    f"reconocibles: {list(tokens.keys())}. "
                    "Esperado 'x_norm_patchtokens' o 'x'."
                )

        if tokens.ndim != 3:
            raise RuntimeError(
                f"ViT backbone devolvió tensor con ndim={tokens.ndim}, esperado 3 "
                f"([B, N, D]). Shape={tuple(tokens.shape)}"
            )

        b, n_total, d = tokens.shape
        expected_patches = self._grid_side * self._grid_side
        # Tokens no-espaciales al inicio (CLS + opcionalmente register tokens).
        n_prefix = n_total - expected_patches
        if n_prefix < 0:
            raise RuntimeError(
                f"ViT devolvió {n_total} tokens, menos que los {expected_patches} "
                f"parches esperados para grid {self._grid_side}×{self._grid_side}. "
                "Revisar expected_input_size/patch_size."
            )
        patch_tokens = tokens[:, n_prefix:, :]  # [B, N_patches, D]

        # Reordenar a grilla espacial: [B, N, D] → [B, D, H, W]
        feature_maps = patch_tokens.transpose(1, 2).reshape(
            b, d, self._grid_side, self._grid_side
        )

        # Si el caller pidió las capas intermedias, retornar la tupla
        # `(feature_maps, list[tokens_layer_i])`. Los hooks ya se ejecutaron
        # durante `forward_features` y `self._intermediate_outputs` contiene
        # los outputs en orden. Validamos que TODOS los slots se llenaron, como
        # defensa frente a hooks ejecutados parcialmente.
        if return_intermediate_layers:
            if self._intermediate_layer_indices is None:
                raise RuntimeError(
                    "return_intermediate_layers=True pero no se llamó "
                    "enable_intermediate_layers(indices) previamente. El caller "
                    "debe configurar las capas a capturar antes del primer "
                    "forward."
                )
            if any(t is None for t in self._intermediate_outputs):
                raise RuntimeError(
                    "Los hooks de capas intermedias no se ejecutaron para todos "
                    "los slots durante forward_features. "
                    f"Estado: {[t is None for t in self._intermediate_outputs]}"
                )
            return feature_maps, list(self._intermediate_outputs)
        return feature_maps


def _apply_linear_drop_path_schedule(backbone: nn.Module, max_rate: float) -> int:
    """Aplica un schedule lineal de Stochastic Depth al backbone.

    Itera sobre los módulos `timm.layers.DropPath` del backbone y les asigna
    `drop_prob` creciente linealmente desde 0 en la primera capa hasta
    `max_rate` en la última. Coincide con el default del paper EfficientNet
    original (Tan & Le 2019) y con Huang et al. "Deep Networks with
    Stochastic Depth" ECCV 2016.

    EfficientNet y DINOv2 ViT usan el mismo atributo `.drop_prob` en el
    módulo `DropPath` de timm — el iter aplica uniformemente.

    Retorna el número de capas a las que se aplicó el schedule (para
    debugging).
    """
    try:
        from timm.layers import DropPath as TimmDropPath
    except ImportError:
        # Fallback: algunas versiones de timm exportan desde timm.models.layers
        from timm.models.layers import DropPath as TimmDropPath  # type: ignore

    drop_path_layers = [m for m in backbone.modules() if isinstance(m, TimmDropPath)]
    if not drop_path_layers:
        warnings.warn(
            "No se encontraron capas DropPath en el backbone, así que el "
            "schedule lineal no tiene efecto. Verificar que el backbone use "
            "Stochastic Depth (EfficientNet/DINOv2 deberían tenerlo con "
            "drop_path_rate > 0).",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0

    schedule = compute_stoch_depth_schedule(
        num_layers=len(drop_path_layers), drop_path_rate=max_rate
    )
    for layer, rate in zip(drop_path_layers, schedule):
        layer.drop_prob = float(rate)
    return len(drop_path_layers)


def build_backbone(
    name: str,
    pretrained: bool = True,
    mode: str = "binary",
    drop_path_rate: float = 0.2,
    drop_path_schedule: str = "uniform",
    use_grad_checkpointing: bool = False,
) -> BackboneWrapper:
    """
    Construye un backbone envuelto que devuelve feature maps `[B, C, H, W]`.

    Args:
        name: clave en `_SUPPORTED_BACKBONES`. Ej: "effnet_b3", "dinov2_vitl14".
        pretrained: cargar pesos preentrenados (siempre True en producción).
        mode: "binary" o "multiclass". Solo afecta ViT (selecciona img_size
            compatible con patch_size cercano a 300/380).
        drop_path_rate: Stochastic Depth. Se mantiene en 0.2 por compatibilidad
            con los checkpoints EffNet existentes; un valor mayor (p. ej. 0.3
            para multiclase) se pasa explícito desde train.py.
        use_grad_checkpointing: si True, llama
            `backbone.set_grad_checkpointing(True)` sobre el modelo timm recién
            creado (ViT y CNN). Recalcula las activaciones en el backward,
            reduciendo la VRAM del backbone en ~30-40% a cambio de ~+25-35% de
            tiempo. Por defecto False, sin efecto alguno. Se aplica ANTES de
            envolver en BackboneWrapper: opera dentro de forward_features, de
            forma transparente al wrapper y al output `[B, C, H, W]`.

    Returns:
        BackboneWrapper listo para consumir por `SkinCancerModel`.
    """
    if name not in _SUPPORTED_BACKBONES:
        raise ValueError(
            f"Backbone '{name}' no soportado. Opciones: {list(_SUPPORTED_BACKBONES.keys())}"
        )
    if drop_path_schedule not in ("uniform", "linear"):
        raise ValueError(
            f"drop_path_schedule debe ser 'uniform' o 'linear', "
            f"recibido {drop_path_schedule!r}"
        )
    cfg = _SUPPORTED_BACKBONES[name]
    kind = cfg["kind"]

    if kind == "cnn":
        backbone = timm.create_model(
            cfg["timm_name"],
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
        )
        if drop_path_schedule == "linear":
            _applied = _apply_linear_drop_path_schedule(backbone, drop_path_rate)
            # Se usa `warnings.warn` en lugar de `print` para que el evento sea
            # capturable por el handler de logging del servicio y por la
            # auditoría clínica; un `print` se perdería en la salida estándar.
            warnings.warn(
                f"Stochastic Depth con schedule lineal aplicado a {_applied} capas (CNN).",
                RuntimeWarning,
                stacklevel=2,
            )
        if use_grad_checkpointing:
            # Recalculo de activaciones en el backward. timm EffNet expone
            # set_grad_checkpointing; solo se invoca con el flag activo, por lo
            # que el comportamiento por defecto no cambia.
            backbone.set_grad_checkpointing(True)
        return BackboneWrapper(
            backbone=backbone,
            kind="cnn",
            num_features=backbone.num_features,
            expected_input_size=cfg["native_size"],
        )

    # ViT (DINOv2)
    if mode == "binary":
        expected_size = cfg["native_size_binary"]
    elif mode == "multiclass":
        expected_size = cfg["native_size_multiclass"]
    else:
        raise ValueError(f"mode debe ser 'binary' o 'multiclass', recibido: {mode!r}")

    patch_size = cfg["patch_size"]
    # timm usa `img_size` para interpolar embeddings de posición si difiere del
    # tamaño de pretrain (224 para DINOv2). Pasar el tamaño efectivo evita que
    # el forward haga checks de shape y que los pos-emb estén desalineados.
    backbone = timm.create_model(
        cfg["timm_name"],
        pretrained=pretrained,
        num_classes=0,
        global_pool="",
        img_size=expected_size,
        drop_path_rate=drop_path_rate,
    )
    if drop_path_schedule == "linear":
        _applied = _apply_linear_drop_path_schedule(backbone, drop_path_rate)
        # Mismo motivo que en la rama CNN: warning capturable, no print.
        warnings.warn(
            f"Stochastic Depth con schedule lineal aplicado a {_applied} capas (ViT).",
            RuntimeWarning,
            stacklevel=2,
        )
    # Sanity check: calcular tokens esperados
    expected_patches = (expected_size // patch_size) ** 2
    if expected_patches <= 0:
        raise RuntimeError(
            f"Configuración inválida: expected_patches={expected_patches} "
            f"para {name} con size={expected_size}, patch={patch_size}"
        )

    if use_grad_checkpointing:
        # Recalculo de activaciones en el backward para los bloques transformer
        # del ViT-L, que son el grueso del pico de VRAM. timm ViT
        # (vit_*_dinov2) expone set_grad_checkpointing; solo se invoca con el
        # flag activo. Se aplica sobre el modelo timm ANTES de envolver en
        # BackboneWrapper: opera dentro de forward_features, de forma
        # transparente al wrapper y al output [B, C, H, W].
        backbone.set_grad_checkpointing(True)

    return BackboneWrapper(
        backbone=backbone,
        kind="vit",
        num_features=backbone.num_features,
        expected_input_size=expected_size,
        patch_size=patch_size,
    )


def list_supported_backbones() -> list[str]:
    """Utilidad para CLI/tests: enumera los nombres válidos."""
    return list(_SUPPORTED_BACKBONES.keys())


def compute_compatible_num_heads(num_features: int, target: int = 8) -> int:
    """Calcula el mayor `num_heads ≤ target` que divide a `num_features` sin
    resto. Lo usa `MetaCrossAttentionFusion` y cualquier módulo que instancie
    `nn.MultiheadAttention` sobre la dimensión del backbone. Lanza error si no
    existe divisor válido (imposible para `num_features ≥ 1`, porque 1 siempre
    divide).

    Emite `RuntimeWarning` si el resultado es estrictamente menor que `target`.
    Eso señala que la dimensión del backbone es poco amigable para multi-head
    attention (p. ej. un número primo o divisible solo por números pequeños).
    Con `heads=1` la MHA degenera a un producto escalar simple — no rompe el
    forward, pero pierde la capacidad de atender a subespacios distintos. La
    decisión de adaptar o fallar se delega al caller; el warning expone esa
    degradación que de otro modo pasaría inadvertida.

    Ejemplos:
        - 1536 (EffNet-B3) → 8 (sin warning)
        - 1792 (EffNet-B4) → 8 (sin warning)
        - 1024 (ViT-L)     → 8 (sin warning)
        - 768  (ViT-B)     → 8 (sin warning)
        - 1280 (EffNet-B1) → 8 (1280/8=160, sin warning)
        - 1281 (3×7×61)    → 7 (warning heads<target, no degenera a 1)
        - 1279 (primo)     → 1 (warning heads<target, MHA degenerada)
    """
    if num_features < 1:
        raise ValueError(f"num_features debe ser >= 1, recibido: {num_features}")
    if target < 1:
        raise ValueError(f"target debe ser >= 1, recibido: {target}")
    heads = min(target, num_features)
    while heads > 1 and num_features % heads != 0:
        heads -= 1
    if heads < target:
        warnings.warn(
            f"compute_compatible_num_heads: num_features={num_features} no es "
            f"divisible por target={target}. Adaptado a heads={heads}. "
            f"{'La atención degenera a producto escalar simple (no multi-head).' if heads == 1 else 'La capacidad multi-head se reduce.'}",
            RuntimeWarning,
            stacklevel=2,
        )
    return heads


class AttentionPoolViT(nn.Module):
    """Attention pooling para ViT mediante un token de consulta aprendible.

    Arquitecturas tipo CLIP, DINOv2 y BiT usan una versión del CLS token como
    representación global. Es una alternativa al GeM pooling sobre los patch
    tokens: permite que el modelo aprenda un pooling específico por token en
    lugar de promediar uniformemente.

    **Contrato con el pipeline:**
    - Input: `[B, 1+N, D]` o `[B, N, D]` tokens (se asume que el primer
      token es CLS si el shape tiene N+1).
    - Output: `[B, D, 1, 1]` — compatible con la línea `GeM → BN → concat
      meta → classifier` del `SkinCancerModel` SIN modificar el classifier.

    **Mecánica:**
    1. Query = CLS token aprendible (1, 1, D).
    2. Key/Value = patch tokens del encoder ViT.
    3. `MultiheadAttention(Q, K, V)` produce `[B, 1, D]`.
    4. Reshape a `[B, D, 1, 1]` para compat downstream.

    **Por qué `[B, D, 1, 1]` y no `[B, D]`:** el resto del pipeline
    (`SkinCancerModel`) espera feature maps 4D antes del GeM. GeM sobre un
    tensor `[B, D, 1, 1]` es idempotente (pooling sobre 1 posición), lo
    que permite insertar AttentionPoolViT ANTES del GeM sin refactorizar el
    forward. En la práctica, cuando `use_attention_pool=True`, GeM se
    desactiva (el pooling ya se hizo via atención).

    **Init estable:** el query CLS se inicializa con `trunc_normal_(std=0.02)`
    (convención timm); los pesos de la MHA mantienen la inicialización
    default de PyTorch. En los primeros steps produce algo cercano al mean
    de los tokens — similar a avg pooling como punto de partida.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        safe_heads = compute_compatible_num_heads(dim, target=num_heads)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=safe_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: `[B, N, D]` o `[B, 1+N, D]`. Se usa el sequence completo
                como K/V (incluyendo CLS si está presente).
        Returns:
            pooled: `[B, D, 1, 1]` listo para consumir el classifier.
        """
        if tokens.ndim != 3:
            raise ValueError(
                f"AttentionPoolViT espera tokens con ndim=3, recibido "
                f"ndim={tokens.ndim} shape={tuple(tokens.shape)}"
            )
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)  # [B, 1, D]
        out, _ = self.attn(q, tokens, tokens, need_weights=False)
        out = self.norm(out)  # [B, 1, D]
        # Reshape a [B, D, 1, 1] para compatibilidad con GeM/classifier
        return out.transpose(1, 2).unsqueeze(-1)  # [B, D, 1, 1]


def compute_stoch_depth_schedule(num_layers: int, drop_path_rate: float) -> "list[float]":
    """Schedule lineal de Stochastic Depth por capa.

    Referencia: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
    EfficientNet paper default: schedule lineal desde 0 en capas tempranas
    hasta `drop_path_rate` en capas profundas (las capas más cercanas a la
    salida son las que más se benefician de Stochastic Depth).

    El `drop_path_rate` uniforme que usa timm por default es un simplificador;
    el schedule lineal tiende a mejorar AUC marginal sin coste extra.

    Args:
        num_layers: número de capas del backbone.
        drop_path_rate: rate máximo (capa final).

    Returns:
        Lista de `num_layers` floats crecientes linealmente desde 0 hasta
        `drop_path_rate`, ambos inclusivos.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers debe ser >= 1, recibido {num_layers}")
    if not (0.0 <= drop_path_rate <= 1.0):
        raise ValueError(
            f"drop_path_rate debe estar en [0, 1], recibido {drop_path_rate}"
        )
    if num_layers == 1:
        return [drop_path_rate]
    return [i * drop_path_rate / (num_layers - 1) for i in range(num_layers)]


def smoke_test_backbone(name: str, batch: int = 2, device: str = "cpu") -> dict:
    """Smoke test reutilizable para CI y diagnóstico.

    Construye un backbone, corre un forward sintético y valida que el output
    tenga shape `[B, C, H, W]` finito. Primer paso de diagnóstico cuando se
    añade un nuevo backbone al factory. Retorna dict con `shape`,
    `num_features`, `all_finite`.
    """
    wrapper = build_backbone(name=name, pretrained=False, mode="binary")
    wrapper = wrapper.to(device).eval()
    size = wrapper.expected_input_size
    x = torch.randn(batch, 3, size, size, device=device)
    with torch.no_grad():
        out = wrapper(x)
    if out.ndim != 4:
        raise RuntimeError(f"smoke_test {name}: output ndim={out.ndim}, esperado 4")
    if not torch.isfinite(out).all():
        raise RuntimeError(f"smoke_test {name}: output contiene NaN/Inf")
    return {
        "name": name,
        "shape": tuple(out.shape),
        "num_features": wrapper.num_features,
        "all_finite": True,
    }
