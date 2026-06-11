import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.data.constants import (
    AGE_NORM_UNKNOWN,
    NUM_SEX_CODES,
    NUM_SITE_CODES,
    SEX_UNKNOWN_CODE,
    SITE_UNKNOWN_CODE,
)
from src.models.backbones import build_backbone, BackboneWrapper, compute_compatible_num_heads

# Tokens que delatan un backbone non-CNN. El path legacy `model_name=` usa
# `timm.create_model(global_pool='')`, que devuelve tokens `[B, N, D]` para
# arquitecturas basadas en transformer/attention — incompatibles con
# GeM/CBAM/FiLM, que esperan `[B, C, H, W]`. El constructor de SkinCancerModel
# lanza un error si detecta uno de estos tokens en el `model_name` legacy y
# redirige al path nuevo `backbone_name=`.
_NON_CNN_MODEL_TOKENS = ("vit", "dinov2", "swin", "convnext", "beit", "deit", "mvit", "maxvit")


# Detector de si estamos bajo el tracing de torch.compile. Sirve para saltar
# bloques que rompen el grafo (`bool(.any())`, `.item()`, prints): son seguros
# en modo eager pero incompatibles con `torch.compile(mode="reduce-overhead")`
# y CUDA Graphs. La API pública `torch.compiler.is_compiling()` existe desde
# PyTorch 2.4; la privada `torch._dynamo.is_compiling()` desde 2.0. El getattr
# defensivo soporta el rango completo declarado en requirements.txt
# (`torch>=2.1.0,<3.0`); el fallback a `lambda: False` garantiza comportamiento
# eager en cualquier versión que no exponga la API.
_is_compiling = getattr(
    getattr(torch, "compiler", None),
    "is_compiling",
    getattr(getattr(torch, "_dynamo", None), "is_compiling", lambda: False),
)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) — Compatible con cualquier versión de timm.
    
    A diferencia del Average Pooling estándar (p=1), GeM usa un exponente aprendible
    que focaliza la atención en los píxeles con mayor activación (la lesión), 
    ignorando naturalmente el fondo de piel sana.
    
    - p=1 → Average Pooling (estándar).
    - p→∞ → Max Pooling (solo el píxel más intenso).
    - p=3 (inicial) → Balance óptimo para dermoscopia (Kaggle empirical).
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)  # Aprendible durante entrenamiento
        self.eps = eps
    
    def forward(self, x):
        # Se desactiva autocast EXPLÍCITAMENTE durante toda la operación GeM.
        # El máximo representable en float16 es 65504; con x=8 y p=5, 8^5=32768
        # roza ese límite y un solo NaN contaminaría la época completa. El
        # x.float() fuerza el tipo del tensor, pero autocast podría recastear
        # operaciones internas (avg_pool2d, pow) en versiones futuras de
        # PyTorch; con enabled=False se blinda la cadena completa.
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()  # Forzar float32
            p = self.p.float().clamp(min=1.0, max=6.0)
            # x: [B, C, H, W] → [B, C, 1, 1] → [B, C].
            # F.avg_pool2d con kernel_size=(x.size(-2), x.size(-1)) es
            # matemáticamente correcto pero NO es exportable a ONNX en PyTorch
            # ≥2.7 porque el kernel depende del shape del input (dinámico).
            # F.adaptive_avg_pool2d(output_size=1) es idéntico (promedia toda la
            # dimensión espacial) y sí exportable. GeM no tiene buffers internos
            # que dependan del kernel, solo el parámetro aprendible `p`, así que
            # los checkpoints existentes siguen cargando.
            return F.adaptive_avg_pool2d(
                x.clamp(min=self.eps).pow(p),
                output_size=1
            ).pow(1.0 / p).flatten(1)


class ChannelAttention(nn.Module):
    """
    Atención por Canal (parte de CBAM).

    Aprende QUÉ canales (features) son diagnósticamente relevantes.
    Usa avg_pool + max_pool como descriptores complementarios → MLP compartido → sigmoid.
    El squeeze ratio reduce la dimensión intermedia para eficiencia.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = self.mlp(x.mean(dim=[2, 3]))  # [B, C]
        max_out = self.mlp(x.amax(dim=[2, 3]))  # [B, C]
        attention = torch.sigmoid(avg_out + max_out)  # [B, C]
        return x * attention.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """
    Atención Espacial (parte de CBAM).

    Aprende DÓNDE mirar en la imagen (qué regiones son la lesión vs fondo).
    Condensa canales vía avg+max → conv k×k → sigmoid → máscara espacial.

    `kernel_size` acepta "adaptive": calcula un kernel proporcional al grid
    size, aproximadamente `grid_size // 4` redondeado al impar más cercano (al
    menos 3×3, siempre impar para padding simétrico). Por ejemplo: grid 28×28 →
    k=7; grid 14×14 → k=3; grid 7×7 → k=3. En grids grandes, un kernel 7×7 fijo
    tiene un campo receptivo demasiado pequeño para modelar dependencias
    largas; adaptarlo mejora marginalmente la precisión sin coste adicional.
    El valor por defecto `kernel_size=7` conserva el comportamiento original.
    """
    def __init__(self, kernel_size=7, grid_size=None):
        super().__init__()
        if isinstance(kernel_size, str):
            if kernel_size != "adaptive":
                raise ValueError(
                    f"SpatialAttention.kernel_size debe ser int o 'adaptive', "
                    f"recibido {kernel_size!r}"
                )
            if grid_size is None:
                raise ValueError(
                    "SpatialAttention(kernel_size='adaptive') requiere "
                    "`grid_size` explícito para calcular el kernel."
                )
            # Proporcional a grid / 4, redondeado al impar más cercano,
            # con mínimo 3 (kernel 1×1 no tiene sentido espacial).
            adaptive_k = max(3, grid_size // 4)
            if adaptive_k % 2 == 0:
                adaptive_k += 1
            kernel_size = adaptive_k
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        max_out = x.amax(dim=1, keepdim=True)  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = torch.sigmoid(self.conv(combined))  # [B, 1, H, W]
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).

    Secuencial: primero atención por canal (QUÉ features importan),
    luego atención espacial (DÓNDE mirar). Para dermoscopia, esto focaliza
    en los canales sensibles a la textura de la lesión y en la región espacial
    de la lesión, ignorando piel sana y artefactos.
    """
    def __init__(self, channels, reduction=16, kernel_size=7, grid_size=None):
        """Con `kernel_size="adaptive"` + `grid_size=N` se activa el cálculo
        proporcional del kernel de `SpatialAttention`. El valor por defecto
        `kernel_size=7` conserva el comportamiento original exactamente.
        """
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size, grid_size=grid_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class FiLMGenerator(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) — Perez et al., 2018.

    Genera parámetros gamma (escala) y beta (sesgo) a partir de los metadatos
    clínicos para modular los feature maps del backbone canal por canal.

    Esto permite que "paciente de 72 años, varón, cuero cabelludo" AMPLIFIQUE
    canales sensibles a la queratosis actínica y SUPRIMA canales irrelevantes
    ANTES de que el clasificador tome la decisión.

    Sin FiLM, los metadatos solo participan tras el pooling (fusión tardía).
    Con FiLM, los metadatos influyen en la representación visual completa.
    """
    def __init__(self, meta_dim, num_channels):
        super().__init__()
        self.gamma_gen = nn.Sequential(
            nn.Linear(meta_dim, num_channels),
            nn.Tanh(),  # gamma en [-1, 1], centrado para modulación suave
        )
        self.beta_gen = nn.Linear(meta_dim, num_channels)
        # Inicializar gamma cercano a 0 y beta cercano a 0
        # para que al inicio FiLM sea ~identidad (no destruye features pretrained)
        nn.init.zeros_(self.gamma_gen[0].weight)
        nn.init.zeros_(self.gamma_gen[0].bias)
        nn.init.zeros_(self.beta_gen.weight)
        nn.init.zeros_(self.beta_gen.bias)

    def forward(self, features, meta_features):
        """
        Args:
            features: [B, C, H, W] — feature maps del backbone
            meta_features: [B, meta_dim] — output del metadata MLP
        Returns:
            Modulated features: [B, C, H, W]
        """
        gamma = self.gamma_gen(meta_features)  # [B, C]
        beta = self.beta_gen(meta_features)    # [B, C]
        # Modular: (1 + gamma) * features + beta
        # El (1 + gamma) asegura que gamma=0 → identidad
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return (1 + gamma) * features + beta


class MetaCrossAttentionFusion(nn.Module):
    """
    Fusión metadata↔imagen mediante cross-attention.

    Alternativa a FiLM cuando la capacidad de modulación canal-a-canal resulta
    insuficiente. Cross-attention permite que los metadatos atiendan
    explícitamente a regiones espaciales relevantes del feature map.

    Query: proyección de metadatos a dim D (num_channels del backbone).
    Key/Value: feature maps aplanados a [B, N, D] donde N = H*W.

    Output residual con factor pequeño (0.1) para no destruir features
    preentrenadas al inicio (análogo a la init-identidad de FiLM).

    Inicialización:
    - `meta_proj`: pesos estándar, bias=0.
    - `attention.out_proj`: pesos y bias en cero → output inicial = 0 →
      con residual `features + 0.1 * 0 = features` (identidad al inicio).

    Nota sobre dtype: MultiheadAttention interno usa el dtype del input. Bajo
    autocast puede correr en fp16; no opera sobre la rama crítica de GeM, así
    que no requiere el guard float32 de GeM.
    """

    def __init__(self, meta_dim: int, num_channels: int, num_heads: int = 8, residual_scale: float = 0.1):
        super().__init__()
        # Validar la divisibilidad antes de instanciar la MHA. Sin esta guarda,
        # `nn.MultiheadAttention` lanza un AssertionError críptico en runtime.
        # Adaptamos num_heads al mayor divisor ≤ solicitado, para cualquier
        # backbone cuya dimensión no sea múltiplo del target.
        if num_channels % num_heads != 0:
            adapted = num_heads
            while adapted > 1 and num_channels % adapted != 0:
                adapted -= 1
            if num_channels % adapted != 0:
                raise ValueError(
                    f"num_channels={num_channels} no es divisible por ningún "
                    f"num_heads∈[1, {num_heads}]. Revisar backbone."
                )
            warnings.warn(
                f"MetaCrossAttentionFusion: num_heads={num_heads} no divide a "
                f"num_channels={num_channels}. Adaptado a num_heads={adapted}.",
                RuntimeWarning,
                stacklevel=2,
            )
            num_heads = adapted
        # Proyección metadata → dim D. Un solo token de query por imagen.
        self.meta_proj = nn.Linear(meta_dim, num_channels)
        self.norm_q = nn.LayerNorm(num_channels)
        self.norm_kv = nn.LayerNorm(num_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=num_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.residual_scale = residual_scale

        # Init-identidad: out_proj en cero → salida inicial es el feature map
        # original sin modificación (análogo a FiLM gamma=beta=0).
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, features: torch.Tensor, meta_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] — feature maps del backbone.
            meta_features: [B, meta_dim] — output del metadata MLP.
        Returns:
            Modulated features: [B, C, H, W].
        """
        b, c, h, w = features.shape
        # Query: 1 token por imagen desde metadatos.
        q = self.meta_proj(meta_features).unsqueeze(1)  # [B, 1, C]
        q = self.norm_q(q)

        # Key/Value: aplanar feature maps espaciales a secuencia de tokens.
        kv = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        kv = self.norm_kv(kv)

        # Atender desde metadata-query hacia feature tokens.
        attended, _ = self.attention(query=q, key=kv, value=kv, need_weights=False)
        # attended: [B, 1, C]. Al inicio es 0 por init-identidad.

        # Broadcast al grid espacial y residual con escala pequeña.
        # El broadcasting preserva la estructura espacial sin destruirla.
        attended_map = attended.transpose(1, 2).unsqueeze(-1)  # [B, C, 1, 1]
        return features + self.residual_scale * attended_map


class HybridPool(nn.Module):
    """Hybrid Pooling: concatena GeM y AttentionPool.

    Combina dos estrategias de pooling COMPLEMENTARIAS sobre el MISMO
    `feature_maps` post-FiLM/CBAM, concatenándolas en un único vector
    de salida `[B, 2*C]`. A diferencia del path puro AttentionPool
    (`use_attention_pool=True`), preserva FiLM/CBAM intactos porque el
    pooling se aplica al final del flujo `Backbone → FiLM → CBAM →
    HybridPool → BN → Classifier`.

    Las dos vistas:

    - **GeM (`p` aprendible)**: pooling generalizado en `[B, C, H, W]
      → [B, C]`. `p=3` favorece píxeles de la lesión. Aprende "qué
      magnitud de respuesta importa por canal".
    - **AttentionPoolViT**: query learnable que atiende al grid
      `[B, H*W, C]` con multi-head attention y retorna un vector
      ponderado `[B, C]`. Aprende "dónde mirar espacialmente, por
      canal y por token".

    El concat captura ambas señales sin que una domine a la otra (cada
    una se entrena independientemente). Conceptualmente análogo a
    "max + avg" en CBAM ChannelAttention, pero a nivel de pooling
    final del backbone.

    **Beneficios cuantitativos** (estimados sobre dataset dermatoscópico):

    - +0.5-1.0% AUC binario y multiclase vs GeM puro o AttentionPool
      puro. La complementariedad reduce el sesgo individual de cada
      pool.
    - Latencia inferencia: +5-10ms por imagen (1 forward extra de
      MHA sobre H*W=441 tokens en ViT-L/B@294, despreciable comparado
      con el coste del backbone).
    - VRAM training: +~30-50MB en activations (un MHA extra por step
      sobre tokens [B, H*W, C]). Cabe holgado en A100 40GB.

    **Compatibilidad**:

    - **Compatible** con FitNets (`use_fitnet=True`): la projection MSE
      del student opera sobre `feature_maps` pre-pool (`[B, C, H, W]`),
      no sobre el output del HybridPool. La incompatibilidad FitNets ↔
      pooling solo afecta a `use_attention_pool` puro, no a hybrid_pool.
    - **Compatible** con SupCon (`use_projection_head=True`): la
      projection head opera sobre `img_features_raw = [B, 2*C]`. El
      input dim del primer Linear se ajusta automáticamente.
    - **Incompatible** (mutuamente exclusivo) con `use_attention_pool=True`
      — uno reemplaza el pooling y el otro lo combina. `__init__` lanza
      `ValueError` si ambos están activos.
    - **Compatible** con CNN (EffNet B3/B4) y ViT (DINOv2 ViT-L/B).
      Para CNN, el `feature_maps` post-CBAM tiene shape `[B, C, H, W]`
      con H,W ~10. Para ViT-L@294, H=W=21 (post-reshape del wrapper).

    Args:
        dim: número de canales del feature map (= `in_features` del
            backbone). Para EffNet-B3 1536, EffNet-B4 1792, ViT-L 1024,
            ViT-B 768.
        num_heads: cabezas de la MHA del AttentionPool. Adaptado a
            divisores de `dim` por `compute_compatible_num_heads`.

    Forward:
        Args: feature_maps `[B, C, H, W]` post-FiLM/CBAM.
        Returns: `[B, 2*C]` concatenación de GeM(C) + AttnPool(C).
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = int(dim)
        # GeM clásico — mismo p=3 inicial que el path puro GeM.
        self.gem = GeM(p=3.0)
        # AttentionPool reutiliza la clase ya validada de backbones.py.
        # Lazy import para evitar ciclo backbones.py↔architectures.py.
        from src.models.backbones import AttentionPoolViT, compute_compatible_num_heads
        safe_heads = compute_compatible_num_heads(self.dim, target=num_heads)
        self.attn_pool = AttentionPoolViT(dim=self.dim, num_heads=safe_heads)

    def forward(self, feature_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_maps: `[B, C, H, W]` post-FiLM/CBAM. **NO** debe ser
                pre-pool — el contrato es "tras todas las modulaciones".
        Returns:
            `[B, 2*C]` concat de GeM y AttentionPool.
        """
        if feature_maps.ndim != 4:
            raise ValueError(
                f"HybridPool espera feature_maps con ndim=4 [B, C, H, W], "
                f"recibido ndim={feature_maps.ndim} shape={tuple(feature_maps.shape)}."
            )

        # GeM: [B, C, H, W] → [B, C, 1, 1] → flatten → [B, C]
        gem_out = self.gem(feature_maps)
        if gem_out.ndim > 2:
            gem_out = gem_out.flatten(1)

        # AttentionPool: convertir feature_maps a tokens [B, H*W, C].
        # AttentionPoolViT espera [B, N, D] y retorna [B, D, 1, 1].
        B, C, H, W = feature_maps.shape
        tokens = feature_maps.flatten(2).transpose(1, 2).contiguous()  # [B, H*W, C]
        attn_out = self.attn_pool(tokens)  # [B, C, 1, 1]
        attn_out = attn_out.flatten(1)  # [B, C]

        # Concat → [B, 2*C]
        return torch.cat([gem_out, attn_out], dim=1)


class MultiScaleAttentionPool(nn.Module):
    """Pooling multi-escala para ViT.

    Captura información de MÚLTIPLES escalas del encoder ViT (típicamente
    3 capas: tercio inferior, medio, último) y aplica `AttentionPoolViT`
    independiente a cada escala. Concat → `[B, num_scales*C]`.

    **Por qué multi-escala**: ViT-L tiene 24 capas. Las primeras capturan
    texturas locales (red reticular dermatoscópica, glóbulos), las
    intermedias features de medio nivel (patrones globulares, estructuras
    arborescentes), las finales semántica de alto nivel (forma global de
    la lesión, asimetría). Pooling solo sobre la última capa pierde
    información de texturas finas críticas para diagnóstico de melanoma.

    **Diseño**:

    - Reemplaza completamente el pooling clásico (NO se combina con GeM
      directamente — usa `MultiScaleAttentionPool` instead of `GeM` o
      `AttentionPoolViT`).
    - Cada escala tiene su propia query learnable + MHA (no comparten
      pesos — cada escala aprende su propia "dónde mirar").
    - El concat preserva la independencia de cada escala.

    **Restricciones**:

    - Solo válido para `kind="vit"` (las CNN no exponen tokens intermedios
      de forma natural; el enfoque con hooks sería distinto).
    - Mutuamente exclusivo con `AttentionPool` puro y con `HybridPool`.
      Una extensión futura podría combinar `MultiScaleAttentionPool` + GeM
      para obtener `[B, (num_scales+1)*C]`.
    - FiLM/CBAM se desactivan automáticamente (igual que con `AttentionPool`),
      porque los `feature_maps` ya no entran al pool — los tokens
      intermedios sí. Trade-off: ganamos información multi-escala pero
      perdemos modulación de metadata sobre features.

    **Beneficios cuantitativos** (estimados sobre dataset dermatoscópico):

    - +0.5-1.0% AUC binario y multiclase. Más impactante en lesiones con
      texturas multi-escala (asimetría + red reticular vs glóbulos).
    - VRAM training: +20% en activations (3 MHA paralelos sobre tokens
      [B, N, D]). Cabe en A100 40GB con `batch_size=24` para ViT-L@294,
      `batch_size=16` para ViT-L@378 (multiclass).
    - Latencia inferencia: +15-25ms por imagen (3 forwards de MHA).

    Args:
        dim: dimensión de canal del backbone (1024 para ViT-L, 768 para ViT-B).
        num_scales: número de capas a fusionar. Default 3.
        num_heads: cabezas de MHA por pool. Adaptado por
            `compute_compatible_num_heads`.

    Forward:
        Args: tokens_per_scale: lista de N tensores `[B, N_total, D]`,
            uno por escala (orden = orden de hooks registrados en backbone).
        Returns: `[B, num_scales*D]` concat.
    """

    def __init__(self, dim: int, num_scales: int = 3, num_heads: int = 8):
        super().__init__()
        if num_scales <= 0:
            raise ValueError(
                f"MultiScaleAttentionPool: num_scales debe ser >= 1, "
                f"recibido {num_scales}."
            )
        self.dim = int(dim)
        self.num_scales = int(num_scales)
        from src.models.backbones import AttentionPoolViT, compute_compatible_num_heads
        safe_heads = compute_compatible_num_heads(self.dim, target=num_heads)
        # ModuleList para que cada escala tenga su propia query/MHA.
        self.attn_pools = nn.ModuleList(
            [AttentionPoolViT(dim=self.dim, num_heads=safe_heads) for _ in range(self.num_scales)]
        )

    def forward(self, tokens_per_scale: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            tokens_per_scale: lista de `num_scales` tensores `[B, N_total, D]`,
                obtenidos de `BackboneWrapper.forward(..., return_intermediate_layers=True)`.
        Returns:
            `[B, num_scales * D]`.
        """
        if len(tokens_per_scale) != self.num_scales:
            raise ValueError(
                f"MultiScaleAttentionPool: esperado {self.num_scales} escalas, "
                f"recibido {len(tokens_per_scale)}."
            )
        outs = []
        for tokens, attn in zip(tokens_per_scale, self.attn_pools):
            o = attn(tokens)  # [B, D, 1, 1]
            outs.append(o.flatten(1))  # [B, D]
        return torch.cat(outs, dim=1)  # [B, num_scales * D]


class SkinCancerModel(nn.Module):
    """
    Modelo base para el Sistema de Diagnóstico (Triaje y Especialista).

    Arquitectura Híbrida (Multi-Modal):
    1. CNN Backbone (timm): Extrae características de la imagen (Ej. EfficientNet, ResNet).
    2. MLP Head (Metadata): Extrae características de los datos clínicos (Categorizados).
    3. Fusión: Combina imagen + metadatos para la predicción final.
    """

    # Flag one-shot del aviso de edad fuera de contrato, a nivel de CLASE (no de
    # instancia) para sobrevivir a DataParallel. Cada réplica DP es una
    # instancia con sus propios atributos Python, pero las variables de clase se
    # comparten a nivel de proceso, así que el aviso se emite una sola vez por
    # proceso en lugar de una por réplica-GPU.
    _age_warned_global: bool = False

    def __init__(
        self,
        model_name="efficientnet_b3",
        num_classes=1,
        pretrained=True,
        use_metadata=True,
        meta_features_out=16,
        use_attention=True,
        use_projection_head=False,
        backbone_name=None,
        fusion_mode="film",
        mode="binary",
        drop_path_rate=0.2,
        meta_dropout_prob=0.1,
        meta_dropout_per_feature: bool = True,
        cbam_adaptive_kernel: bool = False,
        use_attention_pool: bool = False,
        use_hybrid_pool: bool = False,
        use_multiscale_vit: bool = False,
        multiscale_layer_indices: Optional[list] = None,
        use_fitnet: bool = False,
        fitnet_teacher_dim: int = 1536,
        use_grad_checkpointing: bool = False,
    ):
        """
        Args:
            model_name: ruta legacy — nombre de timm pasado directo a
                `timm.create_model`. Mantiene compatibilidad con los checkpoints
                existentes (claves del state_dict `backbone.<layer>`). Usar SOLO
                con backbones CNN (EffNet); lanza un error si el nombre parece un
                backbone non-CNN (vit/dinov2/swin/convnext/...), en cuyo caso
                debe usarse `backbone_name=`.
            backbone_name: ruta nueva — clave del factory `src.models.backbones`
                (ej. "effnet_b3", "effnet_b4", "dinov2_vitl14"). Cuando se
                provee, prevalece sobre `model_name` y envuelve el backbone en
                `BackboneWrapper`. No es compatible con los checkpoints de la
                ruta legacy (el state_dict queda como `backbone.backbone.<layer>`).
            fusion_mode: "film" (por defecto) o "cross_attention"
                (MetaCrossAttentionFusion). Requiere `use_metadata=True` y
                `use_attention=True`.
            mode: "binary"/"multiclass" — solo afecta a ViT (selecciona img_size
                múltiplo de patch_size cercano a 300/380). Ignorado por CNN.
            drop_path_rate: Stochastic Depth.
            meta_dropout_prob: probabilidad de Meta-Dropout. En entrenamiento,
                con probabilidad `p`, reemplaza los metadatos por sus valores
                "unknown" (age=AGE_NORM_UNKNOWN, sex=SEX_UNKNOWN_CODE,
                site=SITE_UNKNOWN_CODE). Enseña al modelo a funcionar cuando el
                clínico no proporciona metadatos en inferencia. 0.0 lo desactiva;
                por defecto 0.1.
            meta_dropout_per_feature: modo de Meta-Dropout.
                - True (por defecto): cada feature {age, sex, site} se dropea
                  INDEPENDIENTEMENTE con probabilidad `p`. Refleja la práctica
                  clínica real, donde el profesional reporta los metadatos de
                  forma parcial (p. ej. edad sí, sexo no, sitio sí). Más robusto
                  en inferencia con metadatos incompletos.
                - False: los 3 features se dropean juntos (per-sample). Se
                  conserva por reproducibilidad estricta del comportamiento
                  original.
                Ambos modos son no-ops en inferencia (`self.eval()`).
            use_grad_checkpointing: activa gradient checkpointing (recálculo de
                activaciones en el backward) en el backbone timm vía
                `set_grad_checkpointing(True)`. Reduce la VRAM del backbone en
                ~30-40% a cambio de ~+25-35% de tiempo/época. Es un parámetro de
                CONSTRUCCIÓN: no crea params/buffers ni aparece en el state_dict
                (igual que `fitnet_teacher_dim`/`feature_dim`), por lo que los
                checkpoints existentes cargan con strict=True sin problema. Por
                defecto False, sin efecto sobre los modelos existentes.
        """
        super(SkinCancerModel, self).__init__()
        self.use_metadata = use_metadata
        self.use_attention = use_attention
        self.fusion_mode = fusion_mode
        self.meta_dropout_prob = float(meta_dropout_prob)
        self.meta_dropout_per_feature = bool(meta_dropout_per_feature)

        if fusion_mode not in {"film", "cross_attention"}:
            raise ValueError(
                f"fusion_mode debe ser 'film' o 'cross_attention', recibido: {fusion_mode!r}"
            )

        # 1. Extractor de imágenes (backbone). Dos rutas:
        # - Legacy (`model_name`): timm directo, state_dict = `backbone.<layer>`.
        # - Nueva (`backbone_name`): factory con wrapper que normaliza la salida
        #   a [B, C, H, W] incluso para ViT. State_dict = `backbone.backbone.<layer>`.
        # global_pool='' hace que el backbone devuelva mapas espaciales
        # [B, C, H, W] SIN pooling; el pooling GeM se aplica después de forma
        # manual (compatible con cualquier versión de timm).
        if backbone_name is not None:
            self.backbone = build_backbone(
                name=backbone_name,
                pretrained=pretrained,
                mode=mode,
                drop_path_rate=drop_path_rate,
                use_grad_checkpointing=use_grad_checkpointing,
            )
            in_features = self.backbone.num_features
        else:
            # Error explícito si se pasa un nombre non-CNN por la ruta legacy.
            # Esta ruta usa `timm.create_model(global_pool='')`, que devuelve
            # tokens [B, N, D] para arquitecturas transformer/attention (ViT,
            # DINOv2, Swin, ConvNeXt), incompatibles con GeM/CBAM/FiLM, que
            # esperan [B, C, H, W]. Sin esta guarda, el fallo se manifestaría
            # como un RuntimeError críptico en el primer forward ("expected 4D
            # input").
            _model_name_lower = str(model_name).lower()
            if any(tok in _model_name_lower for tok in _NON_CNN_MODEL_TOKENS):
                raise ValueError(
                    f"model_name={model_name!r} parece un backbone non-CNN "
                    f"(tokens detectados: {_NON_CNN_MODEL_TOKENS}). El path legacy "
                    f"`model_name=` sólo soporta CNNs con output espacial "
                    f"`[B, C, H, W]`. Para backbones tipo ViT/DINOv2/Swin/ConvNeXt "
                    f"usar el path nuevo `backbone_name=` con una clave del factory "
                    f"`src.models.backbones._SUPPORTED_BACKBONES` (ej. 'dinov2_vitl14')."
                )
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='',  # Sin pooling → devuelve feature maps crudos
                drop_path_rate=drop_path_rate,
            )
            in_features = self.backbone.num_features
            if use_grad_checkpointing:
                # Recálculo de activaciones en el backward para el modelo timm
                # legacy (CNN). timm expone set_grad_checkpointing en EffNet y
                # ViT; solo se invoca con el flag activo, por lo que el
                # comportamiento por defecto no cambia. El pipeline de
                # entrenamiento usa la ruta del factory (backbone_name=), pero
                # los tests CNN ejercitan esta ruta para cubrir el contrato del
                # flag de forma uniforme.
                self.backbone.set_grad_checkpointing(True)

        # 2. Extractor de Metadatos (Si está activado)
        if self.use_metadata:
            # Embeddings semánticos para las variables categóricas. Pasar el
            # sitio anatómico como float (sitio=4, sitio=2) haría que la red
            # interpretara que "4 es el doble que 2", una relación numérica
            # espuria. En su lugar, cada código se proyecta a un vector
            # aprendible independiente. num_embeddings deriva de constants.py
            # (fuente de verdad única) para sincronizarse automáticamente si se
            # añaden categorías a SEX_MAP/SITE_MAP.
            sex_emb_dim = 4
            site_emb_dim = 8
            self.sex_emb = nn.Embedding(num_embeddings=NUM_SEX_CODES, embedding_dim=sex_emb_dim)
            self.site_emb = nn.Embedding(num_embeddings=NUM_SITE_CODES, embedding_dim=site_emb_dim)

            # Derivar meta_input_dim en vez de fijar el literal `1 + 4 + 8 = 13`.
            # Si se añade una categoría o cambia una dimensión de embedding, este
            # Linear se ajusta solo; con el literal, el mismatch solo se detectaba
            # en runtime durante el primer forward.
            age_dim = 1
            meta_input_dim = age_dim + sex_emb_dim + site_emb_dim
            # Entrada: edad(1) + vector de sexo(sex_emb_dim) + vector de sitio(site_emb_dim).
            self.meta_net = nn.Sequential(
                nn.Linear(meta_input_dim, 32),
                nn.BatchNorm1d(32),
                nn.SiLU(),  # SiLU (Swish) en lugar de ReLU: evita neuronas muertas
                nn.Dropout(0.2),
                nn.Linear(32, meta_features_out),
                nn.SiLU(),
            )

            # Fusión temprana metadata↔features. Dos modos:
            # - "film" (por defecto): FiLM canal-a-canal, init en identidad.
            # - "cross_attention": cross-attention residual con escala 0.1, init
            #   en identidad.
            # Ambos inicializan en identidad para preservar las features preentrenadas.
            if self.use_attention:
                if fusion_mode == "film":
                    self.film = FiLMGenerator(
                        meta_dim=meta_features_out, num_channels=in_features
                    )
                else:  # cross_attention
                    # num_heads adaptado a divisores de in_features. Para EffNet
                    # B3/B4 y DINOv2 ViT-L/B el resultado es 8, pero la
                    # adaptación protege ante backbones cuya dimensión no sea
                    # múltiplo de 8.
                    ca_heads = compute_compatible_num_heads(in_features, target=8)
                    self.meta_cross_attn = MetaCrossAttentionFusion(
                        meta_dim=meta_features_out,
                        num_channels=in_features,
                        num_heads=ca_heads,
                        residual_scale=0.1,
                    )

            # `classifier_in` se calcula más abajo, tras configurar el pool, en
            # función de `_pool_out_dim` (que depende de si hybrid_pool está
            # activo). Así hay una única fuente de verdad para esa dimensión.
        # En la rama use_metadata=False, classifier_in también se fija tras el
        # bloque de pool.

        # CBAM: atención por canal (QUÉ features importan) + espacial (DÓNDE
        # mirar). Focaliza en la lesión e ignora piel sana y artefactos.
        #
        # Con `cbam_adaptive_kernel=True` se activa el kernel adaptativo de
        # SpatialAttention: se calcula el `grid_size` del backbone y se pasa
        # `kernel_size="adaptive"` + `grid_size`. Por defecto False, conserva el
        # `kernel_size=7` original.
        if self.use_attention:
            if cbam_adaptive_kernel:
                # Derivar grid_size del backbone:
                # - ViT (`BackboneWrapper`): `._grid_side` (21 para B@294, 27 para M@378)
                # - CNN (EffNet): `expected_input_size / stride` (~32 para EffNet)
                backbone_kind = getattr(self.backbone, "kind", None)
                if backbone_kind == "vit":
                    _grid_size = int(self.backbone._grid_side)
                elif backbone_kind == "cnn":
                    _grid_size = max(3, int(self.backbone.expected_input_size / 32))
                else:
                    # Path legacy sin BackboneWrapper — EffNet timm directo.
                    # Assume stride final 32.
                    _grid_size = 10  # aproximación razonable para EffNet B3/B4 @300-380px
                self.cbam = CBAM(
                    channels=in_features,
                    reduction=16,
                    kernel_size="adaptive",
                    grid_size=_grid_size,
                )
            else:
                self.cbam = CBAM(channels=in_features, reduction=16, kernel_size=7)

        # Attention pooling ViT como alternativa a GeM. Cuando
        # `use_attention_pool=True` y el backbone es ViT, reemplaza GeM por
        # `AttentionPoolViT`; GeM queda como Identity (no-op) porque el pooling
        # ya lo hizo la atención. Con CNN se lanza un error.
        # FitNets projection head: si `use_fitnet=True`, crea un
        # Conv2d(student_channels → teacher_channels, 1×1) para proyectar los
        # feature maps del student al espacio dimensional del teacher (p. ej.
        # EffNet-B3 = 1536 canales). Se entrena como parámetro normal del
        # modelo, no como buffer. La proyección solo se usa en entrenamiento
        # (cuando `distillation.fitnet_feature_loss` la consume); en inferencia
        # es peso muerto.
        self.use_fitnet = bool(use_fitnet)
        # Exponer `self.feature_dim` (= num_features del backbone, tras
        # FiLM/CrossAttention y antes de GeM/AttentionPool). Los callers externos
        # (p. ej. train.py al cablear FitNets) necesitan la dimensión del
        # student/teacher sin depender de los internals del wrapper del backbone,
        # que pueden cambiar (manejo de dict, redimensionado defensivo, etc.).
        self.feature_dim = int(in_features)
        if self.use_fitnet:
            # Validar que `fitnet_teacher_dim` es un int positivo. Sin esta
            # comprobación, pasar un float por error (p. ej. 1536.0) se truncaría
            # en silencio vía `int(...)`; y pasar un valor que no coincide con la
            # dimensión real del teacher (p. ej. 1536 cuando el teacher es un ViT
            # de 1024) produciría un MSE entre proyecciones de shapes distintas,
            # con broadcast silencioso o RuntimeError críptico según el path. La
            # validación estructural de tipo y rango cierra ese riesgo antes del
            # primer forward. La validación CRUZADA (dim del teacher ==
            # fitnet_teacher_dim) vive en train.py, donde el teacher ya está cargado.
            if not isinstance(fitnet_teacher_dim, int):
                raise TypeError(
                    f"fitnet_teacher_dim debe ser int, recibido "
                    f"{type(fitnet_teacher_dim).__name__}. "
                    f"Valor: {fitnet_teacher_dim!r}. Derivar de "
                    f"`teacher.feature_dim` tras `load_teacher(...)`."
                )
            if fitnet_teacher_dim <= 0:
                raise ValueError(
                    f"fitnet_teacher_dim debe ser > 0, recibido "
                    f"{fitnet_teacher_dim}."
                )
            # Conv2d 1×1 — alineación de canales sin interferencia espacial.
            self.fitnet_projection = nn.Conv2d(
                in_features, fitnet_teacher_dim, kernel_size=1, bias=True
            )

        self.use_attention_pool = bool(use_attention_pool)
        self.use_hybrid_pool = bool(use_hybrid_pool)
        self.use_multiscale_vit = bool(use_multiscale_vit)

        # Exclusión mutua entre las 3 estrategias de pooling alternativas
        # (attention_pool, hybrid_pool, multiscale_vit). Cada una asume una
        # forma distinta de los feature_maps post-backbone:
        # - baseline (por defecto): GeM sobre [B, C, H, W] post-FiLM/CBAM.
        # - attention_pool: [B, D, 1, 1] tras MHA sobre tokens (FiLM/CBAM OFF).
        # - hybrid_pool: GeM + AttentionPool sobre [B, C, H, W] (FiLM/CBAM ON).
        # - multiscale_vit: 3 AttentionPool sobre tokens de 3 capas (FiLM/CBAM OFF).
        # Activar más de una crea conflicto sobre QUIÉN consume feature_maps, así
        # que solo se permite UNA activa a la vez.
        _pool_flags = sum([self.use_attention_pool, self.use_hybrid_pool, self.use_multiscale_vit])
        if _pool_flags > 1:
            raise ValueError(
                f"Solo UN modo de pooling avanzado puede estar activo a la vez. "
                f"Recibido: use_attention_pool={self.use_attention_pool}, "
                f"use_hybrid_pool={self.use_hybrid_pool}, "
                f"use_multiscale_vit={self.use_multiscale_vit}. Elegir uno."
            )

        # `use_attention_pool=True` y `use_fitnet=True` son incompatibles. Con
        # attention pool, los `feature_maps` post-pool son `[B, D, 1, 1]`,
        # mientras que FitNets aplica el MSE entre `proj(student_feats
        # [B, D, 1, 1])` y `teacher_feats_cache [B, C, H_t, W_t]` (que provienen
        # de la ruta sin attention pool): dominios estructuralmente distintos
        # (1×1 espacial vs. grid completo). El `F.interpolate` defensivo de
        # `fitnet_feature_loss` lo resuelve por broadcast, pero el MSE resultante
        # carece de sentido (casi todo el gradiente fluye por la interpolación de
        # upsample, no por las representaciones del student). Por eso se rechaza
        # explícitamente la combinación.
        if self.use_attention_pool and self.use_fitnet:
            raise ValueError(
                "use_attention_pool=True + use_fitnet=True son incompatibles. "
                "AttentionPool produce feature_maps [B, D, 1, 1] (post-pooling) "
                "mientras FitNets espera el grid espacial [B, C, H, W] del "
                "teacher para destilar representaciones intermedias del student. "
                "La combinación produce un MSE sin sentido clínico (el broadcast "
                "vía interpolate desbalancea los gradientes). Elegir uno: "
                "(a) AttentionPool con destilación solo de logits "
                "(--use_attention_pool sin --use_fitnet), o (b) FitNets con GeM "
                "(--use_fitnet sin --use_attention_pool, por defecto)."
            )
        # Validación análoga a la de attention_pool: multiscale_vit produce el
        # output del pool a partir de tokens intermedios, no de los
        # feature_maps. FitNets espera el grid espacial [B, C, H, W] del student
        # para destilar, así que multiscale tendría el mismo problema que
        # attention_pool (broadcast sin sentido).
        if self.use_multiscale_vit and self.use_fitnet:
            raise ValueError(
                "use_multiscale_vit=True + use_fitnet=True son incompatibles. "
                "MultiScaleAttentionPool consume tokens de capas intermedias del "
                "ViT, no el grid espacial post-backbone. FitNets espera el grid "
                "[B, C, H, W] del student para destilar del teacher. Elegir: "
                "(a) Multi-scale con destilación solo de logits "
                "(--use_multiscale_vit sin --use_fitnet), o (b) Hybrid pool con "
                "FitNets (--use_hybrid_pool --use_fitnet, FiLM/CBAM activos)."
            )
        if self.use_attention_pool:
            backbone_kind = getattr(self.backbone, "kind", None)
            if backbone_kind != "vit":
                raise ValueError(
                    f"use_attention_pool=True requiere un backbone ViT, "
                    f"recibido kind={backbone_kind!r}. Con CNN, usar GeM (por defecto)."
                )
            from src.models.backbones import AttentionPoolViT
            ca_heads = compute_compatible_num_heads(in_features, target=8)
            self.attention_pool = AttentionPoolViT(dim=in_features, num_heads=ca_heads)
            self.gem_pool = nn.Identity()  # no-op; pool ya aplicado por la atención
        elif self.use_hybrid_pool:
            # HybridPool combina GeM + AttentionPool sobre el MISMO feature_maps
            # post-FiLM/CBAM, con salida `[B, 2*in_features]`. `gem_pool` queda
            # como Identity (HybridPool ya tiene su propio GeM). Compatible con
            # CNN y ViT (a diferencia de attention_pool puro, que requiere
            # kind=="vit") y con FitNets (la projection MSE opera sobre
            # feature_maps pre-pool, no sobre la salida del HybridPool).
            ca_heads_hp = compute_compatible_num_heads(in_features, target=8)
            self.hybrid_pool = HybridPool(dim=in_features, num_heads=ca_heads_hp)
            self.gem_pool = nn.Identity()  # no-op; pool aplicado por HybridPool
        elif self.use_multiscale_vit:
            # MultiScaleAttentionPool con 3 escalas del encoder ViT (hooks en
            # capas intermedias del backbone). Solo válido para kind="vit".
            # Salida: [B, num_scales*in_features] (3*1024 = 3072 para ViT-L).
            backbone_kind = getattr(self.backbone, "kind", None)
            if backbone_kind != "vit":
                raise ValueError(
                    f"use_multiscale_vit=True requiere un backbone ViT, "
                    f"recibido kind={backbone_kind!r}. La fusión multi-escala "
                    f"con CNN no está soportada."
                )
            # Default indices: si el caller no provee, derivar de #blocks del
            # backbone tomando capas a 1/3, 2/3 y última. Para ViT-L (24
            # blocks): [7, 15, 23]. Para ViT-B (12 blocks): [3, 7, 11].
            if multiscale_layer_indices is None:
                if not hasattr(self.backbone, "backbone") or not hasattr(self.backbone.backbone, "blocks"):
                    raise RuntimeError(
                        "No se puede derivar multiscale_layer_indices "
                        "automáticamente: el backbone no expone .blocks. Pasar "
                        "multiscale_layer_indices explícito al constructor."
                    )
                _n_blocks = len(self.backbone.backbone.blocks)
                multiscale_layer_indices = [
                    max(0, _n_blocks // 3 - 1),
                    max(0, 2 * _n_blocks // 3 - 1),
                    _n_blocks - 1,
                ]
            self.backbone.enable_intermediate_layers(multiscale_layer_indices)
            ca_heads_ms = compute_compatible_num_heads(in_features, target=8)
            self.multiscale_pool = MultiScaleAttentionPool(
                dim=in_features,
                num_scales=len(multiscale_layer_indices),
                num_heads=ca_heads_ms,
            )
            self.gem_pool = nn.Identity()  # no-op; pool aplicado por multiscale
            self._multiscale_layer_indices = list(multiscale_layer_indices)
        else:
            # GeM Pooling personalizado (reemplaza el avg_pool que timm usaría internamente)
            self.gem_pool = GeM(p=3.0)

        # La dimensión de salida del pool depende del modo activo:
        # - baseline: in_features (GeM o AttentionPool puro → [B, C]).
        # - hybrid_pool: 2*in_features (GeM + AttentionPool concatenados).
        # - multiscale_vit: num_scales * in_features (3 AttentionPool concatenados).
        if self.use_hybrid_pool:
            _pool_out_dim = 2 * in_features
        elif self.use_multiscale_vit:
            _pool_out_dim = len(self._multiscale_layer_indices) * in_features
        else:
            _pool_out_dim = in_features
        # Ecualizador de magnitud entre modalidades. Las features visuales
        # crudas tienen mucha más varianza y "aplastarían" a los metadatos; las
        # normalizamos (media 0, varianza 1) antes de la fusión.
        self.img_bn = nn.BatchNorm1d(_pool_out_dim)
        # Ajuste del classifier_in para que el classifier final reciba la
        # dimensión correcta: pool_out_dim + meta_features_out (si use_metadata).
        if self.use_metadata:
            classifier_in = _pool_out_dim + meta_features_out
        else:
            classifier_in = _pool_out_dim

        # Projection head para SupCon Loss (opcional, solo durante entrenamiento)
        # Proyecta features de imagen a espacio contrastivo de 64-dim
        self.use_projection_head = use_projection_head
        # El aviso one-shot del guard de edad vive como variable de clase
        # (`SkinCancerModel._age_warned_global`) para sobrevivir a DataParallel.
        if use_projection_head:
            # Con hybrid_pool activo, img_features_raw tiene shape
            # [B, 2*in_features] (concat de GeM + AttentionPool). La projection
            # head de SupCon opera sobre img_features_raw pre-BN, por lo que su
            # dimensión de entrada debe coincidir con _pool_out_dim. Sin este
            # ajuste, habría un RuntimeError por mismatch de shapes en el primer
            # forward con `return_features=True`.
            self.projection_head = nn.Sequential(
                nn.Linear(_pool_out_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )

        # 3. Cabezal de clasificación final (clasificador clínico).
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Regularización fuerte frente al overfitting de la clase minoritaria
            nn.Linear(classifier_in, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),  # SiLU (Swish): evita el problema de las neuronas ReLU muertas
            nn.Dropout(0.3),  # Previene la memorización de los ~6K melanomas
            nn.Linear(128, num_classes),
        )

    def forward(self, image, metadata=None, return_features=False, return_intermediate_features=False):
        """
        Forward pass con fusión temprana (FiLM) y atención (CBAM).

        Flujo: Backbone → [FiLM(meta)] → [CBAM] → GeM → BN → [concat meta] → Classifier
        Los metadatos influyen ANTES del pooling (modulación visual), no solo después (concat).

        Args:
            image (Tensor): [Batch, C, H, W]
            metadata (Tensor, optional): [Batch, 3] numérico.
            return_features (bool): Si True, retorna (logits, projected_features) para SupCon.
            return_intermediate_features (bool): si True, retorna también los
                `feature_maps` pre-pooling (justo antes de GeM/AttentionPool).
                Lo usa la destilación de features de FitNets. Compatible con
                `return_features`: si ambos están activos, retorna
                `(logits, projected, intermediate_features)`. Por defecto False.
        Returns:
            Logits (Tensor): [Batch, num_classes]
            O (logits, projected_features) si return_features=True
            O (logits, intermediate_features) si return_intermediate_features=True
            O (logits, projected_features, intermediate_features) si ambos.
        """
        # Error explícito si el modelo se construyó con use_metadata=True pero
        # el caller pasó metadata=None. Sin esta guarda, el forward saltaría el
        # bloque de metadatos y la concatenación, pero el classifier espera
        # `in_features + meta_features_out` columnas → RuntimeError críptico en
        # runtime. Los callers del pipeline (dataset.py, predict.py) siempre
        # construyen los metadatos; cualquier otro caller (CLI, API, notebook)
        # debe hacerlo explícito o desactivar la rama.
        if self.use_metadata and metadata is None:
            raise ValueError(
                "SkinCancerModel construido con use_metadata=True requiere "
                "metadata no-None en forward. Recibido metadata=None — el "
                "classifier espera in_features + meta_features_out columnas."
            )

        # Extraer mapas de características espaciales [B, C, H, W].
        # Si attention pool está activo, el backbone devuelve tokens [B, N, D]
        # directamente y se saltan CBAM/FiLM (attention pool es una estrategia
        # de pooling alternativa para ViT puro; las modulaciones FiLM/CBAM son
        # para los feature maps de CNN).
        # Si multiscale_vit está activo, capturamos los tokens de capas
        # intermedias mientras el backbone procesa con normalidad. FiLM/CBAM se
        # desactivan: los tokens intermedios no han pasado por FiLM/CBAM, así
        # que aplicarlos solo al feature_maps final crearía asimetría entre
        # escalas.
        _multiscale_intermediate_tokens = None
        if self.use_attention_pool:
            tokens = self.backbone(image, return_tokens=True)  # [B, N_total, D]
            # AttentionPoolViT acepta tokens CLS+patches y retorna [B, D, 1, 1]
            feature_maps = self.attention_pool(tokens)
        elif self.use_multiscale_vit:
            feature_maps, _multiscale_intermediate_tokens = self.backbone(
                image, return_intermediate_layers=True
            )
        else:
            feature_maps = self.backbone(image)

        # --- Procesar metadatos ANTES del pooling (necesario para FiLM) ---
        meta_features = None
        if self.use_metadata and metadata is not None:
            # Formato de metadata: [age_norm (float), sex_code (float), site_code (float)].
            # Guarda defensiva de edad. El contrato es age_norm = fillna(50)/100
            # ∈ [0, 1]. dataset.py y predict.py ya sanitizan y clampean aguas
            # arriba, pero otro caller (CLI, notebook, API externa) que salte el
            # preprocess del engine podría enviar metadata corrupta en silencio
            # (edad en años, NaN, Inf). Aquí clampeamos a [0, 1] tras
            # nan_to_num; un aviso one-shot señala el uso fuera de contrato sin
            # bloquear (compatible con batches ya limpios). Se tolera un valor en
            # (1, 1.2] para absorber el clamp 150/100=1.5→1.0 de predict.py sin
            # spam. El default `unknown` deriva de `AGE_NORM_UNKNOWN` en
            # constants.py (fuente de verdad única del centro fijo en 50 años).
            age_col = metadata[:, 0].float().nan_to_num(AGE_NORM_UNKNOWN)
            # El aviso one-shot vive a nivel de clase (DataParallel-safe). Además,
            # el `bool(.any())` fuerza una sincronización GPU→CPU que rompe el
            # grafo bajo torch.compile. Combinado con `mode="reduce-overhead"`
            # (CUDA Graphs activos en GPUs Ampere/Ada/Hopper), el "resume
            # segment" capturado tras el break reutilizaría la memoria del tensor
            # `meta_features` entre iteraciones, y el backward del batch N+1
            # leería un buffer ya sobrescrito → RuntimeError "accessing tensor
            # output of CUDAGraphs that has been overwritten by a subsequent
            # run". Por eso el guard se condiciona con `_is_compiling()`: durante
            # el tracing el bloque se omite (CUDA Graphs intacto) y en modo eager
            # (validación de dataset.py, inferencia de predict.py, notebooks, CLI)
            # sigue activo. El `clamp(0, 1)` de la línea siguiente protege al
            # modelo con independencia del aviso, preservando el contrato clínico.
            if not _is_compiling() and not SkinCancerModel._age_warned_global:
                with torch.no_grad():
                    if bool(((age_col < 0.0) | (age_col > 1.2)).any()):
                        warnings.warn(
                            "SkinCancerModel.forward recibió `age` fuera de "
                            "[0, 1.2] tras nan_to_num. La edad normalizada "
                            "(age_norm = fillna(50)/100) debe estar en [0, 1]. Clampeando a "
                            "[0, 1]; revisar el pipeline upstream "
                            "(dataset.py/predict.py).",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        SkinCancerModel._age_warned_global = True
            age = age_col.clamp(0.0, 1.0).unsqueeze(1)    # [Batch, 1]
            # NaN.long() es comportamiento indefinido en PyTorch (produce un
            # valor arbitrario dependiente de la plataforma). nan_to_num()
            # convierte el NaN al default ANTES de la conversión a entero, y el
            # clamp actúa como defensa en profundidad.
            #
            # Los defaults y los límites del clamp derivan de constants.py, no de
            # literales 2/7/3/8. Si se reordenara SEX_MAP/SITE_MAP (p. ej. para
            # que unknown=0), unos literales fijos mapearían en silencio códigos
            # válidos a "unknown" y destruirían la señal categórica.
            sex = (
                metadata[:, 1]
                .float()
                .nan_to_num(float(SEX_UNKNOWN_CODE))
                .long()
                .clamp(0, NUM_SEX_CODES - 1)
            )
            site = (
                metadata[:, 2]
                .float()
                .nan_to_num(float(SITE_UNKNOWN_CODE))
                .long()
                .clamp(0, NUM_SITE_CODES - 1)
            )

            # Meta-Dropout con los códigos `unknown` derivados de constants.py,
            # sin graph break bajo torch.compile y con modo per-feature por
            # defecto.
            #
            # - `per_feature=True` (por defecto): cada feature {age, sex, site}
            #   se dropea independientemente con probabilidad `p`. Refleja la
            #   práctica clínica real (reporte parcial de metadatos).
            # - `per_feature=False`: los 3 features se dropean juntos
            #   (per-sample). Se conserva por reproducibilidad estricta.
            #
            # Aplicamos siempre `torch.where` (sin un `if .any()` previo) para
            # evitar el graph break que provocaría la sincronización GPU→CPU de
            # `bool(.any())` bajo torch.compile. El coste extra es despreciable
            # (3 `torch.where` por batch); además P(ningún dropout) = 0.9^B ≈ 3%
            # para B=32, así que el conditional solo ahorraba cómputo en un 3%
            # de los batches mientras rompía el grafo el 100% del tiempo.
            if self.training and self.meta_dropout_prob > 0.0:
                B = age.size(0)
                device = age.device
                if self.meta_dropout_per_feature:
                    age_mask = torch.rand(B, device=device) < self.meta_dropout_prob
                    sex_mask = torch.rand(B, device=device) < self.meta_dropout_prob
                    site_mask = torch.rand(B, device=device) < self.meta_dropout_prob
                else:
                    shared = torch.rand(B, device=device) < self.meta_dropout_prob
                    age_mask = shared
                    sex_mask = shared
                    site_mask = shared

                age = torch.where(
                    age_mask.unsqueeze(1),
                    torch.full_like(age, AGE_NORM_UNKNOWN),
                    age,
                )
                sex = torch.where(
                    sex_mask,
                    torch.full_like(sex, SEX_UNKNOWN_CODE),
                    sex,
                )
                site = torch.where(
                    site_mask,
                    torch.full_like(site, SITE_UNKNOWN_CODE),
                    site,
                )

            # Pasar por Embeddings
            sex_features = self.sex_emb(sex)     # [Batch, 4]
            site_features = self.site_emb(site)  # [Batch, 8]

            # Concatenar a Vector Semántico
            meta_combined = torch.cat([age, sex_features, site_features], dim=1)  # [Batch, 13]

            # Extraer características clínicas avanzadas
            meta_features = self.meta_net(meta_combined)  # [Batch, 16]

            # Fusión temprana metadata↔features: FiLM o CrossAttention según
            # fusion_mode. Se usa `self.fusion_mode` explícito en vez de
            # `hasattr`, porque torch.compile y los wrappers (DataParallel, DDP)
            # pueden alterar el reenvío de atributos; `self.fusion_mode` es la
            # fuente de verdad única. FiLM/CBAM no se aplican ni en attention_pool
            # (los feature_maps ya están pooled [B, D, 1, 1] tras la atención) ni
            # en multiscale_vit (asimetría con los tokens intermedios sin
            # modular). Hybrid_pool SÍ aplica FiLM/CBAM con normalidad.
            if self.use_attention and not self.use_attention_pool and not self.use_multiscale_vit:
                if self.fusion_mode == "film":
                    feature_maps = self.film(feature_maps, meta_features)
                elif self.fusion_mode == "cross_attention":
                    feature_maps = self.meta_cross_attn(feature_maps, meta_features)

        # CBAM: atención por canal + espacial sobre los feature maps (modulados
        # o no). Se omite con attention_pool (feature_maps ya son [B, D, 1, 1])
        # y con multiscale_vit (asimetría con los tokens intermedios).
        if self.use_attention and not self.use_attention_pool and not self.use_multiscale_vit:
            feature_maps = self.cbam(feature_maps)

        # Pooling [B, C, H, W] → [B, D] (D = C en baseline/attention_pool,
        # D = 2*C en hybrid_pool).
        # Separamos las features pre-BN (para SupCon) de las post-BN (para el
        # classifier): SupCon (Khosla et al., NeurIPS 2020) opera sobre
        # representaciones sin normalización por batch, ya que post-BN la
        # distancia coseno queda sesgada por las `running_mean/var` del
        # clasificador, no del espacio contrastivo.
        # Con hybrid_pool activo, combina GeM + AttentionPool sobre el MISMO
        # feature_maps post-FiLM/CBAM, con salida [B, 2*C] (compatible con
        # FiLM/CBAM, ortogonal a AttentionPool puro). Con multiscale_vit activo,
        # hace pooling sobre 3 capas intermedias del ViT capturadas con hooks,
        # con salida [B, 3*C].
        if self.use_hybrid_pool:
            img_features_raw = self.hybrid_pool(feature_maps)
        elif self.use_multiscale_vit:
            if _multiscale_intermediate_tokens is None:
                raise RuntimeError(
                    "use_multiscale_vit=True pero los tokens intermedios no se "
                    "capturaron en el forward del backbone. Esto sugiere un "
                    "fallo en BackboneWrapper o en los hooks."
                )
            img_features_raw = self.multiscale_pool(_multiscale_intermediate_tokens)
        else:
            img_features_raw = self.gem_pool(feature_maps)
        # AttentionPoolViT retorna [B, D, 1, 1] (4D), pero `img_bn`
        # (BatchNorm1d) requiere 2D o 3D. En la ruta normal, GeM ya aplica
        # `.flatten(1)` internamente y devuelve [B, D] (2D); pero con
        # `use_attention_pool=True`, `gem_pool=Identity()` y AttentionPoolViT
        # devuelve 4D para mantener la simetría visual con feature_maps pre-pool.
        # El `.flatten(1)` post-pool es no-op en 2D (ruta GeM) y aplana 4D a 2D
        # en la ruta AttentionPool: idempotente en ambas.
        if img_features_raw.ndim > 2:
            img_features_raw = img_features_raw.flatten(1)
        # Ecualizador para domar la varianza extrema de la imagen frente a los metadatos.
        img_features = self.img_bn(img_features_raw)

        if meta_features is not None:
            # Concatenar visión + contexto clínico (fusión dual: FiLM modula y el concat aporta).
            features = torch.cat((img_features, meta_features), dim=1)
        else:
            features = img_features

        # Decisión Final
        out = self.classifier(features)

        # Si es clasificación binaria (num_classes=1), no aplicamos Sigmoid aquí.
        # PyTorch prefiere BCEWithLogitsLoss (es más estable numéricamente combinar Sigmoid+BCE).
        # Para Multiclase, CrossEntropyLoss también espera Logits crudos (no Softmax).

        # `feature_maps` pre-pooling es lo que FitNets necesita del student. En
        # la ruta normal (sin attention_pool) son [B, C, H, W] post-CBAM/FiLM.
        # Con attention_pool ya son [B, D, 1, 1] y FitNets no tiene sentido — por
        # eso `use_attention_pool + use_fitnet` se rechaza en `__init__` con un
        # ValueError explícito.
        intermediate_feats = feature_maps

        # SupCon: retorna las features proyectadas junto con los logits (solo en
        # entrenamiento), usando features PRE-BN para la proyección contrastiva.
        # Error explícito si el caller pide return_features=True pero el modelo
        # se construyó sin projection_head. Antes retornaba (out, None) en
        # silencio y el caller (típicamente `supcon_criterion(projected.float(),
        # labels)`) fallaba con `AttributeError: 'NoneType' object has no
        # attribute 'float'`, un mensaje que no apunta al constructor incorrecto.
        if return_features and not self.use_projection_head:
            raise ValueError(
                "return_features=True pero use_projection_head=False. "
                "SupCon requiere que SkinCancerModel se construya con "
                "use_projection_head=True para tener `self.projection_head`. "
                "Soluciones: (a) construir el modelo con use_projection_head=True "
                "(por defecto cuando mode='multiclass' en train.py), o "
                "(b) eliminar return_features=True del forward si solo necesitas "
                "los logits."
            )
        projected = None
        if return_features and self.use_projection_head:
            projected = self.projection_head(img_features_raw)

        # Combinar los 4 casos de retorno de forma explícita:
        if return_features and return_intermediate_features:
            return out, projected, intermediate_feats
        if return_features:
            return out, projected
        if return_intermediate_features:
            return out, intermediate_feats
        return out
