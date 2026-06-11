"""
Self-distillation estilo Born-Again Networks (Furlanello et al., ICML 2018).

Variante simplificada: en cada batch, además del loss supervisado se añade
un término de KL-divergence entre la distribución de predicción suavizada
por temperatura y el target suavizado (label smoothing uniforme).

Objetivo clínico
----------------
Estabilizar la convergencia del especialista multiclase en clases raras
(Dermatofibroma, SCC). La KL con soft target ayuda a evitar sobreconfianza
local en las clases dominantes y mejora la calibración (ECE) sin coste
computacional adicional (no requiere un teacher separado como la destilación
binario→multiclase de `distillation.py`).

Fórmula
-------
Sea `z` los logits del modelo, `T=4.0` la temperatura, `y` el target entero
y `n_classes` el número de clases:

    p_soft = softmax(z / T)
    q_soft = label_smooth(one_hot(y), ε=1/n_classes)
    kd_loss = T² · KL(q_soft || p_soft)
    loss = 0.9 · CE/Focal/Seesaw(z, y) + 0.1 · kd_loss

El factor `T²` es la normalización estándar de Hinton para que los gradientes
del término KD sean comparables a los del término supervisado.

Activación tardía
-----------------
El término KD se aplica SOLO desde `warmup_epochs=5` (multiclase) para no
destabilizar el inicio. Antes del warmup las predicciones son
aproximadamente uniformes y la KL introduce ruido no-informativo.

Compatibilidad
--------------
- MixUp/CutMix: con `lam < 1`, el target "duro" no es one-hot sino una
  combinación — `self_distillation_step` recibe `targets_a`, `targets_b`,
  `lam` y construye el soft target mezclado.
- SupCon: ortogonal; KD opera sobre logits, SupCon sobre features. Ambos
  pueden coexistir (losses aditivos con pesos distintos).
- No usa class_weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_KD_TEMPERATURE = 4.0
DEFAULT_KD_WEIGHT = 0.1
DEFAULT_WARMUP_EPOCHS = 5


def _build_soft_target(
    targets_a: torch.Tensor,
    targets_b: torch.Tensor | None,
    lam: float,
    num_classes: int,
    smoothing: float,
) -> torch.Tensor:
    """
    Construye un target suavizado [B, C] compatible con MixUp/CutMix.

    Si `targets_b` es None se asume batch sin MixUp (`lam=1.0`).
    Aplica label smoothing uniforme: cada clase recibe `smoothing/num_classes`
    de masa y la clase dominante recibe `1 - smoothing + smoothing/num_classes`.

    Se valida el contrato de forma explícita: `targets_b=None` con `lam<1.0`
    produciría `sum(soft) = smoothing + lam*(1-smoothing) < 1` — una distribución
    inválida que `F.kl_div(..., reduction="batchmean")` consumiría sin avisar,
    con gradientes sesgados. `train.py` respeta el contrato, pero esta
    comprobación protege ante cualquier cambio futuro.
    """
    if targets_b is None and not (1.0 - 1e-6 <= lam <= 1.0 + 1e-6):
        raise ValueError(
            f"targets_b=None requiere lam=1.0 (batch sin MixUp), "
            f"recibido lam={lam}. El soft target resultante no suma 1 y "
            f"rompe la interpretación de F.kl_div(reduction='batchmean')."
        )
    device = targets_a.device
    smooth_value = smoothing / num_classes
    soft = torch.full(
        (targets_a.size(0), num_classes),
        fill_value=smooth_value,
        device=device,
        dtype=torch.float32,
    )
    one_hot_a = F.one_hot(targets_a.long(), num_classes=num_classes).float()
    soft = soft + lam * (1.0 - smoothing) * one_hot_a
    if targets_b is not None and lam < 1.0:
        one_hot_b = F.one_hot(targets_b.long(), num_classes=num_classes).float()
        soft = soft + (1.0 - lam) * (1.0 - smoothing) * one_hot_b
    return soft


def self_distillation_loss(
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor | None = None,
    lam: float = 1.0,
    temperature: float = DEFAULT_KD_TEMPERATURE,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Calcula la contribución KD (sin el peso ni el término supervisado).

    El caller es responsable de combinar:
        loss_total = (1 - kd_weight) * loss_sup + kd_weight * kd_loss

    Args:
        logits: [B, C] logits crudos del modelo.
        targets_a, targets_b, lam: análogos a MixUp (`targets_b=None, lam=1`
            para batches sin mezcla).
        temperature: T del paper (default 4.0).
        label_smoothing: ε del soft target (default 0.1).

    Returns:
        Escalar KL * T² listo para sumar al loss supervisado.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits debe ser [B, C], recibido shape={tuple(logits.shape)}")
    num_classes = logits.size(1)

    soft_target = _build_soft_target(
        targets_a=targets_a,
        targets_b=targets_b,
        lam=lam,
        num_classes=num_classes,
        smoothing=label_smoothing,
    )

    # El cómputo se fuerza a float32 (autocast desactivado): con T=4.0, el
    # `logits / T` puede saturar el log_softmax bajo fp16 en GPUs antiguas.
    # En bf16 es inocuo, pero mantener este guard preserva la consistencia con
    # las demás losses del proyecto (Seesaw, BalancedSoftmax, FocalLoss).
    with torch.amp.autocast('cuda', enabled=False):
        logits_f32 = logits.float()
        soft_target_f32 = soft_target.float()
        # KL(target || pred) con logits escalados por T. Usamos `log_softmax` del
        # modelo y el target como distribución directa (no log) porque KLDivLoss
        # espera input en log-space y target en espacio de probabilidad.
        log_probs = F.log_softmax(logits_f32 / temperature, dim=1)
        # KL(q || p) = sum q * (log q - log p). El término `q * log q` es constante
        # respecto a los parámetros del modelo, pero lo incluimos para que el valor
        # del loss sea interpretable (distribucional, no solo cross-entropy).
        kl = F.kl_div(log_probs, soft_target_f32, reduction="batchmean")

        # Normalización T² del paper original (Hinton et al., 2015): los gradientes
        # del término soft con temperatura escalan con 1/T², T² recupera la escala
        # comparable al término hard.
        return kl * (temperature ** 2)


class SelfDistillationScheduler:
    """
    Helper para aplicar el loss KD solo tras warmup.

    Uso en `train.py`:

        kd = SelfDistillationScheduler(warmup_epochs=5, kd_weight=0.1)
        ...
        for epoch in range(epochs):
            for batch in loader:
                logits = model(...)
                loss_sup = criterion(logits, targets)
                loss = kd.combine(epoch, logits, targets_a, targets_b, lam, loss_sup)
                loss.backward()

    Si `epoch < warmup_epochs` devuelve `loss_sup` intacto.
    """

    def __init__(
        self,
        warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
        kd_weight: float = DEFAULT_KD_WEIGHT,
        temperature: float = DEFAULT_KD_TEMPERATURE,
        label_smoothing: float = 0.1,
    ):
        if not 0.0 <= kd_weight < 1.0:
            raise ValueError(f"kd_weight debe estar en [0, 1), recibido {kd_weight}")
        self.warmup_epochs = warmup_epochs
        self.kd_weight = kd_weight
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def is_active(self, epoch: int) -> bool:
        return epoch >= self.warmup_epochs and self.kd_weight > 0.0

    def combine(
        self,
        epoch: int,
        logits: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor | None,
        lam: float,
        loss_sup: torch.Tensor,
    ) -> torch.Tensor:
        if not self.is_active(epoch):
            return loss_sup
        kd = self_distillation_loss(
            logits=logits,
            targets_a=targets_a,
            targets_b=targets_b,
            lam=lam,
            temperature=self.temperature,
            label_smoothing=self.label_smoothing,
        )
        return (1.0 - self.kd_weight) * loss_sup + self.kd_weight * kd
