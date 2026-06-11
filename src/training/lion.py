"""Lion optimizer (Chen et al., NeurIPS 2023).

Referencia: Chen et al., "Symbolic Discovery of Optimization Algorithms",
arXiv 2302.06675, aceptado NeurIPS 2023. https://arxiv.org/abs/2302.06675

Lion ("EvoLved Sign Momentum") fue descubierto mediante búsqueda evolutiva
sobre el espacio de programas simbólicos de optimización. A diferencia de
AdamW que mantiene dos estadísticos exponenciales por parámetro (m1, m2),
Lion mantiene UN solo momentum y usa el signo (sign) del EMA para actualizar
pesos. Esto reduce la memoria del optimizador a la mitad (~15% menos VRAM
total en training runs típicos) con rendimiento comparable o ligeramente
mejor en benchmarks ImageNet, LM, y detección.

**Actualización por paso (Chen 2023, eq. 2):**

    c_t = β1 · m_{t-1} + (1 - β1) · g_t        # dirección del update
    m_t = β2 · m_{t-1} + (1 - β2) · g_t        # EMA del momentum
    p_t = p_{t-1} - lr · (sign(c_t) + wd · p_{t-1})

Default hyperparameters del paper:
- `betas=(0.9, 0.99)` — interpolación entre gradiente actual y momentum.
  El paper recomienda `β1 > β2` para estabilidad (opuesto a AdamW donde
  β1 < β2). Razón: con `sign()` la magnitud se descarta; el orden importa.
- `lr` recomendado: ~10x más pequeño que AdamW equivalente (el sign update
  tiene magnitud 1.0 por parámetro, mientras AdamW escala por `1/sqrt(v)`
  que suele ser ≪ 1). Para este proyecto: `lr=1e-5` en vez de `1e-4`.
- `weight_decay` recomendado: ~10x más grande que AdamW equivalente para
  compensar el lr reducido. Para este proyecto: `1e-1` en vez de `1e-2`.

**Trade-offs en SKIN-PROYECT:**
- Ventaja: -15% VRAM del optimizer → permite batch size ligeramente mayor en
  Colab T4 (16GB) para el modelo multiclase B4@380.
- Ventaja: signo-basado → robusto a gradientes outlier (útil bajo AMP fp16).
- Coste: requiere tuning del LR (~10x menor que AdamW). Un run con el `lr` de
  AdamW directamente diverge.
- Coste: NO tiene bias correction como Adam — primeros steps pueden ser
  inestables sin warmup. Compatible con el LinearLR warmup existente.

**Contrato con el pipeline:**
- Drop-in replacement del `torch.optim.AdamW` en `train.py`.
- Compatible con SAM wrapper (Lion implementa `torch.optim.Optimizer`).
- Compatible con `GradScaler` AMP (acepta gradientes escalados via
  `unscale_` upstream).
- Compatible con `param_groups` (LR diferencial backbone/head preservado).
- NO implementa `fused` kernel — 100% PyTorch puro.
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion optimizer — Sign-Momentum EvoLved.

    Args:
        params: iterable de parámetros o param_groups (estándar optimizer).
        lr: learning rate. **Usar ~10x más pequeño que AdamW equivalente.**
        betas: `(β1, β2)` donde β1 es el coef del update y β2 el momentum EMA.
            Default `(0.9, 0.99)` del paper.
        weight_decay: decoupled weight decay. **DEBE ser explícito.** Usar
            ~10x más grande que AdamW equivalente (ej. `0.1` si AdamW usaba
            `0.01`). Pasar `0.0` explícitamente para desactivar la
            regularización (uso experimental, no recomendado en clínica).

    `weight_decay` es obligatorio y explícito: si no se especifica, el modelo
    entrenaría sin regularización (overfitting silencioso, invisible en la
    curva de train loss). Por eso el constructor rechaza un `weight_decay`
    implícito en lugar de asumir un valor por defecto.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: "tuple[float, float]" = (0.9, 0.99),
        weight_decay: "float | None" = None,
    ):
        if lr <= 0.0:
            raise ValueError(f"Lion: lr debe ser > 0, recibido {lr}")
        if not (0.0 <= betas[0] <= 1.0 and 0.0 <= betas[1] <= 1.0):
            raise ValueError(
                f"Lion: betas deben estar en [0,1], recibido {betas}"
            )
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(
                f"Lion: weight_decay debe ser >= 0 si se especifica, "
                f"recibido {weight_decay}"
            )
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        # Tras el merge de defaults con los param_groups, TODOS los grupos deben
        # tener weight_decay explícito (float >= 0.0). Un None post-merge indica
        # que ni el default ni el param_group lo especificaron → el step haría
        # `p.mul_(1.0 - lr * None)` y fallaría. Se valida en el constructor, no
        # en el primer step.
        for idx, group in enumerate(self.param_groups):
            self._validate_weight_decay_group(group, idx)

    @staticmethod
    def _validate_weight_decay_group(group, idx):
        """Validación centralizada de weight_decay por grupo.

        Se invoca tanto desde `__init__` como desde `add_param_group`, para
        garantizar que CUALQUIER grupo — preexistente o añadido post-init —
        tenga weight_decay explícito. Un caller que invoque
        `optimizer.add_param_group({"params": [...], "lr": 1e-5})` sin
        weight_decay hereda `defaults["weight_decay"]=None` → el primer `step()`
        fallaría con `p.mul_(1.0 - lr * None)`. La validación bloquea la adición
        antes de que ocurra.
        """
        wd = group.get("weight_decay")
        if wd is None:
            raise ValueError(
                f"Lion: param_group[{idx}] no tiene weight_decay. "
                "Pasar `weight_decay=X` en el constructor (valor por defecto "
                "para todos los grupos) o fijarlo explícitamente en el "
                "param_group pasado a `add_param_group`. Recomendación del "
                "paper: ~10x el weight_decay de AdamW equivalente (ej. 0.1 si "
                "AdamW usaba 0.01). `weight_decay=0.0` explícito desactiva "
                "regularización (experimental, no recomendado en clínica)."
            )
        if wd < 0.0:
            raise ValueError(
                f"Lion: param_group[{idx}] tiene weight_decay={wd} < 0"
            )

    def add_param_group(self, param_group):
        """Re-valida weight_decay en grupos añadidos después del constructor.

        `__init__` valida los grupos iniciales, pero `Optimizer.add_param_group`
        heredaría `defaults["weight_decay"]=None` sin avisar → el primer `step()`
        fallaría con un mensaje críptico. Aquí se re-aplica la validación.

        La validación debe ser atómica: `super().add_param_group(param_group)`
        mergea defaults y AÑADE el grupo a `self.param_groups` ANTES de que la
        validación corra. Si esta lanza `ValueError`, el grupo inválido quedaría
        residual con `weight_decay=None` → estado inconsistente: (a) el siguiente
        `step()` fallaría con `p.mul_(1.0 - lr * None)`; (b) un reintento añadiría
        un grupo extra, incrementando `len(param_groups)` sin que el caller lo
        note. El try/except remueve el grupo residual antes de re-lanzar.
        """
        super().add_param_group(param_group)
        # El nuevo grupo es el último tras `super().add_param_group`.
        new_idx = len(self.param_groups) - 1
        try:
            self._validate_weight_decay_group(self.param_groups[new_idx], new_idx)
        except Exception:
            # Remover el grupo residual para restaurar el estado previo. Se
            # captura `Exception` (no solo `ValueError`) para que cualquier
            # validación futura (tipo de params, shapes...) que lance otra
            # excepción también preserve la atomicidad. `raise` sin argumento
            # propaga la excepción original intacta → el caller ve el error real.
            del self.param_groups[new_idx]
            raise

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lion no soporta gradientes sparse (use SparseAdam)"
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]

                # Weight decay desacoplado (estilo AdamW, aplicado ANTES del update)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Dirección del update: c = β1·m + (1-β1)·g
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1)
                # Aplicar el update basado en signo (sign)
                p.add_(torch.sign(update), alpha=-lr)

                # Update del momentum EMA: m = β2·m + (1-β2)·g
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
