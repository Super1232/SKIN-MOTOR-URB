"""
Sharpness-Aware Minimization (SAM) — Foret et al., ICLR 2021.

Motivación
----------
Los optimizadores estándar (SGD, Adam, AdamW) buscan mínimos de loss sin
reparar en la curvatura del entorno. Mínimos "afilados" generalizan mal: un
pequeño shift en los pesos (inherente a cualquier despliegue con cambio de
hardware, quantization o data drift) degrada el modelo bruscamente. SAM busca
mínimos "planos" — regiones donde la loss se mantiene baja incluso tras una
perturbación worst-case de los pesos.

En detección de melanoma esto importa clínicamente: el modelo se desplegará
en dispositivos distintos al de entrenamiento (quantization ONNX INT8,
distribución de cámaras dermatoscópicas variada), y mínimos planos reducen la
brecha train→prod.

Mecánica
--------
Cada paso real de optimización requiere DOS forwards:

    1. Forward(w)           → loss_1 → backward → gradiente ∇L(w)
    2. Perturbación:  w_adv = w + ε · ∇L(w) / ‖∇L(w)‖
       (ε = ρ: norma de la perturbación worst-case en dirección del gradiente)
    3. Forward(w_adv)       → loss_2 → backward → gradiente ∇L(w_adv)
    4. Restaurar:     w   = w_adv - ε · ∇L(w) / ‖∇L(w)‖
    5. Step real con ∇L(w_adv):  w  ← w - lr · ∇L(w_adv)

El optimizador base (AdamW en nuestro pipeline) se usa internamente en el
paso 5. SAM solo orquesta las perturbaciones.

Variantes
---------
- **Standard SAM** (`adaptive=False`): perturbación uniforme sobre todos los
  parámetros.
- **ASAM** (Adaptive SAM, `adaptive=True`, Kwon et al. 2021): escala la
  perturbación por `|w|`, lo que hace la norma invariante a reescalados del
  modelo. Empíricamente mejor para ViTs.

Integración con AMP, grad accumulation y MixUp
----------------------------------------------
SAM requiere coordinación cuidadosa con el loop de entrenamiento estándar:

1. **AMP**: el GradScaler debe desescalar el gradiente antes del primer
   `first_step()` (para que la norma sea correcta en fp32). El segundo
   forward también usa autocast, y su backward se escala con el mismo
   scaler. El `scaler.step(base_optimizer)` se llama después del `second_step`.

2. **Grad accumulation**: SAM solo tiene sentido como par atómico
   first_step+second_step aplicado al gradiente ACUMULADO. Esto significa que
   dentro de un ciclo de accum_steps se acumulan gradientes de `accum_steps`
   forwards normales, y SOLO al final del ciclo se ejecuta la secuencia
   first_step → segundo forward/backward → second_step. Alternativa (más cara
   pero más fiel al paper): ejecutar SAM en cada mini-batch, duplicando el
   coste. Este pipeline aplica la primera variante (SAM por ciclo completo).

3. **MixUp/CutMix**: ambos forwards deben usar el MISMO batch mezclado con el
   MISMO lambda. NO resamplear λ entre el primer y segundo forward — eso
   rompe la suposición de worst-case del paper.

API de uso (train.py pseudo-código)
-----------------------------------
```python
from src.training.sam import SAM

base_optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)
optimizer = SAM(base_optimizer, rho=0.05, adaptive=False)

for batch_idx, (images, labels, meta) in enumerate(loader):
    # ---- Forward 1 (normal) ----
    with autocast("cuda"):
        logits = model(images, meta)
        loss = criterion(logits, labels) / accum_steps
    scaler.scale(loss).backward()

    if (batch_idx + 1) % accum_steps == 0:
        scaler.unscale_(optimizer.base_optimizer)  # desescalar para norma
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.first_step(zero_grad=True)

        # ---- Forward 2 (en pesos perturbados, MISMO batch mezclado) ----
        with autocast("cuda"):
            logits_adv = model(images, meta)
            loss_adv = criterion(logits_adv, labels) / accum_steps
        scaler.scale(loss_adv).backward()
        scaler.unscale_(optimizer.base_optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # second_step restaura los pesos y deja gradiente adversarial listo
        optimizer.second_step(zero_grad=False)
        scaler.step(optimizer.base_optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
```

Coste
-----
~+80-100% tiempo de training por época (dos forwards por step real). Para
compensarlo se recomienda reducir el número de épocas (p. ej. binary 20→15,
multiclass 30→20). Como alternativa más barata: ASAM (rho adaptativo, ~+60%)
o desactivar SAM y confiar en EMA+SWA.

Referencias
-----------
- Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
  Generalization", ICLR 2021. https://arxiv.org/abs/2010.01412
- Kwon et al., "ASAM: Adaptive Sharpness-Aware Minimization", ICML 2021.
"""

from __future__ import annotations

from collections import OrderedDict

import torch


class SAM(torch.optim.Optimizer):
    """
    Wrapper SAM sobre un optimizador base (ej. AdamW).

    No implementa `step()` directamente — el usuario debe invocar
    `first_step()` y `second_step()` explícitamente, porque el loop de
    training tiene que ejecutar el segundo forward entre medias.

    Nota de diseño: deliberadamente NO invocamos `super().__init__(params, defaults)`
    porque `param_groups` se comparte con `base_optimizer` (permite que
    `clip_grad_norm_` y los schedulers operen sin cambios). Como consecuencia,
    poblamos manualmente los atributos que `torch.optim.Optimizer` instalaría
    (`_step_count`, hooks dict) para que schedulers y utilities que los leen
    en versiones recientes de PyTorch no rompan.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        norm_eps: float = 1e-12,
    ):
        if rho < 0.0:
            raise ValueError(f"rho debe ser >= 0, recibido {rho}")
        # `norm_eps` debe ser estrictamente positivo: si fuese 0, la división
        # `rho / (grad_norm + 0)` se indefine cuando `grad_norm` es exactamente
        # 0 (caso raro pero alcanzable con un modelo congelado). Se valida en
        # el constructor para detectar el error antes del primer paso.
        if norm_eps <= 0.0:
            raise ValueError(f"norm_eps debe ser > 0, recibido {norm_eps}")
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        # `norm_eps` (no `eps`): AdamW ya tiene su propio `eps` en defaults.
        # Usamos un nombre disjunto para evitar colisión al mergear defaults.
        self.norm_eps = norm_eps

        # Compartir param_groups con el optimizador base. Esto permite que
        # `torch.nn.utils.clip_grad_norm_(params)` y `scheduler.step(optimizer)`
        # funcionen sin cambios.
        self.param_groups = base_optimizer.param_groups
        self.defaults = {
            **base_optimizer.defaults,
            "sam_rho": rho,
            "sam_adaptive": adaptive,
            "sam_norm_eps": norm_eps,
        }
        # state local de SAM (no interfiere con el state del base_optimizer)
        self.state: dict = {}

        # Atributos que `torch.optim.Optimizer.__init__` instalaría y que
        # algunos schedulers/utilities leen en PyTorch >=2.x. Los poblamos a
        # mano porque no llamamos al constructor base (ver nota de diseño de la
        # clase). `step()` se sobreescribe para forzar el uso de
        # first_step/second_step en vez de un paso estándar.
        self._step_count = 0
        self._optimizer_step_pre_hooks: OrderedDict = OrderedDict()
        self._optimizer_step_post_hooks: OrderedDict = OrderedDict()
        self._patch_step_function = lambda: None

    def step(self, closure=None):
        raise NotImplementedError(
            "SAM usa first_step()/second_step() explícitos; no llamar step() "
            "directamente. Ver docstring del módulo para el contrato."
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> bool:
        """
        Paso 1 de SAM: calcula ε · ∇L(w) / ‖∇L(w)‖ y añade al peso.

        Guarda la perturbación exacta en `self.state[p]["e_w"]` para que
        `second_step` pueda restaurar los pesos originales.

        Si `grad_norm` no es finita (NaN/Inf por overflow en backward, GeM
        degenerado, etc.) se aborta SIN mutar pesos ni poblar `self.state`:
        una perturbación NaN haría imposible el rollback en `second_step`
        (NaN − NaN = NaN). Retorna `True` si la perturbación se aplicó, `False`
        si se saltó; el caller usa el retorno para gatear el segundo forward y
        el step posterior. `second_step` es idempotente si `self.state` está
        vacío.
        """
        grad_norm = self._grad_norm()
        if not torch.isfinite(grad_norm):
            # Gradiente acumulado no finito → no se perturba. Se limpian los
            # grads si el caller pidió zero_grad para preservar el contrato.
            # Se llama a `base_optimizer.zero_grad` directamente (no
            # `self.zero_grad`, que además borraría `self.state`): aquí no hay
            # state que preservar porque salimos antes de poblarlo, pero se
            # mantiene la misma API que el branch exitoso de abajo, donde el
            # state SÍ debe conservarse.
            if zero_grad:
                self.base_optimizer.zero_grad(set_to_none=True)
            return False
        scale = self.rho / (grad_norm + self.norm_eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Escala adaptativa (ASAM): la perturbación se proporcional al
                # tamaño del peso, haciéndola invariante a reescalados
                # (importante en ViT con LayerNorm).
                if self.adaptive:
                    e_w = (torch.pow(p, 2)) * p.grad * scale
                else:
                    e_w = p.grad * scale
                p.add_(e_w)  # escalar a w_adv
                # Guardar perturbación exacta para revertirla en second_step.
                state = self.state.setdefault(p, {})
                state["e_w"] = e_w

        # Se invoca `base_optimizer.zero_grad` directamente, no `self.zero_grad`:
        # esta última borra `self.state`, donde acabamos de guardar los `e_w`,
        # y haría imposible el rollback. La limpieza de state en `self.zero_grad`
        # está pensada para callers externos que resetean SAM entre ciclos; aquí
        # se preserva el invariante interno first_step→second_step.
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)
        return True

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        Paso 2 de SAM: restaura pesos originales.

        El caller debe llamar después a `base_optimizer.step()` (o
        `scaler.step(base_optimizer)` si AMP) con el gradiente adversarial
        que dejó el segundo backward.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, None)
                if state is None or "e_w" not in state:
                    continue
                p.sub_(state["e_w"])  # revertir a w
                # Liberar memoria del buffer de perturbación
                del state["e_w"]

        # `base_optimizer.zero_grad` directo. Borrar `self.state` aquí sería
        # correcto (ya revertimos los pesos), pero se mantiene la misma API
        # interna que first_step por consistencia.
        if zero_grad:
            self.base_optimizer.zero_grad(set_to_none=True)

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Limpia también `self.state`, donde `first_step` guarda los `e_w` para
        # el rollback posterior. Sin esto, un caller que invoque `zero_grad`
        # ENTRE `first_step` y `second_step` dejaría a `second_step` aplicando
        # un rollback obsoleto con pesos corruptos. `train.py` respeta el
        # contrato first_step(zero_grad=True) → second_step(zero_grad=False),
        # pero esta limpieza protege el invariante ante cualquier cambio.
        self.state.clear()
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        """Delegado al base_optimizer (SAM no guarda state recurrente)."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.base_optimizer.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _grad_norm(self) -> torch.Tensor:
        """
        Norma L2 global del gradiente acumulado en todos los param_groups.

        Usa el device del primer parámetro con gradiente como referencia (SAM
        debe soportar multi-GPU con DataParallel). Si no hay gradientes,
        devuelve tensor cero para evitar división por cero.
        """
        shared_device = None
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if shared_device is None:
                    shared_device = p.grad.device
                g = p.grad
                if self.adaptive:
                    # Escala por |w| para invarianza a reescalado
                    g = g * torch.abs(p)
                norms.append(g.norm(p=2).to(shared_device))
        if not norms:
            # Si todos los grads son None, `shared_device` queda en None. Se usa
            # el device del primer parámetro como fallback (no CPU): un tensor en
            # CPU provocaría un device mismatch con parámetros en GPU.
            fallback_device = next(
                (p.device for group in self.param_groups for p in group["params"]),
                "cpu",
            )
            return torch.zeros((), device=fallback_device)
        return torch.stack(norms).norm(p=2)
