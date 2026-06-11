"""Test-time Entropy Minimization (TENT, Wang et al., ICLR 2021).

Adapta los parámetros affine (gamma, beta) de un subconjunto de capas
BatchNorm durante la inferencia, minimizando la entropía de la predicción
sobre el batch actual. Mejora la robustez ante cambios de dominio (cámara
distinta de la del conjunto de validación, cohorte o iluminación diferentes).

Paper: "Tent: Fully Test-time Adaptation by Entropy Minimization"
    https://arxiv.org/abs/2006.10726

Contratos clínicos:

- **Desactivado por defecto**: el `ClinicalInferenceEngine` lo expone como
  parámetro opcional (`tent_binary=False` / `tent_multiclass=False`). Con TENT
  desactivado, la inferencia es determinista bit a bit (mismo input produce el
  mismo output), requisito para la trazabilidad clínica.
- **Adaptación episódica**: se toma un snapshot de los parámetros objetivo al
  inicio de cada petición y se restaura al final. No hay aprendizaje continuo —
  la siguiente petición parte siempre del modelo original. El aprendizaje
  continuo rompería la trazabilidad clínica y permitiría que peticiones
  anteriores influyeran en la actual.
- **Solo parámetros affine de BN** (`weight`=gamma, `bias`=beta): `running_mean`
  y `running_var` permanecen inmutables (las capas BN operan en modo `eval()`
  durante la adaptación, usando estadísticas fijas). No se modifica nada más
  del modelo.
- **BN del backbone: adaptación adicional opcional** (`adapt_backbone=False` por
  defecto). Solo aplica a backbones CNN — DINOv2 (ViT) usa LayerNorm en el
  backbone, donde el TENT clásico no es aplicable. Adaptar el BN del backbone
  puede degradar la transferencia de los pesos preentrenados (ImageNet/JFT),
  de ahí que sea opcional.
- **Trazabilidad**: el adaptador expone `last_loss_pre`,
  `last_loss_post` y `last_steps_used` para que el caller (p. ej. el
  `ClinicalInferenceEngine`) los registre en el report. Sin estos campos, no
  se podría verificar a posteriori qué adaptación se aplicó en cada petición.

Limitaciones conocidas:

- Con TENT activo, la garantía formal del `TemperatureScaler` (calibrado antes
  de la adaptación) deja de cumplirse y la calibración puede quedar subóptima.
  La recalibración con TENT activo es una mejora futura (script aparte).
- TENT combinado con ensemble pierde coherencia matemática: cada modelo del
  ensemble se adaptaría de forma distinta. El engine emite `RuntimeWarning`
  cuando ambos están activos a la vez.
- TENT y MC Dropout se ordenan correctamente: primero TENT (BN objetivo en
  eval, gamma/beta entrenables) y después MC Dropout (Dropout en train, BN en
  eval con los pesos ya adaptados, N pases).
"""

import warnings
from contextlib import contextmanager
from typing import Iterable, List, Optional

import torch
import torch.nn as nn


# Default target modules — las 3 BatchNorm1d del trunk de SkinCancerModel,
# presentes en CNN y ViT. Verificadas en `architectures.py`:
#   - "img_bn"          BatchNorm1d(in_features)  — post-GeM/AttentionPool
#   - "meta_net.1"      BatchNorm1d(32)            — metadata MLP
#   - "classifier.2"    BatchNorm1d(128)           — classifier head
DEFAULT_TENT_TARGETS: tuple[str, ...] = ("img_bn", "meta_net.1", "classifier.2")


def _entropy_from_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    """Entropía media del batch en espacio probabilidad (no logit).

    Binario: entropía binaria de Bernoulli `H(p) = -p log p - (1-p) log(1-p)`,
        con `p = sigmoid(logit)` clampada a `[1e-7, 1-1e-7]` para evitar
        `log(0) = -inf`.
    Multiclase: entropía de Shannon `H(p) = -sum p_i log p_i` sobre softmax.

    Args:
        logits: shape `[B]` o `[B, 1]` (binario) o `[B, C]` (multiclase).
        mode: "binary" o "multiclass".

    Returns:
        Escalar (entropía media del batch). Para minimizar.
    """
    if mode == "binary":
        # Squeeze dim trailing 1 si existe → [B]
        l = logits.squeeze(-1) if logits.ndim == 2 and logits.shape[-1] == 1 else logits
        if l.ndim != 1:
            raise ValueError(
                f"TENT binario espera logits con shape [B] o [B, 1], "
                f"recibido {tuple(logits.shape)}."
            )
        p = torch.sigmoid(l).clamp(min=1e-7, max=1.0 - 1e-7)
        h = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return h.mean()
    elif mode == "multiclass":
        if logits.ndim != 2:
            raise ValueError(
                f"TENT multiclase espera logits con shape [B, C], "
                f"recibido {tuple(logits.shape)}."
            )
        log_p = torch.log_softmax(logits, dim=-1)
        p = log_p.exp()
        # Entropía: -sum p log p (por sample), promediado sobre batch.
        return -(p * log_p).sum(dim=-1).mean()
    else:
        raise ValueError(
            f"TENT: el mode debe ser 'binary' o 'multiclass', "
            f"recibido {mode!r}."
        )


class TENTAdapter:
    """Adaptador episodic de BN affine vía minimización de entropía (Wang ICLR 2021).

    Args:
        model: `SkinCancerModel` ya cargado en `eval()`.
        mode: "binary" o "multiclass". Determina cálculo de entropía.
        target_modules: nombres dot-notation de submódulos cuyo BN affine se
            adapta. Cada nombre debe resolver a una `nn.BatchNorm[1d|2d]` o
            `SyncBatchNorm`. Default `DEFAULT_TENT_TARGETS` (las 3 BN1d del
            trunk de `SkinCancerModel`, válidas tanto para CNN como ViT).
        adapt_backbone: si `True` y el backbone es CNN, también adapta
            TODOS los `BatchNorm2d` del backbone. Default `False` (más
            conservador clínicamente — adaptar backbone degrada transferencia
            ImageNet/JFT). Si el backbone es ViT (LayerNorm), emite
            `RuntimeWarning` y se ignora silenciosamente — TENT clásico no
            aplica a LayerNorm.
        lr: learning rate del SGD para los target params. Default `1e-3`.
            Se valida que esté en el rango `0 < lr < 1`.
        n_steps: pasos de optimización por `adapt_context()`. Default 1
            (Wang 2021 sugiere 1 paso típicamente suficiente). Max 10
            para forzar la asunción episodic — más pasos rompen el
            contrato y aproximan continual learning.

    Audit fields (poblados tras cada `adapt_context()`):
        last_loss_pre: entropía media del batch ANTES de adapt.
        last_loss_post: entropía media del batch DESPUÉS de adapt.
        last_steps_used: pasos efectivos. Si `< n_steps` indica abort por
            loss no-finita en algún paso.

    Uso clínico (vía `ClinicalInferenceEngine`):
        adapter = TENTAdapter(model, mode="binary")
        with adapter.adapt_context(image, metadata) as adapted_model:
            logits = adapted_model(image, metadata)
        # Aquí los pesos están restaurados (episodic).
        # adapter.last_loss_pre/post/steps_used disponibles para audit.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str,
        target_modules: Optional[Iterable[str]] = None,
        adapt_backbone: bool = False,
        lr: float = 1e-3,
        n_steps: int = 1,
    ):
        # Validación de configuración: la construcción se detiene con error si
        # los parámetros están fuera de rango, antes de adaptar nada.
        if mode not in ("binary", "multiclass"):
            raise ValueError(
                f"TENT: el mode debe ser 'binary' o 'multiclass', "
                f"recibido {mode!r}."
            )
        if not (0.0 < lr < 1.0):
            raise ValueError(
                f"tent_lr debe estar en (0, 1), recibido {lr}. "
                f"El valor por defecto 1e-3 es seguro; un lr alto desestabiliza "
                f"los parámetros affine de BN."
            )
        if not (1 <= n_steps <= 10):
            raise ValueError(
                f"tent_n_steps debe estar en [1, 10], recibido "
                f"{n_steps}. Más de 10 pasos rompe la asunción episódica — "
                f"recalibrar el modelo en lugar de hacer adaptación continua."
            )

        self.model = model
        self.mode = mode
        self.lr = float(lr)
        self.n_steps = int(n_steps)
        self.adapt_backbone = bool(adapt_backbone)

        if target_modules is None:
            target_modules = DEFAULT_TENT_TARGETS
        self.target_module_names: List[str] = list(target_modules)

        # Resolver nombres dot-notation a módulos vivos.
        self._target_bn_modules: List[nn.Module] = self._collect_target_bn_modules()

        # Si adapt_backbone=True, se añaden las capas BN del backbone. El backbone
        # construido por la factory expone `BackboneWrapper.kind` para discriminar
        # CNN vs ViT; el camino legacy (timm directo) no tiene `.kind` y solo se
        # usa con CNN. Por tanto: si `kind` es "cnn" o está ausente, se trata como
        # CNN; si `kind == "vit"`, se emite warning y se omite (el backbone ViT usa
        # LayerNorm, no BatchNorm).
        if self.adapt_backbone:
            backbone_obj = getattr(self.model, "backbone", None)
            backbone_kind = getattr(backbone_obj, "kind", None)
            if backbone_kind == "vit":
                warnings.warn(
                    "tent_adapt_backbone=True se ignora con backbone ViT. "
                    "DINOv2 usa LayerNorm en el backbone, no BatchNorm. El TENT "
                    "clásico no aplica al backbone ViT — la adaptación se "
                    "limita a los target_modules del trunk (img_bn, meta_net, "
                    "classifier).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                # backbone_kind in {"cnn", None} — CNN o camino legacy timm directo,
                # que solo se usa con CNN. Si el backbone no tiene BN (caso raro:
                # arquitectura sin BatchNorm), `_collect_backbone_bn_modules`
                # devuelve [] y el conjunto de targets queda como estaba. No se
                # detiene con error: adapt_backbone es una opción best-effort.
                self._target_bn_modules.extend(
                    self._collect_backbone_bn_modules()
                )

        if not self._target_bn_modules:
            raise RuntimeError(
                f"TENT no encontró ningún BatchNorm en target_modules="
                f"{self.target_module_names}. Verificar que el modelo es "
                f"`SkinCancerModel` y que las capas BN no fueron eliminadas "
                f"en un refactor."
            )

        # Identificar params adaptables: SOLO weight (gamma) y bias (beta).
        # NO running_mean/running_var (esos se preservan via BN.eval()).
        self._adaptable_params: List[nn.Parameter] = []
        for bn in self._target_bn_modules:
            if bn.weight is not None:
                self._adaptable_params.append(bn.weight)
            if bn.bias is not None:
                self._adaptable_params.append(bn.bias)

        # Audit fields (None hasta primera invocación de adapt_context).
        self.last_loss_pre: Optional[float] = None
        self.last_loss_post: Optional[float] = None
        self.last_steps_used: int = 0

    def _collect_target_bn_modules(self) -> List[nn.Module]:
        """Resuelve nombres dot-notation a módulos. Lanza error si no es BN."""
        modules: List[nn.Module] = []
        for name in self.target_module_names:
            mod = self.model
            for part in name.split("."):
                if not hasattr(mod, part):
                    raise AttributeError(
                        f"TENT target_module '{name}' no existe en el "
                        f"modelo (parte '{part}' no encontrada). El modelo "
                        f"podría haber sido refactorizado — actualizar "
                        f"`DEFAULT_TENT_TARGETS` en `src/inference/tent.py`."
                    )
                mod = getattr(mod, part)
            if not isinstance(
                mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
            ):
                raise TypeError(
                    f"TENT target_module '{name}' no es BatchNorm "
                    f"(es {type(mod).__name__}). LayerNorm/GroupNorm requieren "
                    f"variantes de TENT no implementadas (p. ej. TENT-LN o MEMO)."
                )
            modules.append(mod)
        return modules

    def _collect_backbone_bn_modules(self) -> List[nn.Module]:
        """Para `adapt_backbone=True` con CNN: todos los BN del backbone."""
        backbone = getattr(self.model, "backbone", None)
        if backbone is None:
            return []
        return [
            m
            for m in backbone.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
        ]

    def _snapshot(self) -> List[torch.Tensor]:
        """Snapshot de los params adaptables (clone). Lista paralela a
        `self._adaptable_params` — el orden importa para `_restore()`.
        """
        return [p.detach().clone() for p in self._adaptable_params]

    def _restore(self, snapshot: List[torch.Tensor]) -> None:
        """Restaura los params adaptables desde snapshot. Idempotente."""
        with torch.no_grad():
            for p, s in zip(self._adaptable_params, snapshot):
                p.copy_(s)

    @contextmanager
    def _bn_modes(self):
        """Context manager: pone target BN en `eval()` para que usen running
        stats fijos (NO batch stats), preservando el invariante TENT de no
        modificar `running_mean/running_var`. Restaura modos originales al
        salir.

        Importante: `eval()` mode NO desactiva los gradientes a `weight`/`bias`.
        Los params siguen entrenables — la diferencia eval vs train es solo
        si BN usa `running_stats` (eval) o `batch stats` con update de
        `running_*` (train).
        """
        # Snapshot training modes de TODOS los módulos (no solo target).
        # Esto permite restaurar exactamente el estado original.
        original_modes = {id(m): m.training for m in self.model.modules()}
        try:
            # Forzar todo a eval (nadie acumula batch stats durante TENT).
            self.model.eval()
            yield
        finally:
            # Restaurar modos originales.
            for m in self.model.modules():
                m.train(original_modes.get(id(m), False))

    @contextmanager
    def adapt_context(self, image: torch.Tensor, metadata: torch.Tensor):
        """Context manager: adapta los target params via TENT y los restaura
        al salir. Yields el modelo adaptado para uso por el caller.

        Args:
            image: tensor de entrada `[B, 3, H, W]`. En deployment clínico,
                típicamente `B=1`.
            metadata: tensor de metadata `[B, 3]` (`[age_norm, sex_code,
                site_code]`). El forward de `SkinCancerModel` requiere
                metadata si se construyó con `use_metadata=True` — el
                caller debe garantizarlo.

        Yields:
            El mismo `self.model` con pesos adaptados in-place. Al salir
            del `with`, los pesos se restauran al estado pre-adapt.

        Side effects:
            - Modifica los target params durante el `with` (revierte al salir).
            - Pone el modelo en `eval()` durante el `with` (restaura al salir).
            - Pone los target params en `requires_grad=True` durante el `with`
              (restaura al salir).
            - Optimizer SGD interno se construye y descarta.
            - Audit fields (`last_loss_pre`, `last_loss_post`, `last_steps_used`)
              se actualizan al salir.

        Comportamiento ante errores:
            - Si entropía produce NaN/Inf en algún paso: warning + abort
              + restore. `last_steps_used` < `n_steps`.
            - Si excepción durante el yield del caller: restore garantizado
              en el `finally`.
        """
        snapshot = self._snapshot()

        # Snapshot del estado requires_grad de TODOS los params (no solo los
        # objetivo). Bajo torch.enable_grad() o torch.no_grad() el estado actual
        # puede ser arbitrario; para no afectar otras operaciones del engine
        # (MC Dropout, explicabilidad) se restaura exactamente al salir.
        prev_grad_states = {
            id(p): p.requires_grad for p in self.model.parameters()
        }
        # Frozen todos los params, luego activar solo los target.
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self._adaptable_params:
            p.requires_grad_(True)

        # SGD sin momentum sobre target params. SGD vs Adam: SGD es
        # determinista simple, sin estado interno entre invocaciones (Adam
        # mantiene running averages que requerirían snapshot adicional).
        # Para 1-3 pasos de adaptación affine, SGD es suficiente.
        optimizer = torch.optim.SGD(
            self._adaptable_params, lr=self.lr, momentum=0.0
        )

        steps_done = 0
        loss_pre = float("nan")
        loss_post = float("nan")
        try:
            with self._bn_modes():
                # Forward inicial para audit (entropía pre-adapt).
                with torch.no_grad():
                    logits_pre = self.model(image, metadata)
                    h_pre = _entropy_from_logits(logits_pre, self.mode)
                    if torch.isfinite(h_pre):
                        loss_pre = float(h_pre.item())

                # Loop de adaptación.
                for step in range(self.n_steps):
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(image, metadata)
                    loss = _entropy_from_logits(logits, self.mode)
                    if not torch.isfinite(loss):
                        warnings.warn(
                            f"TENT paso {step+1}/{self.n_steps} "
                            f"produjo una loss no finita ({float(loss.item())}). "
                            f"Se aborta TENT y se restaura el snapshot. Revisar "
                            f"AMP autocast, pesos del modelo o inputs.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        break
                    loss.backward()
                    optimizer.step()
                    steps_done = step + 1

                # Forward final para audit (entropía post-adapt).
                with torch.no_grad():
                    logits_post = self.model(image, metadata)
                    h_post = _entropy_from_logits(logits_post, self.mode)
                    if torch.isfinite(h_post):
                        loss_post = float(h_post.item())

                # Si abortamos, mantenemos los pesos adaptados de los pasos
                # válidos previos (steps_done indica cuántos). Pero el restore
                # final SIEMPRE devuelve al snapshot — episodic strict.
                yield self.model
        finally:
            # Restaurar grad states + pesos snapshot. Estricto episodic.
            self._restore(snapshot)
            for p in self.model.parameters():
                p.requires_grad_(prev_grad_states.get(id(p), False))
            # Update audit fields (visibles al caller post-yield).
            self.last_loss_pre = loss_pre
            self.last_loss_post = loss_post
            self.last_steps_used = steps_done


__all__ = [
    "TENTAdapter",
    "DEFAULT_TENT_TARGETS",
]
