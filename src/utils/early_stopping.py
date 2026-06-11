import torch
import numpy as np

from .safe_io import save_checkpoint_safe


class EarlyStopping:
    """
    Detiene el entrenamiento temprano si la métrica de validación no mejora.
    Esto previene el overfitting clínico: el modelo memoriza el train set pero
    pierde capacidad de generalizar a nuevos pacientes (test set).

    Selección de checkpoint — dos trackers separados
    -------------------------------------------------
    - ``best_score``: mejor métrica VISTA (para el contador de patience y la
      parada temprana). Avanza ante cualquier mejora, esté el save bloqueado o no.
    - ``best_saved_score``: métrica CORRESPONDIENTE al checkpoint que realmente
      está en disco. Solo avanza cuando se escribe un save.

    El split es necesario porque las épocas con el save bloqueado
    (``save_blocked=True`` cuando el health check de clases malignas falla)
    deben resetear el contador sin contaminar la decisión de guardado. Sin él,
    una época bloqueada que mejora la métrica "quemaría" el umbral de save y las
    épocas posteriores sanas con métrica inferior a ese máximo bloqueado no
    guardarían, aunque fueran mejores que el checkpoint en disco.

    Override del baseline forzado
    -----------------------------
    La primera época siempre guarda un baseline aunque falle el health check
    (red de seguridad para no quedarse sin ningún checkpoint). Cuando ese
    baseline inicial es forzado (no sano), la primera época sana posterior lo
    sobreescribe **siempre**, independientemente de la comparación de métricas:
    un modelo sano con métrica ligeramente menor es clínicamente preferible a
    uno no-sano (colapso de clase silencioso) con métrica aparentemente mejor.
    Tras el override se aplica la comparación estándar contra ``best_saved_score``.
    """

    def __init__(
        self,
        patience=5,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
        mode="min",
    ):
        """
        Args:
            patience (int): Cuántas épocas esperar después de la última mejora.
            verbose (bool): Si es True, imprime mensajes en cada mejora.
            delta (float): Cambio mínimo para ser considerado "mejora".
            path (str): Ruta donde guardar el mejor modelo.
            mode (str): "min" para minimizar loss, "max" para maximizar AUC.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None            # mejor métrica vista (counter)
        self.best_saved_score = None      # mejor métrica en disco (save)
        self._initial_save_done = False   # ya hubo un save inicial (forzado o no)
        self._initial_save_is_forced_baseline = False  # override pendiente
        # Métrica del baseline forzado cuando es el único checkpoint en disco.
        # Tras un baseline forzado, `best_saved_score` queda None para señalar el
        # override pendiente; sin este tracker, `best_for_comparison` caería a
        # `best_score` (mejor VISTA, incluidas épocas bloqueadas que avanzaron el
        # contador con métricas infladas) y el comparador SWA/EMA vs estándar
        # elegiría el checkpoint equivocado para el deployment clínico. Separa
        # "métrica realmente en disco" de "mejor métrica vista".
        self._forced_baseline_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode

    def _is_better(self, val_metric, reference):
        """Comparación unificada contra best_score o best_saved_score."""
        if reference is None:
            return True
        if self.mode == "min":
            return val_metric < reference - self.delta
        return val_metric > reference + self.delta

    @property
    def best_for_comparison(self):
        """Métrica del checkpoint que REALMENTE está en disco.

        Jerarquía de fuentes (de más a menos fiable):
        1. ``best_saved_score`` — hay un checkpoint sano en disco (caso normal).
        2. ``_forced_baseline_score`` — solo hay un baseline forzado (no sano)
           en disco (primera época con health check fallido y posteriores aún
           bloqueadas). No debe usarse ``best_score`` aquí, porque incluiría
           épocas bloqueadas con métricas infladas y el comparador SWA/EMA vs
           estándar desplegaría el checkpoint no-sano cuando un SWA sano con
           métrica inferior al "máximo bloqueado" debería ganar.
        3. ``best_score`` — último recurso degenerado (ni save ni baseline
           forzado). No alcanzable en el flujo normal de entrenamiento.
        """
        if self.best_saved_score is not None:
            return self.best_saved_score
        # Si solo hay baseline forzado en disco, usar esa métrica real, no
        # `best_score` (puede reflejar épocas bloqueadas con métrica inflada).
        if self._forced_baseline_score is not None:
            return self._forced_baseline_score
        return self.best_score

    def __call__(self, val_metric, model, save_blocked=False):
        """
        Args:
            val_metric: métrica de validación (AUC o loss).
            model: modelo a guardar. IMPORTANTE: pasar ``raw_model`` (sin
                DataParallel/DistributedDataParallel/torch.compile wrappers) —
                ``save_checkpoint`` desenvuelve los 3 wrappers como salvaguarda
                adicional, pero el contrato preferido es pasarlo ya desenvuelto.
            save_blocked: si True, el modelo puede haber mejorado la métrica
                pero NO se guarda checkpoint porque alguna clase maligna no pasa
                el health check (AUC < 0.80 o recall < 0.30). El counter SÍ se
                resetea si hay mejora (el training sigue buscando una época sana)
                pero el checkpoint no se persiste.
        """
        # --- 1. Actualizar best_score + counter (independiente del save) ---
        is_better_seen = self._is_better(val_metric, self.best_score)
        if self.best_score is None or is_better_seen:
            self.best_score = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True

        # --- 2. Decidir save (separado del counter) ---
        if save_blocked:
            if not self._initial_save_done:
                # La primera época SIEMPRE guarda un baseline (aunque el health
                # check falle) como red de seguridad. Se marca como baseline
                # forzado para que el primer checkpoint sano posterior lo
                # sobreescriba.
                self.trace_func(
                    f"BASELINE FORZADO: clase maligna con AUC/recall crítico. "
                    f"Métrica={val_metric:.4f} — guardando baseline forzado "
                    f"(será reemplazado por el primer checkpoint sano)."
                )
                self.save_checkpoint(val_metric, model)
                self._initial_save_done = True
                self._initial_save_is_forced_baseline = True
                # NO se actualiza best_saved_score → queda None como señal del
                # override pendiente. Se registra la métrica del baseline forzado
                # en un tracker propio para que `best_for_comparison` devuelva la
                # métrica realmente en disco, sin caer a `best_score` (que puede
                # subir con épocas bloqueadas posteriores).
                self._forced_baseline_score = val_metric
            else:
                # Épocas bloqueadas posteriores: solo log, no save.
                if is_better_seen:
                    self.trace_func(
                        f"CHECKPOINT BLOQUEADO: clase maligna con AUC/recall "
                        f"crítico. Métrica mejoró a {val_metric:.4f} pero "
                        f"checkpoint NO guardado (esperando una época sana)."
                    )
            return

        # --- No bloqueado: puede guardar ---
        if not self._initial_save_done:
            # Primera época y sana: guardado normal.
            self.save_checkpoint(val_metric, model)
            self._initial_save_done = True
            self.best_saved_score = val_metric
            return

        if self._initial_save_is_forced_baseline:
            # Override del baseline forzado (no sano) con el primer checkpoint
            # sano posterior, independientemente de la comparación de métricas:
            # un modelo sano es clínicamente preferible a uno no-sano con métrica
            # aparentemente mejor (colapso de clase silencioso).
            self.trace_func(
                f"REEMPLAZO DE BASELINE FORZADO: primer checkpoint sano con "
                f"val={val_metric:.4f} reemplaza el baseline forzado "
                f"(seguridad > métrica en la selección clínica)."
            )
            self.save_checkpoint(val_metric, model)
            self.best_saved_score = val_metric
            self._initial_save_is_forced_baseline = False
            # Tras el override, el baseline forzado ya no está en disco →
            # limpiar el tracker para que `best_for_comparison` use
            # `best_saved_score` (fuente primaria) en adelante.
            self._forced_baseline_score = None
            return

        # Comparación estándar contra el checkpoint actualmente en disco.
        if self._is_better(val_metric, self.best_saved_score):
            self.save_checkpoint(val_metric, model)
            self.best_saved_score = val_metric

    def save_checkpoint(self, val_metric, model):
        """Guarda el modelo cuando la métrica de validación mejora."""
        if self.verbose:
            metric_type = "Loss" if self.mode == "min" else "Metric"
            prev = self.best_saved_score
            prev_str = f"{prev:.6f}" if prev is not None else "None"
            self.trace_func(
                f"Validation {metric_type} improved ({prev_str} --> {val_metric:.6f}). "
                f"Saving {metric_type} model ..."
            )

        # Desenvolver `nn.DataParallel`/`DistributedDataParallel` para evitar el
        # prefijo 'module.' en los pesos. Esto previene crashes silenciosos
        # durante la inferencia en CPU/1-GPU.
        model_to_save = (
            model.module
            if isinstance(
                model,
                (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
            )
            else model
        )
        # Salvaguarda adicional para `torch.compile`: `OptimizedModule` envuelve
        # el modelo original en `_orig_mod`. Si un caller olvida pasar
        # `raw_model`, sin esto el state_dict se guardaría con prefijo
        # `_orig_mod.` → incompatible con el engine de inferencia.
        model_to_save = getattr(model_to_save, "_orig_mod", model_to_save)
        # Guardado atómico con validación `isfinite` (helper consolidado en
        # `src/utils/safe_io.py`, reutilizado por `train.py` para SWA/EMA/SOUP y
        # por este callsite sin cross-imports utils ↔ training). Sin él, el
        # best_checkpoint estándar quedaría vulnerable al mismo vector que
        # SWA/EMA/SOUP: una desconexión durante `torch.save` sobrescribiría el
        # checkpoint previo válido con uno corrupto (parcialmente escrito), o un
        # BN degenerado / AMP overflow persistiría NaN en disco como "best".
        save_checkpoint_safe(
            model_to_save.state_dict(),
            self.path,
            checkpoint_label="best_standard",
        )
