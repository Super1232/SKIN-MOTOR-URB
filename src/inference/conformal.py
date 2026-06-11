"""
Conformal Prediction para inferencia clínica con garantía estadística de
cobertura marginal.

Motivación clínica
------------------
Un modelo calibrado (Temperature Scaling) produce probabilidades fiables EN
PROMEDIO, pero no garantiza que cualquier paciente individual esté cubierto
por una región de confianza con una tasa de error especificada. Conformal
Prediction (Vovk et al. 2005; Romano et al. 2020 para APS) resuelve esto:

    "Para un nivel de error α=0.05, el set predicho contendrá la clase real
     en ≥95% de los casos (cobertura marginal), sin suposiciones sobre la
     distribución."

La única hipótesis es el intercambio (exchangeability) entre calibración y
test — que nuestro val split satisface por diseño (StratifiedGroupKFold).

Dos variantes implementadas
---------------------------

1. **Split Conformal (binario)**: para el modelo de triaje. Score
   `1 - p_true_class`. Se calcula un quantile global sobre el val set y en
   test se predice el set `{k : score(x, k) ≤ q}`. Para binario el set puede
   ser `{benigno}`, `{maligno}` o `{benigno, maligno}` (ambiguo → AMBER).

2. **APS (Adaptive Prediction Sets)**: para el especialista multiclase.
   Score acumulativo sobre probabilidades ordenadas descendentemente; el set
   resultante se adapta a la dificultad del input (más clases en casos
   ambiguos). Romano et al., NeurIPS 2020.

Uso en predict.py
-----------------
```python
# 1) Entrenar/calibrar conformal sobre el val set (una vez).
calib = BinaryConformal(alpha=0.05)
calib.fit(val_probs, val_labels)
calib.save("models/checkpoints/conformal_binary.pt")

# 2) Inferencia por imagen.
pred_set = calib.predict(test_prob)  # e.g. {'benigno'} o {'benigno','maligno'}
```

Garantía
--------
Si val y test son intercambiables, P(y_test ∈ pred_set) ≥ 1 - α. Cuidado con
domain shift (hospital nuevo, cámara distinta) — la garantía se degrada y es
necesario recalibrar sobre datos del nuevo dominio.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch


def _finite_sample_quantile(n: int, alpha: float) -> float:
    """
    Quantile conformal corregido para muestras finitas.

    En un calib set de tamaño n, para alcanzar cobertura marginal ≥ 1-α se
    usa el quantile de índice ⌈(n+1)(1-α)⌉ / n. Este cap a 1.0 evita
    excepciones cuando (n+1)(1-α) > n (α muy pequeño o n muy chico).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha debe estar en (0, 1), recibido {alpha}")
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    return float(min(q_level, 1.0))


class BinaryConformal:
    """
    Split Conformal Prediction para el triaje binario.

    Score de nonconformity: `1 - p_verdadera_clase`. Imagen con p(maligno)=0.9
    y label=maligno → score=0.1 (fácil). p(maligno)=0.1 y label=maligno →
    score=0.9 (difícil).

    Atributos tras `fit()`:
        alpha: nivel de error objetivo (ej. 0.05 → 95% cobertura).
        threshold: quantile del score sobre el calib set. Clases con
            `score(x, k) ≤ threshold` entran en el set predicho.
        n_calib: tamaño del calib set (para auditoría).
    """

    def __init__(self, alpha: float = 0.05):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha debe estar en (0, 1), recibido {alpha}")
        self.alpha = alpha
        self.threshold: float | None = None
        self.n_calib: int = 0
        # Hash SHA256 del val CSV usado durante la calibración. Lo asigna
        # `scripts/calibrate_conformal.py` tras `fit()` y lo persiste `save()`.
        # `predict.py` compara el hash al cargar el predictor; si diverge, emite
        # `RuntimeWarning` (no detiene la ejecución: un warning capturable por el
        # handler del servicio permite una degradación controlada sin interrumpir
        # la producción clínica en caliente).
        self.val_csv_hash: str = ""

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            probs: [N] — probabilidades P(maligno) ya calibradas (tras temperature).
            labels: [N] — etiquetas binarias en {0, 1}.
        """
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if probs.shape != labels.shape:
            raise ValueError(
                f"probs y labels deben tener misma forma, recibido "
                f"{probs.shape} vs {labels.shape}"
            )
        if probs.size == 0:
            raise ValueError("calib set vacío")
        # Validación de dominios que detiene la ejecución ante entradas inválidas.
        # Sin esto, labels con valores distintos de {0,1} (un sentinel -1 por NaN,
        # un bug upstream con 2, label smoothing mal filtrado) producirían scores
        # corruptos -> threshold conformal roto -> cobertura real muy por debajo
        # del 97% sin aviso. Es el artefacto central de seguridad del triaje.
        if not np.isfinite(probs).all():
            raise ValueError("BinaryConformal.fit: probs contiene NaN/Inf")
        unique_labels = np.unique(labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"BinaryConformal.fit: labels debe contener solo {{0, 1}}, "
                f"encontrado {unique_labels.tolist()}"
            )

        # Score por sample: 1 - p(true_class)
        p_true = np.where(labels == 1, probs, 1.0 - probs)
        scores = 1.0 - p_true
        q_level = _finite_sample_quantile(n=scores.size, alpha=self.alpha)
        # `np.quantile` con el método por defecto es suficiente para conformal;
        # q_level ya incorpora la corrección (n+1)(1-α)/n.
        self.threshold = float(np.quantile(scores, q_level, method="higher"))
        self.n_calib = int(scores.size)

    def predict(self, prob_malignant: float) -> List[str]:
        """
        Devuelve el set conformal para una única predicción.

        Args:
            prob_malignant: P(maligno) calibrada ∈ [0, 1].
        Returns:
            Lista con 1 o 2 elementos en {"benigno", "maligno"}. Si contiene
            ambos, la imagen cae en zona ambigua → estado clínico AMBER.
        """
        if self.threshold is None:
            raise RuntimeError("BinaryConformal no calibrado — llama a fit() primero")
        # Se re-valida alpha en predict(). El constructor y load ya validan alpha
        # y la finitud del threshold, pero un caller que mute `self.alpha` tras
        # fit() dejaría el threshold congelado e incoherente con el nivel
        # declarado: el report clínico diría "97% cobertura" sin garantía real.
        if not (0.0 < self.alpha < 1.0):
            raise RuntimeError(
                f"BinaryConformal.alpha inválido ({self.alpha!r}), "
                f"debe estar en (0, 1). Recalibrar el predictor."
            )
        # Validación de rango de la entrada. El contrato público es
        # `prob ∈ [0, 1]`. Sin esta comprobación, un bug upstream (vector scaling
        # corrupto, calibración degenerada, o un caller que envíe `prob=1.5`)
        # pasaría inadvertido: `(1 - 1.5) <= threshold` podría evaluar a True por
        # accidente y reportar un set incorrecto como AMBER en lugar de detenerse.
        # Se usa una validación tolerante al tipo: `isinstance(..., (int, float))`
        # rechazaría `np.float32` y tensores torch 0-d (habituales en pipelines de
        # deep learning), porque solo `np.float64` hereda de `float` nativo. El
        # patrón es: (a) rechazar str/bytes explícitamente (un string numérico
        # como "0.3" suele ser un bug upstream — JSON sin cast, CSV crudo, input
        # de UI sin validar — que `float("0.3")` convertiría en silencio);
        # (b) try/except `float()` para aceptar Python int/float, np.floating,
        # np.integer y tensores torch 0-d; (c) comprobar isnan/isinf sobre el
        # valor ya convertido.
        if isinstance(prob_malignant, (str, bytes, bytearray)):
            raise ValueError(
                f"BinaryConformal.predict requiere prob_malignant "
                f"numérico (Python int/float, np.floating, torch 0-d), recibido "
                f"{type(prob_malignant).__name__}. Verificar que el caller aplicó "
                "la conversión a número (no pasar strings crudos del request HTTP)."
            )
        try:
            prob_value = float(prob_malignant)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"BinaryConformal.predict requiere prob_malignant "
                f"numérico (Python int/float, np.floating, torch 0-d), recibido "
                f"{type(prob_malignant).__name__}."
            ) from e
        if math.isnan(prob_value) or math.isinf(prob_value):
            raise ValueError(
                f"BinaryConformal.predict requiere prob_malignant "
                f"finito, recibido {prob_value}."
            )
        if not (0.0 <= prob_value <= 1.0):
            raise ValueError(
                f"BinaryConformal.predict requiere "
                f"prob_malignant ∈ [0, 1], recibido {prob_value}. "
                "Verificar calibración upstream (TemperatureScaler) y que "
                "el caller pase la sigmoide aplicada, no logits crudos."
            )
        # Score conformal por clase candidata: 1 - p(clase).
        #   maligno: score = 1 - p(maligno)             → incluir si ≤ threshold
        #   benigno: score = 1 - p(benigno) = p(maligno) → incluir si ≤ threshold
        # Incluir una clase ⇔ el modelo le asigna masa suficiente para caer
        # dentro del set conformal calibrado al nivel 1-alpha.
        # Se usa `prob_value` (float puro) en vez de `prob_malignant` para que la
        # aritmética sea consistente con independencia del tipo de entrada.
        result: List[str] = []
        if (1.0 - prob_value) <= self.threshold:
            result.append("maligno")
        if prob_value <= self.threshold:
            result.append("benigno")
        # Edge case: si ambos scores exceden el threshold (input muy raro,
        # probabilidades ~0.5 con threshold muy bajo), devolvemos ambos como
        # fallback conservador en vez de un set vacío. Set vacío en conformal
        # es señal de dominio fuera de distribución; clínicamente lo tratamos
        # como AMBER (revisión humana).
        if not result:
            result = ["benigno", "maligno"]
        return result

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "alpha": self.alpha,
                "threshold": self.threshold,
                "n_calib": self.n_calib,
                "val_csv_hash": self.val_csv_hash,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BinaryConformal":
        state = torch.load(path, map_location="cpu", weights_only=True)
        threshold_val = state["threshold"]
        if threshold_val is None or not math.isfinite(float(threshold_val)):
            raise ValueError(
                f"Conformal checkpoint corrupto: threshold={threshold_val!r} no es finito"
            )
        obj = cls(alpha=float(state["alpha"]))
        obj.threshold = float(threshold_val)
        obj.n_calib = int(state["n_calib"])
        # Compatible con checkpoints antiguos que no incluían `val_csv_hash`.
        obj.val_csv_hash = str(state.get("val_csv_hash", ""))
        return obj


class APSConformal:
    """
    Adaptive Prediction Sets (Romano et al., NeurIPS 2020) para multiclase.

    Para cada sample, ordena las clases por probabilidad descendente, acumula
    masas y el score de nonconformity de la clase verdadera es la masa
    acumulada HASTA incluir la clase verdadera. El quantile sobre el calib
    set define un threshold adaptativo: en test, se incluye en el set toda
    clase necesaria para que la masa acumulada alcance el threshold.

    Ventaja sobre el conformal clásico con score `1-p`: los sets se adaptan
    al "shape" de la distribución — imágenes confiadas (argmax cerca de 1)
    producen sets de 1 clase, imágenes ambiguas producen sets de 2-3 clases.

    Randomización
    -------------
    El paper original usa randomización U(0,1) sobre la masa justo antes de
    llegar a la clase verdadera para garantizar cobertura exacta. Aquí
    usamos la variante determinista (sin randomización) que sobrecubre
    ligeramente — preferible en contexto clínico donde determinismo importa.
    """

    def __init__(self, alpha: float = 0.05, class_names: Sequence[str] | None = None):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha debe estar en (0, 1), recibido {alpha}")
        self.alpha = alpha
        self.class_names = list(class_names) if class_names is not None else None
        self.threshold: float | None = None
        self.n_calib: int = 0
        # Hash SHA256 del val CSV usado durante la calibración (igual que en
        # BinaryConformal). Ver su docstring para la semántica.
        self.val_csv_hash: str = ""

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            probs: [N, C] — probabilidades calibradas por clase.
            labels: [N] — índices de clase verdaderos en [0, C-1].
        """
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if probs.ndim != 2:
            raise ValueError(f"probs debe ser 2D [N, C], recibido shape={probs.shape}")
        if probs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"N mismatch: probs={probs.shape[0]}, labels={labels.shape[0]}"
            )

        n, num_classes = probs.shape
        if self.class_names is not None and len(self.class_names) != num_classes:
            raise ValueError(
                f"class_names tiene {len(self.class_names)} elementos pero "
                f"probs tiene {num_classes} clases"
            )
        # Validación que detiene la ejecución, simétrica a la de BinaryConformal.
        # Probs con NaN (batch corrupto en el val set durante la calibración)
        # produciría un threshold APS NaN: en inferencia, `cumsum >= NaN` es False
        # y el set incluiría SIEMPRE todas las clases — cobertura inflada pero sin
        # valor informativo.
        if not np.isfinite(probs).all():
            raise ValueError("APSConformal.fit: probs contiene NaN/Inf")
        if labels.size > 0:
            lo, hi = int(labels.min()), int(labels.max())
            if lo < 0 or hi >= num_classes:
                raise ValueError(
                    f"APSConformal.fit: labels fuera de rango [0, {num_classes - 1}]: "
                    f"min={lo}, max={hi}"
                )
        # Ordenar cada fila por probabilidad descendente.
        order = np.argsort(-probs, axis=1)  # [N, C]
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_probs, axis=1)  # masa acumulada

        # Para cada sample, encontrar la posición de la clase verdadera en el
        # ranking ordenado, y tomar la masa acumulada HASTA esa posición.
        rank_of_true = np.argmax(order == labels[:, None], axis=1)  # [N]
        scores = cum[np.arange(n), rank_of_true]  # [N]

        q_level = _finite_sample_quantile(n=n, alpha=self.alpha)
        self.threshold = float(np.quantile(scores, q_level, method="higher"))
        self.n_calib = n

    def predict(self, probs: np.ndarray) -> List[int]:
        """
        Args:
            probs: [C] o [1, C] — probabilidades calibradas de una imagen.
        Returns:
            Lista de índices de clase en el prediction set.
        """
        if self.threshold is None:
            raise RuntimeError("APSConformal no calibrado — llama a fit() primero")
        # Se re-valida alpha en predict(), igual que en BinaryConformal. Un alpha
        # mutado tras fit() dejaría el threshold y el nivel declarado
        # inconsistentes → reporte multiclase con garantía inválida.
        if not (0.0 < self.alpha < 1.0):
            raise RuntimeError(
                f"APSConformal.alpha inválido ({self.alpha!r}), "
                f"debe estar en (0, 1). Recalibrar el predictor."
            )
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        # Comprobación de NaN en inferencia. Un probs con NaN/Inf produciría un
        # set no determinista (`cumsum >= threshold_nan` es False) e incluiría
        # ninguna o todas las clases según el orden.
        if not np.isfinite(probs).all():
            raise ValueError("APSConformal.predict: probs contiene NaN/Inf")
        # Validación de rango y normalización. APS asume probs que suman ~1
        # (cumsum monotónico hasta un threshold ≤ 1). Probs no normalizadas (drift
        # numérico tras Vector Scaling, softmax mal aplicado upstream, o un caller
        # que pase logits) producirían sets matemáticamente inválidos: un
        # cumsum=0.9 nunca alcanzaría threshold=0.95 y el set incluiría todas las
        # clases por agotamiento del bucle, violando la garantía conformal sin
        # aviso. Política: rechazar valores fuera de [0, 1], auto-renormalizar el
        # drift menor (sum ∈ [0.95, 1.05]) con warning, y detenerse ante drift
        # mayor.
        if (probs < 0.0).any() or (probs > 1.0).any():
            raise ValueError(
                f"APSConformal.predict requiere probs ∈ [0, 1], "
                f"recibido min={probs.min():.6f}, max={probs.max():.6f}. "
                "Verificar que el caller aplica softmax/sigmoid antes "
                "de llamar predict (no pasar logits crudos)."
            )
        _sum = float(probs.sum())
        if not math.isclose(_sum, 1.0, abs_tol=5e-2):
            raise ValueError(
                f"APSConformal.predict requiere probs.sum() ≈ 1, "
                f"recibido sum={_sum:.6f} (drift > 5%). Verificar softmax "
                "upstream y que num_classes coincide con la longitud del array."
            )
        if not math.isclose(_sum, 1.0, abs_tol=1e-5):
            warnings.warn(
                f"APSConformal.predict probs.sum()={_sum:.6f} "
                "(drift menor) — auto-renormalizando antes del cumsum.",
                RuntimeWarning,
                stacklevel=2,
            )
            probs = probs / _sum
        order = np.argsort(-probs)
        cum = 0.0
        chosen: List[int] = []
        for idx in order:
            cum += probs[idx]
            chosen.append(int(idx))
            if cum >= self.threshold:
                break
        return chosen

    def predict_named(self, probs: np.ndarray) -> List[str]:
        """
        Wrapper que traduce índices a nombres de clase si `class_names` se
        proporcionó en el constructor.
        """
        idxs = self.predict(probs)
        if self.class_names is None:
            return [str(i) for i in idxs]
        return [self.class_names[i] for i in idxs]

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "alpha": self.alpha,
                "threshold": self.threshold,
                "n_calib": self.n_calib,
                "class_names": self.class_names,
                "val_csv_hash": self.val_csv_hash,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "APSConformal":
        state = torch.load(path, map_location="cpu", weights_only=True)
        threshold_val = state["threshold"]
        if threshold_val is None or not math.isfinite(float(threshold_val)):
            raise ValueError(
                f"Conformal checkpoint corrupto: threshold={threshold_val!r} no es finito"
            )
        loaded_names = state.get("class_names")
        obj = cls(alpha=float(state["alpha"]), class_names=loaded_names)
        obj.threshold = float(threshold_val)
        obj.n_calib = int(state["n_calib"])
        # Compatible con checkpoints antiguos que no incluían `val_csv_hash`.
        obj.val_csv_hash = str(state.get("val_csv_hash", ""))
        if loaded_names is not None and len(loaded_names) < 2:
            raise ValueError(
                f"Conformal checkpoint corrupto: class_names tiene "
                f"{len(loaded_names)} elementos (minimo 2)"
            )
        return obj


class ConformalRiskControl:
    """Conformal Risk Control para controlar directamente la tasa de falsos negativos.

    Referencia: Angelopoulos et al., "Conformal Risk Control", arXiv 2022.
    https://arxiv.org/abs/2208.02814

    Motivación clínica: el Split Conformal (`BinaryConformal`) garantiza
    cobertura marginal `P(y ∈ pred_set) ≥ 1 - α`. Pero en detección de
    melanoma, lo que importa clínicamente es la **tasa de falsos negativos**
    de la clase maligna (FNR): de todos los melanomas reales, ¿qué fracción
    perdió el sistema? El FNR es el riesgo directo que queremos acotar.

    Conformal Risk Control extiende la garantía a cualquier función de
    riesgo monótona. Para el caso binario con threshold de decisión `t`,
    definimos el FNR empírico como:

        L(t) = (1/N_pos) · sum_{y_i=1} I[prob_i < t]

    monótona NO-CRECIENTE en `t` (menor threshold → más positivos → menor FNR).

    Algoritmo (variante simplificada finite-sample Split Conformal, no
    exactamente la fórmula del Angelopoulos 2022 Theorem 1 — ver nota):
    1. Ordenar las probabilidades de las muestras POSITIVAS (malignas) en
       orden ascendente.
    2. Tomar el cuantil `floor((n_pos + 1) * target_fnr)` como threshold.
    3. En inferencia: predecir "maligno" si `prob >= threshold`.

    **Nota de precisión del algoritmo (importante para revisión clínica):**
    Angelopoulos et al. 2022 Theorem 1 prueba la garantía para la elección
    `λ = inf{λ : (n·L̂(λ) + B)/(n+1) ≤ α}` con `B` un upper bound de la
    pérdida (B=1 para FNR). Esta implementación usa el quantile finite-sample
    `floor((n+1)·α)` que es la forma canónica del Split Conformal de Vovk,
    NO exactamente la fórmula extendida del paper. Para `α ≥ 0.05` los
    valores numéricos resultantes difieren en menos de `1/n` → efecto
    clínico despreciable en sets de calibración realistas (n>500). Si se
    necesita la garantía formal estricta del paper, sustituir `idx` por
    la búsqueda binaria sobre la lambda que satisface la desigualdad
    `(n·L̂ + 1)/(n+1) ≤ target_fnr`.

    Garantía aproximada: `E_{cal, test}[FNR_test] ≈ target_fnr` bajo
    exchangeability, con desviación `O(1/n)` respecto al bound formal.

    Cuándo usar vs BinaryConformal:
    - `BinaryConformal` → cobertura de set → bueno para "no lo sé" ambiguo.
    - `ConformalRiskControl` → control directo FNR → bueno para elegir un
      umbral de decisión clínicamente conservador con garantía formal.
    """

    def __init__(self, target_fnr: float = 0.1):
        if not (0.0 < target_fnr < 1.0):
            raise ValueError(
                f"target_fnr debe estar en (0, 1), recibido {target_fnr}"
            )
        self.target_fnr = float(target_fnr)
        self.threshold: float | None = None
        self.n_calib_pos: int = 0
        self.n_calib_neg: int = 0
        self.val_csv_hash: str = ""

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Calibra el threshold sobre el conjunto de calibración."""
        probs = np.asarray(probs, dtype=np.float64).ravel()
        labels = np.asarray(labels).astype(int).ravel()

        if not np.isfinite(probs).all():
            raise ValueError(
                "ConformalRiskControl.fit: probs contiene valores no-finitos"
            )
        unique_labels = np.unique(labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"ConformalRiskControl.fit: labels deben ser 0/1, "
                f"recibido uniques={unique_labels.tolist()}"
            )
        if probs.shape != labels.shape:
            raise ValueError(
                f"ConformalRiskControl.fit: shape mismatch "
                f"probs={probs.shape}, labels={labels.shape}"
            )

        pos_mask = labels == 1
        n_plus = int(pos_mask.sum())
        if n_plus < 2:
            raise ValueError(
                f"ConformalRiskControl.fit: se requieren ≥2 muestras "
                f"positivas, solo {n_plus}"
            )

        pos_probs = np.sort(probs[pos_mask])
        idx = int(np.floor((n_plus + 1) * self.target_fnr))
        idx = max(0, min(idx, n_plus - 1))
        self.threshold = float(pos_probs[idx])
        self.n_calib_pos = n_plus
        self.n_calib_neg = int((labels == 0).sum())

    def predict(self, prob: float) -> str:
        """Predice "maligno" si `prob >= threshold`, else "benigno"."""
        if self.threshold is None:
            raise RuntimeError(
                "ConformalRiskControl: llame `fit()` antes de `predict()`"
            )
        # Se re-valida target_fnr en predict(), igual que alpha en
        # BinaryConformal/APSConformal. Mutarlo tras fit() dejaría el threshold
        # incoherente con el FNR declarado → riesgo de auditoría clínica (la
        # autoridad sanitaria exige un FNR acotado, no solo declarado).
        if not (0.0 < self.target_fnr < 1.0):
            raise RuntimeError(
                f"ConformalRiskControl.target_fnr inválido "
                f"({self.target_fnr!r}), debe estar en (0, 1). Recalibrar."
            )
        # Validación de la entrada, simétrica a BinaryConformal.predict. Sin
        # guards, `float(prob) >= self.threshold` esconde fallos graves: `float(nan)
        # >= X` es False → retornaría "benigno" en silencio, de modo que un modelo
        # que devuelve NaN (overflow AMP, GeM degenerado, calibración corrupta)
        # produciría "benigno" para un posible melanoma sin señal en el audit
        # trail. `float(inf) >= X` es True → falso positivo "maligno". Los strings
        # numéricos ("0.5") serían aceptados por float() ocultando bugs upstream, y
        # probs fuera de [0, 1] (calibración corrupta) también pasarían. Esta
        # validación es crítica porque este predictor alimenta el flag clínico
        # `defer_to_human`, y su fallo silencioso invalidaría toda la cascada de
        # derivación humana. Se aplican los mismos tres checks que en
        # BinaryConformal.predict.
        if isinstance(prob, (str, bytes, bytearray)):
            raise ValueError(
                f"ConformalRiskControl.predict requiere prob numérico "
                f"(Python int/float, np.floating, torch 0-d), recibido "
                f"{type(prob).__name__}. Verificar que el caller aplicó la "
                "conversión a número (no pasar strings crudos del request HTTP)."
            )
        try:
            prob_value = float(prob)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"ConformalRiskControl.predict requiere prob numérico, "
                f"recibido {type(prob).__name__}."
            ) from e
        if math.isnan(prob_value) or math.isinf(prob_value):
            raise ValueError(
                f"ConformalRiskControl.predict requiere prob finito, "
                f"recibido {prob_value}. Un NaN aquí sería catastrófico: "
                "`NaN >= threshold` es False → retorna 'benigno' → melanoma "
                "perdido silenciosamente. Verificar pipeline AMP/calibración upstream."
            )
        if not (0.0 <= prob_value <= 1.0):
            raise ValueError(
                f"ConformalRiskControl.predict requiere prob ∈ [0, 1], "
                f"recibido {prob_value}. Verificar calibración upstream "
                "(TemperatureScaler con clamp [0.3, 5.0]) y que el "
                "caller pase la sigmoide aplicada, no logits crudos."
            )
        return "maligno" if prob_value >= self.threshold else "benigno"

    def save(self, path: str | Path) -> None:
        if self.threshold is None:
            raise RuntimeError("ConformalRiskControl: no se puede guardar sin fit")
        torch.save(
            {
                "target_fnr": self.target_fnr,
                "threshold": self.threshold,
                "n_calib_pos": self.n_calib_pos,
                "n_calib_neg": self.n_calib_neg,
                "val_csv_hash": self.val_csv_hash,
                "__class__": "ConformalRiskControl",
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConformalRiskControl":
        state = torch.load(path, map_location="cpu", weights_only=True)
        threshold_val = state["threshold"]
        if threshold_val is None or not math.isfinite(float(threshold_val)):
            raise ValueError(
                f"ConformalRiskControl checkpoint corrupto: "
                f"threshold={threshold_val!r} no es finito"
            )
        obj = cls(target_fnr=float(state["target_fnr"]))
        obj.threshold = float(threshold_val)
        obj.n_calib_pos = int(state.get("n_calib_pos", 0))
        obj.n_calib_neg = int(state.get("n_calib_neg", 0))
        obj.val_csv_hash = str(state.get("val_csv_hash", ""))
        return obj
