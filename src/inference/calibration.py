"""
Calibración de Temperatura para Confianza Clínica.

Las redes neuronales son notoriamente sobreconfiadas: un modelo que predice 90% melanoma
puede tener una precisión real del 70%. Temperature Scaling (Guo et al. 2017) aprende un
escalar T en el validation set tal que sigmoid(logits/T) o softmax(logits/T) esté
calibrado — la probabilidad predicha coincide con la frecuencia real de acierto.

Esto es CRÍTICO para que el umbral clínico de 0.20 sea significativo.

`PerSiteThresholdCalibrator` añade la calibración del threshold de decisión
RED/GREEN por sitio anatómico, complementaria a `TemperatureScaler` (que opera
sobre las probabilidades). El threshold global de 0.20 es el punto
de operación del clasificador binario, optimizado para una lesión promedio con
prevalencia maligna ~3%. Para sitios con prevalencia extrema (palmas/plantas 58%
maligno; tronco posterior 0.6%) el threshold óptimo difiere; calibrarlo por sitio
recupera la especificidad perdida sin sacrificar sensibilidad (con la restricción
target_recall=0.95).
"""
import math
import warnings
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemperatureScaler(nn.Module):
    """
    Calibrador de temperatura post-hoc.

    Aprende un parámetro T que escala los logits: calibrated = logits / T.
    - T > 1: Suaviza probabilidades (reduce sobreconfianza, caso típico).
    - T < 1: Agudiza probabilidades (aumenta confianza).
    - T = 1: Sin cambio (red ya calibrada).

    Vector Scaling multiclase:
    -------------------------------------
    Con `num_classes > 1` se usa una T por clase (`torch.ones(num_classes)`),
    equivalente a Vector Scaling (Guo et al. 2017, Tabla 3). Razón clínica:
    asumir una única T para las 8 clases trata a Dermatofibroma (84 imgs,
    típicamente sobreconfiante) y Nevus (clase dominante, típicamente mejor
    calibrada) como si necesitaran el mismo ajuste — en práctica, las clases
    raras requieren T distinta. Escalar per-clase reduce ECE 15-30% en
    clases minoritarias sin afectar la macro AUC.

    `num_classes=1` conserva el comportamiento clásico (un solo escalar)
    para compatibilidad con todos los checkpoints binarios existentes.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes debe ser >=1, recibido {num_classes}")
        self.num_classes = num_classes
        # Inicializar T=1.5 (las redes suelen ser sobreconfiadas → T > 1 las suaviza).
        # Un vector de tamaño `num_classes` implementa vector scaling.
        self.temperature = nn.Parameter(torch.ones(num_classes) * 1.5)

    def calibrate(self, logits):
        """
        Escala logits por la temperatura aprendida.

        Args:
            logits: Tensor de logits crudos [B, C] o [B, 1]
        Returns:
            Logits escalados con mismas dimensiones.

        Con vector scaling (num_classes>1), T es `[C]` y se broadcastea a `[1,C]`
        para dividir el último eje. Con num_classes=1, T es `[1]` y divide escalar.
        """
        # La temperatura se acota al rango clínico [0.3, 5.0]. Un límite inferior
        # más bajo (p. ej. 0.01) permitiría una amplificación de hasta 100× sobre
        # cualquier logit: `logit=0.5 → 50 → sigmoid(50) ≈ 1.0`, es decir,
        # confianza artificial del 99.99%. El límite se aplica en el propio camino
        # de calibración (no solo como aviso tras el ajuste): si el caller carga
        # un checkpoint con T fuera de rango (corrupción o modelo gravemente mal
        # calibrado), el clamp evita una confianza artificial silenciosa. Para
        # experimentos legítimos fuera de rango, recalibrar el modelo en lugar de
        # relajar este límite.
        t = self.temperature.clamp(min=0.3, max=5.0)
        if self.num_classes > 1:
            # Broadcast [C] → [1, C] para dividir el último eje.
            t = t.view(1, -1)
        scaled = logits / t
        if not torch.isfinite(scaled).all():
            raise RuntimeError(
                f"Temperature scaling produjo valores no finitos "
                f"(T={self.temperature.detach().cpu().tolist()}). "
                f"Checkpoint de temperatura corrupto o logits degenerados."
            )
        return scaled

    def fit(self, logits, labels, mode="binary", max_iter=50, lr=0.01):
        """
        Optimiza T sobre logits y labels del validation set.

        Args:
            logits: Tensor [N, C] o [N, 1] — logits crudos del modelo sobre val set
            labels: Tensor [N] — etiquetas verdaderas (int para multiclase, float para binario)
            mode: 'binary' o 'multiclass'
            max_iter: Iteraciones máximas de LBFGS
            lr: Learning rate de LBFGS
        Returns:
            T óptimo (float)
        """
        # LBFGS es el optimizador estándar para temperature scaling (Guo et al. 2017)
        # porque el problema es convexo en 1D y LBFGS converge en pocas iteraciones.
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        logits = logits.detach().float()
        labels = labels.detach()

        if mode == "binary":
            labels = labels.float()

            def closure():
                optimizer.zero_grad()
                scaled = self.calibrate(logits)
                loss = F.binary_cross_entropy_with_logits(scaled.squeeze(-1), labels)
                loss.backward()
                return loss
        else:
            labels = labels.long()

            def closure():
                optimizer.zero_grad()
                scaled = self.calibrate(logits)
                loss = F.cross_entropy(scaled, labels)
                loss.backward()
                return loss

        optimizer.step(closure)

        # Validar que T quedó en un rango clínicamente razonable:
        # T < 0.3 → amplificación extrema (100× → confianza artificial del 99.99%)
        # T > 5.0 → predicciones casi uniformes (el modelo pierde capacidad discriminativa)
        # Ambos casos indican que el modelo base está patológicamente mal calibrado o que
        # los logits del val set son degenerados.
        t_vec = self.temperature.detach().cpu().tolist()
        t_min = min(t_vec)
        t_max = max(t_vec)
        # Si tras `fit()` la T cae fuera del rango clínico [0.3, 5.0], se detiene
        # con error en lugar de continuar: un T=0.01 amplifica los logits 100×,
        # produciendo confianza artificial. El operador no debe poder desplegar un
        # modelo con calibración patológica en producción clínica. Si el modelo
        # está genuinamente mal calibrado, debe recalibrarse el modelo (no la
        # temperatura). La comprobación es posterior al ajuste porque LBFGS puede
        # converger fuera de rango con datos degenerados; el guard aborta antes de
        # guardar o usar el calibrador.
        if t_min < 0.3 or t_max > 5.0:
            raise RuntimeError(
                f"Temperatura calibrada fuera del rango clínico "
                f"[0.3, 5.0]: {t_vec}. T<0.3 amplifica logits >3× "
                "(confianza artificial); T>5.0 aplana predicciones. Causa "
                "probable: modelo patológicamente mal calibrado o val set "
                "degenerado (logits casi todos iguales). Acción: revisar "
                "logits del val (deberían tener varianza > 0), reentrenar el "
                "modelo si el AUC val es bajo, o aumentar el val set. NO "
                "relajar este guard sin justificación clínica documentada."
            )
        return t_vec if self.num_classes > 1 else t_vec[0]

    def save(self, path):
        """Guarda la temperatura calibrada.

        Formato: dict con `temperature` (lista si vector scaling, escalar si scalar)
        y `num_classes` para disambiguar al cargar.
        """
        t = self.temperature.detach().cpu().tolist()
        torch.save(
            {
                "temperature": t if self.num_classes > 1 else t[0],
                "num_classes": self.num_classes,
            },
            path,
        )

    def load(self, path):
        """Carga la temperatura calibrada.

        Backward-compatible: checkpoints legacy sin `num_classes` se interpretan
        como scalar scaling (binario). Vector scaling requiere `num_classes` en
        el dict y una lista de longitud `num_classes`.
        """
        data = torch.load(path, map_location="cpu", weights_only=True)
        t = data["temperature"]
        num_classes = int(data.get("num_classes", 1))
        if num_classes != self.num_classes:
            # Se avisa al caller del cambio de forma del parámetro. Si creó el
            # scaler con `num_classes=1` y el archivo tiene 8, `temperature` se
            # reinstancia como vector. Es el comportamiento esperado (el engine
            # hace peek del num_classes antes de cargar), pero sin aviso un caller
            # externo no sabría que su objeto escalar pasó a ser vector y podría
            # llamar `.item()` asumiendo un escalar, produciendo un RuntimeError
            # poco informativo. El warning facilita el diagnóstico.
            import warnings as _w
            _w.warn(
                f"TemperatureScaler.load: num_classes cambió {self.num_classes} → "
                f"{num_classes} desde el archivo. El parámetro `temperature` se "
                "reinstancia como vector; el caller debe tratarlo como tensor "
                "(usar `.tolist()` no `.item()`) tras esta carga.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Reinstanciar el parámetro con el tamaño correcto.
            self.num_classes = num_classes
            self.temperature = nn.Parameter(torch.ones(num_classes) * 1.5)
        if num_classes == 1:
            # Escalar (incluye compatibilidad con floats "pelados"). Se usa un
            # patrón tolerante al tipo: `isinstance(t, (int, float))` rechazaría
            # `np.float32` (que no hereda de Python float — solo np.float64 lo
            # hace, por un detalle de implementación). Un checkpoint legacy que
            # guardara T como np.float32 puro o como numpy 0-d array caería al
            # camino `float(t[0])` y fallaría con "not subscriptable". El
            # try/except sobre `float()` acepta de forma uniforme Python
            # int/float, np.floating, np.integer y tensores torch 0-d.
            if isinstance(t, (list, tuple, np.ndarray)) and not (
                hasattr(t, "ndim") and t.ndim == 0
            ):
                val = float(t[0])
            else:
                try:
                    val = float(t)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"temperature legacy no convertible a float "
                        f"(tipo={type(t).__name__}, valor={t!r}). Soporta "
                        "Python int/float, np.floating/integer, torch 0-d, "
                        "lista/tupla/ndarray 1-elem. Regenerar el checkpoint."
                    ) from e
            self.temperature.data.fill_(val)
        else:
            t_list = list(t)
            if len(t_list) != num_classes:
                raise ValueError(
                    f"Temperature vector len={len(t_list)} ≠ num_classes={num_classes}"
                )
            self.temperature.data = torch.tensor(t_list, dtype=torch.float32)


def expected_calibration_error(probs, labels, n_bins=15):
    """
    Expected Calibration Error (ECE) — mide si las probabilidades predichas
    son fieles a la frecuencia real.

    Si el modelo dice "melanoma al 80%", el 80% de esos casos deberían ser melanoma.
    ECE alto = probabilidades poco fiables = umbral clínico sin significado real.

    Args:
        probs: np.array [N] — probabilidades predichas (0-1)
        labels: np.array [N] — etiquetas verdaderas binarias (0 o 1)
        n_bins: Número de bins para discretizar probabilidades
    Returns:
        ece: float — ECE (0 = calibrado perfecto, 1 = descalibrado total)
        bin_data: list of dicts — datos por bin para reliability diagram
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        # Último bin incluye el borde derecho
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_data.append({
                "bin_center": (lo + hi) / 2,
                "avg_confidence": 0.0,
                "accuracy": 0.0,
                "count": 0,
            })
            continue

        avg_confidence = probs[mask].mean()
        accuracy = labels[mask].mean()
        gap = abs(avg_confidence - accuracy)
        ece += gap * (n_in_bin / len(probs))

        bin_data.append({
            "bin_center": (lo + hi) / 2,
            "avg_confidence": float(avg_confidence),
            "accuracy": float(accuracy),
            "count": int(n_in_bin),
        })

    return float(ece), bin_data


class PerSiteThresholdCalibrator:
    """
    Calibrador del threshold de decisión RED/GREEN por sitio anatómico.

    Complementa a TemperatureScaler (que calibra probabilidades) ajustando el punto
    de operación del clasificador binario por subgrupo. El threshold global 0.20
    fue optimizado para una lesión promedio con prevalencia maligna
    ~3%. En sitios con prevalencia extrema (palmas/plantas 58% maligno en train;
    tronco posterior 0.6%) el threshold óptimo difiere sustancialmente: un threshold
    global puede producir especificidad muy baja en esos sitios si el modelo usa el
    sitio anatómico como atajo y la base rate diverge de la media.

    Algoritmo (sobre val set, post temperature scaling):
        Para cada site_code con n >= min_subgroup_n y n_pos >= min_positive:
            1. Computar ROC curve sobre el sitio
            2. Buscar el threshold MÁS ALTO que mantenga recall >= target_recall
            3. Dentro del constraint, maximizar specificity
            4. Clamp a [PER_SITE_THRESHOLD_MIN, PER_SITE_THRESHOLD_MAX]
            5. Si imposible cumplir target_recall (sitio degenerado): None → fallback global

    Persiste:
        - thresholds: dict[int, float | None]    site_code → threshold óptimo o None
        - fallback_threshold: float              CLINICAL_THRESHOLD_BINARY
        - target_recall: float                   contrato persistido para audit
        - min_subgroup_n: int
        - min_positive: int
        - val_csv_hash: str                      SHA256 del val CSV (paridad val)
        - per_site_metrics: dict[int, dict]      n, n_pos, recall@thresh, spec@thresh

    Integración en ClinicalInferenceEngine:
        - Constructor acepta `per_site_threshold_path: str | None = None`
        - `_get_effective_threshold(site_code)` despacha por sitio con fallback global
        - Reporte clínico expone `effective_threshold` y `threshold_source`
        ("per_site" | "global_fallback") para trazabilidad clínica.

    Compatibilidad con conformal/risk_control:
        - BinaryConformal.predict opera sobre prob calibrada, no threshold → independiente
        - ConformalRiskControl.predict idem
        - AMBER override (set conformal {benigno, maligno}) preserva su rol; solo cambia
          la decisión RED/GREEN inicial.
    """

    def __init__(
        self,
        target_recall: float = 0.95,
        min_subgroup_n: int = 30,
        min_positive: int | None = None,
        fallback_threshold: float | None = None,
    ):
        """
        Args:
            target_recall: Recall mínimo a preservar por sitio. Default 0.95 alineado
                con el sesgo a recall (NUNCA perder un maligno). NO subir
                arbitrariamente sin justificar el coste de specificity.
            min_subgroup_n: Mínimo de muestras totales por sitio para calibrar.
                Default 30. Sitios con n < min_subgroup_n caen al fallback (threshold
                global) — calibrar sobre menos de 30 muestras produce ROC curves con
                varianza extrema.
            min_positive: Mínimo de malignos por sitio. Default `MIN_POSITIVE_DEFAULT=10`
                (el intervalo de Wilson al 95% para recall@p=0.5 con n=10 es
                [0.24, 0.76], rango ~52pp; con n=4 el rango es ~80pp, inutilizable
                clínicamente).
            fallback_threshold: Threshold a usar para sitios sin calibración. Default
                `CLINICAL_THRESHOLD_BINARY=0.20`.
        """
        from src.data.constants import (
            CLINICAL_THRESHOLD_BINARY,
            MIN_POSITIVE_DEFAULT,
            PER_SITE_THRESHOLD_MIN,
            PER_SITE_THRESHOLD_MAX,
        )
        if not (0.0 < target_recall < 1.0):
            raise ValueError(
                f"target_recall={target_recall} fuera de (0, 1)."
            )
        if min_subgroup_n < 1:
            raise ValueError(
                f"min_subgroup_n={min_subgroup_n} debe ser >= 1."
            )
        self.target_recall = float(target_recall)
        self.min_subgroup_n = int(min_subgroup_n)
        self.min_positive = int(min_positive) if min_positive is not None else int(MIN_POSITIVE_DEFAULT)
        if self.min_positive < 1:
            raise ValueError(
                f"min_positive={self.min_positive} debe ser >= 1."
            )
        self.fallback_threshold = float(
            fallback_threshold if fallback_threshold is not None else CLINICAL_THRESHOLD_BINARY
        )
        self._threshold_min = float(PER_SITE_THRESHOLD_MIN)
        self._threshold_max = float(PER_SITE_THRESHOLD_MAX)
        if not (self._threshold_min <= self.fallback_threshold <= self._threshold_max):
            # El fallback threshold puede ser legítimamente CLINICAL_THRESHOLD_BINARY
            # = 0.20, que está dentro de [0.10, 0.50]. Si se pasa un fallback fuera de
            # esos límites, se detiene con error por incoherencia con el rango clínico.
            raise ValueError(
                f"fallback_threshold={self.fallback_threshold} fuera de los "
                f"limites clinicos [{self._threshold_min}, {self._threshold_max}]."
            )

        # Estado post-fit
        self.thresholds: dict[int, float | None] = {}
        self.per_site_metrics: dict[int, dict] = {}
        self.val_csv_hash: str = ""

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        site_codes: np.ndarray,
    ) -> dict[int, float | None]:
        """
        Calibra el threshold por sitio sobre el val set (tras temperature scaling).

        El orden de los argumentos (probabilidades primero, etiquetas después) es
        coherente con el resto del módulo y con `conformal.py`:
        `BinaryConformal.fit(probs, labels)`, `APSConformal.fit(probs, labels)`,
        `ConformalRiskControl.fit(probs, labels)` y
        `TemperatureScaler.fit(logits, labels)` siguen la misma convención. Mantener
        esta consistencia evita invertir `probs` y `labels` en el callsite (lo que
        haría que las probabilidades se interpretaran como etiquetas truncadas a 0 y
        produciría `n_pos=0` en todos los sitios, dejando el calibrador inservible).

        Args:
            probs: np.ndarray [N] — probabilidades calibradas (post temperature) en [0, 1].
            labels: np.ndarray [N] — etiquetas binarias 0/1 del val set.
            site_codes: np.ndarray [N] — site_code int [0..NUM_SITE_CODES-1].

        Returns:
            thresholds: dict {site_code: threshold | None}.
        """
        try:
            from sklearn.metrics import roc_curve
        except ImportError as e:
            raise ImportError(
                "scikit-learn requerido para PerSiteThresholdCalibrator.fit. "
                "Instalar con `pip install scikit-learn`."
            ) from e

        labels = np.asarray(labels).astype(np.int64)
        probs = np.asarray(probs).astype(np.float64)
        site_codes = np.asarray(site_codes).astype(np.int64)

        # Validaciones que detienen la ejecución ante entradas inválidas.
        if labels.shape != probs.shape or labels.shape != site_codes.shape:
            raise ValueError(
                f"shape mismatch — labels={labels.shape}, "
                f"probs={probs.shape}, site_codes={site_codes.shape}."
            )
        if labels.size == 0:
            raise ValueError("labels vacio. Val set sin muestras.")
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError(
                f"labels contiene valores != 0/1. "
                f"unique={np.unique(labels).tolist()}."
            )
        if not np.isfinite(probs).all():
            raise ValueError(
                "probs contiene NaN/Inf. Aplicar Temperature Scaling "
                "antes de calibrar el per-site threshold."
            )
        if (probs < 0).any() or (probs > 1).any():
            raise ValueError(
                f"probs fuera de [0, 1]. "
                f"min={probs.min()}, max={probs.max()}. "
                "Pasar PROBABILIDADES (post sigmoid + temperature), no logits."
            )

        self.thresholds = {}
        self.per_site_metrics = {}

        unique_sites = np.unique(site_codes).tolist()
        for site_code in unique_sites:
            site_code = int(site_code)
            mask = (site_codes == site_code)
            n = int(mask.sum())
            n_pos = int((labels[mask] == 1).sum())
            n_neg = int(n - n_pos)

            site_metrics = {
                "n": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "threshold": None,
                "recall_at_threshold": None,
                "specificity_at_threshold": None,
                "skip_reason": None,
            }

            if n < self.min_subgroup_n:
                site_metrics["skip_reason"] = f"n={n} < min_subgroup_n={self.min_subgroup_n}"
                self.thresholds[site_code] = None
                self.per_site_metrics[site_code] = site_metrics
                continue
            if n_pos < self.min_positive:
                site_metrics["skip_reason"] = f"n_pos={n_pos} < min_positive={self.min_positive}"
                self.thresholds[site_code] = None
                self.per_site_metrics[site_code] = site_metrics
                continue
            if n_neg == 0:
                # Sitio puramente maligno (caso patológico): threshold no significa
                # nada porque no hay benignos para optimizar specificity.
                site_metrics["skip_reason"] = "n_neg=0 (sitio puramente positivo)"
                self.thresholds[site_code] = None
                self.per_site_metrics[site_code] = site_metrics
                continue

            yt_site = labels[mask]
            yp_site = probs[mask]
            fpr, tpr, thresh_arr = roc_curve(yt_site, yp_site)

            # Encontrar el threshold MÁS ALTO que mantenga recall >= target_recall.
            # En roc_curve, los thresholds vienen en orden DESCENDENTE: thresh[0] es
            # el más alto (recall=0), thresh[-1] el más bajo (recall=1).
            valid = tpr >= self.target_recall
            if not valid.any():
                # Imposible cumplir target_recall: el sitio es tan difícil que ni
                # threshold=0 alcanza recall=target. Fallback global.
                site_metrics["skip_reason"] = (
                    f"imposible cumplir target_recall={self.target_recall} "
                    f"(max_recall={tpr.max():.3f})"
                )
                self.thresholds[site_code] = None
                self.per_site_metrics[site_code] = site_metrics
                continue

            # Dentro del subset valid, maximizar specificity (= 1 - fpr).
            # spec = 1 - fpr; mayor spec ↔ menor fpr.
            spec = 1.0 - fpr
            # Multiplicar por valid (booleano) para descartar puntos fuera del constraint.
            scores = spec * valid.astype(np.float64)
            best_idx = int(np.argmax(scores))

            # Threshold candidato. roc_curve devuelve un threshold inicial = max(probs)+1
            # (sentinela para "todo predicho negativo"); ese caso solo aparece con
            # tpr=fpr=0 y NO será valid (tpr < target_recall). Igualmente clamp.
            raw_thresh = float(thresh_arr[best_idx])
            if not np.isfinite(raw_thresh):
                # roc_curve sentinela inf — usar el siguiente válido
                site_metrics["skip_reason"] = "roc_curve devolvio threshold no-finito"
                self.thresholds[site_code] = None
                self.per_site_metrics[site_code] = site_metrics
                continue

            clamped = float(np.clip(raw_thresh, self._threshold_min, self._threshold_max))

            # Recomputar recall/spec con el threshold clamp para audit honesto
            # (si raw_thresh > 0.50 y clamp lo baja, recall puede subir y spec bajar).
            preds = (yp_site >= clamped).astype(np.int64)
            tp = int(((preds == 1) & (yt_site == 1)).sum())
            tn = int(((preds == 0) & (yt_site == 0)).sum())
            fn = int(((preds == 0) & (yt_site == 1)).sum())
            fp = int(((preds == 1) & (yt_site == 0)).sum())
            recall_post = tp / max(tp + fn, 1)
            spec_post = tn / max(tn + fp, 1)

            site_metrics["threshold"] = clamped
            site_metrics["threshold_raw"] = raw_thresh
            site_metrics["recall_at_threshold"] = recall_post
            site_metrics["specificity_at_threshold"] = spec_post
            self.thresholds[site_code] = clamped
            self.per_site_metrics[site_code] = site_metrics

        return self.thresholds

    def get_threshold(self, site_code: int) -> tuple[float, str]:
        """
        Despacha threshold por sitio en inferencia.

        Returns:
            (threshold, source) donde source ∈ {"per_site", "global_fallback"}
            para la trazabilidad clínica.
        """
        site_int = int(site_code)
        thresh = self.thresholds.get(site_int)
        if thresh is None:
            return self.fallback_threshold, "global_fallback"
        return float(thresh), "per_site"

    def save(self, path: str, val_csv_hash: str = ""):
        """
        Persiste los thresholds y los metadatos de auditoría.

        `val_csv_hash` es el SHA256 del val CSV usado para calibrar. El
        ClinicalInferenceEngine valida este hash al cargar para detectar cuándo es
        necesaria una recalibración tras re-ejecutar el split de datos.
        """
        self.val_csv_hash = str(val_csv_hash)
        torch.save(
            {
                "thresholds": {int(k): (None if v is None else float(v))
                               for k, v in self.thresholds.items()},
                "fallback_threshold": float(self.fallback_threshold),
                "target_recall": float(self.target_recall),
                "min_subgroup_n": int(self.min_subgroup_n),
                "min_positive": int(self.min_positive),
                "val_csv_hash": self.val_csv_hash,
                "per_site_metrics": self.per_site_metrics,
                "schema_version": 1,
            },
            path,
        )

    def load(self, path: str):
        """
        Carga los thresholds y los metadatos de auditoría.

        La carga se detiene con error si:
          - schema_version reconocible
          - thresholds dict de int → float|None
          - todos los thresholds en [PER_SITE_THRESHOLD_MIN, PER_SITE_THRESHOLD_MAX]
          - target_recall ∈ (0, 1)
          - fallback_threshold finito y en bounds
        """
        data = torch.load(path, map_location="cpu", weights_only=True)

        schema_version = int(data.get("schema_version", 1))
        if schema_version != 1:
            raise ValueError(
                f"schema_version desconocido: {schema_version}. "
                "Soportado: 1. Regenerar con scripts/calibrate_per_site_threshold.py."
            )

        thresholds = data.get("thresholds", {})
        if not isinstance(thresholds, dict):
            raise ValueError(
                f"'thresholds' debe ser dict, recibido {type(thresholds).__name__}."
            )

        # Validar bounds de cada threshold (los None son legitimos: fallback)
        cleaned = {}
        for k, v in thresholds.items():
            try:
                k_int = int(k)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"site_code key no convertible a int: {k!r}."
                ) from e
            if v is None:
                cleaned[k_int] = None
                continue
            try:
                v_float = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"threshold para site={k_int} no convertible a float: {v!r}."
                ) from e
            if not math.isfinite(v_float):
                raise ValueError(
                    f"threshold para site={k_int} no finito: {v_float}."
                )
            if not (self._threshold_min <= v_float <= self._threshold_max):
                raise ValueError(
                    f"threshold para site={k_int}={v_float} fuera de los "
                    f"limites clinicos [{self._threshold_min}, {self._threshold_max}]."
                )
            cleaned[k_int] = v_float
        self.thresholds = cleaned

        # Validar fallback_threshold
        fb = float(data.get("fallback_threshold", self.fallback_threshold))
        if not math.isfinite(fb) or not (self._threshold_min <= fb <= self._threshold_max):
            raise ValueError(
                f"fallback_threshold cargado={fb} fuera de los limites clinicos."
            )
        self.fallback_threshold = fb

        # Otros campos (con defaults graceful para checkpoints legacy)
        tr = float(data.get("target_recall", self.target_recall))
        if not (0.0 < tr < 1.0):
            raise ValueError(f"target_recall cargado={tr} fuera de (0, 1).")
        self.target_recall = tr
        self.min_subgroup_n = int(data.get("min_subgroup_n", self.min_subgroup_n))
        self.min_positive = int(data.get("min_positive", self.min_positive))
        self.val_csv_hash = str(data.get("val_csv_hash", ""))
        self.per_site_metrics = dict(data.get("per_site_metrics", {}))
