"""
Métricas de Evaluación Avanzadas y Análisis de Equidad.

El AUC por sí solo es insuficiente para la evaluación clínica. Este módulo provee:
- Expected Calibration Error (ECE): ¿Las probabilidades son fiables?
- Curvas Precision-Recall por clase: ¿Qué tradeoff PR tiene cada diagnóstico?
- Matrices de confusión normalizadas: ¿Qué clases se confunden?
- Diagramas de fiabilidad: visualización de la calibración.
- Análisis de equidad: ¿El modelo sesga por sexo, edad o sitio anatómico?
"""
import numpy as np
from pathlib import Path

# AGE_UNKNOWN_YEARS se importa de constants.py para no hardcodear el literal
# `50` en fairness_report: es la fuente de verdad única del centro de edad.
from src.data.constants import AGE_UNKNOWN_YEARS, MULTICLASS_INDEX_MAP

try:
    import matplotlib
    matplotlib.use("Agg")  # Backend sin GUI para servidores/Colab
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
        recall_score,
        precision_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def expected_calibration_error(probs, labels, n_bins=15):
    """
    Expected Calibration Error (ECE).

    Mide si las probabilidades predichas coinciden con la frecuencia real.
    ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|

    Args:
        probs: np.array [N] — probabilidades predichas (0-1)
        labels: np.array [N] — etiquetas verdaderas (0 o 1)
        n_bins: Número de bins
    Returns:
        ece: float, bin_data: list of dicts
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
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


def expected_calibration_error_adaptive(probs, labels, n_bins=15):
    """Adaptive-mass (equi-mass) Expected Calibration Error.

    Referencia: Nixon et al., "Measuring Calibration in Deep Learning",
    CVPR Workshops 2019. https://arxiv.org/abs/1904.01685

    El ECE estándar usa bins equi-ancho en `[0, 1]`. Cuando la distribución
    de probabilidades está concentrada (típico en triaje binario post
    Temperature Scaling, donde la mayoría de muestras tienen prob ≈ 0.05 ó
    prob ≈ 0.95), los bins centrales quedan casi vacíos y los bins
    extremos dominan el error — sesgo sistemático hacia "el modelo está
    bien calibrado" incluso cuando hay miscalibración local en el régimen
    clínico (prob ≈ 0.20).

    La variante **equi-mass** coloca fronteras de bin en los cuantiles de
    `probs`, garantizando que cada bin tenga ~`N/n_bins` muestras. Esto da
    una estimación más robusta del ECE, especialmente en los regímenes
    donde hay muchos datos.

    Args:
        probs: `[N]` probabilidades calibradas en `[0, 1]`.
        labels: `[N]` enteros `{0, 1}` — etiquetas reales.
        n_bins: número de bins equi-mass. Default 15 (match del ECE estándar).

    Returns:
        float en `[0, 1]` — ECE equi-mass ponderado por la fracción de
        muestras de cada bin (aunque aquí todos los bins tienen ~igual peso
        por construcción).

    **Nota:** si la distribución tiene empates exactos (muchas muestras con
    la misma prob), los cuantiles pueden colapsar bins → bin count efectivo
    menor que `n_bins`. La función maneja este caso fusionando bins
    consecutivos con fronteras idénticas.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    labels = np.asarray(labels).astype(int).ravel()
    if probs.shape != labels.shape:
        raise ValueError(
            f"probs y labels deben tener el mismo shape, recibido "
            f"probs={probs.shape} labels={labels.shape}"
        )
    if probs.size == 0:
        return 0.0

    # Cuantiles para bordes de bin (equi-mass)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(probs, quantiles)
    # Fusionar fronteras duplicadas para evitar bins de ancho cero
    edges = np.unique(edges)
    if edges.size < 2:
        # Degenerado: todas las probs son iguales
        avg_prob = float(probs.mean())
        avg_acc = float(labels.mean())
        return abs(avg_prob - avg_acc)

    # Asegurar que el último edge cubre max exactamente
    edges[0] = min(edges[0], probs.min())
    edges[-1] = max(edges[-1], probs.max()) + 1e-12

    ece = 0.0
    n_total = probs.size
    for i in range(edges.size - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == edges.size - 2:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_prob = probs[mask].mean()
        bin_acc = labels[mask].mean()
        weight = mask.sum() / n_total
        ece += weight * abs(bin_prob - bin_acc)

    return float(ece)


def plot_reliability_diagram(probs, labels, n_bins=15, title="Reliability Diagram", save_path=None):
    """
    Diagrama de fiabilidad: barras de confianza vs precisión real.
    La diagonal perfecta = calibración perfecta.

    Args:
        probs: np.array [N] probabilidades
        labels: np.array [N] etiquetas
        n_bins: Número de bins
        title: Título del gráfico
        save_path: Path para guardar (None = no guardar)
    Returns:
        fig: matplotlib Figure (o None si matplotlib no disponible)
    """
    if not HAS_MATPLOTLIB:
        return None

    ece, bin_data = expected_calibration_error(probs, labels, n_bins)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    centers = [bd["bin_center"] for bd in bin_data]
    accuracies = [bd["accuracy"] for bd in bin_data]
    confidences = [bd["avg_confidence"] for bd in bin_data]
    counts = [bd["count"] for bd in bin_data]

    # Barras de precisión
    ax.bar(centers, accuracies, width=1.0 / n_bins, alpha=0.7,
           edgecolor="black", color="steelblue", label="Accuracy")
    # Línea de calibración perfecta
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True,
                          title="Confusion Matrix", save_path=None):
    """
    Matriz de confusión normalizada — identifica confusiones específicas entre clases.
    Ej: ¿El modelo confunde Actinic Keratosis con Benign Keratosis?

    Args:
        y_true: np.array [N] etiquetas verdaderas (índices)
        y_pred: np.array [N] predicciones (índices)
        class_names: list of str — nombres de clases
        normalize: Si True, normaliza filas a [0, 1]
        title: Título
        save_path: Path para guardar
    Returns:
        cm: np.array, fig: matplotlib Figure (o None)
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn requerido para confusion_matrix")

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Guard div/0 para clases sin muestras
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm = cm.astype(float) / row_sums
    else:
        cm = cm.astype(float)

    fig = None
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Anotar celdas
        thresh = cm.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    return cm, fig


def plot_per_class_pr_curves(y_true, y_probs, class_names, save_path=None):
    """
    Curvas Precision-Recall por clase — revela el tradeoff PR de cada diagnóstico.

    Args:
        y_true: np.array [N] etiquetas verdaderas (índices)
        y_probs: np.array [N, C] probabilidades por clase
        class_names: list of str
        save_path: Path para guardar
    Returns:
        ap_scores: dict {class_name: average_precision}
        fig: matplotlib Figure (o None)
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn requerido para precision_recall_curve")

    n_classes = len(class_names)
    ap_scores = {}

    fig = None
    if HAS_MATPLOTLIB:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for i in range(n_classes):
        binary_labels = (np.asarray(y_true) == i).astype(int)
        # Skip si la clase no tiene muestras
        if binary_labels.sum() == 0:
            ap_scores[class_names[i]] = 0.0
            continue

        class_probs = np.asarray(y_probs)[:, i]
        precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
        ap = average_precision_score(binary_labels, class_probs)
        ap_scores[class_names[i]] = float(ap)

        if HAS_MATPLOTLIB:
            ax.plot(recall, precision, linewidth=2,
                    label=f"{class_names[i]} (AP={ap:.3f})")

    if HAS_MATPLOTLIB and fig is not None:
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Per-Class Precision-Recall Curves", fontsize=14)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    return ap_scores, fig


def _binary_subgroup_metrics(yt, yp, threshold):
    """Métricas clínicas completas para un subgrupo binario.

    Además de AUC y recall (sensitivity), computa specificity, PPV, NPV y F1
    en el threshold clínico. Un subgrupo puede tener el mismo recall que otro
    pero specificity dramáticamente menor → más falsos positivos, peor utilidad
    clínica. Reportar solo recall oculta ese sesgo.

    Returns dict listo para integrar en el report (valores None si la clase
    positiva o negativa está ausente en el subgrupo).
    """
    n = int(yt.size)
    out = {"n": n, "threshold": float(threshold)}

    if len(np.unique(yt)) < 2:
        # Subgrupo con una sola clase: AUC y specificity no definidas.
        out.update({"auc": None, "recall": None, "specificity": None,
                    "ppv": None, "npv": None, "f1": None})
        return out

    out["auc"] = float(roc_auc_score(yt, yp))
    preds = (yp >= threshold).astype(int)

    tp = int(((preds == 1) & (yt == 1)).sum())
    tn = int(((preds == 0) & (yt == 0)).sum())
    fp = int(((preds == 1) & (yt == 0)).sum())
    fn = int(((preds == 0) & (yt == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else None  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    ppv = tp / (tp + fp) if (tp + fp) > 0 else None
    npv = tn / (tn + fn) if (tn + fn) > 0 else None
    if recall is not None and ppv is not None and (recall + ppv) > 0:
        f1 = 2 * recall * ppv / (recall + ppv)
    else:
        f1 = None

    out["recall"] = float(recall) if recall is not None else None
    out["specificity"] = float(specificity) if specificity is not None else None
    out["ppv"] = float(ppv) if ppv is not None else None
    out["npv"] = float(npv) if npv is not None else None
    out["f1"] = float(f1) if f1 is not None else None
    out["tp"], out["tn"], out["fp"], out["fn"] = tp, tn, fp, fn
    return out


def _multiclass_subgroup_metrics(yt, yp_probs, class_names):
    """Per-class recall stratified para multiclase.

    Macro AUC por subgrupo oculta colapsos de clase específicos. Por ejemplo:
    un subgrupo con macro AUC 0.98 pero melanoma_recall 0.30 tiene un sesgo
    catastrófico invisible. Reportar per-class recall expone el sesgo.

    Schema backward-compatible: el campo `auc` (macro OVR) se mantiene con el
    mismo nombre del schema previo. `recall_per_class` es aditivo.
    """
    n = int(yt.size)
    out = {"n": n}

    if len(np.unique(yt)) < 2:
        out.update({"auc": None, "recall_per_class": {}})
        return out

    try:
        auc_macro = roc_auc_score(yt, yp_probs, multi_class="ovr", average="macro")
        out["auc"] = float(auc_macro)  # Backward compat: key `auc` = macro OVR
    except ValueError:
        out["auc"] = None

    preds = np.argmax(yp_probs, axis=1)
    recall_per_class = {}
    for idx, name in enumerate(class_names):
        class_mask = yt == idx
        n_class = int(class_mask.sum())
        if n_class == 0:
            recall_per_class[name] = {"n": 0, "recall": None}
            continue
        correct = int((preds[class_mask] == idx).sum())
        recall_per_class[name] = {"n": n_class, "recall": correct / n_class}
    out["recall_per_class"] = recall_per_class
    return out


def fairness_report(
    y_true,
    y_probs,
    metadata_df,
    mode="binary",
    threshold=0.5,
    bias_gap_threshold: float = 0.05,
    min_subgroup_n: int = 10,
    min_positive: int = 10,
    class_names=None,
):
    """
    Análisis de equidad — estratifica métricas por subgrupos demográficos.

    Los sistemas dermatológicos rinden peor en pieles oscuras (Fitzpatrick V-VI).
    Sin datos de fototipo, usamos proxies: sexo, edad, sitio anatómico.
    Un gap en AUC (o specificity/recall_per_class) mayor a
    ``bias_gap_threshold`` entre subgrupos indica sesgo potencial.

    Args:
        y_true: np.array [N] etiquetas verdaderas.
        y_probs: np.array [N] probabilidades (binario) o [N, C] (multiclase).
        metadata_df: DataFrame con columnas 'age_approx', 'sex', 'anatom_site'.
        mode: 'binary' o 'multiclass'.
        threshold: umbral de decisión para recall/specificity binario (default 0.5).
            En modo binario conviene pasar el umbral clínico (0.20)
            para detectar sesgo en el punto operativo real, no en 0.50.
        bias_gap_threshold: umbral de gap para marcar sesgo potencial. Default
            0.05 (5 puntos porcentuales). Aplicado a AUC, specificity (binario) y
            recall de clases malignas (multiclase).
        min_subgroup_n: mínimo de muestras totales por subgrupo para considerar
            sus métricas fiables (default 10). Subgrupos más pequeños se reportan
            con `None` en sus métricas.
        min_positive: mínimo de muestras POSITIVAS (malignos en binario, clase
            target en multiclase) por subgrupo para considerar sus métricas
            fiables. Default 10. El intervalo de Wilson al 95% para recall=0.5
            con n_pos=10 es [0.24, 0.76] (rango 52pp); con n_pos=4 es
            [0.10, 0.90] (rango 80pp, inutilizable clínicamente). Sin este
            guard, subgrupos como `sex=unknown` (n=372 en val pero solo 4
            malignos) producen warnings espurios de "gap recall=0.475" que son
            ruido estadístico, saturan logs y desensibilizan al operador clínico
            ante warnings reales. Aplicado solo en modo binario (multiclase tiene
            per-class recall, gestionado de forma distinta).
        class_names: multiclase — lista de nombres de clase en orden de índice.
            Si None, se deriva de `MULTICLASS_INDEX_MAP`.

    Returns:
        report: dict con métricas estratificadas y flags de sesgo. Schema
        backward-compatible con la versión previa (campos existentes siguen
        presentes); los nuevos campos (`specificity`, `ppv`, `npv`, `f1`,
        `recall_per_class`) son aditivos.
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn requerido para fairness_report")

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    report = {
        "stratified_metrics": {},
        "bias_flags": [],
        "bias_gap_threshold": float(bias_gap_threshold),
        "min_subgroup_n": int(min_subgroup_n),
        "min_positive": int(min_positive),
        "mode": mode,
    }

    # Derivar class_names de constants.py si no se pasa (fuente única de verdad).
    if mode == "multiclass" and class_names is None:
        class_names = [MULTICLASS_INDEX_MAP[i] for i in range(len(MULTICLASS_INDEX_MAP))]

    def _compute_metrics(mask):
        """Computa métricas para un subgrupo (binario o multiclase)."""
        if mask.sum() < min_subgroup_n:
            return None
        # Guard adicional sobre n_positive en modo binario. Un subgrupo con
        # n=372 (pasa min_subgroup_n=10) pero solo 4 malignos produce un Wilson
        # CI 95% para recall=0.5 de [10%, 90%] — inutilizable clínicamente.
        # Excluirlo del bias_flag check evita warnings espurios que saturan logs
        # y desensibilizan al operador. Multiclase NO se filtra aquí porque tiene
        # per-class recall (cada clase target tiene su propio n_positive natural,
        # con el filtrado por clase ya implícito).
        if mode == "binary":
            n_pos = int(np.asarray(y_true)[mask].astype(int).sum())
            if n_pos < min_positive:
                return None
        yt = y_true[mask]
        yp = y_probs[mask]
        if mode == "binary":
            return _binary_subgroup_metrics(yt, yp, threshold)
        return _multiclass_subgroup_metrics(yt, yp, class_names)

    def _collect_gaps(groups: dict, metric_key: str):
        """Helper: extrae valores de `metric_key` de todos los subgrupos y
        computa (max - min). Devuelve None si <2 valores válidos."""
        values = [g[metric_key] for g in groups.values()
                  if isinstance(g, dict) and g.get(metric_key) is not None]
        if len(values) < 2:
            return None
        return max(values) - min(values)

    def _check_subgroup_bias(dim_name: str, groups: dict):
        """Detecta sesgo por dimensión, con métricas apropiadas al modo."""
        if mode == "binary":
            gap_auc = _collect_gaps(groups, "auc")
            gap_spec = _collect_gaps(groups, "specificity")
            gap_rec = _collect_gaps(groups, "recall")
            if gap_auc is not None and gap_auc > bias_gap_threshold:
                report["bias_flags"].append(
                    f"SESGO POTENCIAL por {dim_name}: gap AUC = {gap_auc:.3f} "
                    f"(> {bias_gap_threshold:.2f})"
                )
            if gap_spec is not None and gap_spec > bias_gap_threshold:
                report["bias_flags"].append(
                    f"SESGO POTENCIAL por {dim_name}: gap specificity = {gap_spec:.3f} "
                    f"(> {bias_gap_threshold:.2f}, threshold clínico={threshold})"
                )
            if gap_rec is not None and gap_rec > bias_gap_threshold:
                report["bias_flags"].append(
                    f"SESGO POTENCIAL por {dim_name}: gap recall = {gap_rec:.3f} "
                    f"(> {bias_gap_threshold:.2f}, threshold clínico={threshold})"
                )
        else:
            gap_auc = _collect_gaps(groups, "auc")
            if gap_auc is not None and gap_auc > bias_gap_threshold:
                report["bias_flags"].append(
                    f"SESGO POTENCIAL por {dim_name}: gap macro AUC = {gap_auc:.3f} "
                    f"(> {bias_gap_threshold:.2f})"
                )
            # Gap de recall por clase (especialmente las malignas).
            per_class_values = {}  # class_name → list of (subgroup, recall)
            for subgroup_name, sg in groups.items():
                rpc = sg.get("recall_per_class", {}) if isinstance(sg, dict) else {}
                for cname, cdata in rpc.items():
                    if cdata.get("recall") is None:
                        continue
                    # Guard `n < min_positive` per-clase, simétrico con el path
                    # binario. Sin esto, subgrupos con n_class pequeño en clases
                    # raras (Vascular Lesion, Dermatofibroma con n_class=1-4 en
                    # val) generan warnings espurios "gap recall=1.0" puramente
                    # estadísticos: el Wilson CI 95% para recall=0 con n=2 va de
                    # [0, 0.84], ruido inutilizable clínicamente que el operador
                    # acaba descartando — y entonces pierde un sesgo REAL en
                    # clases mayoritarias (Melanoma con n_class=50+ y gap=0.15)
                    # entre el ruido. El Wilson CI requiere n>=10 para una
                    # interpretación clínica fiable, independientemente del modo.
                    # `cdata` siempre contiene la clave "n" (poblada en
                    # _multiclass_subgroup_metrics).
                    if int(cdata.get("n", 0)) < min_positive:
                        continue
                    per_class_values.setdefault(cname, []).append(
                        (subgroup_name, cdata["recall"])
                    )
            for cname, values in per_class_values.items():
                if len(values) < 2:
                    continue
                recalls = [v[1] for v in values]
                gap = max(recalls) - min(recalls)
                if gap > bias_gap_threshold:
                    report["bias_flags"].append(
                        f"SESGO POTENCIAL por {dim_name}: gap recall '{cname}' = "
                        f"{gap:.3f} (> {bias_gap_threshold:.2f})"
                    )

    # --- Estratificar por sexo ---
    if "sex" in metadata_df.columns:
        sex_groups = {}
        for sex_val in metadata_df["sex"].dropna().unique():
            mask = (metadata_df["sex"] == sex_val).values
            metrics = _compute_metrics(mask)
            if metrics:
                sex_groups[str(sex_val)] = metrics
        report["stratified_metrics"]["sex"] = sex_groups
        _check_subgroup_bias("sexo", sex_groups)

    # --- Estratificar por grupos de edad ---
    if "age_approx" in metadata_df.columns:
        age_groups = {}
        age_raw = metadata_df["age_approx"]
        # Ratio de imputación: si es masivo, la estratificación por edad es
        # poco fiable (todos los NaN caen en el mismo bin via fillna).
        nan_ratio = float(age_raw.isna().mean())
        # Centro fijo desde AGE_UNKNOWN_YEARS (constants.py) en vez del literal
        # `50`: es la fuente de verdad única del centro de edad.
        ages = age_raw.fillna(AGE_UNKNOWN_YEARS)
        # Último bin abierto por arriba: edades sentinel/outlier (>=150) no
        # deben caer silenciosamente fuera del reporte de fairness.
        bins = [(0, 30, "0-30"), (30, 50, "30-50"), (50, 70, "50-70"),
                (70, float("inf"), "70+")]
        for lo, hi, label in bins:
            mask = ((ages >= lo) & (ages < hi)).values
            metrics = _compute_metrics(mask)
            if metrics:
                age_groups[label] = metrics
        report["stratified_metrics"]["age"] = age_groups
        report["stratified_metrics"]["age_nan_ratio"] = nan_ratio
        if nan_ratio > 0.30:
            report["bias_flags"].append(
                f"FAIRNESS: {nan_ratio:.1%} edades imputadas — estratificacion "
                f"por edad poco fiable"
            )
        _check_subgroup_bias("edad", age_groups)

    # --- Estratificar por sitio anatomico ---
    if "anatom_site" in metadata_df.columns:
        site_groups = {}
        for site_val in metadata_df["anatom_site"].dropna().unique():
            mask = (metadata_df["anatom_site"] == site_val).values
            metrics = _compute_metrics(mask)
            if metrics:
                site_groups[str(site_val)] = metrics
        report["stratified_metrics"]["anatom_site"] = site_groups
        _check_subgroup_bias("sitio anatómico", site_groups)

    return report


# =============================================================================
# Métricas clínicas complementarias al AUC
# =============================================================================
#
# AUC estándar mide discriminación sobre todo el rango de operación. En
# dermatología clínica interesan regiones específicas:
#   - Spec fijo al 95% (el régimen en el que el sistema NO genera fatiga de
#     alertas al dermatólogo).
#   - FPR bajo (partial AUC con max_fpr=0.05 — estándar dermatológico).
#   - Decision curve analysis (net benefit vs estrategias triviales).
#   - Brier score (calibración + discriminación en una sola métrica).
# Estas funciones son standalone, usan sklearn si está disponible, si no
# degradan graciosamente.


def sensitivity_at_specificity(y_true, y_probs, target_spec: float = 0.95):
    """
    Sensibilidad (recall de la clase positiva) en el umbral que fija
    especificidad en `target_spec`.

    Interpretación clínica: "Si exigimos falsos positivos ≤ 5% (spec=0.95),
    ¿qué fracción de melanomas detectamos?". Métrica directamente ligada a
    la utilidad de un sistema de triaje — un AUC alto con sensibilidad baja
    a spec=0.95 es inútil operacionalmente.

    Args:
        y_true: [N] binarios {0, 1}.
        y_probs: [N] probabilidades (típicamente ya calibradas).
        target_spec: nivel objetivo de especificidad (default 0.95).
    Returns:
        dict con `sensitivity`, `threshold`, `actual_spec` (especificidad
        alcanzada — puede ser ligeramente mayor que target_spec si el set
        es discreto y no hay threshold exacto).
    """
    if not HAS_SKLEARN:
        return {"sensitivity": None, "threshold": None, "actual_spec": None}
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int).ravel()
    y_probs = np.asarray(y_probs, dtype=np.float64).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    spec = 1.0 - fpr
    # Buscar el threshold con spec >= target_spec y mayor sensibilidad.
    valid = spec >= target_spec
    if not valid.any():
        # No hay threshold que satisfaga spec objetivo — reportar el mejor.
        idx = int(np.argmax(spec))
    else:
        # Entre los válidos, maximizar TPR.
        valid_idx = np.where(valid)[0]
        idx = int(valid_idx[np.argmax(tpr[valid_idx])])
    return {
        "sensitivity": float(tpr[idx]),
        "threshold": float(thresholds[idx]),
        "actual_spec": float(spec[idx]),
    }


def auc_at_specificity(y_true, y_probs, target_spec: float = 0.95):
    """
    AUC restringida a la región de especificidad ≥ `target_spec`.

    Calculada como área bajo la porción de la curva ROC donde FPR ≤
    1 - target_spec, normalizada al rectángulo [0, 1-target_spec] × [0, 1].
    Esto hace la métrica comparable al AUC estándar (rango [0, 1]) pero
    sensible solo al régimen clínico útil.

    Args:
        y_true, y_probs: binarios y probabilidades.
        target_spec: umbral de especificidad para definir la región (0.95).
    Returns:
        float en [0, 1] o None si sklearn no está disponible.
    """
    if not HAS_SKLEARN:
        return None
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int).ravel()
    y_probs = np.asarray(y_probs, dtype=np.float64).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    max_fpr = 1.0 - target_spec
    # Recortar FPR/TPR a la región FPR ≤ max_fpr. Añadir punto interpolado
    # exactamente en max_fpr para evitar sesgo de cuadratura.
    if fpr[-1] <= max_fpr:
        # Toda la curva cae dentro de la región — devolver AUC estándar.
        from sklearn.metrics import auc as sk_auc
        return float(sk_auc(fpr, tpr) / max_fpr)
    idx = int(np.searchsorted(fpr, max_fpr, side="right"))
    fpr_clip = np.concatenate([fpr[:idx], [max_fpr]])
    # Interpolación lineal del TPR en el límite.
    if idx == 0:
        tpr_at_limit = tpr[0]
    else:
        t0, t1 = tpr[idx - 1], tpr[idx]
        f0, f1 = fpr[idx - 1], fpr[idx]
        if f1 == f0:
            tpr_at_limit = t1
        else:
            tpr_at_limit = t0 + (t1 - t0) * (max_fpr - f0) / (f1 - f0)
    tpr_clip = np.concatenate([tpr[:idx], [tpr_at_limit]])

    from sklearn.metrics import auc as sk_auc
    partial = sk_auc(fpr_clip, tpr_clip)
    # Normalizar al área máxima teóricamente alcanzable (rectángulo
    # max_fpr × 1) para que la métrica quede en [0, 1].
    return float(partial / max_fpr)


def partial_auc(y_true, y_probs, max_fpr: float = 0.05):
    """
    Partial AUC hasta max_fpr (estándar dermatológico: max_fpr=0.05).

    Wrapper sobre `sklearn.metrics.roc_auc_score(..., max_fpr=...)` que
    devuelve el valor re-normalizado a [0.5, 1.0] (McClish transformation,
    hecha internamente por sklearn). Útil como métrica de paper.

    Returns:
        float en [0.5, 1.0] o None si sklearn no está disponible.
    """
    if not HAS_SKLEARN:
        return None
    y_true = np.asarray(y_true).astype(int).ravel()
    y_probs = np.asarray(y_probs, dtype=np.float64).ravel()
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return None
    return float(roc_auc_score(y_true, y_probs, max_fpr=max_fpr))


def partial_auc_above_sensitivity(y_true, y_probs, min_sens: float = 0.80):
    """
    Partial AUC restringida a la región de **alta sensibilidad** (TPR ≥
    ``min_sens``). Estándar del reto ISIC 2020 / ISIC 2024 SLICE-3D
    (pAUC above 80% TPR).

    Por qué esta métrica y no `partial_auc` (max_fpr=0.05):
    -------------------------------------------------------
    `partial_auc` mide el régimen de alta ESPECIFICIDAD (FPR ≤ 0.05). El
    triaje binario de melanoma se despliega en el umbral clínico 0.20
    (sesgo a recall — NUNCA perder un melanoma; los falsos
    positivos los filtra el especialista de Nivel 2). El régimen operativo
    real es de **alta sensibilidad**, no de alta especificidad. Seleccionar
    el checkpoint por `partial_auc` empuja el modelo a la esquina conservadora
    de la ROC y degrada el recall@0.20 (el recall@0.20 puede caer mientras
    pAUC@spec≥0.95 sube). El estándar de la literatura melanoma-ML (ISIC 2020)
    es pAUC@TPR≥0.80.

    Implementación (truco oficial ISIC 2020/2024): pAUC sobre TPR ≥ ``min_sens``
    del problema original ≡ pAUC sobre FPR ≤ ``1 - min_sens`` del problema con
    etiquetas y predicciones invertidas (``1 - y_true``, ``-y_probs``). Esto
    reusa `sklearn.metrics.roc_auc_score(..., max_fpr=...)` (mismo motor que
    `partial_auc`, interpolación robusta de sklearn) y devuelve el valor
    McClish-normalizado a [0.5, 1.0] — **misma escala que `partial_auc`**, lo
    que preserva la consistencia de escala que `train.py` exige para
    `EarlyStopping.best_saved_score` (no mezclar ~0.5-1.0 pAUC con 0-1 AUC).

    Verificación de la simetría (ROC monótona):
        TPR_orig ≥ min_sens  ⟺  FPR_flipped ≤ (1 - min_sens)

    Args:
        y_true: [N] binarios {0, 1}.
        y_probs: [N] probabilidades (típicamente ya calibradas).
        min_sens: sensibilidad mínima de la región (default 0.80, ISIC 2020).
    Returns:
        float en [0.5, 1.0] o None si sklearn no está disponible / el set no
        tiene ambas clases (idéntico contrato a `partial_auc`).
    """
    if not HAS_SKLEARN:
        return None
    y_true = np.asarray(y_true).astype(int).ravel()
    y_probs = np.asarray(y_probs, dtype=np.float64).ravel()
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return None
    max_fpr = 1.0 - float(min_sens)
    if not (0.0 < max_fpr <= 1.0):
        # min_sens fuera de (0, 1] → región degenerada. Se devuelve None: el
        # caller (train.py) lo trata como error fatal con selection_metric activo.
        return None
    # Flip: convierte la región de alta sensibilidad del problema original en
    # la región de baja FPR del problema invertido (estándar ISIC 2020/2024).
    v_true = 1 - y_true
    v_pred = -y_probs
    return float(roc_auc_score(v_true, v_pred, max_fpr=max_fpr))


def brier_score(y_true, y_probs) -> float:
    """
    Brier score = mean((p - y)²).

    Combina calibración y discriminación: un score bajo requiere que las
    probabilidades estén cerca de la etiqueta verdadera. Rango [0, 1],
    menor es mejor. Complementa ECE — ECE solo mide gap de calibración
    promediado, Brier castiga tanto sobre-confianza como sub-confianza.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_probs = np.asarray(y_probs, dtype=np.float64).ravel()
    return float(np.mean((y_probs - y_true) ** 2))


def decision_curve_analysis(y_true, y_probs, thresholds=None):
    """
    Decision Curve Analysis (Vickers & Elkin, 2006).

    Para cada threshold de probabilidad t, calcula el "net benefit":

        NB(t) = (TP/N) - (FP/N) * (t / (1-t))

    Compara el modelo con dos estrategias triviales:
    - "treat all": biopsiar/derivar a todos (NB = prev - (1-prev) * t/(1-t))
    - "treat none": no biopsiar (NB = 0)

    Un modelo es clínicamente útil SOLO si su NB supera ambas estrategias
    triviales en el rango de umbrales relevantes (típicamente 0.05-0.30 para
    screening dermatológico, donde el coste de un falso negativo supera el
    de un falso positivo).

    Args:
        y_true: [N] binarios.
        y_probs: [N] probabilidades calibradas.
        thresholds: iterable de thresholds o None (default
            np.arange(0.05, 0.5, 0.05)).
    Returns:
        dict con arrays `thresholds`, `net_benefit_model`,
        `net_benefit_treat_all` para plotting.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_probs = np.asarray(y_probs, dtype=np.float64).ravel()
    if thresholds is None:
        thresholds = np.arange(0.05, 0.50, 0.05)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    n = y_true.size
    prevalence = y_true.mean()
    nb_model = np.zeros_like(thresholds)
    nb_all = np.zeros_like(thresholds)

    for i, t in enumerate(thresholds):
        pred = (y_probs >= t).astype(np.float64)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        odds = t / max(1.0 - t, 1e-8)
        nb_model[i] = (tp / n) - (fp / n) * odds
        # "Treat all": TP = positivos reales, FP = negativos reales.
        nb_all[i] = prevalence - (1.0 - prevalence) * odds

    return {
        "thresholds": thresholds,
        "net_benefit_model": nb_model,
        "net_benefit_treat_all": nb_all,
        "net_benefit_treat_none": np.zeros_like(thresholds),
    }


def selective_prediction_curve(
    y_true,
    y_probs,
    confidence_score=None,
    mode: str = "binary",
    n_coverage_levels: int = 21,
):
    """Curva de selective prediction (accuracy vs coverage).

    Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks",
    NeurIPS 2017 (arXiv:1705.08500). Permite al clínico elegir el punto
    operativo según contexto:
    - Coverage=1.0 (no rechazo): accuracy base del modelo.
    - Coverage=0.8 (20% deferidos al especialista): accuracy típicamente
      0.03-0.08 puntos mayor porque se rechazan los casos más inciertos.
    - Coverage=0.5: accuracy casi perfecta pero 50% de trabajo humano.

    Permite auditar el trade-off entre cobertura del modelo y precisión
    resultante. Complementa el threshold clínico 0.20 con
    una dimensión de ABSTENCIÓN explícita.

    Args:
        y_true: array [N] — labels binarios (0/1) o índices de clase [0, C-1].
        y_probs: array [N] o [N, C] — probabilidades calibradas. En binario,
            `y_probs` debe ser P(maligno) escalar por muestra. En multiclase,
            matrix [N, C].
        confidence_score: array [N] opcional. Si None, se deriva del contexto:
            binario → `|y_probs - 0.5| * 2` (distancia al umbral normalizada);
            multiclase → max prob por sample.
        mode: "binary" o "multiclass".
        n_coverage_levels: número de puntos en la curva (default 21 → pasos
            de 5% de coverage, incluye 0.0 y 1.0 como extremos).

    Returns:
        dict con keys:
          - "coverage": list[float] — fracción de muestras aceptadas.
          - "accuracy": list[float | None] — accuracy sobre las muestras aceptadas.
            `None` cuando coverage=0 (sin muestras → indefinido).
          - "threshold": list[float | None] — threshold de confianza aplicado.
            `None` cuando coverage=0 (sin muestras → indefinido).
          - "n_accepted": list[int] — muestras efectivamente aceptadas.
          - "mode": "binary" | "multiclass".

    Sin coste de entrenamiento: se calcula post-hoc sobre el conjunto de
    validación al cierre del entrenamiento. El JSON se persiste junto a los
    reportes de equidad (fairness).
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    if mode == "binary":
        # y_probs es P(maligno); predicción = argmax entre {0, 1} con threshold 0.5.
        # El threshold clínico 0.20 NO se usa aquí — la curva es agnóstica al
        # umbral operativo; el clínico lo calibra a posteriori leyendo la curva.
        if y_probs.ndim > 1:
            y_probs = y_probs.reshape(-1)
        pred = (y_probs >= 0.5).astype(np.int64)
        if confidence_score is None:
            # |p - 0.5| * 2 → [0, 1], máximo en los extremos (prob certero).
            confidence_score = np.abs(y_probs - 0.5) * 2.0
        correct = (pred == y_true.astype(np.int64))
    elif mode == "multiclass":
        if y_probs.ndim != 2:
            raise ValueError(
                f"mode='multiclass' requiere y_probs 2D [N, C], "
                f"recibido shape {y_probs.shape}"
            )
        pred = np.argmax(y_probs, axis=1)
        if confidence_score is None:
            confidence_score = np.max(y_probs, axis=1)
        correct = (pred == y_true.astype(np.int64))
    else:
        raise ValueError(f"mode='{mode}' no válido (use 'binary' o 'multiclass')")

    if len(confidence_score) != len(y_true):
        raise ValueError(
            f"confidence_score len={len(confidence_score)} != "
            f"y_true len={len(y_true)}"
        )

    # Coverage levels de 0 (nada aceptado, trivial) a 1 (todo aceptado).
    # np.linspace(0, 1, 21) → [0.0, 0.05, 0.10, ..., 1.0]
    coverages = np.linspace(0.0, 1.0, n_coverage_levels)
    n_total = len(y_true)

    # Ordenamos por confianza descendente: a coverage=k/N aceptamos los k más
    # confiados y rechazamos el resto.
    order = np.argsort(-confidence_score)
    sorted_correct = correct[order]
    sorted_conf = confidence_score[order]

    result_coverage = []
    result_accuracy = []
    result_threshold = []
    result_n_accepted = []

    for cov in coverages:
        n_accept = int(round(cov * n_total))
        if n_accept == 0:
            # Usar `None` en vez de `float("nan")`/`float("inf")`:
            # `json.dumps(float("inf"))` produce `"Infinity"`, que NO es JSON
            # válido según RFC 7159 — `json.dumps(..., allow_nan=False)` falla
            # con ValueError, y parsers strict (jq, Go, Rust, dashboards)
            # rechazan el archivo como corrupto. El fichero
            # `selective_prediction_curve_{mode}.json` se genera al cierre del
            # training y debe ser portable.
            # Semánticamente equivalente: coverage=0 ⇒ no hay muestras aceptadas
            # ⇒ accuracy/threshold indefinidos. `None → "null"` es JSON estándar
            # y los consumidores Python siguen pudiendo filtrar `[r for r in
            # data["threshold"] if r is not None]`.
            result_coverage.append(0.0)
            result_accuracy.append(None)
            result_threshold.append(None)
            result_n_accepted.append(0)
            continue
        accepted_correct = sorted_correct[:n_accept]
        threshold = float(sorted_conf[n_accept - 1])
        accuracy = float(accepted_correct.mean())
        result_coverage.append(float(n_accept) / float(n_total))
        result_accuracy.append(accuracy)
        result_threshold.append(threshold)
        result_n_accepted.append(int(n_accept))

    return {
        "coverage": result_coverage,
        "accuracy": result_accuracy,
        "threshold": result_threshold,
        "n_accepted": result_n_accepted,
        "mode": mode,
        "n_total": int(n_total),
    }
