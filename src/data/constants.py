"""
Constantes compartidas del pipeline de datos.

Fuente de verdad única para mapeos de etiquetas y metadatos clínicos.
Importar desde aquí en dataset.py, 07_filter_multiclass.py, y predict.py
para evitar divergencias silenciosas.
"""

# Mapeo de diagnósticos para el clasificador multiclase (8 clases)
MULTICLASS_LABEL_MAP = {
    "Melanoma": 0,
    "Nevus": 1,
    "Basal Cell Carcinoma": 2,
    "Squamous Cell Carcinoma": 3,
    "Benign Keratosis": 4,
    "Actinic Keratosis": 5,
    "Vascular Lesion": 6,
    "Dermatofibroma": 7,
}

# Mapeo inverso (índice -> nombre) para inferencia
MULTICLASS_INDEX_MAP = {v: k for k, v in MULTICLASS_LABEL_MAP.items()}

# Mapeo binario (triaje)
BINARY_LABEL_MAP = {"benign": 0, "malignant": 1}

# Mapeos de metadatos clínicos.
# SEX_MAP incluye aliases defensivos para las abreviaturas clínicas estándar: los
# sistemas hospitalarios y el estándar HL7 FHIR emiten "M"/"F" en lugar de
# "male"/"female". Sin estos aliases, tras normalizar a minúsculas, el valor "m" no
# se encontraría en el mapa y caería a "unknown", perdiendo la señal demográfica
# que alimenta la fusión de metadatos. Los aliases "u" y "" cubren el campo
# "Unknown" abreviado de sistemas legacy y los campos vacíos. Todos apuntan a
# códigos ya existentes (0/1/2), de modo que NUM_SEX_CODES —derivado de
# max(values)+1, no de len()— no cambia y los checkpoints guardados siguen siendo
# compatibles.
SEX_MAP = {
    "male": 0,
    "m": 0,
    "female": 1,
    "f": 1,
    "unknown": 2,
    "u": 2,
    "": 2,
}

SITE_MAP = {
    "anterior torso": 0,
    "chest": 0,
    "torso": 0,
    "lateral torso": 0,
    "upper extremity": 1,
    "arms": 1,
    "posterior torso": 2,
    "back": 2,
    "lower extremity": 3,
    "legs": 3,
    "head/neck": 4,
    "palms/soles": 5,
    "oral/genital": 6,
    "unknown": 7,
    "nan": 7,
    # Aliases de HAM10000: su columna `localization` usa una ontología distinta de
    # la de ISIC. Sin estos aliases, ~4000 muestras caerían silenciosamente a
    # unknown (código 7), degradando la señal anatómica que alimenta FiLM. El mismo
    # problema con "torso"/"chest"/"lateral torso" había costado 2657 muestras de ISIC.
    "head": 4,
    "scalp": 4,
    "ear": 4,
    "face": 4,
    "neck": 4,
    "trunk": 0,
    "abdomen": 0,
    # hand/foot/acral -> palms/soles: dermatológicamente comparten patrón glabro
    # y son el sitio clave del melanoma acral lentiginoso (distribución única).
    "hand": 5,
    "foot": 5,
    "acral": 5,
    "genital": 6,
}

# Diagnósticos excluidos del clasificador multiclase
MULTICLASS_EXCLUDE_LIST = [
    "Benign (Unspecified)",
    "Malignant (Unspecified)",
    "Unknown",
    "Benign Other",
    "Melanoma (Potential)",
]

# Método de normalización de color aplicado en el preprocesado.
# Único valor soportado: "shades_of_gray" (constancia de color Minkowski p=6),
# validado empíricamente como suficiente sobre el dataset completo. Lo consumen
# dataset.py y predict.py para mantener la paridad entre entrenamiento e inferencia.
# build_stain_transform() en preprocessing.py lanza ValueError si el valor no es
# 'shades_of_gray', en lugar de degradar en silencio.
STAIN_METHOD = "shades_of_gray"


# Índices de clases malignas en MULTICLASS_LABEL_MAP.
# Usado por: predict.py (estado CRITICAL), distillation.py (teacher-student loss).
# AK (5) es precancerosa, NO maligna -> queda fuera.
# Derivado de los nombres del map para sobrevivir reordenamientos futuros.
MALIGNANT_INDICES = [
    MULTICLASS_LABEL_MAP["Melanoma"],
    MULTICLASS_LABEL_MAP["Basal Cell Carcinoma"],
    MULTICLASS_LABEL_MAP["Squamous Cell Carcinoma"],
]

# Nombres de diagnósticos malignos, derivados de MALIGNANT_INDICES y
# MULTICLASS_INDEX_MAP. Los consume 02_unify.py para generar `benign_malignant` sin
# literales hardcoded: añadir una clase maligna a MALIGNANT_INDICES se propaga
# automáticamente aquí y a 02_unify.py, predict.py (CRITICAL) y distillation.py
# (teacher-student).
#
# Validación de integridad en tiempo de import: si alguien reordena
# MULTICLASS_LABEL_MAP sin actualizar esta lista, la validación falla antes de que
# cualquier inferencia pueda producir un diagnóstico incorrecto. Usamos
# `raise ValueError` y no `assert` porque `assert` se desactiva con `python -O`.
_EXPECTED_MALIGNANT_NAMES = {"Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"}
for _idx in MALIGNANT_INDICES:
    if _idx not in MULTICLASS_INDEX_MAP:
        raise ValueError(
            f"MALIGNANT_INDICES contiene indice {_idx} que no existe en MULTICLASS_INDEX_MAP. "
            f"Indices validos: {list(MULTICLASS_INDEX_MAP.keys())}"
        )
    if MULTICLASS_INDEX_MAP[_idx] not in _EXPECTED_MALIGNANT_NAMES:
        raise ValueError(
            f"MALIGNANT_INDICES[{_idx}] = {MULTICLASS_INDEX_MAP[_idx]!r} no es una malignidad "
            f"conocida. Esperadas: {_EXPECTED_MALIGNANT_NAMES}"
        )

# Validación inversa: detectar clases con nombre maligno que NO estén en
# MALIGNANT_INDICES. Si se añadiera, p.ej., "Merkel Cell Carcinoma" al label_map
# sin actualizar MALIGNANT_INDICES, predict.py no dispararía CRITICAL para esa clase.
# Keywords que indican malignidad clínica inequívoca:
_MALIGNANT_KEYWORDS = {"carcinoma", "melanoma", "sarcoma", "lymphoma"}
# Excepciones explícitas: clases cuyo nombre contiene un keyword pero NO son malignas
# (vacío por ahora; documentar aquí si se añade alguna, p.ej. "keratoacanthoma").
_MALIGNANT_KEYWORD_EXCEPTIONS: set = set()
for _name, _idx in MULTICLASS_LABEL_MAP.items():
    _name_lower = _name.lower()
    if (
        any(kw in _name_lower for kw in _MALIGNANT_KEYWORDS)
        and _idx not in MALIGNANT_INDICES
        and _name not in _MALIGNANT_KEYWORD_EXCEPTIONS
    ):
        raise ValueError(
            f"'{_name}' (idx={_idx}) contiene keyword maligno pero NO esta en "
            f"MALIGNANT_INDICES={MALIGNANT_INDICES}. Si es maligna, agregarla a "
            f"MALIGNANT_INDICES. Si es precancerosa/benigna, agregarla a "
            f"_MALIGNANT_KEYWORD_EXCEPTIONS en constants.py."
        )

MALIGNANT_DIAGNOSIS_NAMES = frozenset(MULTICLASS_INDEX_MAP[i] for i in MALIGNANT_INDICES)


# PRECANCEROUS y REFERRAL: separación clínica de dos conceptos.
#
# Fundamento clínico (guías AAD/EADV, WHO 2018):
#   La queratosis actínica (AK) es una lesión precancerosa: 0.5-1% de progresión
#   anual a SCC y 6-10% de riesgo acumulado a lo largo de la vida. Requiere
#   intervención dermatológica (criocirugía, 5-FU, imiquimod, terapia fotodinámica).
#   Tratarla como benigna en el triaje binario sería clínicamente engañoso: un
#   paciente con AK y triaje prob<0.20 caería en GREEN y progresaría a SCC sin
#   detección.
#
# Separación de conceptos:
#   - MALIGNANT_INDICES: cáncer invasivo confirmado (Mel/BCC/SCC). Dispara CRITICAL
#     en la cascada → fast-track oncológico. Se mantiene puro para no saturar
#     oncología con lesiones no invasivas.
#   - PRECANCEROUS_INDICES: lesiones del continuum del cáncer pero NO invasivas.
#     Disparan AMBER_PRECANCEROUS → intervención dermatológica, sin urgencia
#     oncológica.
#   - REFERRAL_INDICES = MALIGNANT ∪ PRECANCEROUS: todo lo que requiere derivación a
#     especialista. Lo usa el triaje binario (02_unify.py deriva benign_malignant
#     desde aquí) y la destilación teacher-student (el alumno aprende que la AK suma
#     a la probabilidad binaria del profesor).
#
# MALIGNANT y REFERRAL coexisten porque modelan dos preguntas clínicas distintas:
# "¿es cáncer invasivo?" (MALIGNANT, define CRITICAL) frente a "¿necesita
# dermatólogo?" (REFERRAL, define la derivación).
PRECANCEROUS_INDICES = [
    MULTICLASS_LABEL_MAP["Actinic Keratosis"],
]

PRECANCEROUS_DIAGNOSIS_NAMES = frozenset(
    MULTICLASS_INDEX_MAP[i] for i in PRECANCEROUS_INDICES
)

REFERRAL_INDICES = MALIGNANT_INDICES + PRECANCEROUS_INDICES
REFERRAL_DIAGNOSIS_NAMES = MALIGNANT_DIAGNOSIS_NAMES | PRECANCEROUS_DIAGNOSIS_NAMES

# Validación en tiempo de import de PRECANCEROUS_INDICES, análoga a la de
# MALIGNANT_INDICES. Usamos `raise ValueError` y no `assert` para que `python -O`
# no pueda desactivar esta comprobación de integridad clínica.
_EXPECTED_PRECANCEROUS_NAMES = {"Actinic Keratosis"}
for _idx in PRECANCEROUS_INDICES:
    if _idx not in MULTICLASS_INDEX_MAP:
        raise ValueError(
            f"PRECANCEROUS_INDICES contiene indice {_idx} que no existe en "
            f"MULTICLASS_INDEX_MAP. Indices validos: "
            f"{list(MULTICLASS_INDEX_MAP.keys())}"
        )
    if MULTICLASS_INDEX_MAP[_idx] not in _EXPECTED_PRECANCEROUS_NAMES:
        raise ValueError(
            f"PRECANCEROUS_INDICES[{_idx}] = {MULTICLASS_INDEX_MAP[_idx]!r} "
            f"no es una lesion precancerosa conocida. Esperadas: "
            f"{_EXPECTED_PRECANCEROUS_NAMES}"
        )

# Validación inversa: detectar clases con nombre claramente precanceroso que NO
# estén en PRECANCEROUS_INDICES (análoga a la de MALIGNANT_INDICES). Si se añadiera
# "Bowen Disease" o "SCC In Situ" al label_map sin actualizar PRECANCEROUS_INDICES,
# predict.py no dispararía AMBER_PRECANCEROUS para esa clase y caería en silencio
# como benigna en la cascada, con el riesgo clínico que eso implica.
# Keywords que indican precancerosidad clínica inequívoca:
_PRECANCEROUS_KEYWORDS = {"actinic", "in situ", "in-situ", "bowen"}
# Excepciones explícitas: clases cuyo nombre contiene un keyword pero NO son
# precancerosas (vacío por ahora; documentar aquí si surge alguna).
_PRECANCEROUS_KEYWORD_EXCEPTIONS: set = set()
for _name, _idx in MULTICLASS_LABEL_MAP.items():
    _name_lower = _name.lower()
    if (
        any(kw in _name_lower for kw in _PRECANCEROUS_KEYWORDS)
        and _idx not in PRECANCEROUS_INDICES
        and _idx not in MALIGNANT_INDICES  # malignas confirmadas tienen su propio bucket
        and _name not in _PRECANCEROUS_KEYWORD_EXCEPTIONS
    ):
        raise ValueError(
            f"'{_name}' (idx={_idx}) contiene keyword precanceroso pero NO "
            f"esta en PRECANCEROUS_INDICES={PRECANCEROUS_INDICES} ni en "
            f"MALIGNANT_INDICES={MALIGNANT_INDICES}. Si es precancerosa, "
            f"agregarla a PRECANCEROUS_INDICES en constants.py. Si es benigna, "
            f"agregarla a _PRECANCEROUS_KEYWORD_EXCEPTIONS."
        )

# Propiedad de inclusión: MALIGNANT ⊆ REFERRAL. Garantiza que comprobar
# `pred_idx in REFERRAL_INDICES` cubre todos los `pred_idx in MALIGNANT_INDICES`.
# Si alguien rompe la derivación `REFERRAL = MALIGNANT + PRECANCEROUS`, esta
# comprobación lo detecta antes de que la destilación o el triaje binario operen
# sobre datos inconsistentes.
if not set(MALIGNANT_INDICES).issubset(set(REFERRAL_INDICES)):
    raise ValueError(
        f"REFERRAL_INDICES={REFERRAL_INDICES} debe contener TODOS los "
        f"MALIGNANT_INDICES={MALIGNANT_INDICES}. Propiedad: MALIGNANT ⊆ "
        f"REFERRAL."
    )
if not set(PRECANCEROUS_INDICES).issubset(set(REFERRAL_INDICES)):
    raise ValueError(
        f"REFERRAL_INDICES={REFERRAL_INDICES} debe contener TODOS los "
        f"PRECANCEROUS_INDICES={PRECANCEROUS_INDICES}. Propiedad: "
        f"PRECANCEROUS ⊆ REFERRAL."
    )
# Propiedad de disjunción: MALIGNANT ∩ PRECANCEROUS = ∅. Una misma clase no puede
# ser a la vez cáncer invasivo y precancerosa: son niveles clínicos distintos. Si
# se violara, la cascada en predict.py daría un estado ambiguo (las ramas CRITICAL
# y AMBER_PRECANCEROUS compiten por cuál se evalúa primero).
if set(MALIGNANT_INDICES) & set(PRECANCEROUS_INDICES):
    raise ValueError(
        f"MALIGNANT_INDICES={MALIGNANT_INDICES} y "
        f"PRECANCEROUS_INDICES={PRECANCEROUS_INDICES} no pueden compartir "
        f"clases. Una clase es cancer invasivo (CRITICAL) O precancerosa "
        f"(AMBER_PRECANCEROUS), nunca ambas."
    )

# Umbral clínico del triaje binario, sesgado hacia recall para minimizar los
# melanomas no detectados. Si prob >= CLINICAL_THRESHOLD_BINARY, la imagen se deriva
# al especialista multiclase. Solo se aplica en inferencia (predict.py), nunca en
# las métricas de entrenamiento (que usan 0.50). No subir sin justificación clínica
# documentada.
CLINICAL_THRESHOLD_BINARY = 0.20

# Tamaños de imagen nativos por tipo de backbone y modo de entrenamiento.
# CNN (EfficientNet): 300px binario, 380px multiclase (resoluciones nativas de B3/B4).
# ViT (DINOv2): 294px binario, 378px multiclase (múltiplos de patch_size=14).
# Alimentar 300px a un ViT con patch_size=14 fuerza un resize interno 300→294
# que degrada features dermatoscópicas finas (doble interpolación).
# Consumido por dataset.py (get_transforms) y predict.py (preprocess).
IMAGE_SIZES = {
    "cnn": {"binary": 300, "multiclass": 380},
    "vit": {"binary": 294, "multiclass": 378},
}

# Backbones por defecto, centralizados. Los usan predict.py y
# scripts/calibrate_conformal.py como fallback para checkpoints antiguos (sin el
# wrapper build_backbone). Mantener sincronizados si se cambia de backbone.
DEFAULT_BINARY_BACKBONE = "tf_efficientnet_b3.ns_jft_in1k"
DEFAULT_MULTICLASS_BACKBONE = "tf_efficientnet_b4.ns_jft_in1k"

# Dimensiones derivadas para sincronizar los tamaños de embedding en
# architectures.py. Ambos mapas tienen aliases, así que `len()` NO refleja el número
# de códigos únicos; el tamaño correcto del Embedding es `max(values)+1`:
#   SEX_MAP:  len()=7  (aliases m/f/u/vacío),  max(values)+1=3  (male/female/unknown)
#   SITE_MAP: len()=25 (aliases HAM10000+ISIC), max(values)+1=8 (códigos 0..7)
# Usar `len()` produciría tablas de embedding más grandes de lo necesario y, peor,
# rompería la compatibilidad con los checkpoints guardados: un sex_emb de shape
# (3, 4) no cargaría en un modelo que esperase (7, 4).
NUM_SEX_CODES = max(SEX_MAP.values()) + 1     # 3
NUM_SITE_CODES = max(SITE_MAP.values()) + 1   # 8

# Códigos `unknown` derivados como fuente de verdad única para el Meta-Dropout y los
# clamps defensivos de SkinCancerModel. Sin ellos, architectures.py hardcodearía los
# literales 2/7/0.5: si alguien reordenara SEX_MAP/SITE_MAP (p.ej.
# {unknown: 0, male: 1, female: 2}), el Meta-Dropout inyectaría el código equivocado
# en entrenamiento y el clamp mapearía códigos válidos a "unknown" en silencio.
#
# Contrato: SEX_UNKNOWN_CODE / SITE_UNKNOWN_CODE siempre apuntan al índice del valor
# textual "unknown" en cada mapa, sea cual sea la numeración.
SEX_UNKNOWN_CODE = SEX_MAP["unknown"]         # 2 (actual)
SITE_UNKNOWN_CODE = SITE_MAP["unknown"]       # 7 (actual)

# Normalización de edad: `age_norm = fillna(50) / 100 ∈ [0, 1]`. El centro fijo de
# 50 años es el valor clínicamente neutral para edad desconocida. Nunca usar la
# media del batch: introduciría fuga de información entre train, val y test.
# AGE_UNKNOWN_YEARS es la fuente canónica; AGE_NORM_UNKNOWN es su forma normalizada,
# que consume architectures.py.
AGE_UNKNOWN_YEARS = 50.0
AGE_NORM_UNKNOWN = AGE_UNKNOWN_YEARS / 100.0  # 0.5

# Constantes para el WeightedRandomSampler balanceado por subgrupo y para la
# calibración de umbral por sitio anatómico. El sampler base de dataset.py balancea
# solo por clase (frecuencia inversa en binario, Effective Number β=0.999 en
# multiclase), sin estratificar por subgrupo demográfico. Cuando un sitio tiene una
# tasa base de malignidad extrema (palms/soles, 58% en train), el modelo aprende a
# usar site_code como atajo y produce especificidad ~0% en ese sitio.
# AGE_BUCKETS_DEFAULT replica los bins de fairness_report (evaluation.py) para que el
# muestreo de entrenamiento y la auditoría de equidad usen los mismos rangos:
# cualquier cambio debe mantener ambos sincronizados.
# Tuplas (lo, hi, label). El hi=float('inf') del último bin captura edades
# sentinela/outlier (>=150) para que no queden fuera de la auditoría en silencio.
AGE_BUCKETS_DEFAULT = (
    (0.0, 30.0, "0-30"),
    (30.0, 50.0, "30-50"),
    (50.0, 70.0, "50-70"),
    (70.0, float("inf"), "70+"),
)
# Mínimo de muestras por subgrupo para considerar fiable el rebalanceo. Los grupos
# con n < MIN_GROUP_N_DEFAULT NO se rebalancean (factor=1.0), para evitar memorizar
# unas pocas muestras únicas (el oversampling extremo provoca memorización). El valor
# 50 es empírico: oral/genital (n=56 en train) pasa; un sitio futuro con n=10 no, y
# cae al peso de clase base.
MIN_GROUP_N_DEFAULT = 50
# Suavizado del balance por grupo: 0.7 modera la corrección (0 preserva el baseline
# exacto, 1 fuerza balance total). Con palms/soles, 0.7 reduce el ratio
# maligno/benigno de los pesos de 56× a 3.3× sin memorización catastrófica.
GROUP_BALANCE_SMOOTH_DEFAULT = 0.7

# Límites de la calibración de umbral por sitio anatómico. CLINICAL_THRESHOLD_BINARY
# (0.20) es el umbral global; el umbral por sitio puede divergir, pero nunca fuera del
# rango clínicamente aceptable: por debajo de 0.10 trivializaría el triaje (todo
# positivo) y por encima de 0.50 contradiría el sesgo hacia recall. El calibrador
# clampea a este rango el umbral óptimo que encuentra.
PER_SITE_THRESHOLD_MIN = 0.10
PER_SITE_THRESHOLD_MAX = 0.50
# Mínimo de positivos por subgrupo: con menos de 10 malignos, el intervalo de
# confianza de Wilson al 95% es inutilizable (~80 puntos de rango para recall=0.5).
# Lo usan fairness_report y PerSiteThresholdCalibrator para excluir sitios con
# muestreo insuficiente.
MIN_POSITIVE_DEFAULT = 10

# Validaciones de integridad en tiempo de import.
if not (0 <= SEX_UNKNOWN_CODE < NUM_SEX_CODES):
    raise ValueError(
        f"SEX_UNKNOWN_CODE={SEX_UNKNOWN_CODE} fuera de rango [0, {NUM_SEX_CODES - 1}]. "
        "Revisar SEX_MAP en constants.py."
    )
if not (0 <= SITE_UNKNOWN_CODE < NUM_SITE_CODES):
    raise ValueError(
        f"SITE_UNKNOWN_CODE={SITE_UNKNOWN_CODE} fuera de rango [0, {NUM_SITE_CODES - 1}]. "
        "Revisar SITE_MAP en constants.py."
    )
if not (0.0 <= AGE_NORM_UNKNOWN <= 1.0):
    raise ValueError(
        f"AGE_NORM_UNKNOWN={AGE_NORM_UNKNOWN} fuera de [0, 1]. "
        "Centro fijo de edad: age_norm = fillna(50)/100."
    )

# Validar los límites de la calibración por sitio y del balance por subgrupo.
if not (0.0 < PER_SITE_THRESHOLD_MIN < PER_SITE_THRESHOLD_MAX < 1.0):
    raise ValueError(
        f"PER_SITE_THRESHOLD_MIN={PER_SITE_THRESHOLD_MIN}, MAX={PER_SITE_THRESHOLD_MAX} "
        f"deben cumplir 0 < MIN < MAX < 1."
    )
if MIN_GROUP_N_DEFAULT < 1:
    raise ValueError(
        f"MIN_GROUP_N_DEFAULT={MIN_GROUP_N_DEFAULT} debe ser >= 1."
    )
if MIN_POSITIVE_DEFAULT < 1:
    raise ValueError(
        f"MIN_POSITIVE_DEFAULT={MIN_POSITIVE_DEFAULT} debe ser >= 1."
    )
# Validar AGE_BUCKETS_DEFAULT: bins consecutivos sin overlap, primero comienza en 0,
# último termina en inf, todos los bins son no-vacios (lo < hi).
_prev_hi = 0.0
for _i, (_lo, _hi, _label) in enumerate(AGE_BUCKETS_DEFAULT):
    if _lo != _prev_hi:
        raise ValueError(
            f"AGE_BUCKETS_DEFAULT[{_i}]: bin '{_label}' empieza en {_lo} pero "
            f"el anterior termina en {_prev_hi}. Los bins deben ser consecutivos "
            "sin solapamiento."
        )
    if _lo >= _hi:
        raise ValueError(
            f"AGE_BUCKETS_DEFAULT[{_i}]: bin '{_label}' tiene lo={_lo} >= hi={_hi}. "
            "Los bins deben ser no vacíos."
        )
    _prev_hi = _hi
if AGE_BUCKETS_DEFAULT[-1][1] != float("inf"):
    raise ValueError(
        "AGE_BUCKETS_DEFAULT: el último bin debe terminar en float('inf') para "
        "capturar edades sentinela/outlier sin perderlas silenciosamente."
    )
del _prev_hi, _i, _lo, _hi, _label


# Fuente de verdad única para la codificación de metadata clínica.
# El entrenamiento (dataset.py) y la inferencia (predict.py) deben aplicar
# exactamente la misma transformación de `(sex, anatom_site, age)` crudos a códigos
# numéricos. El punto delicado es `.strip()`: un valor con espacios sobrantes
# (`"male "`, típico de CSVs exportados desde Excel o de datos tecleados a mano)
# caería en silencio al código `unknown` si no se recorta, degradando la señal
# demográfica que alimenta FiLM de forma invisible en las métricas globales.
#
# Contrato: `encode_clinical_metadata_scalar` es la única forma válida de convertir
# un triple `(sex, anatom_site, age)` crudo a los valores numéricos que consume el
# modelo. Es defensiva ante None, NaN (float de Python o escalar de numpy), espacios,
# mayúsculas/minúsculas y tipos no esperados. La usa `predict.py::preprocess`.
# `dataset.py::__init__` aplica la misma secuencia `str → strip → lower → map →
# fallback`, pero vectorizada con pandas por rendimiento sobre las 442K filas; no
# existe una versión vectorizada como función aparte porque pandas ya ofrece
# `.str.strip().str.lower().map()` de forma nativa. `02_unify.py` normaliza el CSV
# fuente con la misma cadena para que 06_verify.py no vea categorías duplicadas tipo
# `"torso "` frente a `"torso"`.
import math as _math


def normalize_metadata_string_scalar(value, fallback: str = "unknown") -> str:
    """Normaliza un valor escalar de metadata a string lowercase sin whitespace.

    Defensivo ante None, NaN (Python/numpy), tipos no-string. Devuelve `fallback`
    si el valor es None, NaN, o si la conversión a str falla. Mantiene paridad
    exacta con la cadena vectorizada de pandas en `dataset.py`.
    """
    if value is None:
        return fallback
    try:
        as_float = float(value)
        if _math.isnan(as_float):
            return fallback
    except (TypeError, ValueError):
        pass
    try:
        normalized = str(value).strip().lower()
    except Exception:
        return fallback
    if normalized == "" or normalized == "nan":
        return fallback
    return normalized


def encode_clinical_metadata_scalar(sex, anatom_site, age) -> tuple:
    """Codifica (sex, anatom_site, age) -> (age_norm, sex_code, site_code).

    El orden de retorno `(age_norm, sex_code, site_code)` coincide con el orden del
    vector de metadata que consume el modelo (`meta = [age_norm, sex_code,
    site_code]`): predict.py y dataset.py lo desempaquetan en ese mismo orden, un
    invariante posicional que ambos módulos deben respetar.

    Garantías:
      - `sex_code` ∈ [0, NUM_SEX_CODES); "unknown" si no está en SEX_MAP.
      - `site_code` ∈ [0, NUM_SITE_CODES); "unknown" si no está en SITE_MAP.
      - `age_norm` ∈ [0, 1] tras el clamp. Centro fijo AGE_NORM_UNKNOWN si `age` es
        None/NaN/no convertible (nunca la media del batch).
      - El aviso de edades fuera del rango clínico [0, 120] lo emite el caller
        (predict.py por muestra, dataset.py de forma agregada en __init__). Esta
        función solo clampea, no registra avisos.

    Es la fuente de verdad única para esta codificación, consumida por predict.py y
    validada contra la cadena vectorizada de dataset.py.
    """
    sex_key = normalize_metadata_string_scalar(sex, fallback="unknown")
    site_key = normalize_metadata_string_scalar(anatom_site, fallback="unknown")
    sex_code = int(SEX_MAP.get(sex_key, SEX_UNKNOWN_CODE))
    site_code = int(SITE_MAP.get(site_key, SITE_UNKNOWN_CODE))

    try:
        age_f = float(age) if age is not None else AGE_UNKNOWN_YEARS
        if _math.isnan(age_f):
            age_f = AGE_UNKNOWN_YEARS
    except (TypeError, ValueError):
        age_f = AGE_UNKNOWN_YEARS
    # Clamp a [0, 1] tras división por 100. El caller decide si emite warning
    # por edades fuera del rango clínico [0, 120].
    age_norm = max(0.0, min(1.0, age_f / 100.0))

    return age_norm, sex_code, site_code
