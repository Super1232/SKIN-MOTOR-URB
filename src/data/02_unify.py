"""
Script de Unificación de Datasets (Data Fusion).

Este módulo resuelve el problema de la "Heterogeneidad de Fuentes".
Fusionamos múltiples bases de datos públicas (ISIC 2019, 2020, 2024 y HAM10000) en un esquema común.

Desafíos abordados:
1. Estandarización de Etiquetas: Mapeo de terminologías dispares (ej. 'nv' vs 'Nevus').
2. Normalización de Metadatos: Unificación de formatos para Edad, Sexo y Sitio Anatómico.
3. Filtrado de Calidad: Eliminación de diagnósticos ambiguos ('Unknown') para reducir el ruido de etiqueta.

Salida: data/processed/02_all_data.csv
"""

import pandas as pd
import warnings
from pathlib import Path
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.data.constants import (
    MULTICLASS_LABEL_MAP,
    MULTICLASS_EXCLUDE_LIST,
    REFERRAL_DIAGNOSIS_NAMES,
)


def _drop_missing_paths(df, dataset_name, diagnosis_col="diagnosis", fatal_pct=5.0):
    """
    Elimina filas con image_path NaN y audita la pérdida.

    Una pérdida >5% es fatal: indica un dataset mal montado (zip corrupto en
    Colab, symlink roto, etc.). Una pérdida asimétrica por diagnóstico
    introduciría sesgo silencioso en el entrenamiento.
    """
    before = len(df)
    missing = df["image_path"].isna()
    lost = int(missing.sum())
    if lost > 0:
        pct = 100.0 * lost / max(before, 1)
        # Usamos `warnings.warn` y no `print`: el aviso debe ser capturable por un
        # handler de logging estructurado y por pytest.warns(), mientras que un print
        # se perdería mezclado en stdout con la barra de progreso.
        if diagnosis_col in df.columns:
            dist = df.loc[missing, diagnosis_col].value_counts().to_dict()
            warnings.warn(
                f"[{dataset_name}] {lost}/{before} imagenes perdidas ({pct:.1f}%) por image_path faltante. "
                f"Distribucion por diagnostico: {dist}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"[{dataset_name}] {lost}/{before} imagenes perdidas ({pct:.1f}%) por image_path faltante.",
                RuntimeWarning,
                stacklevel=2,
            )
        if pct > fatal_pct:
            raise ValueError(
                f"[{dataset_name}] Perdida masiva de imagenes ({pct:.1f}% > {fatal_pct}%). "
                f"Probable dataset mal montado o zip corrupto. Abortando para evitar "
                f"entrenamiento con subconjunto sesgado."
            )
    return df.dropna(subset=["image_path"]).reset_index(drop=True)


# Configuración de Mapeos (Ontología Unificada)
DX_MAP = {
    "nv": "Nevus",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "ak": "Actinic Keratosis",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma",
    "scc": "Squamous Cell Carcinoma",
    "unk": "Unknown",
}


def get_image_path_dict(folder_path):
    """Indexa recursivamente las imágenes para mapear ID -> Ruta Relativa desde data/raw/."""
    print(f"Indexando imágenes en {folder_path}...")
    image_paths = {}
    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.png"]
    count = 0
    collisions = 0
    for ext in patterns:
        for path in Path(folder_path).rglob(ext):
            img_id = path.stem
            # Almacenar ruta relativa desde la raíz del proyecto.
            # Esto garantiza compatibilidad entre máquinas (local, Colab, etc.)
            # sin necesitar parches en dataset.py.
            abs_path = str(path.resolve())
            marker = "data/raw/"
            if marker in abs_path:
                new_path = abs_path[abs_path.find(marker):]
            else:
                new_path = str(path)
            # Guard: detectar colisiones de nombre entre datasets distintos
            if img_id in image_paths and image_paths[img_id] != new_path:
                if collisions < 5:  # limitar el logging para no inundar la consola
                    warnings.warn(
                        f"COLISION img_id '{img_id}': {image_paths[img_id]} -> {new_path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                collisions += 1
            image_paths[img_id] = new_path
            count += 1
    if collisions > 0:
        raise ValueError(
            f"COLISIÓN DE IDs DE IMAGEN: {collisions} img_ids colisionan entre datasets. "
            f"Esto corrompe el mapeo image→path (el último sobrescribe al primero). "
            f"Investigar duplicados entre datasets o renombrar imágenes."
        )
    print(f" {count} imágenes indexadas.")
    return image_paths


def process_isic_2019(base_path):
    """Procesamiento específico para la estructura de ISIC 2019."""
    print("\n Procesando ISIC 2019 (+Metadata)...")
    path = Path(base_path)

    gt_file = list(path.rglob("*GroundTruth.csv"))[0]
    meta_file = list(path.rglob("*Metadata.csv"))[0]

    df_gt = pd.read_csv(gt_file)
    df_meta = pd.read_csv(meta_file)

    df = pd.merge(df_gt, df_meta, on="image")

    # Decodificación One-Hot a Etiqueta única
    diag_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

    def get_label(row):
        for col in diag_cols:
            if row[col] == 1.0:
                return col
        return "UNK"

    df["diagnosis_code"] = df.apply(get_label, axis=1)

    # Validar integridad del one-hot encoding: cada fila debe tener exactamente 1 columna activa.
    # Si hay multi-hot (>1) solo se tomó la primera; si hay zero-hot (0) se asignó UNK.
    hot_counts = df[diag_cols].sum(axis=1)
    multi_hot = df[hot_counts > 1]
    zero_hot = df[hot_counts == 0]
    if not multi_hot.empty:
        # Varias columnas activas en datos one-hot indican corrupción de la fuente.
        # Tomar en silencio la primera enmascararía datos corruptos que asignarían
        # diagnósticos incorrectos, así que abortamos.
        raise ValueError(
            f"ISIC 2019: {len(multi_hot)} filas con >1 columna diagnóstica activa "
            f"(one-hot corrupto — multi-hot indica datos fuente corruptos). "
            f"No se puede continuar de forma segura. "
            f"Imágenes afectadas: {multi_hot['image'].head(10).tolist()}"
        )
    if not zero_hot.empty:
        # Un encoding sin ninguna columna activa indica corrupción del dataset: estas
        # filas caerían en silencio a "Unknown" y se filtrarían más adelante, perdiendo
        # datos sin trazabilidad. Abortamos para detectar la corrupción a tiempo.
        raise ValueError(
            f"ISIC 2019: {len(zero_hot)} filas con 0 columnas diagnósticas activas "
            f"(one-hot corrupto). Esto indica corrupción del dataset — no se puede "
            f"continuar de forma segura. Imágenes afectadas: "
            f"{zero_hot['image'].head(10).tolist()}"
        )

    df["diagnosis"] = (
        df["diagnosis_code"].str.lower().map(DX_MAP).fillna(df["diagnosis_code"])
    )

    # Binarización clínica para el triaje. El campo `benign_malignant` del CSV
    # unificado modela "¿necesita derivación dermatológica?" (REFERRAL), no "¿es cáncer
    # invasivo confirmado?" (MALIGNANT). Lo derivamos de REFERRAL_DIAGNOSIS_NAMES
    # (= malignas ∪ precancerosas) para que el modelo binario aprenda a marcar también
    # la AK (precancerosa, con 6-10% de riesgo de progresión a SCC): sin esto, una AK
    # con prob<0.20 caería en GREEN y el paciente no recibiría tratamiento. La rama
    # CRITICAL de la cascada sigue usando solo las clases malignas invasivas.
    df["benign_malignant"] = df["diagnosis"].apply(
        lambda x: "malignant" if x in REFERRAL_DIAGNOSIS_NAMES else "benign"
    )

    # Normalización de columnas
    if "age_approx" in df.columns:
        df["age"] = df["age_approx"]
    if "anatom_site_general" in df.columns:
        df["anatom_site"] = df["anatom_site_general"]

    # Gestión de IDs de paciente (clave para evitar leakage entre splits). Solo usamos
    # lesion_id como respaldo si patient_id no existe: si ambos existen, el patient_id
    # real agrupa mejor (un paciente puede tener varios lesion_id, y usarlos
    # fragmentaría sus imágenes entre splits).
    if "patient_id" not in df.columns and "lesion_id" in df.columns:
        df["patient_id"] = df["lesion_id"]

    # ISIC 2019 tiene 2.084 filas (de 25.331) sin lesion_id, por lo que tras el respaldo
    # anterior siguen sin patient_id. Las eliminamos explícitamente, con auditoría, en
    # lugar de dejar que 04_split.py las trate como pacientes únicos (lo que fragmentaría
    # a pacientes reales con varias imágenes entre splits y crearía un leakage
    # indetectable). La pérdida es 2.084/25.331 = 8,2% de ISIC 2019, ~0,47% del total
    # unificado. Los otros datasets no entran aquí (ISIC 2020/2024 proveen patient_id
    # real y HAM10000 tiene lesion_id completo).
    n_missing_pid = int(df["patient_id"].isna().sum())
    if n_missing_pid > 0:
        pct = 100.0 * n_missing_pid / len(df)
        dist = (
            df.loc[df["patient_id"].isna(), "diagnosis"]
            .value_counts()
            .to_dict()
        )
        warnings.warn(
            f"[ISIC_2019] Eliminando {n_missing_pid} filas ({pct:.2f}%) "
            f"sin patient_id ni lesion_id. Distribucion por diagnostico: {dist}",
            RuntimeWarning,
            stacklevel=2,
        )
        df = df.dropna(subset=["patient_id"]).reset_index(drop=True)

    img_map = get_image_path_dict(base_path)
    df["image_path"] = df["image"].map(img_map)
    df = _drop_missing_paths(df, "ISIC_2019")
    df["dataset"] = "ISIC_2019"

    cols = [
        "image",
        "image_path",
        "diagnosis",
        "benign_malignant",
        "age",
        "sex",
        "anatom_site",
        "patient_id",
        "dataset",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    return df[cols]


def process_isic_2020(base_path):
    print("\nProcesando ISIC 2020 (+Metadata)...")
    path = Path(base_path)
    train_csv = list(path.rglob("train.csv"))[0]
    df = pd.read_csv(train_csv)

    # Pasamos a minúsculas antes del map para que los códigos en mayúsculas ("SCC",
    # "MEL") pasen por DX_MAP en vez de llegar crudos al clean_map.
    #
    # El `.str.title()` posterior es obligatorio para la derivación binaria: ISIC 2020
    # usa nombres completos en `diagnosis` ("melanoma", "nevus", "seborrheic keratosis")
    # que no coinciden con los códigos cortos de DX_MAP, de modo que `DX_MAP.get(x, x)`
    # los deja en minúsculas. La derivación de más abajo compara contra
    # REFERRAL_DIAGNOSIS_NAMES (en formato título), así que sin el `.str.title()` los 584
    # melanomas de ISIC 2020 quedarían marcados como benignos, contaminando el dataset
    # binario. El title-case es idempotente con el del merge final y con clean_map.
    df["diagnosis"] = (
        df["diagnosis"]
        .astype(str)
        .str.lower()
        .map(lambda x: DX_MAP.get(x, x))
        .str.title()
    )

    # Derivamos benign_malignant del diagnóstico normalizado contra
    # REFERRAL_DIAGNOSIS_NAMES (= malignas ∪ precancerosas), igual que en los demás
    # datasets: así el triaje binario marca también la AK precancerosa. Tomar
    # `benign_malignant` directamente del CSV fuente arriesgaría inconsistencias entre
    # datasets.
    source_bm = df.get("benign_malignant")
    df["benign_malignant"] = df["diagnosis"].apply(
        lambda x: "malignant" if x in REFERRAL_DIAGNOSIS_NAMES else "benign"
    )
    if source_bm is not None:
        # Filtrar NaN antes de comparar: source_bm.str.lower() retorna NaN para filas NaN,
        # y NaN != "benign" siempre es True → infla el conteo de mismatches sin serlo.
        valid_source = source_bm.notna()
        mismatches = (
            (df.loc[valid_source, "benign_malignant"] != source_bm.loc[valid_source].str.lower()).sum()
        )
        n_nan_source = (~valid_source).sum()
        # Ambos avisos se emiten como warnings (capturables por el handler de logging).
        # El de NaN en la fuente es informativo (no una discrepancia real); si hace
        # falta distinguirlos, se filtran por el texto del mensaje.
        if n_nan_source > 0:
            warnings.warn(
                f"ISIC 2020: {n_nan_source} filas con benign_malignant fuente NaN "
                f"(ignoradas en cross-check).",
                RuntimeWarning,
                stacklevel=2,
            )
        if mismatches > 0:
            warnings.warn(
                f"ISIC 2020: {mismatches} filas con benign_malignant derivado != fuente CSV. "
                f"Usando derivado (canonico).",
                RuntimeWarning,
                stacklevel=2,
            )

    # Metadata
    df["age"] = df["age_approx"]
    if "anatom_site_general_challenge" in df.columns:
        df["anatom_site"] = df["anatom_site_general_challenge"]

    img_map = get_image_path_dict(base_path)
    df["image_path"] = df["image_name"].map(img_map)
    df = _drop_missing_paths(df, "ISIC_2020")
    df["dataset"] = "ISIC_2020"
    df = df.rename(columns={"image_name": "image"})

    # Guard defensivo, igual que en ISIC 2019 y HAM10000. Hoy el CSV oficial de ISIC
    # 2020 provee `patient_id` directamente (esta rama no se activa), pero si en el
    # futuro se republicara con `lesion_id` en vez de `patient_id`, el bucle de más
    # abajo crearía `patient_id=None` para todas las filas y 04_split.py fragmentaría
    # pacientes entre splits, con leakage indetectable. Mantenerlo en los cuatro
    # datasets cierra ese vector.
    if "patient_id" not in df.columns and "lesion_id" in df.columns:
        df["patient_id"] = df["lesion_id"]

    cols = [
        "image",
        "image_path",
        "diagnosis",
        "benign_malignant",
        "age",
        "sex",
        "anatom_site",
        "patient_id",
        "dataset",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def process_ham10000(base_path):
    print("\nProcesando HAM10000 (+Metadata)...")
    path = Path(base_path)
    meta_csv = list(path.rglob("*metadata.csv"))[0]
    df = pd.read_csv(meta_csv)

    df["diagnosis"] = df["dx"].map(DX_MAP)
    # benign_malignant se deriva del diagnóstico normalizado contra
    # REFERRAL_DIAGNOSIS_NAMES, igual que en los demás datasets. Usar una lista literal
    # de códigos haría divergir las etiquetas entre datasets (HAM10000 marcaba la AK
    # como maligna mientras ISIC no); centralizar la derivación elimina esa divergencia.
    df["benign_malignant"] = df["diagnosis"].apply(
        lambda x: "malignant" if x in REFERRAL_DIAGNOSIS_NAMES else "benign"
    )

    # Metadata
    df["anatom_site"] = df["localization"]
    # Solo usamos lesion_id como respaldo si patient_id no existe (igual que en ISIC
    # 2019): sobrescribir un patient_id real con lesion_id fragmentaría pacientes con
    # varias lesiones entre splits, con leakage indetectable.
    if "patient_id" not in df.columns and "lesion_id" in df.columns:
        df["patient_id"] = df["lesion_id"]

    img_map = get_image_path_dict(base_path)
    df["image_path"] = df["image_id"].map(img_map)
    df = _drop_missing_paths(df, "HAM10000")
    df["dataset"] = "HAM10000"
    df = df.rename(columns={"image_id": "image"})

    cols = [
        "image",
        "image_path",
        "diagnosis",
        "benign_malignant",
        "age",
        "sex",
        "anatom_site",
        "patient_id",
        "dataset",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def process_isic_2024(base_path):
    print("\nProcesando ISIC 2024 (+Metadata)...")
    path = Path(base_path)
    meta_files = list(path.rglob("train-metadata.csv"))
    if not meta_files:
        return pd.DataFrame()

    df = pd.read_csv(meta_files[0], low_memory=False)

    # Función para extraer diagnóstico granular de iddx_full
    def parse_diagnosis(row):
        # Prioridad: Si conocemos el subtipo, lo usamos.
        full_dx = str(row.get("iddx_full", "")).lower()

        # Mapeo por palabras clave. El orden importa: "melanoma" se comprueba ANTES que
        # "nevus", porque un diagnóstico como "nevus-associated melanoma" contiene ambas
        # palabras y clasificarlo como benigno sería peligroso.
        if "melanoma" in full_dx:
            return "Melanoma"
        # "lentigo maligna" es un melanoma in situ; no contiene "melanoma" en el nombre,
        # así que el check anterior no lo captura y debe ir ANTES del "lentigo" genérico.
        if "lentigo maligna" in full_dx:
            return "Melanoma"
        # Cubrimos también el plural "nevi": ISIC 2024 usa el singular casi siempre,
        # pero variantes como "multiple nevi" caerían al fallback si no se contemplan.
        if "nevus" in full_dx or "nevi" in full_dx:
            return "Nevus"
        if "basal cell" in full_dx:
            return "Basal Cell Carcinoma"
        if "squamous cell" in full_dx:
            return "Squamous Cell Carcinoma"
        # La enfermedad de Bowen es un SCC in situ; no contiene "squamous cell" en el
        # nombre, así que sin esta entrada caería al fallback y se filtraría de
        # multiclase, perdiendo señal de una clase ya escasa.
        if "bowen" in full_dx:
            return "Squamous Cell Carcinoma"
        if "seborrheic keratosis" in full_dx:
            return "Benign Keratosis"
        if "lichenoid keratosis" in full_dx:
            return "Benign Keratosis"
        if "solar lentigo" in full_dx:
            return "Benign Keratosis"
        # Catch-all para lentigos benignos: "lentigo nos", "lentigo simplex", etc.
        # DESPUÉS de "lentigo maligna" (→ Melanoma) y "solar lentigo" (→ BK, ya arriba).
        if "lentigo" in full_dx:
            return "Benign Keratosis"
        if "dermatofibroma" in full_dx:
            return "Dermatofibroma"
        if "angioma" in full_dx or "vascular" in full_dx:
            return "Vascular Lesion"
        if "actinic keratosis" in full_dx:
            return "Actinic Keratosis"
        # Variantes ortográficas de "café-au-lait": el vocabulario de ISIC 2024 usa
        # guiones, pero otros catálogos pueden escribirlo sin guiones, con acento o con
        # guiones bajos. Comprobamos la presencia conjunta de ambas palabras, robusto a
        # los separadores.
        if ("cafe" in full_dx or "café" in full_dx) and "lait" in full_dx:
            return "Benign Other"
        # "atypical melanocytic proliferation" — lesión precancerosa. No contiene
        # keywords de malignidad estándar. Se clasifica como "Melanoma (Potential)"
        # (excluida de multiclase, tratada como maligna en binario para derivar a especialista).
        if "atypical melanocytic" in full_dx:
            return "Melanoma (Potential)"

        # Fallback: Si no hay keyword, usamos target
        if row["target"] == 1:
            return "Malignant (Unspecified)"
        return "Benign (Unspecified)"

    print("   Extrayendo diagnósticos detallados (iddx_full)...")
    df["diagnosis"] = df.apply(parse_diagnosis, axis=1)

    # Derivamos benign_malignant del diagnóstico normalizado contra
    # REFERRAL_DIAGNOSIS_NAMES, igual que en los demás datasets (la AK precancerosa
    # cuenta como derivación). "Malignant (Unspecified)" y "Melanoma (Potential)" son
    # casos especiales del fallback de parse_diagnosis (target=1 sin keyword, o
    # "atypical melanocytic proliferation"): ambos deben derivarse al especialista.
    df["benign_malignant"] = df["diagnosis"].apply(
        lambda x: "malignant"
        if x in REFERRAL_DIAGNOSIS_NAMES
        or x == "Malignant (Unspecified)"
        or x == "Melanoma (Potential)"
        else "benign"
    )
    # Cross-check contra target original para auditoría
    if "target" in df.columns:
        target_bm = df["target"].apply(lambda x: "malignant" if x == 1 else "benign")
        mismatches = df["benign_malignant"] != target_bm
        n_mismatches = mismatches.sum()
        if n_mismatches > 0:
            # El cross-check detectó discrepancias entre el valor derivado (canónico) y
            # el target original del CSV; lo emitimos como warning para que quede
            # trazable en logs estructurados.
            mismatch_dx = df.loc[mismatches, "diagnosis"].value_counts()
            warnings.warn(
                f"ISIC 2024: {n_mismatches} filas con benign_malignant derivado "
                f"!= target original. Diagnosticos discrepantes: "
                f"{mismatch_dx.to_dict()}",
                RuntimeWarning,
                stacklevel=2,
            )

    if "age_approx" in df.columns:
        df["age"] = df["age_approx"]
    if "anatom_site_general" in df.columns:
        df["anatom_site"] = df["anatom_site_general"]

    img_map = get_image_path_dict(base_path)
    df["image_path"] = df["isic_id"].map(img_map)
    df = _drop_missing_paths(df, "ISIC_2024")
    df["dataset"] = "ISIC_2024"
    df = df.rename(columns={"isic_id": "image"})

    # Guard defensivo, igual que en los otros tres datasets. Hoy ISIC 2024 provee
    # `patient_id` en el CSV fuente, pero si una revisión futura eliminara la columna,
    # el bucle de abajo crearía `patient_id=None` para todas las filas y 04_split.py
    # trataría cada imagen como un paciente único, con leakage indetectable. 04_split.py
    # aborta además si falta más del 0,1% de patient_id, pero preferimos rescatar aquí
    # cuando el dato existe como lesion_id (un paciente puede
    # tener varios lesion_id, limitación conocida del agrupamiento por paciente).
    if "patient_id" not in df.columns and "lesion_id" in df.columns:
        df["patient_id"] = df["lesion_id"]

    cols = [
        "image",
        "image_path",
        "diagnosis",
        "benign_malignant",
        "age",
        "sex",
        "anatom_site",
        "patient_id",
        "dataset",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def main():
    root = Path("data/raw")
    processed_dir = Path("data/processed")

    dfs = []
    if (root / "isic2019").exists():
        dfs.append(process_isic_2019(root / "isic2019"))
    if (root / "isic2020").exists():
        dfs.append(process_isic_2020(root / "isic2020"))
    if (root / "ham10000").exists():
        dfs.append(process_ham10000(root / "ham10000"))
    if (root / "isic2024").exists():
        dfs.append(process_isic_2024(root / "isic2024"))

    if not dfs:
        raise FileNotFoundError(
            "No se encontró ninguno de los 4 datasets raw "
            "(isic2019/isic2020/ham10000/isic2024) en data/raw/. "
            "Verifica que data/raw/ contenga al menos un subdirectorio con datos. "
            "En Colab, verifica que Drive este montado correctamente y que "
            "skin_data_cloud.zip se haya descomprimido en /content/data/raw/."
        )

    print("\nFusionando Datasets...")
    final_df = pd.concat(dfs, ignore_index=True)

    print("Normalizando valores de metadatos y etiquetas...")

    # Limpieza de Tipos de Datos
    final_df["age"] = pd.to_numeric(final_df["age"], errors="coerce")
    # `sex` y `anatom_site` se normalizan con la cadena
    # `.fillna("unknown") → .astype(str) → .str.strip() → .str.lower()`.
    # El orden es CRÍTICO: el fillna debe ir ANTES del astype(str), porque astype(str)
    # convierte NaN/None en los strings literales "nan"/"none", que el fillna ya no
    # detectaría como faltantes; el CSV quedaría entonces con categorías espurias que no
    # están en SEX_MAP/SITE_MAP. dataset.py lo rescata en runtime, pero 06_verify.py
    # contaría "nan" y "unknown" como categorías distintas al auditar la cobertura de
    # SITE_MAP. Esta cadena debe ser idéntica a la de dataset.py::__init__ y
    # encode_clinical_metadata_scalar (constants.py) para que, p.ej., "male " con
    # espacios produzca "male" en los tres sitios.
    final_df["sex"] = final_df["sex"].fillna("unknown").astype(str).str.strip().str.lower()
    final_df["anatom_site"] = final_df["anatom_site"].fillna("unknown").astype(str).str.strip().str.lower()
    final_df["benign_malignant"] = final_df["benign_malignant"].str.lower()
    final_df["diagnosis"] = final_df["diagnosis"].astype(str).str.title()

    # Mapeo Final de Diagnósticos para Consistencia
    clean_map = {
        "Nv": "Nevus",
        "Mel": "Melanoma",
        "Bcc": "Basal Cell Carcinoma",
        "Scc": "Squamous Cell Carcinoma",
        "Bkl": "Benign Keratosis",
        "Akiec": "Actinic Keratosis",
        "Ak": "Actinic Keratosis",
        "Df": "Dermatofibroma",
        "Vasc": "Vascular Lesion",
        "Unknown": "Unknown",
        "Benign": "Benign (Unspecified)",
        "Malignant": "Malignant (Unspecified)",
        "Seborrheic Keratosis": "Benign Keratosis",
        "Lichenoid Keratosis": "Benign Keratosis",
        "Solar Lentigo": "Benign Keratosis",
        "Lentigo Nos": "Benign Keratosis",
        "Cafe-Au-Lait Macule": "Benign Other",
        "Atypical Melanocytic Proliferation": "Melanoma (Potential)",
    }

    final_df["diagnosis"] = final_df["diagnosis"].replace(clean_map)

    # Validación cruzada: detectar diagnósticos que constants.py no reconoce. Si aparece
    # un nombre que no está ni en MULTICLASS_LABEL_MAP ni en MULTICLASS_EXCLUDE_LIST,
    # probablemente hay un typo en DX_MAP/clean_map que provocaría pérdida silenciosa de
    # datos en 07_filter_multiclass.py.
    known_dx = set(MULTICLASS_LABEL_MAP.keys()) | set(MULTICLASS_EXCLUDE_LIST) | {"Unknown"}
    unknown_dx = set(final_df["diagnosis"].unique()) - known_dx
    if unknown_dx:
        # Abortamos: continuar solo emitiría un aviso y 07_filter_multiclass eliminaría
        # esas filas en silencio.
        counts_by_dx = {
            dx: int((final_df["diagnosis"] == dx).sum()) for dx in unknown_dx
        }
        raise ValueError(
            f"Diagnosticos no reconocidos en constants.py: {counts_by_dx}. "
            f"Actualizar DX_MAP/clean_map en 02_unify.py o MULTICLASS_LABEL_MAP/"
            f"MULTICLASS_EXCLUDE_LIST en constants.py antes de continuar. "
            f"Downstream estas filas desapareceran sin auditoria."
        )

    # --- Filtrado de calidad de etiquetas: eliminación de "Unknown" ---
    # Las etiquetas desconocidas introducen ruido severo en el entrenamiento supervisado;
    # preferimos reducir el tamaño del dataset a cambio de garantizar etiquetas limpias.
    # La eliminación se audita con porcentaje, desglose por dataset y un umbral de seguridad.
    n_total = len(final_df)
    unk_mask = final_df["diagnosis"] == "Unknown"
    n_unk = int(unk_mask.sum())
    if n_unk > 0:
        pct = 100.0 * n_unk / n_total
        # La eliminación de muestras modifica el dataset; la emitimos como warning para
        # que quede trazable en logs estructurados y sea capturable por pytest.warns().
        if "dataset" in final_df.columns:
            breakdown = final_df.loc[unk_mask, "dataset"].value_counts()
            breakdown_str = "; ".join(
                f"{ds}: {count}/{int((final_df['dataset'] == ds).sum())} ({100.0 * count / int((final_df['dataset'] == ds).sum()):.1f}%)"
                for ds, count in breakdown.items()
            )
            warnings.warn(
                f"[FILTRO DE PRECISION] Eliminando {n_unk}/{n_total} ({pct:.2f}%) "
                f"imagenes con diagnostico 'Unknown'. Desglose por dataset: {breakdown_str}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"[FILTRO DE PRECISION] Eliminando {n_unk}/{n_total} ({pct:.2f}%) "
                f"imagenes con diagnostico 'Unknown'.",
                RuntimeWarning,
                stacklevel=2,
            )
        if pct > 10.0:
            raise ValueError(
                f"Eliminación de 'Unknown' supera 10% ({pct:.2f}%). "
                f"Revisar datasets de origen — posible corrupción masiva."
            )
        final_df = final_df[~unk_mask]

    # Auditoría de cobertura de SITE_MAP. Si un dataset nuevo trae sitios anatómicos no
    # cubiertos, caerían en silencio a "unknown" (código 7); el caso de "torso" llegó a
    # costar 2657 muestras.
    if "anatom_site" in final_df.columns:
        from src.data.constants import SITE_MAP

        raw_sites = set(
            final_df["anatom_site"]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .unique()
        )
        known_sites = set(SITE_MAP.keys())
        unmapped_sites = raw_sites - known_sites - {"unknown", "nan", ""}
        if unmapped_sites:
            counts = {
                s: int((final_df["anatom_site"].astype(str).str.lower().str.strip() == s).sum())
                for s in unmapped_sites
            }
            total_unmapped = sum(counts.values())
            pct_unmapped = 100.0 * total_unmapped / len(final_df)
            # Aviso como warning: los sitios no mapeados caen a "unknown" (código 7) si
            # no se auditan.
            warnings.warn(
                f"[AUDIT SITE_MAP] {total_unmapped} ({pct_unmapped:.2f}%) muestras con sitio "
                f"no cubierto por SITE_MAP (caeran a 'unknown'): {counts}",
                RuntimeWarning,
                stacklevel=2,
            )
            if pct_unmapped > 5.0:
                raise ValueError(
                    f"Más del 5% de muestras ({pct_unmapped:.2f}%) tienen sitio anatómico "
                    f"no mapeado. Actualizar SITE_MAP en constants.py antes de continuar."
                )

    print(f"\nDataset Unificado Creado (con Metatadas):")
    print(f"   - Total Muestras: {len(final_df)}")
    print(f"   - Columnas: {list(final_df.columns)}")
    print(
        f"   - Distribución Diagnósticos:\n{final_df['diagnosis'].value_counts().head(10)}"
    )

    processed_dir.mkdir(exist_ok=True, parents=True)
    final_df.to_csv(processed_dir / "02_all_data.csv", index=False)
    print(f"\nGuardado en: {processed_dir / '02_all_data.csv'}")


if __name__ == "__main__":
    main()
