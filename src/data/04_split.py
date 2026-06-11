import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path


def main():
    """
    Script de Data Splitting.

    Objetivo: Generar particiones de Train, Validation y Test robustas.

    Desafío: 'Patient Leakage'.
    Las imágenes dermatoscópicas a menudo vienen en series del mismo paciente/lesión.
    Si separamos aleatoriamente imágenes del mismo paciente entre Train y Test,
    el modelo podría "memorizar" características biológicas del paciente (piel, vello)
    en lugar de aprender a diagnosticar la lesión.

    Solución Implementada: StratifiedGroupKFold
    1. GroupKFold: Garantiza que TODAS las imágenes de un mismo 'patient_id' permanezcan en el mismo grupo.
    2. Stratified: Intenta mantener la misma proporción de clases (Benigno/Maligno) en todos los cortes.
    """
    print("Cargando dataset limpio...")
    df = pd.read_csv("data/processed/03_all_data_cleaned.csv")

    # Ordenamiento determinista del DataFrame antes de splittear: asegura que el
    # split sea bit a bit idéntico en cualquier máquina (reproducibilidad científica).
    df = df.sort_values(by="image_path").reset_index(drop=True)

    # Relleno de IDs faltantes para evitar perder datos
    if "patient_id" not in df.columns:
        print("ERROR CRÍTICO: Falta columna patient_id")
        return

    missing_patients = df["patient_id"].isnull().sum()
    print(f"Total imágenes: {len(df)}")

    if missing_patients > 0:
        # Abortamos si falta patient_id en una fracción no trivial de filas. Un fallback
        # a image_id trataría cada imagen sin ID como un paciente único, lo que
        # fragmentaría a pacientes reales (varias imágenes del mismo paciente caerían en
        # splits distintos) y produciría un leakage invisible para el check posterior.
        # El umbral clínico es 0.1% (≈442 filas sobre ~442k imágenes).
        pct_missing = missing_patients / len(df) * 100
        if pct_missing > 0.1:
            raise ValueError(
                f"{missing_patients} filas ({pct_missing:.2f}%) sin patient_id. "
                "Umbral clinico: max 0.1%. El fallback a image_id fragmenta pacientes "
                "reales entre splits → leakage no detectable downstream. "
                "Corregir en 02_unify.py antes de splittear."
            )
        print(
            f"[ADVERTENCIA] {missing_patients} filas ({pct_missing:.2f}%) sin patient_id — "
            "fallback a image_id (dentro de tolerancia del 0.1%)."
        )
        df["patient_id"] = df["patient_id"].fillna(df["image"])

    # --- Configuración del Split ---
    # Estratificación compuesta: benign_malignant + dataset.
    # Estratificar solo por benign_malignant dejaría que la distribución
    # INTER-DATASET fuera sesgada en val/test (HAM10000 es ~99% Nevus, ISIC 2020
    # tiene otra mezcla, etc.), de modo que un split podría sobre-representar una
    # fuente concreta y las métricas de validación no reflejarían el rendimiento
    # real en datos de cualquier dispositivo. Con la clave compuesta, cada
    # combinación (fuente, benignidad) mantiene su proporción global en los 3 splits.
    if "dataset" not in df.columns:
        raise ValueError(
            "Columna 'dataset' ausente. Se necesita para estratificación compuesta. "
            "Revisar 02_unify.py (cada dataset debe etiquetar su origen)."
        )
    df["_strat_key"] = (
        df["benign_malignant"].astype(str) + "__" + df["dataset"].astype(str)
    )
    target = df["_strat_key"]
    groups = df["patient_id"]  # Variable de agrupación (anti-leakage)

    print("\nDistribución por clave de estratificación (benign_malignant__dataset):")
    print(df["_strat_key"].value_counts().to_string())

    # Usamos GroupKFold con 10 splits.
    # Esto divide el dataset en 10 partes ("folds") independientes de pacientes.
    sgkf = StratifiedGroupKFold(n_splits=10)

    folds = list(sgkf.split(df, target, groups))

    # Estrategia 80/10/10:
    # Fold 0 -> Usado para Validación (10%)
    # Fold 1 -> Usado para Test (10%)
    # Folds 2-9 -> Usados para Entrenamiento (80%)

    # Índices del Fold 0, fold0_train_idx = 90%, fold0_test_idx = 10% de todos los datos
    fold0_train_idx, fold0_test_idx = folds[0]
    # Índices del Fold 1, fold1_train_idx = 90%, fold1_test_idx = 10% de todos los datos
    fold1_train_idx, fold1_test_idx = folds[1]

    # Asignación final
    val_indices = fold0_test_idx
    test_indices = fold1_test_idx

    # Verificación de seguridad: No debe haber solapamiento entre Val y Test
    # (GroupKFold garantiza esto por defecto, pero verificamos por si acaso).
    # Nota: assert se puede desactivar con `python -O`. Usar raise para checks de seguridad.
    overlap = set(val_indices) & set(test_indices)
    if len(overlap) > 0:
        raise ValueError(
            f"Solapamiento entre Val y Test: {len(overlap)} índices compartidos. "
            "Esto indica un fallo en StratifiedGroupKFold."
        )

    # Entrenamiento = Todo lo que no está ni en Val ni en Test
    all_indices = set(range(len(df)))
    train_indices = list(all_indices - set(val_indices) - set(test_indices))

    # Eliminar la clave de estratificación temporal antes de guardar (columna auxiliar, no se persiste).
    df = df.drop(columns=["_strat_key"])

    df_train = df.iloc[train_indices]
    df_val = df.iloc[val_indices]
    df_test = df.iloc[test_indices]

    # Auditoría: distribución por dataset en cada split.
    for name, subset in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"\n{name} — distribución por dataset × benign_malignant:")
        dist = subset.groupby(["dataset", "benign_malignant"]).size()
        print(dist.to_string())

    print(f"\nResultados del Split:")
    print(f"Train: {len(df_train)} imágenes ({len(df_train)/len(df):.1%})")
    print(f"Val:   {len(df_val)} imágenes ({len(df_val)/len(df):.1%})")
    print(f"Test:  {len(df_test)} imágenes ({len(df_test)/len(df):.1%})")

    # --- Verificación de Fugas (Leakage Check) ---
    train_patients = set(df_train["patient_id"])
    val_patients = set(df_val["patient_id"])
    test_patients = set(df_test["patient_id"])

    leakage_tv = train_patients.intersection(val_patients)
    leakage_tt = train_patients.intersection(test_patients)
    # Defensa en profundidad: 06_verify.py ya comprueba val/test, pero fallar aquí evita
    # generar CSVs con leakage si alguien omite esa verificación posterior.
    leakage_vt = val_patients.intersection(test_patients)

    if len(leakage_tv) == 0 and len(leakage_tt) == 0 and len(leakage_vt) == 0:
        print("\nÉXITO: Aislamiento total de pacientes garantizado.")
    else:
        # Error fatal, no un simple aviso: continuar con leakage corrompe la evaluación
        # del modelo (métricas optimistas que no reflejan el rendimiento en pacientes
        # nuevos).
        raise ValueError(
            f"LEAKAGE DETECTADO: {len(leakage_tv)} pacientes train/val, "
            f"{len(leakage_tt)} pacientes train/test, "
            f"{len(leakage_vt)} pacientes val/test. "
            f"Abortando para proteger integridad clínica."
        )

    # Guardado
    out_dir = Path("data/processed")
    df_train.to_csv(out_dir / "04_train.csv", index=False)
    df_val.to_csv(out_dir / "04_val.csv", index=False)
    df_test.to_csv(out_dir / "04_test.csv", index=False)
    print("\nArchivos guardados en data/processed/ (Prefijo 04_)")


if __name__ == "__main__":
    main()
