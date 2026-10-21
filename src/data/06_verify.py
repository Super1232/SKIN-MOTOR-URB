import pandas as pd
import os
import sys
import warnings
import cv2
from tqdm import tqdm
from pathlib import Path

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.data.constants import MULTICLASS_LABEL_MAP, MULTICLASS_EXCLUDE_LIST


def verify_integrity():
    # Verifica los artefactos FINALES, no los intermedios.
    print(f"---AUDITORIA FINAL DE SEGURIDAD---")

    splits = [
        "04_train.csv",
        "04_val.csv",
        "04_test.csv",
        "07_train_multiclass.csv",
        "07_val_multiclass.csv",
        "07_test_multiclass.csv",
    ]
    base_dir = Path("data/processed")

    all_verified_paths = set()
    skipped_splits = []
    verified_splits = []

    for split in splits:
        csv_path = base_dir / split
        print(f"\nVerificando {split}...")

        if not csv_path.exists():
            print(
                f" [ADVERTENCIA] No existe {csv_path} (¿Quizás no has corrido el filtro multiclass?)"
            )
            skipped_splits.append(split)
            continue

        df = pd.read_csv(csv_path)
        print(f"   - Registros: {len(df)}")
        all_verified_paths.update(df["image_path"].tolist())
        verified_splits.append(split)

        # 1. Validación de Labels
        print(f"   - Validando etiquetas...")

        if "multiclass" in split:
            # Reglas estrictas para Multiclass: No debe haber 'Unspecified' ni 'Unknown'
            # Y debe tener columna 'diagnosis' válida
            # Importado de constants.py (fuente de verdad única)
            forbidden = MULTICLASS_EXCLUDE_LIST
            bad_rows = df[df["diagnosis"].isin(forbidden)]
            if not bad_rows.empty:
                raise ValueError(
                    f"Diagnósticos prohibidos en split multiclase '{split}': "
                    f"{bad_rows['diagnosis'].value_counts().to_dict()}"
                )

            # Importado de constants.py (fuente de verdad única)
            valid_classes = list(MULTICLASS_LABEL_MAP.keys())
            invalid_dx = df[~df["diagnosis"].isin(valid_classes)]
            if not invalid_dx.empty:
                raise ValueError(
                    f"Diagnósticos fuera de las 8 clases permitidas en '{split}': "
                    f"{invalid_dx['diagnosis'].unique().tolist()}"
                )

            # Cross-check de target_idx contra MULTICLASS_LABEL_MAP. Un CSV corrupto
            # podría tener "Melanoma" con target_idx=7 y pasar la validación de nombres;
            # sin este check, el modelo entrenaría con etiquetas incorrectas en silencio.
            if "target_idx" in df.columns:
                expected_idx = df["diagnosis"].map(MULTICLASS_LABEL_MAP)
                mismatches = df[df["target_idx"] != expected_idx]
                if not mismatches.empty:
                    sample = mismatches.head(5)[["diagnosis", "target_idx"]].to_dict("records")
                    raise ValueError(
                        f"target_idx no coincide con MULTICLASS_LABEL_MAP en '{split}': "
                        f"{len(mismatches)} filas. Primeras 5: {sample}"
                    )

        else:
            # Reglas para Binary
            invalid_labels = df[~df["benign_malignant"].isin(["benign", "malignant"])]
            if not invalid_labels.empty:
                raise ValueError(
                    f"Etiquetas binarias inválidas en '{split}': {len(invalid_labels)} filas. "
                    f"Valores encontrados: {invalid_labels['benign_malignant'].unique().tolist()}"
                )

        # 2. Validación de archivos (comprobación rápida)
        print(f"   - Validando existencia de archivos...")
        # Comprueba rutas inexistentes
        paths = df["image_path"].tolist()
        # La comprobación vectorizada es difícil con os.path; se usa un bucle simple.
        missing = []
        for p in tqdm(paths):
            if not os.path.exists(p):
                missing.append(p)

        if missing:
            raise ValueError(
                f"{len(missing)} archivos de imagen no encontrados en disco para '{split}'. "
                f"Primeros 5: {missing[:5]}"
            )

        # 3. Corrupción Check (Muestreo estratificado por dataset)
        # 50 imágenes es 0.015% de 337K — insuficiente para detectar corrupción localizada.
        # El muestreo estratificado por dataset de origen garantiza cobertura de todas las fuentes.
        n_corruption_samples = min(500, len(df))
        if "dataset" in df.columns:
            sample = df.groupby("dataset", group_keys=False).apply(
                lambda x: x.sample(min(max(50, n_corruption_samples // df["dataset"].nunique()), len(x)), random_state=42)
            )
            if len(sample) < n_corruption_samples:
                extra = df.drop(sample.index).sample(min(n_corruption_samples - len(sample), len(df) - len(sample)), random_state=42)
                sample = pd.concat([sample, extra])
        else:
            sample = df.sample(n_corruption_samples, random_state=42)
        print(f"   - Test de corrupción ({len(sample)} muestras estratificadas)...")
        for _, row in sample.iterrows():
            img = cv2.imread(row["image_path"])
            if img is None:
                raise ValueError(
                    f"Imagen corrupta detectada en '{split}': {row['image_path']}"
                )

    # --- Verificación de Patient Leakage + Image Leakage ---
    print("\n   Verificando aislamiento de pacientes e imágenes entre splits...")
    binary_splits = ["04_train.csv", "04_val.csv", "04_test.csv"]
    split_patients = {}
    split_images = {}
    split_dfs = {}  # Reutilizar para consistencia diagnóstica
    for split_name in binary_splits:
        csv_path = base_dir / split_name
        if csv_path.exists():
            df_s = pd.read_csv(csv_path)
            split_dfs[split_name] = df_s
            if "patient_id" in df_s.columns:
                split_patients[split_name] = set(df_s["patient_id"].dropna())
            if "image_path" in df_s.columns:
                split_images[split_name] = set(df_s["image_path"].dropna())

    if len(split_patients) == 3:
        leak_tv = split_patients["04_train.csv"] & split_patients["04_val.csv"]
        leak_tt = split_patients["04_train.csv"] & split_patients["04_test.csv"]
        leak_vt = split_patients["04_val.csv"] & split_patients["04_test.csv"]
        if leak_tv or leak_tt or leak_vt:
            raise ValueError(
                f"Patient leakage detectado: "
                f"train/val={len(leak_tv)}, train/test={len(leak_tt)}, val/test={len(leak_vt)}"
            )
        else:
            print("     [OK] Patient leakage check: PASSED (0 overlap)")
    else:
        print("     [ADVERTENCIA] No se pudieron verificar los 3 splits binarios.")

    if len(split_images) == 3:
        img_leak_tv = split_images["04_train.csv"] & split_images["04_val.csv"]
        img_leak_tt = split_images["04_train.csv"] & split_images["04_test.csv"]
        img_leak_vt = split_images["04_val.csv"] & split_images["04_test.csv"]
        if img_leak_tv or img_leak_tt or img_leak_vt:
            raise ValueError(
                f"Image leakage detectado: "
                f"train/val={len(img_leak_tv)}, train/test={len(img_leak_tt)}, "
                f"val/test={len(img_leak_vt)}. CSVs posiblemente corruptos."
            )
        else:
            print("     [OK] Image leakage check: PASSED (0 overlap)")
    else:
        print("     [ADVERTENCIA] No se pudieron verificar los 3 splits binarios para image leakage.")

    # --- Verificación de Consistencia Diagnóstica por Paciente ---
    # Si un paciente tiene imágenes con diagnósticos MUY distintos (ej: "Melanoma" + "Nevus"),
    # puede indicar error de anotación o confusión de pacientes en la fuente de datos.
    # No es necesariamente un error (un paciente puede tener múltiples lesiones), pero
    # >2 diagnósticos distintos merece auditoría manual.
    # Comprobamos los 3 splits por separado. Como cada paciente vive en un único split,
    # train suele absorber la corrupción del dataset de origen; pero si un dataset futuro
    # afectara sobre todo a val/test, revisar solo train dejaría esa grieta invisible.
    # El umbral del 5% se aplica por split para no enmascarar corrupción localizada con
    # el volumen agregado.
    print("\n   Verificando consistencia diagnóstica por paciente (per-split)...")
    for split_name in binary_splits:
        if split_name not in split_dfs:
            continue
        df_all = split_dfs[split_name]
        if "patient_id" not in df_all.columns or "diagnosis" not in df_all.columns:
            continue
        suspicious_patients = 0
        first_suspects = []
        for patient_id, group in df_all.groupby("patient_id"):
            n_dx = group["diagnosis"].nunique()
            if n_dx > 2:
                if len(first_suspects) < 5:
                    first_suspects.append(
                        f"paciente {patient_id}: {n_dx} dx → {group['diagnosis'].unique().tolist()}"
                    )
                suspicious_patients += 1
        total_patients = df_all["patient_id"].nunique()
        suspicious_pct = 100.0 * suspicious_patients / max(total_patients, 1)
        if suspicious_patients > 0:
            print(
                f"     [ADVERTENCIA] [{split_name}] {suspicious_patients}/{total_patients} "
                f"pacientes ({suspicious_pct:.1f}%) con >2 diagnosticos distintos."
            )
            for s in first_suspects:
                print(f"        · {s}")
            if suspicious_pct >= 5.0:
                raise ValueError(
                    f"Consistencia diagnostica [{split_name}]: {suspicious_patients}/{total_patients} "
                    f"({suspicious_pct:.1f}%) pacientes con >2 diagnosticos distintos. "
                    f"Umbral clinico de 5% excedido — posible corrupcion masiva de labels. "
                    f"Auditar datos de origen antes de continuar."
                )
        else:
            print(f"     [OK] [{split_name}] Ningun paciente con >2 diagnosticos distintos.")

    print(f"\n--- RESUMEN AUDITORIA ---")
    print(f"Total Imágenes Únicas Verificadas: {len(all_verified_paths)}")
    print(f"Splits verificados: {verified_splits}")
    if skipped_splits:
        # Emitimos un warning (capturable por el logging estructurado): una
        # certificación incompleta es un evento crítico de auditoría que no debe
        # perderse mezclado en stdout.
        warnings.warn(
            f"Splits NO verificados (archivos no encontrados): {skipped_splits}",
            RuntimeWarning,
            stacklevel=2,
        )
        print(f"Estado: [ADVERTENCIA] VERIFICACIÓN PARCIAL — ejecutar el pipeline completo antes de entrenar")
    else:
        print(f"Estado: [OK] Integridad de los datos verificada")


if __name__ == "__main__":
    verify_integrity()
