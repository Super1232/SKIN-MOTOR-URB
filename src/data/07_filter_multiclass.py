import pandas as pd
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.constants import MULTICLASS_LABEL_MAP, MULTICLASS_EXCLUDE_LIST


def main():
    """
    Script de Filtrado para el Nivel 2 (Multiclass Specialist).

    Objetivo: "Data Curation for Fine-Grained Classification".
    El "Especialista" no debe entrenarse con etiquetas genéricas ("Benign Unspecified").
    Solo aprende de imágenes con "Ground Truth" específico.

    Este script toma los splits oficiales (train/val/test) y:
    1. Elimina filas con diagnóstico 'Unspecified'.
    2. Elimina filas con diagnóstico 'Unknown' (por seguridad).
    3. Genera archivos `_multiclass.csv`.
    """
    print("--- Generando Datasets para Nivel 2 (Multiclass) ---")

    base_dir = Path("data/processed")
    # Tuplas (Input, Output)
    splits = [
        ("04_train.csv", "07_train_multiclass.csv"),
        ("04_val.csv", "07_val_multiclass.csv"),
        ("04_test.csv", "07_test_multiclass.csv"),
    ]

    # Importados de src/data/constants.py (fuente de verdad única)
    MULTICLASS_MAP = MULTICLASS_LABEL_MAP
    EXCLUDE_LIST = MULTICLASS_EXCLUDE_LIST

    total_kept = 0
    total_dropped = 0

    # Pase 1: validar y filtrar todos los splits en memoria antes de escribir nada.
    # Previene CSVs parciales si un split falla y otro ya se escribió.
    validated = []

    for in_name, out_name in splits:
        in_file = base_dir / in_name
        out_file = base_dir / out_name

        if not in_file.exists():
            print(f"[SKIP] No existe {in_file}")
            continue

        df = pd.read_csv(in_file)
        initial_len = len(df)

        df_filtered = df[~df["diagnosis"].isin(EXCLUDE_LIST)]

        # Auditamos los diagnósticos que no están ni en EXCLUDE_LIST ni en
        # MULTICLASS_MAP. Sin este check, un typo como "Mellanoma" desaparecería en
        # silencio; preferimos abortar.
        remaining_dx = set(df_filtered["diagnosis"].unique())
        unknown_dx = remaining_dx - set(MULTICLASS_MAP.keys())
        if unknown_dx:
            counts = {dx: int((df_filtered["diagnosis"] == dx).sum()) for dx in unknown_dx}
            raise ValueError(
                f"Diagnósticos no reconocidos en {in_name} (ni en EXCLUDE_LIST ni en "
                f"MULTICLASS_LABEL_MAP): {counts}. Corregir en 02_unify.py o actualizar "
                f"constants.py."
            )

        df_filtered = df_filtered[df_filtered["diagnosis"].isin(MULTICLASS_MAP.keys())]
        df_filtered["target_idx"] = df_filtered["diagnosis"].map(MULTICLASS_MAP)

        unmapped = df_filtered[df_filtered["target_idx"].isna()]
        if not unmapped.empty:
            raise ValueError(
                f"Diagnósticos no mapeados a target_idx: "
                f"{unmapped['diagnosis'].unique().tolist()}"
            )

        kept = len(df_filtered)
        dropped = initial_len - kept

        if kept == 0:
            raise ValueError(
                f"El filtrado eliminó TODAS las imágenes en '{in_name}'. "
                "Revisa los nombres de diagnóstico en el CSV de entrada."
            )

        total_kept += kept
        total_dropped += dropped

        print(
            f"[{in_name.upper()}] Original: {initial_len} -> Filtrado: {kept} (Drop: {dropped})"
        )

        validated.append((out_file, df_filtered))

    # Pase 2: escribir todos los CSVs solo si todos los splits pasaron validación.
    for out_file, df_filtered in validated:
        df_filtered.to_csv(out_file, index=False)
        print(f"   -> Guardado: {out_file}")
        print(f"   -> Distribución:\n{df_filtered['diagnosis'].value_counts()}")

    print("\n--- RESUMEN GLOBAL ---")
    print(f"Total Imágenes Aptas para Especialista: {total_kept}")
    print(f"Total Imágenes 'Unspecified' (Solo aptas para Triaje): {total_dropped}")
    print("Listo para el entrenamiento del especialista multiclase (Nivel 2).")


if __name__ == "__main__":
    main()
