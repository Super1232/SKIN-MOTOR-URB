import pandas as pd
import sys
from pathlib import Path
import os

# Añade la raíz del proyecto al path
sys.path.append(os.path.abspath("."))

from src.data.dataset import get_dataloaders
from src.utils.seeding import seed_everything


def check_file(path, description):
    p = Path(path)
    if p.exists():
        print(f"[OK] {description}: Found {path}")
        return True
    else:
        print(f"[ERROR] {description}: MISSING {path}")
        return False


def verify_structure():
    print("\n--- 0. Structure & File Flow Check ---")
    files = [
        "src/data/01_explore.py",
        "src/data/02_unify.py",
        "src/data/03_clean.py",
        "src/data/04_split.py",
        "src/data/05_calc_stats.py",
        "src/data/06_verify.py",
        "src/data/07_filter_multiclass.py",
        "src/data/dataset.py",
        "src/features/preprocessing.py",
        "src/utils/seeding.py",
    ]
    all_ok = True
    for f in files:
        if not check_file(f, f"Script {Path(f).name}"):
            all_ok = False
    return all_ok


def verify_csv_integrity():
    print("\n--- 1. CSV Integrity Check ---")
    files = {
        "Raw Unified": "data/processed/02_all_data.csv",
        "Cleaned": "data/processed/03_all_data_cleaned.csv",
        "Train Split": "data/processed/04_train.csv",
        "Val Split": "data/processed/04_val.csv",
        "Test Split": "data/processed/04_test.csv",
    }

    dfs = {}
    all_ok = True
    for name, path in files.items():
        if check_file(path, name):
            try:
                df = pd.read_csv(path)
                dfs[name] = df
                print(f"   -> Shape: {df.shape}")
                if "patient_id" in df.columns:
                    print(f"   -> Patients: {df['patient_id'].nunique()}")
                else:
                    print(f"   -> [ERROR] No patient_id column in {name} — leakage check impossible")
                    all_ok = False
            except Exception as e:
                print(f"   -> [ERROR] Could not read CSV: {e}")
                all_ok = False
        else:
            all_ok = False

    return all_ok, dfs


def verify_splits(dfs):
    print("\n--- 2. Split Strategy Check ---")
    if "Train Split" not in dfs or "Val Split" not in dfs or "Test Split" not in dfs:
        print("[SKIP] Missing splits.")
        return False

    train = dfs["Train Split"]
    val = dfs["Val Split"]
    test = dfs["Test Split"]

    # Sin patient_id no es posible validar el leakage: abortamos con un mensaje claro
    # en vez de dejar que el script falle con un KeyError críptico más adelante.
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        if "patient_id" not in split_df.columns:
            print(f"[ERROR] patient_id missing in {split_name} — aborting leakage check")
            return False

    # 1. Leakage
    train_patients = set(train["patient_id"].astype(str).unique())
    val_patients = set(val["patient_id"].astype(str).unique())
    test_patients = set(test["patient_id"].astype(str).unique())

    tv = train_patients & val_patients
    tt = train_patients & test_patients
    vt = val_patients & test_patients

    leakage = len(tv) + len(tt) + len(vt)
    if leakage == 0:
        print("[OK] NO Patient Leakage detected.")
    else:
        print(
            f"[ERROR] LEAKAGE DETECTED! (T-V: {len(tv)}, T-T: {len(tt)}, V-T: {len(vt)})"
        )
        return False

    # 2. Ratios
    total = len(train) + len(val) + len(test)
    print(
        f"[INFO] Split Ratios: Train {len(train)/total:.1%}, Val {len(val)/total:.1%}, Test {len(test)/total:.1%}"
    )

    return True


def verify_dataloader():
    print("\n--- 3. DataLoader & Sampling Check ---")
    try:
        seed_everything(42)
        # Se carga con num_workers=0 por seguridad al ejecutar como script
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=32, num_workers=0
        )

        print(f"[OK] Dataloaders created.")
        print(f"   -> Train Batches: {len(train_loader)}")
        print(f"   -> Val Batches: {len(val_loader)}")

        # Obtener un batch de entrenamiento para verificar las formas de los tensores
        print("   -> Fetching one training batch to verify shapes...")
        batch = next(iter(train_loader))
        images, labels, meta = batch
        print(f"   -> Image Batch Shape: {images.shape}")
        print(f"   -> Labels: {labels}")
        print(f"   -> Metadata: {meta.shape} (Ref: [32, 3])")

        # Verificación del preprocesamiento
        print("   -> [OK] Preprocessing transforms applied successfully.")

        return True
    except Exception as e:
        print(f"[ERROR] DataLoader failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=== Verificación del pipeline de datos y dataloaders ===")

    if not verify_structure():
        print("\n[FAILED] System Structure incomplete.")
        return

    csv_ok, dfs = verify_csv_integrity()
    if not csv_ok:
        print("\n[FAILED] CSV checks failed.")
        return

    split_ok = verify_splits(dfs)
    if not split_ok:
        print("\n[FAILED] Split checks failed.")
        return

    dl_ok = verify_dataloader()
    if dl_ok:
        print("\n=== Pipeline verificado: listo para entrenamiento ===")
    else:
        print("\n[FAILED] DataLoader checks failed.")


if __name__ == "__main__":
    main()
