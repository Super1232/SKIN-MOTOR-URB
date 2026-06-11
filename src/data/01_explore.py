import pandas as pd
from pathlib import Path


def explore_dataset(folder_path):
    print(f"\n{'='*50}")
    print(f"Analizando: {folder_path.name}")
    print(f"{'='*50}")

    # 1. Contar Imágenes
    # Buscamos jpg, jpeg, png (case insensitive)
    extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    image_count = 0
    for ext in extensions:
        image_count += len(list(folder_path.glob(f"**/{ext}")))

    print(f"Total Imágenes encontradas: {image_count}")

    # 2. Buscar y Analizar CSVs
    # Usamos rglob para buscar en subcarpetas (ej. ISIC 2024 tiene el csv dentro de una carpeta)
    csv_files = list(folder_path.rglob("*.csv"))
    if not csv_files:
        print("No se encontraron archivos .csv de metadatos.")

    for csv_file in csv_files:
        print(f"\nArchivo de Metadatos: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            print(f"   - Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
            print(f"   - Columnas: {list(df.columns)}")

            # Intentar adivinar la columna objetivo para mostrar distribución
            # Palabras clave comunes en estos datasets
            possible_targets = [
                "diagnosis",
                "benign_malignant",
                "target",
                "category",
                "anatom_site_general_challenge",
                "dx",
            ]

            target_col = None
            for col in df.columns:
                if col in possible_targets:
                    target_col = col
                    break

            print(f"   - Primeras 3 filas:")
            print(df.head(3).to_string(index=False))

            if target_col:
                print(f"\n   - Distribución de '{target_col}':")
                print(df[target_col].value_counts().head(10))
            else:
                print(
                    "\nNo se identificó automáticamente una columna de 'diagnóstico' obvia."
                )

        except Exception as e:
            print(f"Error al leer {csv_file.name}: {e}")


def main():
    base_path = Path("data/raw")

    if not base_path.exists():
        print(f"Error: No existe el directorio {base_path}")
        return

    # Recorrer carpetas en data/raw
    # Ignoramos 'Zip' y archivos sueltos, solo queremos las carpetas de datasets
    subdirs = [d for d in base_path.iterdir() if d.is_dir() and d.name != "Zip"]

    if not subdirs:
        print("No se encontraron subdirectorios en data/raw (aparte de Zip si existe).")

    for subdir in subdirs:
        explore_dataset(subdir)


if __name__ == "__main__":
    main()
