import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


def main():
    """
    Cálculo de Estadísticas de Normalización (Mean & Std) — SOLO REFERENCIA.

    Calcula la media y desviación estándar del dataset dermatoscópico.
    Estos valores NO se usan en entrenamiento ni inferencia: se usan las estadísticas
    de ImageNet para mantener la alineación con los pesos preentrenados del
    EfficientNet.

    Propósito: documentar la distribución cromática del dominio dermatoscópico
    para análisis y comparación con ImageNet. Los resultados son informativos.
    """
    csv_path = Path("data/processed/04_train.csv")
    if not csv_path.exists():
        print("Train CSV no encontrado. Ejecuta el pipeline primero.")
        return

    print("--- Calculando Media y Desviación Estándar (Muestreo Representativo) ---")
    df = pd.read_csv(csv_path)

    # Teorema Central del Límite:
    # 10,000 muestras son estadísticamente suficientes para aproximar la media poblacional
    # con un error insignificante (<0.01), ahorrando tiempo de cómputo.
    SAMPLE_SIZE = 10000
    # Capturamos el total ANTES del muestreo; tras df.sample(), len(df) ya sería SAMPLE_SIZE.
    total_count = len(df)
    if total_count > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
        print(f"Muestreando {SAMPLE_SIZE} imágenes de {total_count} totales.")
    else:
        print(f"Usando todas las {len(df)} imágenes.")

    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    count = 0

    # Algoritmo de paso único (Online algorithm) para varianza
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = row["image_path"]
        img = cv2.imread(path)
        if img is None:
            continue

        # Conversión obligatoria BGR -> RGB (OpenCV lee en BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalización al rango [0, 1]
        img = img / 255.0

        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img**2).sum(axis=(0, 1))
        count += img.shape[0] * img.shape[1]

    mean = pixel_sum / count
    var = (pixel_sq_sum / count) - (mean**2)
    std = np.sqrt(np.maximum(var, 0))

    print("\n--- RESULTADOS ESTADÍSTICOS ---")
    print(f"Media (RGB): {mean}")
    print(f"Std  (RGB): {std}")
    print(
        f"Formato Albumentations: mean={tuple(mean.round(3))}, std={tuple(std.round(3))}"
    )


if __name__ == "__main__":
    main()
