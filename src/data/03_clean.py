import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def clean_duplicates():
    """
    Script de Limpieza con Perceptual Hashing (dhash).

    Problema: 'Duplicate Data Leakage'.
    Los datasets públicos a menudo contienen la misma imagen subida múltiples veces con nombres distintos,
    o versiones ligeramente modificadas (escaladas/comprimidas).
    Si una versión cae en Train y la otra en Test, la métrica de evaluación se infla artificialmente.

    Solución: Difference Hash (dHash).
    A diferencia de hashes criptográficos (MD5, SHA) que cambian con un solo bit,
    el dHash es resistente a cambios visuales menores (tamaño, compresión, brillo).
    Calcula un "fingerprint" basado en los gradientes de luminosidad de la imagen.
    """
    input_csv = Path("data/processed/02_all_data.csv")
    output_csv = Path("data/processed/03_all_data_cleaned.csv")

    print("--- Inicio de Limpieza de Duplicados (Perceptual Hashing) ---")

    if not input_csv.exists():
        print(f"Error: {input_csv} no encontrado.")
        return

    df = pd.read_csv(input_csv)
    print(f"Inicial: {len(df)} imágenes")

    print("Calculando hashes perceptuales...")
    hashes = {}
    duplicates = []
    corrupted_idx = []  # posiciones (0..N-1) de filas corruptas
    duplicate_idx = []  # posiciones (0..N-1) de filas marcadas como duplicadas

    # Iteramos sobre todo el dataset. El df viene de read_csv → RangeIndex(0, N),
    # así que `idx` (label) == posición. Usamos ambas consistentemente.
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img_path = row["image_path"]
            with Image.open(img_path) as img:
                # 1. dhash (Difference Hash):
                # - Reduce a escala de grises y tamaño pequeño (8x9).
                # - Compara cada píxel con su vecino.
                # - Genera un hash binario de 64 bits.
                h = str(imagehash.dhash(img))

                if h in hashes:
                    # Colisión detectada: Es un duplicado visual
                    duplicates.append(
                        {
                            "duplicate_image": row["image"],
                            "duplicate_dataset": row.get("dataset"),
                            "original_image": hashes[h]["image"],
                            "original_dataset": hashes[h]["dataset"],
                            "hash": h,
                        }
                    )
                    duplicate_idx.append(idx)
                else:
                    # Hash nuevo: Es una imagen única — guardamos image+dataset
                    hashes[h] = {"image": row["image"], "dataset": row.get("dataset")}
        except Exception as e:
            corrupted_idx.append(idx)
            tqdm.write(f"Error leyendo {row['image']}: {e}")

    print(f"\nAnálisis Completado.")
    print(f"   - Hashes Únicos: {len(hashes)}")
    print(f"   - Duplicados Detectados: {len(duplicates)}")
    print(f"   - Imágenes Corruptas: {len(corrupted_idx)}")

    # Abortamos si la corrupción excede el 1%: es señal de un dataset mal montado
    # (copia interrumpida en Colab, permisos rotos, ZIP truncado). Preferimos un error
    # explícito a perder datos en silencio.
    if corrupted_idx:
        pct = 100 * len(corrupted_idx) / len(df)
        by_dx = df.iloc[corrupted_idx]["diagnosis"].value_counts().to_dict()
        print(f"   - Pérdida por corrupción: {pct:.2f}% — distribución: {by_dx}")
        if pct > 1.0:
            raise ValueError(
                f"Pérdida masiva de imágenes por corrupción: {pct:.2f}% "
                f"({len(corrupted_idx)}/{len(df)}). Re-verificar data/raw/."
            )

    # Eliminamos duplicados por POSICIÓN DE FILA (label del índice), no por image_id.
    # ISIC 2019 incluye HAM10000 como subconjunto histórico del challenge, así que
    # comparten ~10K image_ids. Filtrar con `df[~df["image"].isin(dup_ids)]` eliminaría
    # AMBAS filas con el mismo image_id, incluida la original que queremos conservar.
    # Usando los labels del índice (RangeIndex 0..N-1 tras read_csv) solo se elimina el
    # segundo encuentro; el primero sobrevive con su metadata, preservando ISIC 2019
    # (el primer dataset que procesa 02_unify.py) ante colisiones entre datasets.
    to_drop_idx = set(duplicate_idx) | set(corrupted_idx)

    if to_drop_idx:
        df_clean = df.drop(index=list(to_drop_idx)).reset_index(drop=True)
        print(f"   - Conteo Tras Limpieza: {len(df_clean)}")
        print(f"   - Filas eliminadas: {len(to_drop_idx)} "
              f"(duplicados dHash: {len(duplicate_idx)}, corruptas: {len(corrupted_idx)})")
        df_clean.to_csv(output_csv, index=False)
        print(f"Guardado dataset limpio en: {output_csv}")

        if duplicates:
            # Reporte de auditoría solo si hubo duplicados
            pd.DataFrame(duplicates).to_csv(
                "data/processed/03_duplicates_report.csv", index=False
            )
    else:
        print("No se encontraron duplicados ni corruptas.")
        df.to_csv(output_csv, index=False)


def update_metadata_only():
    """
    Utilidad para actualizar metadatos sin recalcular hashes (que es costoso).
    Mantiene el filtrado previo.
    """
    cleaned_path = Path("data/processed/03_all_data_cleaned.csv")
    new_all_path = Path("data/processed/02_all_data.csv")

    if not cleaned_path.exists():
        print("Archivo limpio no encontrado. Ejecutando limpieza completa...")
        clean_duplicates()
        return

    print("--- Actualizando Metadatos en Dataset Limpio ---")
    df_clean_old = pd.read_csv(cleaned_path)
    clean_paths = set(df_clean_old["image_path"])

    df_new = pd.read_csv(new_all_path)

    # Filtrar el nuevo CSV usando los paths que ya sabemos que son 'limpios'
    df_final = df_new[df_new["image_path"].isin(clean_paths)].copy()

    df_final.to_csv(cleaned_path, index=False)
    print(f"Actualizado {cleaned_path}")


def cnn_dedup_pass(
    cleaned_csv: str = "data/processed/03_all_data_cleaned.csv",
    embedding_cache: str = "data/processed/embeddings_dedup.npz",
    output_csv: str = "data/processed/03_all_data_cleaned.csv",
    cnn_duplicates_report: str = "data/processed/03_cnn_duplicates_report.csv",
    similarity_threshold: float = 0.97,
    batch_size: int = 64,
    device: str = None,
):
    """
    Segunda pasada de deduplicación basada en embeddings CNN.

    Motivación:
    ----------
    dHash (pasada 1) detecta duplicados por gradientes perceptuales de 8x9 px, lo
    que deja pasar:
    - Versiones recomprimidas JPEG con pérdida distinta de calidad.
    - Conversiones PNG↔JPG que alteran ligeramente el histograma.
    - Resizes agresivos que modifican la estructura de bordes.

    Embeddings de un backbone preentrenado en ImageNet capturan semántica de alto
    nivel (estructura de lesión, color general, patrón dermoscópico) y son
    robustos a estas perturbaciones. Cosine similarity > 0.97 sobre embeddings L2
    normalizados → duplicado semántico casi seguro.

    Estrategia de eficiencia (crítica para 442k imágenes):
    -----------------------------------------------------
    - Comparamos SOLO dentro de cada (dataset, diagnosis) — reduce O(n²) a
      sum_c(n_c²) típicamente 10-100x más barato en escenarios realistas.
    - Cache de embeddings en disco (npz): re-ejecutar sin recalcular.
    - Batch inference con GPU si disponible.

    Limitación conocida: esta pasada NO detecta conflictos de etiqueta
    -----------------------------------------------------------------
    Agrupar por `(dataset, diagnosis)` hace que solo se comparen imágenes que YA
    comparten diagnóstico, por lo que es matemáticamente imposible detectar pares con
    `dx_a ≠ dx_b` en este layout.

    Los conflictos de etiqueta reales (p.ej. una imagen duplicada etiquetada "Melanoma"
    en ISIC 2019 y "Nevus" en ISIC 2020) son una fuente de error frecuente en datasets
    dermatoscópicos públicos, pero detectarlos exige una pasada aparte cross-diagnosis
    (O(n²) por chunks en GPU, o ANN tipo FAISS/HNSW sobre el dataset completo) que no
    está implementada. No incluimos un check simulado para no dar una falsa sensación
    de seguridad.

    Si necesitas detectar conflictos cross-diagnosis, el plan es:
      1) Recalcular embeddings (cache hoy los precomputa a 224px con ImageNet norm)
      2) FAISS IVF-PQ sobre embeddings L2-normalizados, top-5 por imagen
      3) Filtrar pares con `dataset_a == dataset_b AND dx_a != dx_b AND sim > 0.99`
      4) Exportar a `03_conflicts_report.csv` para revisión manual.
    Coste estimado: ~10min en GPU sobre 442K embeddings.

    Por qué opcional (y no default):
    --------------------------------
    - Requiere PyTorch + timm + GPU (o CPU lenta) — dependencias extra.
    - Puede tardar 1-3h sobre 442k imágenes incluso con GPU.
    - Los duplicados extra detectados son ~0.01-0.05% del dataset — mejora
      marginal. Uso recomendado: ejecutar una vez antes del entrenamiento final.

    Lanzamiento:
    -----------
        python src/data/03_clean.py cnn_dedup
    """
    try:
        import torch
        import timm
        import cv2
    except ImportError as exc:
        raise ImportError(
            "cnn_dedup_pass requiere torch, timm, opencv-python. "
            f"Error: {exc}"
        ) from exc

    cleaned_path = Path(cleaned_csv)
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"{cleaned_csv} no existe. Ejecuta primero clean_duplicates() (pasada 1 dHash)."
        )

    df = pd.read_csv(cleaned_path).reset_index(drop=True)
    print(f"--- Pasada 2: CNN Dedup sobre {len(df)} imágenes ---")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------- 1. Cargar o calcular embeddings -------
    cache_path = Path(embedding_cache)
    if cache_path.exists():
        print(f"Cargando embeddings cacheados de {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        cached_paths = list(cache["paths"])
        embeddings = cache["embeddings"]
        path_to_idx = {p: i for i, p in enumerate(cached_paths)}
        missing = [p for p in df["image_path"].values if p not in path_to_idx]
        if missing:
            print(
                f"[ADVERTENCIA] Cache desactualizada: {len(missing)} imgs nuevas no embebidas. "
                "Recalculando embeddings desde cero (descarta cache)."
            )
            embeddings = None
    else:
        embeddings = None

    if embeddings is None:
        print("Cargando backbone tf_efficientnet_b0.ns_jft_in1k para embeddings...")
        backbone = timm.create_model(
            "tf_efficientnet_b0.ns_jft_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        backbone.eval().to(device)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def _load_img(path):
            img = cv2.imread(str(path))
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            img = (img - mean) / std
            return np.transpose(img, (2, 0, 1))  # CHW

        paths_list = df["image_path"].tolist()
        all_embeddings = np.zeros((len(paths_list), backbone.num_features), dtype=np.float32)
        valid_mask = np.zeros(len(paths_list), dtype=bool)

        with torch.no_grad():
            for start in tqdm(range(0, len(paths_list), batch_size), desc="Embeddings"):
                end = min(start + batch_size, len(paths_list))
                batch_imgs = []
                batch_idx = []
                for i in range(start, end):
                    arr = _load_img(paths_list[i])
                    if arr is None:
                        continue
                    batch_imgs.append(arr)
                    batch_idx.append(i)
                if not batch_imgs:
                    continue
                batch_tensor = torch.from_numpy(np.stack(batch_imgs)).to(device)
                feats = backbone(batch_tensor).cpu().numpy()
                feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
                all_embeddings[batch_idx] = feats
                valid_mask[batch_idx] = True

        embeddings = all_embeddings
        cached_paths = paths_list
        path_to_idx = {p: i for i, p in enumerate(cached_paths)}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            paths=np.array(cached_paths, dtype=object),
            embeddings=embeddings,
            valid_mask=valid_mask,
        )
        n_valid = int(valid_mask.sum())
        print(f"Guardados {n_valid}/{len(paths_list)} embeddings en {cache_path}")
    else:
        valid_mask = cache.get("valid_mask", np.ones(len(cached_paths), dtype=bool))

    # ------- 2. Comparación dentro de (dataset, diagnosis) para reducir O(n²) -------
    df_to_process = df.copy()
    df_to_process["_idx"] = df_to_process["image_path"].map(path_to_idx)
    df_to_process = df_to_process[df_to_process["_idx"].notna()].copy()
    df_to_process["_idx"] = df_to_process["_idx"].astype(int)

    duplicate_pairs = []  # (keep_image, drop_image, similarity)

    # Agrupamos por (dataset, diagnosis) para reducir el coste de O(n²) a sum_c(n_c²).
    # Esta estrategia detecta duplicados puros (misma lesión, mismo label) pero NO
    # conflictos cross-diagnosis: dentro de cada grupo es imposible que dx_a ≠ dx_b
    # (ver el docstring para el plan de la pasada de conflictos).
    group_cols = [c for c in ["dataset", "diagnosis"] if c in df_to_process.columns]
    if not group_cols:
        group_cols = ["dataset"] if "dataset" in df_to_process.columns else []
    if group_cols:
        groups = df_to_process.groupby(group_cols)
    else:
        groups = [(("all",), df_to_process)]

    print(f"Comparando embeddings en {len(groups) if hasattr(groups, '__len__') else 'N'} grupos...")
    for key, sub in (groups.items() if hasattr(groups, "items") else groups):
        if len(sub) < 2:
            continue
        idxs = sub["_idx"].values
        sub_valid = valid_mask[idxs]
        idxs = idxs[sub_valid]
        if len(idxs) < 2:
            continue
        sub_emb = embeddings[idxs]  # (k, D) ya normalizado
        sim = sub_emb @ sub_emb.T  # (k, k) cosine similarity
        # Marcar diagonal y triángulo inferior como no-candidatos.
        np.fill_diagonal(sim, -1.0)
        sim[np.tril_indices_from(sim)] = -1.0

        hits_i, hits_j = np.where(sim > similarity_threshold)
        if len(hits_i) == 0:
            continue

        sub_df_indexed = sub.reset_index(drop=True)
        for a, b in zip(hits_i, hits_j):
            row_a = sub_df_indexed.iloc[a]
            row_b = sub_df_indexed.iloc[b]
            s = float(sim[a, b])
            # Dentro del grupo (dataset, diagnosis) siempre dx_a == dx_b, así que aquí
            # no hay rama de conflicto de etiqueta (ver el docstring).
            duplicate_pairs.append(
                {
                    "keep_image": row_a["image"],
                    "drop_image": row_b["image"],
                    "similarity": s,
                }
            )

    print(f"Duplicados CNN detectados: {len(duplicate_pairs)}")
    print(
        "Aviso: esta pasada NO detecta conflictos cross-diagnosis. "
        "Ver el docstring para el plan de la pasada de conflictos separada."
    )

    # ------- 3. Guardar reportes + eliminar duplicados puros -------
    if duplicate_pairs:
        pd.DataFrame(duplicate_pairs).to_csv(cnn_duplicates_report, index=False)
        print(f"Reporte duplicados CNN: {cnn_duplicates_report}")

    if duplicate_pairs:
        drop_ids = {d["drop_image"] for d in duplicate_pairs}
        df_final = df[~df["image"].isin(drop_ids)].reset_index(drop=True)
        print(f"Imágenes antes: {len(df)} → después: {len(df_final)} (-{len(drop_ids)})")
        df_final.to_csv(output_csv, index=False)
        print(f"Dataset actualizado guardado en {output_csv}")
    else:
        print("Sin cambios en el CSV — no hubo duplicados CNN para eliminar.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "update":
        update_metadata_only()
    elif len(sys.argv) > 1 and sys.argv[1] == "cnn_dedup":
        cnn_dedup_pass()
    else:
        clean_duplicates()
