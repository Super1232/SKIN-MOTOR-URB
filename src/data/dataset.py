import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import cv2
cv2.setNumThreads(0)  # IMPORTANTE: evita que OpenCV cree hilos dentro de los workers de PyTorch (deadlock/overhead).
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
from pathlib import Path
import sys

# build_stain_transform se centraliza en preprocessing.py como fuente de verdad
# única: el entrenamiento y la inferencia deben aplicar exactamente la misma
# normalización de color, y duplicar esa lógica en dataset.py y predict.py la haría
# divergir en silencio.
from src.features.preprocessing import HairRemoval, build_stain_transform
from src.data.constants import (
    BINARY_LABEL_MAP,
    MULTICLASS_LABEL_MAP,
    SEX_MAP,
    SITE_MAP,
    IMAGE_SIZES,
    AGE_UNKNOWN_YEARS,
    SEX_UNKNOWN_CODE,
    SITE_UNKNOWN_CODE,
    AGE_BUCKETS_DEFAULT,
    MIN_GROUP_N_DEFAULT,
    GROUP_BALANCE_SMOOTH_DEFAULT,
)


class SkinLesionDataset(Dataset):
    """
    Dataset personalizado PyTorch para el procesamiento de imágenes dermatoscópicas.

    Esta clase maneja:
    1. La lectura robusta de imágenes desde disco.
    2. La codificación de metadatos clínicos (Multimodalidad).
    3. La aplicación de transformaciones y aumentos de datos.
    4. Validación estricta para asegurar la integridad de los datos (Fail-Fast).
    """

    # Limitación de frecuencia del aviso de imagen corrupta, a nivel de CLASE (se
    # comparte entre las réplicas de DataParallel). El fallback cíclico de __getitem__
    # ya evita la recursión infinita con un máximo de 10 reintentos, pero un dataset
    # con un 5-10% de imágenes corruptas distribuidas (habitual en Colab tras descargas
    # parciales del zip) emitiría un aviso en cada encuentro: cientos de miles de
    # líneas que romperían la barra de progreso y harían ilegible el log. El primer
    # encuentro de cada ruta emite un warning; los siguientes quedan silenciados. El
    # conjunto se limita a 1024 entradas, con vaciado cuando se excede, para no
    # acumular memoria en datasets grandes con corrupción dispersa.
    _corrupt_warned_paths: "set[str]" = set()
    _CORRUPT_WARN_CAP: int = 1024

    def __init__(self, csv_file, transform=None, mode="binary"):
        """
        Args:
            csv_file (str): Ruta al archivo CSV.
            transform (callable, optional): Pipeline de transformaciones.
            mode (str): 'binary' (Benigno/Maligno) o 'multiclass' (8 clases).
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode

        # Configuración según el modo (Arquitectura Jerárquica)
        if self.mode == "binary":
            self.label_col = "benign_malignant"
            self.label_map = BINARY_LABEL_MAP
        elif self.mode == "multiclass":
            self.label_col = "diagnosis"  # Leemos el diagnóstico específico
            # Importado de src/data/constants.py (fuente de verdad única)
            self.label_map = MULTICLASS_LABEL_MAP
        else:
            raise ValueError("Modo debe ser 'binary' o 'multiclass'")

        # Codificación determinista de los metadatos clínicos. Usamos diccionarios
        # estáticos (importados de constants.py) en lugar de pd.factorize: factorize
        # asigna códigos según el orden del CSV, de modo que train y validación
        # recibirían codificaciones distintas para el mismo valor (domain shift).
        self.sex_map = SEX_MAP
        self.site_map = SITE_MAP

        # `.str.strip()` antes de `.str.lower()` es defensivo: sin él, un valor con
        # espacios sobrantes (`"male "`, típico de CSVs exportados desde Excel) no
        # casaría en SEX_MAP/SITE_MAP y el `.fillna(...)` posterior lo sanearía en
        # silencio al código "unknown", perdiendo señal demográfica. Esta cadena debe
        # mantener paridad exacta con `encode_clinical_metadata_scalar` (constants.py),
        # que usa la inferencia, para garantizar el mismo encoding en train e
        # inferencia. Los códigos de "unknown" se toman de SEX_UNKNOWN_CODE/
        # SITE_UNKNOWN_CODE (constants.py), nunca como literales.
        self.df["sex_code"] = (
            self.df["sex"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .map(self.sex_map)
            .fillna(SEX_UNKNOWN_CODE)
        )
        self.df["site_code"] = (
            self.df["anatom_site"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .map(self.site_map)
            .fillna(SITE_UNKNOWN_CODE)
        )

        # Defensa en profundidad: tras .map().fillna(), un NaN debería ser imposible.
        # Si aparece, indica que SEX_MAP/SITE_MAP están incompletos o que la cadena
        # .fillna("unknown") → .str.strip() → .str.lower() → .map() → .fillna(default)
        # se alteró. Usamos `raise ValueError` y no `assert`, que se desactiva con
        # `python -O`.
        if not self.df[["sex_code", "site_code"]].notna().all().all():
            raise ValueError(
                "NaN detectado en sex_code o site_code tras mapeo y fillna. "
                "Esto indica un bug en SEX_MAP/SITE_MAP o en la cadena de transformacion."
            )

        # Normalización de edad con centro fijo en 50 años. Usar la media del batch
        # cambiaría la escala entre train y validación (fuga de información). Avisamos
        # si hay edades fuera del rango clínico [0, 120], que suelen indicar datos
        # corruptos.
        extreme_ages = self.df["age"].dropna()
        extreme_mask = (extreme_ages < 0) | (extreme_ages > 120)
        if extreme_mask.any():
            import warnings
            warnings.warn(
                f"{extreme_mask.sum()} muestras tienen edades fuera de rango clinico [0, 120]: "
                f"min={extreme_ages[extreme_mask].min()}, max={extreme_ages[extreme_mask].max()}. "
                "Seran clipped a [0, 1] tras normalizacion."
            )
        # El centro clínico AGE_UNKNOWN_YEARS vive en constants.py como fuente única;
        # importarlo aquí evita que un literal 50.0 quede desincronizado con
        # architectures.py, evaluation.py y predict.py, que también lo usan.
        self.df["age_norm"] = (self.df["age"].fillna(AGE_UNKNOWN_YEARS) / 100.0).clip(0.0, 1.0)

        # Validación temprana: comprobar todas las etiquetas al crear el dataset, no
        # durante __getitem__. Descubrir una etiqueta inválida a mitad del
        # entrenamiento desperdiciaría horas de GPU.
        invalid_mask = ~self.df[self.label_col].isin(self.label_map)
        if invalid_mask.any():
            bad_labels = self.df.loc[invalid_mask, self.label_col].unique().tolist()
            raise ValueError(
                f"ERROR CRITICO: {invalid_mask.sum()} muestras tienen etiquetas invalidas "
                f"en modo '{self.mode}': {bad_labels}. "
                f"Etiquetas validas: {list(self.label_map.keys())}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, _retries=0):
        """Devuelve la muestra (imagen, etiqueta, metadatos) en la posición `idx`.

        Lee la imagen de disco, codifica los metadatos clínicos y aplica el pipeline
        de transformaciones. Si la imagen está corrupta o no se encuentra, salta a la
        siguiente posición de forma cíclica (con un máximo de reintentos) para no
        abortar un entrenamiento largo por un único archivo dañado.
        """
        row = self.df.iloc[idx]
        img_path_raw = row["image_path"]

        # Resolución de ruta portable. El CSV guarda rutas absolutas de la máquina
        # local; si el código corre en Colab u otra máquina, reconstruimos la ruta
        # relativa desde el marcador "data/raw/" y la unimos a la raíz del proyecto.
        if "data/raw/" in img_path_raw:
            rel_path = img_path_raw[img_path_raw.find("data/raw/") :]
            project_root = Path(__file__).resolve().parents[2]
            img_path = str(project_root / rel_path)
        else:
            img_path = img_path_raw

        # La columna de etiqueta depende del modo (benign_malignant en binario,
        # diagnosis en multiclase). Convertimos el nombre a su índice numérico; una
        # etiqueta no reconocida aborta con un mensaje claro.
        label_str = row[self.label_col]

        if label_str not in self.label_map:
            raise ValueError(
                f"ERROR CRÍTICO: Etiqueta inválida '{label_str}' en modo '{self.mode}'."
            )

        label = self.label_map[label_str]

        # Vector de metadatos clínicos [edad_normalizada, código_sexo, código_sitio]
        # que el modelo multimodal fusiona con las features de la imagen. El orden es
        # un invariante: architectures.py lo desempaqueta posicionalmente.
        meta = torch.tensor(
            [row["age_norm"], float(row["sex_code"]), float(row["site_code"])],
            dtype=torch.float32,
        )

        # Lectura de imagen con tolerancia a fallos. Una sola imagen corrupta en disco
        # no debe abortar un entrenamiento de horas: si la lectura falla, saltamos a la
        # siguiente muestra (con reintentos acotados más abajo).
        try:
            if not Path(img_path).exists():
                raise FileNotFoundError()
            image = cv2.imread(img_path)
            if image is None:
                raise IOError()
        except Exception:
            # Límite de reintentos para evitar recursión infinita si hay varias
            # imágenes corruptas consecutivas.
            if _retries >= 10:
                raise RuntimeError(
                    f"No se encontró imagen válida tras 10 reintentos desde idx {idx}. "
                    f"Verificar rutas de imagen en el CSV."
                )
            # Aviso limitado a una vez por ruta (ver _corrupt_warned_paths): un dataset
            # con corrupción dispersa generaría miles de avisos que romperían la barra
            # de progreso y el log del notebook.
            if img_path not in SkinLesionDataset._corrupt_warned_paths:
                if len(SkinLesionDataset._corrupt_warned_paths) >= SkinLesionDataset._CORRUPT_WARN_CAP:
                    # Vaciado defensivo para no acumular memoria en datasets grandes;
                    # tras él, las mismas rutas volverán a avisar una vez.
                    SkinLesionDataset._corrupt_warned_paths.clear()
                SkinLesionDataset._corrupt_warned_paths.add(img_path)
                import warnings as _warnings
                _warnings.warn(
                    f"Imagen corrupta o no encontrada: {img_path}. "
                    f"Saltando y re-muestreando (one-shot por path).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx, _retries + 1)

        # OpenCV carga en BGR; convertimos a RGB, que es lo que esperan los modelos
        # preentrenados en ImageNet.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Aplicación del pipeline de transformaciones (augmentation + preprocesado).
        # Albumentations devuelve un diccionario con la imagen en la clave "image".
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        # Devuelve la imagen, la etiqueta (entero de 64 bits, requerido por la función
        # de pérdida) y el vector de metadatos.
        return image, torch.tensor(label, dtype=torch.long), meta


def get_transforms(phase="train", mode="binary", backbone_kind="cnn"):
    """
    Define el pipeline de transformación de imágenes.

    Estrategia:
    - Preprocesado: limpieza de la imagen (eliminación de vello) y normalización de color.
    - Data augmentation (solo en train): variaciones sintéticas para reducir el overfitting.

    Args:
        phase (str): 'train' (con aumentos) o 'val'/'test' (solo redimensionado y normalización).
        mode (str): 'binary' o 'multiclass' para usar la resolución nativa correcta.
        backbone_kind (str): 'cnn' o 'vit' — selecciona resolución nativa del backbone
            desde IMAGE_SIZES (constants.py). ViT usa múltiplos de patch_size=14 (294/378px)
            para evitar doble interpolación que degrada features dermatoscópicas finas.
    """
    # Estadísticas de normalización de ImageNet, donde se preentrenó el backbone.
    # Usar las estadísticas del propio dataset rompería la alineación con los pesos
    # preentrenados (domain shift).
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transforms = []

    # Resolución nativa del backbone, centralizada en IMAGE_SIZES (constants.py):
    # CNN (EfficientNet B3/B4) 300/380px, ViT (DINOv2) 294/378px (patch_size=14).
    # Los modelos `tf_` de timm se preentrenaron con interpolación bicúbica; usar
    # INTER_AREA destruiría la distribución de altas frecuencias y desalinearía los
    # pesos de ImageNet.
    img_size = IMAGE_SIZES[backbone_kind][mode]
    crop_size = img_size + 32  # margen para el recorte aleatorio

    # El RandomCrop se aplica ANTES de HairRemoval/ShadesOfGray en train para que el
    # preprocesado vea el mismo tamaño que en val/inferencia (que solo redimensionan a
    # img_size). Si ShadesOfGray estimara el iluminante sobre un tamaño distinto en
    # train (crop_size) y en val (img_size), introduciría un domain shift en la
    # corrección de color, agravado por el viñeteado del dermatoscopio (bordes negros)
    # que altera radicalmente el iluminante estimado.
    if phase == "train":
        transforms.append(A.Resize(crop_size, crop_size, interpolation=cv2.INTER_CUBIC))
        transforms.append(A.RandomCrop(img_size, img_size))
    else:
        transforms.append(A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC))

    # --- 1. Preprocesado determinista ---
    # Se aplica siempre, en train y en test, tras el RandomCrop para que ambos vean
    # el mismo tamaño de imagen.
    transforms.append(HairRemoval(p=1.0))  # eliminación de vello (algoritmo DullRazor)
    transforms.append(build_stain_transform())  # normalización de color (STAIN_METHOD)

    # --- 2. Data augmentation (solo en train; el RandomCrop ya se aplicó arriba) ---
    if phase == "train":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(
                    p=0.5
                ),
                A.Rotate(limit=180, p=0.5),
                # Las lesiones cutáneas no tienen orientación natural, así que voltear
                # y rotar aumenta la variabilidad sin introducir sesgo.
                A.CLAHE(p=0.2),  # realza el contraste local para ver texturas sutiles
                # Simula distintas cámaras e iluminaciones variando brillo, contraste,
                # saturación y tono.
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3
                ),
                # CoarseDropout: ocluye parches para que la red no dependa de una sola
                # región de la imagen. El tamaño (hasta 32x32) y el relleno aleatorio
                # imitan artefactos dermatoscópicos reales (burbujas de gel, marcas de
                # regla, vello residual), más realistas que parches negros fijos.
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    fill="random_uniform",
                    p=0.2,
                ),
                # Cambios leves de escala y traslación (la rotación ya la cubre A.Rotate).
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(0.05, 0.05),
                    p=0.3,
                ),
                # Perspective: simula la inclinación del dermatoscopio al fotografiar a
                # mano. El rango es conservador para no distorsionar la morfología.
                A.Perspective(scale=(0.02, 0.05), p=0.1),
                # OpticalDistortion: simula la distorsión barrel/pincushion propia de la
                # óptica de cada dispositivo, con un límite bajo para preservar la
                # geometría diagnóstica.
                A.OpticalDistortion(distort_limit=0.05, p=0.1),
            ]
        )
        # Augmentaciones adicionales para multiclase: más diversidad visual para las
        # clases raras. Con solo 84 imágenes de Dermatofibroma, cada vista debe ser lo
        # más distinta posible; estas deformaciones no lineales simulan variaciones de
        # piel y óptica que el conjunto limitado no cubre.
        if mode == "multiclass":
            transforms.extend(
                [
                    A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.15),
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.15),
                    A.GaussNoise(std_range=(0.01, 0.04), p=0.15),
                    # RandAugment-equivalente vía SomeOf: selecciona 2 de 6 ops con
                    # magnitud controlada. Albumentations no expone RandAugment nativo;
                    # SomeOf(n=2) replica la semántica del paper (Cubuk et al. 2020):
                    # N operaciones muestreadas uniformemente con magnitud fija.
                    # Pool restringido a ops que no rompen la señal dermatoscópica:
                    # sin posterize/solarize (destruyen gradientes sutiles de lesión),
                    # sin equalize (colisiona con CLAHE/ShadesOfGray upstream).
                    A.SomeOf(
                        [
                            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),
                            A.Emboss(alpha=(0.1, 0.3), strength=(0.2, 0.5), p=1.0),
                            A.RandomBrightnessContrast(
                                brightness_limit=0.15, contrast_limit=0.15, p=1.0
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=1.0
                            ),
                            A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                            A.MotionBlur(blur_limit=3, p=1.0),
                        ],
                        n=2,
                        replace=False,
                        p=0.3,
                    ),
                ]
            )

    # --- 3. Normalización final y conversión a tensor ---
    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),  # (pixel - mean) / std por canal
            ToTensorV2(),  # NumPy [H, W, C] -> tensor PyTorch [C, H, W]
        ]
    )

    return A.Compose(transforms)


def seed_worker(worker_id):
    """
    Obliga a cada worker de la CPU a usar una semilla distinta.
    Evita que PyTorch duplique las mismas transformaciones aleatorias en hilos paralelos.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(
    data_dir="data/processed", batch_size=32, num_workers=None, mode="binary",
    backbone_kind="cnn", seed: int = 42,
    group_balance_dims: tuple = (),
    group_balance_smooth: float = GROUP_BALANCE_SMOOTH_DEFAULT,
    min_group_n: int = MIN_GROUP_N_DEFAULT,
):
    """
    Genera los DataLoaders de PyTorch listos para el entrenamiento.

    Args:
        mode (str): 'binary' (usa train.csv) o 'multiclass' (usa train_multiclass.csv).
        seed (int): semilla del generador de PyTorch que inicializa los workers
            (base de `worker_init_fn`) y del WeightedRandomSampler. Propagarla desde la
            CLI es necesario para la diversidad de los deep ensembles: si fuese un
            literal fijo, los workers y el muestreo balanceado verían las mismas
            secuencias entre ejecuciones con semillas distintas, y los modelos del
            ensemble perderían la independencia (Lakshminarayanan 2017) de la que
            depende la garantía de cobertura conformal posterior.
        group_balance_dims (tuple): dimensiones de rebalanceo por subgrupo, subconjunto
            de `("site", "age")`. El valor por defecto `()` preserva el comportamiento
            base (pesos idénticos al cálculo solo-por-clase). Al pasar dimensiones, se
            aplica un factor multiplicativo por muestra que acerca la prevalencia
            maligna de cada subgrupo a la prevalencia global, forzando al modelo a
            aprender features visuales intra-grupo en lugar de usar `site_code` o
            `age_bucket` como atajo. Verificado con palms/soles (58% maligno en train):
            smooth=0.7 reduce el ratio de pesos maligno/benigno de 56× a 3.3×. Es
            ortogonal al Effective Number (β=0.999), que ya balancea por clase.
        group_balance_smooth (float): suavizado ∈ [0, 1] que modera la corrección.
            smooth=0 → factor=1 (idéntico a no rebalancear); smooth=1 → balance total
            (prevalencia resultante ≈ prevalencia global). Por defecto 0.7.
        min_group_n (int): mínimo de muestras por subgrupo para considerar fiable el
            rebalanceo. Los subgrupos con n < min_group_n no se rebalancean (factor=1).
            Por defecto MIN_GROUP_N_DEFAULT=50, que evita memorizar unas pocas muestras
            únicas. No reducir por debajo de 30 sin validación A/B.
        batch_size (int): tamaño de lote enviado a la GPU; bajarlo si hay poca memoria.
        num_workers (int | None): si es None, se detecta automáticamente el número de
            cores (limitado a 8) para no estrangular la alimentación de la GPU.
    """
    import os

    if num_workers is None:
        # Usamos todos los cores disponibles, con un tope de 8: sin tope, una máquina
        # con muchos cores (p.ej. 96) crearía demasiados procesos hijo y agotaría la
        # memoria.
        num_workers = min(os.cpu_count() or 2, 8)
        print(f"[INFO] Usando {num_workers} procesos de carga de datos.")

    data_dir = Path(data_dir)

    # Selección de archivo CSV según el modo
    if mode == "multiclass":
        train_csv = data_dir / "07_train_multiclass.csv"
        val_csv = data_dir / "07_val_multiclass.csv"
        test_csv = data_dir / "07_test_multiclass.csv"
    else:
        train_csv = data_dir / "04_train.csv"
        val_csv = data_dir / "04_val.csv"
        test_csv = data_dir / "04_test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"No se encontró {train_csv}. ¿Ejecutaste 07_filter_multiclass.py?"
        )

    # Instanciación de Datasets con el modo correcto
    train_ds = SkinLesionDataset(
        train_csv, transform=get_transforms("train", mode=mode, backbone_kind=backbone_kind), mode=mode
    )
    val_ds = SkinLesionDataset(val_csv, transform=get_transforms("val", mode=mode, backbone_kind=backbone_kind), mode=mode)
    test_ds = SkinLesionDataset(test_csv, transform=get_transforms("test", mode=mode, backbone_kind=backbone_kind), mode=mode)

    # Se crean tres instancias del Dataset:
    # train_ds recibe augmentations agresivas; val_ds y test_ds solo
    # redimensionado + normalización.

    # Estrategia de muestreo: el dataset está muy desbalanceado, así que usamos un
    # WeightedRandomSampler para sobrerrepresentar las clases minoritarias.
    targets = train_ds.df[train_ds.label_col].map(train_ds.label_map).values

    # Comprobación de seguridad: ninguna etiqueta debe quedar sin mapear (NaN).
    if np.isnan(targets).any():
        raise ValueError("Error Crítico: Etiquetas no mapeadas encontradas en targets.")

    class_counts = np.bincount(targets.astype(int))  # Asegurar int

    if mode == "multiclass":
        # Effective Number of Samples (Cui et al., CVPR 2019)
        # Pure inverse frequency causa ~200x oversampling de Dermatofibroma (84 imgs),
        # forzando al modelo a MEMORIZAR las 84 imágenes en lugar de generalizar.
        # Effective Number modera esto a ~12x: suficiente boost sin memorización.
        #
        # Fórmula: weight = (1-β) / (1 - β^n)  donde n = class_count
        # β=0.999 comprime el rango de pesos exponencialmente:
        #   Nevus(8938): 1x | Melanoma: 1.05x | SCC: 3.4x | Dermatofibroma: 12.4x
        beta = 0.999
        effective_num = 1.0 - np.power(beta, class_counts)
        class_weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    else:
        # Binario: frecuencia inversa simple (funciona bien con 337K benignos vs 6K malignos).
        class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = class_weights[targets.astype(int)]

    # Cast explícito a float32. `class_weights` puede heredar float64 (np.power sobre
    # enteros lo produce); forzar float32 garantiza que el WeightedRandomSampler reciba
    # siempre el mismo tipo, con o sin rebalanceo por subgrupo, lo que mantiene la
    # reproducibilidad entre configuraciones.
    sample_weights = sample_weights.astype(np.float32)

    # Guard: detectar NaN/Inf en pesos (posible si alguna clase tiene 0 muestras → div/0 → inf)
    if not np.all(np.isfinite(sample_weights)):
        bad_count = np.sum(~np.isfinite(sample_weights))
        raise ValueError(
            f"ERROR CRITICO: {bad_count} sample weights son NaN/Inf. "
            f"class_counts={class_counts}, class_weights={class_weights}. "
            "Revisar que todas las clases tengan al menos 1 muestra."
        )

    # Rebalanceo por subgrupo demográfico.
    # Cuando `group_balance_dims` no está vacío, se multiplica el peso de cada muestra
    # por un factor que acerca la prevalencia de su clase dentro de cada subgrupo a la
    # prevalencia global. Así el modelo aprende features visuales intra-grupo en lugar
    # de usar el identificador del grupo como atajo predictivo (la causa de la
    # especificidad ~0% observada en palms/soles).
    #
    # Algoritmo: para cada (dimensión, grupo g, clase c):
    #     ratio_c = target_prev[c] / group_prev[c, g]   (con clamps a [0.01, 0.99])
    #     factor_c = ratio_c ** smooth
    # y se multiplica por factor_c a las muestras de clase c que están en el grupo g.
    # smooth=0 → factor=1 (no rebalancea); smooth=1 → balance total.
    #
    # Es importante iterar por (grupo, clase) aplicando un único factor a cada muestra.
    # Una variante "uno contra el resto" que aplicara el factor a todas las demás
    # clases acumularía un producto de N-1 factores por muestra, distorsionando los
    # pesos hasta un ±40% en multiclase. En binario ambas formulaciones coinciden
    # (solo hay una clase positiva, sin acumulación cruzada).
    #
    # Comprobado con palms/soles binario:
    #   benigno: 1/337K × 1.83 = 5.5e-6
    #   maligno: 1/6000 × 0.105 = 1.8e-5  → ratio 3.3× (frente a 56× sin rebalanceo)
    if group_balance_dims:
        valid_dims = {"site", "age"}
        unknown_dims = set(group_balance_dims) - valid_dims
        if unknown_dims:
            raise ValueError(
                f"group_balance_dims contiene dimensiones desconocidas: "
                f"{sorted(unknown_dims)}. Validas: {sorted(valid_dims)}."
            )
        if not (0.0 <= group_balance_smooth <= 1.0):
            raise ValueError(
                f"group_balance_smooth={group_balance_smooth} fuera de "
                f"[0, 1]. smooth=0 preserva baseline; smooth=1 balance total."
            )
        if min_group_n < 1:
            raise ValueError(
                f"min_group_n={min_group_n} debe ser >= 1."
            )

        targets_int = targets.astype(int)
        group_factors = np.ones(len(targets_int), dtype=np.float64)

        # Resolvemos la columna de cada dimensión. "site_code" ya existe tras __init__;
        # para "age" construimos los buckets desde la edad cruda (los NaN se imputan a
        # AGE_UNKNOWN_YEARS, igual que en la normalización de edad y en fairness_report).
        site_ids = train_ds.df["site_code"].astype(int).values if "site" in group_balance_dims else None
        if "age" in group_balance_dims:
            age_raw = train_ds.df["age"].fillna(AGE_UNKNOWN_YEARS).values
            age_bucket_ids = np.full(len(age_raw), -1, dtype=np.int64)
            for bi, (lo, hi, _label) in enumerate(AGE_BUCKETS_DEFAULT):
                mask_b = (age_raw >= lo) & (age_raw < hi)
                age_bucket_ids[mask_b] = bi
            if (age_bucket_ids == -1).any():
                # No debería ocurrir (el último bin termina en infinito), pero lo
                # comprobamos por defensa en profundidad.
                n_unbinned = int((age_bucket_ids == -1).sum())
                raise ValueError(
                    f"{n_unbinned} muestras quedaron sin age_bucket. "
                    "Revisar AGE_BUCKETS_DEFAULT (el último bin debe terminar en inf)."
                )
        else:
            age_bucket_ids = None

        # Clases a balancear: todas las presentes en el split (binario [0, 1];
        # multiclase [0..7]). np.unique es eficiente sobre arrays grandes.
        classes_to_balance = np.unique(targets_int).tolist()

        # Prevalencia global por clase (constante en todas las dimensiones). El clamp a
        # [0.01, 0.99] evita la división por cero en grupos puramente positivos o negativos.
        target_prev_by_class = {
            int(c): float(np.clip((targets_int == c).mean(), 0.01, 0.99))
            for c in classes_to_balance
        }

        # Iteramos (dimensión, grupo, clase): cada muestra recibe un único factor por
        # (dimensión, su clase), sin acumulación cruzada entre clases.
        for dim_name in group_balance_dims:
            if dim_name == "site":
                group_ids = site_ids
            else:  # "age"
                group_ids = age_bucket_ids
            # group_ids es np.ndarray validado arriba; nunca None aquí.
            for g in np.unique(group_ids):
                mask_g = group_ids == g
                n_g = int(mask_g.sum())
                if n_g < min_group_n:
                    continue
                # Para cada clase, computar factor por (clase, grupo) y aplicar
                # SOLO a samples de esa clase en este grupo. Sin acumulación.
                for c in classes_to_balance:
                    target_prev_c = target_prev_by_class[int(c)]
                    group_prev_c = float(np.clip((targets_int[mask_g] == c).mean(), 0.01, 0.99))
                    ratio_c = target_prev_c / group_prev_c
                    factor_c = ratio_c ** group_balance_smooth

                    # mask_class_in_group: samples DE clase c que están EN grupo g
                    mask_class_in_group = (targets_int == c) & mask_g
                    if not mask_class_in_group.any():
                        continue  # No hay samples de clase c en grupo g (factor irrelevante)
                    group_factors[mask_class_in_group] *= factor_c

        # Los factores deben quedar finitos y positivos.
        if not np.all(np.isfinite(group_factors)):
            raise ValueError(
                f"group_factors contiene NaN/Inf tras el rebalanceo. "
                f"Revisar group_balance_smooth={group_balance_smooth} y los datos del CSV."
            )
        if (group_factors <= 0).any():
            raise ValueError(
                "group_factors contiene valores <= 0 tras el rebalanceo."
            )

        # Se aplican multiplicativamente sobre los pesos existentes, preservando los
        # pesos de clase del Effective Number / frecuencia inversa.
        sample_weights = (sample_weights.astype(np.float64) * group_factors).astype(np.float32)

        # Revalidar finitud tras la multiplicación.
        if not np.all(np.isfinite(sample_weights)):
            raise ValueError(
                "sample_weights contiene NaN/Inf tras combinar con group_factors. "
                "Revisar class_weights y group_factors."
            )

        # Emitimos el evento como warning (no print) para que un backend con logging
        # estructurado pueda capturarlo; en CLI/Colab aparece igualmente por stderr.
        import warnings as _warnings_gb
        _warnings_gb.warn(
            f"Group balance activo: dims={list(group_balance_dims)}, "
            f"smooth={group_balance_smooth}, min_group_n={min_group_n}. "
            f"sample_weights ratio max/min: {sample_weights.max()/max(sample_weights.min(), 1e-12):.1f}×",
            RuntimeWarning,
            stacklevel=2,
        )

    # Muestreador ponderado. Usa un generador propio sembrado con `seed` (separado del
    # generador del DataLoader) para aislar la secuencia de muestreo de la
    # inicialización de los workers. Con replacement=True una misma muestra de una clase
    # rara puede repetirse dentro de una época, y num_samples=len(...) mantiene el
    # tamaño de época.
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
        generator=sampler_gen,
    )

    # Generador determinista del DataLoader, sembrado con `seed` (no un literal fijo).
    # `worker_init_fn=seed_worker` lee `torch.initial_seed()`, que deriva de este
    # generador, de modo que cada worker obtiene una semilla distinta pero reproducible.
    # Propagar la semilla preserva la diversidad de los ensembles entre ejecuciones.
    g = torch.Generator()
    g.manual_seed(seed)

    # Creación de Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, # CRÍTICO: Previene crashes silenciosos en BatchNorm si el último batch = 1
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0) # Mantiene vivos los workers entre épocas (Acelera el salto Train->Val)
    )

    # Val y Test no necesitan balanceo artificial, queremos medir el rendimiento real.
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Configuración para permitir ejecución directa del script
    sys.path.append(str(Path(__file__).resolve().parents[2]))

    print("Comprobando la construcción de los DataLoaders...")
    try:
        tl, vl, tsl = get_dataloaders(batch_size=4, num_workers=0)
        print(f"Train batches: {len(tl)}, Val batches: {len(vl)}")

        batch = next(iter(tl))
        images, labels, meta = batch
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        print(f"Metadata: {meta}")
        print("Construcción de DataLoaders verificada correctamente.")
    except Exception as e:
        print(f"Error: {e}")
