# SKIN-PROYECT: Motor Clínico de Detección de Melanoma

> **Objetivo**: detección temprana de melanoma y clasificación de cáncer de piel mediante una cascada de *Deep Learning* de dos niveles, diseñada dando prioridad a la **sensibilidad** (perder el menor número posible de melanomas) y a saber **cuándo el sistema no sabe** (abstención y derivación a especialista).

Este repositorio contiene el **motor de Inteligencia Artificial** del sistema: el pipeline de datos, las arquitecturas multimodales, el entrenamiento y la inferencia clínica en cascada con calibración estadística y explicabilidad. Procesa cerca de 442.000 imágenes reales de lesiones cutáneas (ISIC 2019/2020/2024 + HAM10000; unas 430.000 tras eliminar duplicados): dermatoscopias y, sobre todo, recortes de fotografía corporal total 3D de ISIC 2024. La clasificación se ejecuta de forma local.

> **Alcance.** Es la primera parte del proyecto, la que presenté como Trabajo de Fin de Grado: el motor de IA, desde los datos hasta el modelo entrenado y evaluado. La segunda parte (seguimiento de la evolución de un mismo lunar en el tiempo, API y aplicación) sigue en desarrollo; todavía no la considero terminada y por eso no está publicada.
>
> **Aviso.** Es un prototipo académico entrenado y evaluado con datos públicos. No es un producto sanitario, no está validado clínicamente y no se ha usado con pacientes.

---

## ¿Qué hace el sistema?

El sistema funciona como una **cascada de dos niveles**:

1. **Nivel 1 — Triaje binario** (¿benigno o sospechoso?): un primer modelo decide, con un umbral sesgado hacia la detección (0,20 global en lugar de 0,50, ajustable por sitio anatómico), si la lesión merece una segunda mirada. Es deliberadamente cauteloso: prefiere derivar un benigno de más que dejar pasar un melanoma.
2. **Nivel 2 — Especialista multiclase** (¿qué es exactamente?): si el triaje sospecha, un segundo modelo más especializado clasifica la lesión en **8 categorías** (Melanoma, Nevus, Carcinoma Basocelular, Carcinoma Espinocelular, Queratosis Benigna, Queratosis Actínica, Lesión Vascular, Dermatofibroma).

Cada decisión puede acompañarse (son opciones del motor) de **probabilidades calibradas**, **conjuntos de predicción con garantía estadística** (*conformal prediction*), **estimación de incertidumbre** y **mapas de calor explicativos**. El resultado final no es solo una etiqueta, sino un **estado clínico** con su acción recomendada.

---

## 1. Ingeniería de datos

El pipeline de datos (`src/data/`) es un flujo numerado y determinista (`01_explore` → `02_unify` → `03_clean` → `04_split` → `05_calc_stats` → `06_verify` → `07_filter_multiclass`):

* **Fusión**: unifica ISIC 2019, ISIC 2020, ISIC 2024 y HAM10000 (~442k imágenes) en un esquema único de etiquetas y metadatos.
* **Deduplicación**: *perceptual hashing* (dHash) para eliminar duplicados visuales entre datasets y evitar fugas de datos.
* **Split anti-fugas**: división 80/10/10 con `StratifiedGroupKFold` agrupando por **paciente** (o por lesión cuando el dataset no identifica al paciente, como en HAM10000 e ISIC 2019), de modo que todas las imágenes de un mismo grupo caen en el mismo conjunto. `06_verify.py` comprueba que no haya solapamiento entre conjuntos.
* **Normalización de color**: *Color Constancy* (Shades of Gray) para homogeneizar la iluminación de distintas cámaras.
* **Eliminación de artefactos**: "afeitado digital" del vello (BlackHat morfológico + *inpainting*).
* **Augmentations**: geométricas y fotométricas con `Albumentations`, **MixUp/CutMix** (*dominant-instance*, λ≥0.5) y un bloque tipo *RandAugment* restringido a operaciones que no destruyen la señal dermatoscópica.

---

## 2. Arquitectura multimodal

**Archivo:** `src/models/architectures.py` — clase `SkinCancerModel`.

* **Backbones soportados** (factory en `src/models/backbones.py`): EfficientNet-B3 (triaje, 300px) y EfficientNet-B4 (especialista, 380px) por defecto, con soporte adicional para **DINOv2 ViT-L/B**.
* **Fusión multimodal**: la imagen se combina con metadatos clínicos (**edad, sexo, localización anatómica**) mediante **FiLM** (modulación condicional de los mapas de características) y atención **CBAM** (qué y dónde mirar).
* **Pooling**: **GeM** (*Generalized Mean*, parámetro `p` aprendible, en float32 por estabilidad numérica) e **Hybrid Pooling** opcional (GeM + atención).
* **Cabeza**: clasificador que produce *logits* crudos (binario `[B,1]`, multiclase `[B,8]`).

---

## 3. Entrenamiento

**Archivo:** `src/training/train.py`.

* **Funciones de pérdida**: `FocalLoss(γ=2.0)` con *label smoothing* médico para el binario y α=0.7 por defecto (0.75 en el entrenamiento del notebook), para penalizar más perder un caso que habría que derivar; `MulticlassFocalLoss(γ=2.0, label_smoothing=0.1)` para el multiclase, con `WeightedRandomSampler` (*Effective Number*, β=0.999) para equilibrar las clases raras.
* **Optimizador**: AdamW con *learning rate* diferencial (backbone × 0.1, cabeza × 1) + *warmup* lineal → *cosine annealing*.
* **Promediado de pesos**: **SWA** (*Stochastic Weight Averaging*, desde el 60% del entrenamiento, con recalibración manual de BatchNorm) y **EMA** (*Exponential Moving Average*), más **Model Soup** opcional.
* **Selección de checkpoint clínica**: por **pAUC en alta sensibilidad** (`pauc_sens80`) en lugar de AUC global, alineada con el régimen de despliegue del triaje, con un *guardrail* de *recall* mínimo.
* **Estabilidad numérica**: precisión mixta (bf16/fp16), *gradient accumulation*, *gradient checkpointing* opcional, guardas anti-NaN en cada paso y reportes automáticos de equidad demográfica (*fairness*).

---

## 4. Inferencia clínica en cascada

**Archivo:** `src/inference/predict.py` — clase `ClinicalInferenceEngine`.

El motor no devuelve una simple etiqueta, sino un **estado clínico** con precedencia **CRITICAL > AMBER_PRECANCEROUS > AMBER > RED > GREEN**:

| Estado | Significado | Acción |
|--------|-------------|--------|
| **GREEN** | Triaje benigno (prob < umbral; 0,20 global) | Sin derivación |
| **RED** | Sospechoso, pero el especialista predice una clase ni maligna ni precancerosa | Derivar a dermatólogo |
| **AMBER** | El sistema no puede distinguir (conjunto conformal ambiguo) | Revisión humana obligatoria |
| **AMBER_PRECANCEROUS** | El especialista predice queratosis actínica (precancerosa) | Intervención dermatológica (criocirugía, 5-FU, PDT…) |
| **CRITICAL** | El especialista predice cáncer invasivo (Melanoma/BCC/SCC) | Vía rápida oncológica |

Componentes de inferencia:

* **TTA-6**: cada imagen se evalúa en 6 orientaciones y se promedian los *logits* (con guardas anti-NaN).
* **Calibración** (`src/inference/calibration.py`): *Temperature/Vector Scaling* (mejor calibración; ECE del triaje en test ≈ 0,06) y umbrales **por sitio anatómico** (la planta del pie no tiene la misma prevalencia que el torso).
* **Conformal Prediction** (`src/inference/conformal.py`): conjuntos de predicción con **garantía de cobertura** (si los datos nuevos se parecen a los de calibración) y una variante simplificada de *risk control* sobre la tasa de falsos negativos, que activa la derivación a humano.
* **Incertidumbre**: **MC Dropout** (estima la confianza epistémica y marca los casos dudosos).
* **Explicabilidad** (`src/inference/explainability.py`): *Occlusion Sensitivity* — mapas de calor que muestran qué regiones de la imagen impulsan la predicción, válido tanto para backbones CNN como ViT.
* **Abstención**: el sistema marca explícitamente `defer_to_human` cuando la confianza es baja o la predicción es ambigua — *sabe cuándo no sabe*.

---

## Resultados

Cifras de `notebooks/Evaluate_Test_Colab.ipynb`, que conserva las salidas de su ejecución en Colab. Se evalúan dos cascadas con el mismo pipeline calibrado: una CNN (EfficientNet-B3 → B4) y un Vision Transformer (DINOv2 ViT-L en los dos niveles). El conjunto de test es el 10 % de los datos, separado por paciente (o por lesión cuando el dataset no identifica al paciente). Las métricas por imagen usan TTA de 4 vistas y el umbral global 0,20.

**Triaje binario (test, 993 lesiones a derivar).** La clase positiva incluye las lesiones malignas y la queratosis actínica (precancerosa):

| Modelo | AUC | Sensibilidad @0,20 | Especificidad @0,20 | Casos a derivar no detectados |
|--------|-----|--------------------|---------------------|-------------------------------|
| EfficientNet-B3 | 0,987 | 0,971 | 0,957 | 29 / 993 |
| DINOv2 ViT-L | 0,989 | 0,981 | 0,946 | 19 / 993 |

El ViT-L detecta más casos a cambio de algo menos de especificidad. La diferencia de AUC entre ambos no es estadísticamente significativa en test (DeLong, p = 0,148), aunque sí en validación (p = 0,033). El ECE tras calibrar es ≈ 0,06 en los dos.

**Especialista multiclase (test, 8 clases):** AUC macro 0,948 (B4) y 0,961 (ViT-L); *accuracy* 0,76 y 0,81.

**Cascada completa** (muestra de 400 imágenes de test, 200 de ellas a derivar): 3 (B3→B4) y 2 (ViT-L) de esos 200 casos quedaron en verde, sin derivar.

### Limitaciones

* **Sin validación externa.** Train y test vienen de los mismos cuatro datasets públicos, con imágenes de dermatoscopia y de fotografía corporal 3D. No se ha probado con otro hospital, otra cámara ni fotos de móvil, y es esperable que el rendimiento baje.
* **Algunas clases se reconocen mal.** El *recall* del especialista en test es bajo en melanoma (0,54 / 0,68), carcinoma espinocelular (0,27 / 0,55; solo 49 casos), queratosis actínica (0,28 / 0,29; 89 casos) y queratosis benigna (0,45 / 0,49). Un error del especialista no deja sin derivar un caso que el triaje ya marcó (pasa a RED), pero la etiqueta concreta no es fiable en esas clases.
* **Diferencias por subgrupo.** Las salidas del notebook marcan sesgo potencial por sitio anatómico (sobre todo en la especificidad del triaje), por edad y, en el especialista, también por sexo. Es una limitación abierta; los umbrales por sitio anatómico son un primer intento de compensarla.
* **La muestra de la cascada está enriquecida** (50 % de casos a derivar frente a ~2 % en el test real), así que esas cifras no representan la prevalencia real.
* **No se incluyen datos ni modelos entrenados**, por tamaño y por las licencias de los datasets.

---

## Invariantes de seguridad clínica

El proyecto documenta y prueba reglas que **nunca** deben violarse, entre ellas:

1. **Umbral binario global = 0,20** (sesgo hacia la sensibilidad, no 0,50).
2. **Separación por paciente** entre train/val/test (o por lesión cuando falta el identificador de paciente).
3. **Normalización con estadísticas de ImageNet** (alineada con los pesos preentrenados).
4. **Queratosis Actínica tratada como precancerosa** (dispara derivación, no se silencia como benigna).
5. **GeM pooling en float32** (evita desbordamiento numérico).

---

## Cómo se desarrolló

Lo desarrollé en solitario con ayuda de IA y de agentes. Las reglas clínicas (umbral 0,20, separación por paciente, queratosis actínica como precancerosa…), las decisiones de diseño y la validación técnica son mías.

Cada defecto encontrado se registró con su causa raíz y su corrección, y muchos además con un test que evita que vuelva a aparecer. Esa batería de tests vive en el repositorio completo del proyecto, que todavía no es público.

---

## Cómo ejecutar

```bash
# 1. Entorno
python3 -m venv skin_env
source skin_env/bin/activate
pip install -r requirements.txt

# 2. Pipeline de datos (en orden estricto, requiere los datasets en data/raw/)
python src/data/01_explore.py
python src/data/02_unify.py
python src/data/03_clean.py
python src/data/04_split.py
python src/data/05_calc_stats.py
python src/data/06_verify.py
python src/data/07_filter_multiclass.py

# 3. Entrenamiento (local)
python src/training/train.py --mode binary --batch_size 32 --epochs 20 --accum_steps 2
python src/training/train.py --mode multiclass --batch_size 32 --epochs 30 --accum_steps 2
```

Los datasets son públicos (ISIC 2019, 2020 y 2024 y HAM10000) y no se incluyen en el repositorio: hay que descargarlos en `data/raw/`.

El entrenamiento intensivo está preparado para **Google Colab** (`notebooks/Run_Training_Colab.ipynb`), y la evaluación y comparación de modelos, en `notebooks/Evaluate_Test_Colab.ipynb`.

---

## Estado actual

Este repositorio contiene el **motor de IA**: pipeline de datos, arquitecturas multimodales, entrenamiento e inferencia clínica en cascada con calibración, *conformal prediction*, incertidumbre y explicabilidad.

Queda fuera la segunda parte del proyecto (seguimiento temporal del mismo lunar, API y aplicación), que sigue en desarrollo. Lo siguiente que haría falta para acercarlo a un uso real es validarlo con datos externos (otro centro, otras cámaras y fotos de móvil) y medir el sesgo por tipo de piel.

---

## Estructura del proyecto

```text
src/
├── data/        # Pipeline 01-07 + constants.py (mapas de etiquetas) + dataset.py
├── features/    # Preprocesado: HairRemoval + ShadesOfGray
├── models/      # SkinCancerModel + backbones (CNN/ViT) + GeM/Hybrid/CBAM/FiLM
├── training/    # train.py + distillation + SAM + self-distillation + Lion
├── inference/   # ClinicalInferenceEngine + calibration + conformal + explainability
└── utils/       # losses + evaluation + early_stopping + seeding
scripts/         # Calibración (temperature, conformal, per-site threshold)
notebooks/       # Entrenamiento y evaluación en Google Colab
```

---

## Licencia

Código bajo licencia GPL-3.0 (ver `LICENSE`). Los datasets tienen sus propias licencias.
