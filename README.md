# SKIN-PROYECT: Motor Clínico de Detección de Melanoma

> **Objetivo**: detección temprana de melanoma y clasificación de cáncer de piel mediante una cascada de *Deep Learning* de dos niveles, diseñada con prioridad clínica absoluta en la **sensibilidad** (no perder ningún melanoma) y en saber **cuándo el sistema no sabe** (abstención y derivación a especialista).

Este repositorio contiene el **motor de Inteligencia Artificial** del sistema: el pipeline de datos, las arquitecturas multimodales, el entrenamiento y la inferencia clínica en cascada con calibración estadística y explicabilidad. Procesa cerca de 442.000 imágenes dermatoscópicas reales (ISIC 2019/2020/2024 + HAM10000) y ejecuta la clasificación de forma local.

---

## ¿Qué hace el sistema?

El sistema imita el razonamiento de un dermatólogo mediante una **cascada de dos niveles**:

1. **Nivel 1 — Triaje binario** (¿benigno o sospechoso?): un primer modelo decide, con un umbral sesgado hacia la detección (0.20, no 0.50), si la lesión merece una segunda mirada. Es deliberadamente cauteloso: prefiere derivar un benigno de más que dejar pasar un melanoma.
2. **Nivel 2 — Especialista multiclase** (¿qué es exactamente?): si el triaje sospecha, un segundo modelo más especializado clasifica la lesión en **8 categorías** (Melanoma, Nevus, Carcinoma Basocelular, Carcinoma Espinocelular, Queratosis Benigna, Queratosis Actínica, Lesión Vascular, Dermatofibroma).

Cada decisión se acompaña de **probabilidades calibradas**, **conjuntos de predicción con garantía estadística** (*conformal prediction*), **estimación de incertidumbre** y **mapas de calor explicativos**. El resultado final no es solo una etiqueta, sino un **estado clínico** con su acción recomendada.

---

## 1. Ingeniería de datos

El pipeline de datos (`src/data/`) es un flujo numerado y determinista (`01_explore` → `02_unify` → `03_clean` → `04_split` → `05_calc_stats` → `06_verify` → `07_filter_multiclass`):

* **Fusión**: unifica ISIC 2019, ISIC 2020, ISIC 2024 y HAM10000 (~442k imágenes) en un esquema único de etiquetas y metadatos.
* **Deduplicación**: *perceptual hashing* (dHash) para eliminar duplicados visuales entre datasets y evitar fugas de datos.
* **Split anti-fugas**: división 80/10/10 con `StratifiedGroupKFold` agrupando por **paciente**, de modo que todas las imágenes de un paciente caen en el mismo conjunto (cero *patient leakage*, verificado de forma estricta).
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

* **Funciones de pérdida**: `FocalLoss(α=0.7, γ=2.0)` con *label smoothing* médico para el binario (penaliza ~2,3× más perder un melanoma); `MulticlassFocalLoss(γ=2.0, label_smoothing=0.1)` para el multiclase, con `WeightedRandomSampler` (*Effective Number*, β=0.999) para equilibrar las clases raras.
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
| **GREEN** | Triaje benigno (prob < 0.20) | Sin derivación |
| **RED** | Sospechoso, lesión de interés no maligna/precancerosa | Derivar a dermatólogo |
| **AMBER** | El sistema no puede distinguir (conjunto conformal ambiguo) | Revisión humana obligatoria |
| **AMBER_PRECANCEROUS** | Queratosis Actínica (precancerosa) | Intervención dermatológica (criocirugía, 5-FU, PDT…) |
| **CRITICAL** | Cáncer invasivo confirmado (Melanoma/BCC/SCC) | Vía rápida oncológica |

Componentes de inferencia:

* **TTA-6**: cada imagen se evalúa en 6 orientaciones y se promedian los *logits* (con guardas anti-NaN).
* **Calibración** (`src/inference/calibration.py`): *Temperature/Vector Scaling* (probabilidades fiables) y umbrales **por sitio anatómico** (la planta del pie no tiene la misma prevalencia que el torso).
* **Conformal Prediction** (`src/inference/conformal.py`): conjuntos de predicción con **garantía de cobertura** y control directo de la tasa de falsos negativos (*risk control*).
* **Incertidumbre**: **MC Dropout** (estima la confianza epistémica y marca los casos dudosos).
* **Explicabilidad** (`src/inference/explainability.py`): *Occlusion Sensitivity* — mapas de calor que muestran qué regiones de la imagen impulsan la predicción, válido tanto para backbones CNN como ViT.
* **Abstención**: el sistema marca explícitamente `defer_to_human` cuando la confianza es baja o la predicción es ambigua — *sabe cuándo no sabe*.

---

## Invariantes de seguridad clínica

El proyecto documenta y prueba reglas que **nunca** deben violarse, entre ellas:

1. **Umbral binario = 0.20** (sesgo hacia la sensibilidad, no 0.50).
2. **Separación por paciente** entre train/val/test (cero fugas).
3. **Normalización con estadísticas de ImageNet** (alineada con los pesos preentrenados).
4. **Queratosis Actínica tratada como precancerosa** (dispara derivación, no se silencia como benigna).
5. **GeM pooling en float32** (evita desbordamiento numérico).

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

El entrenamiento intensivo está preparado para **Google Colab** (`notebooks/Run_Training_Colab.ipynb`).

---

## Estado actual y alcance de la entrega

Este repositorio entrega el **motor de IA completo y funcional**: pipeline de datos, arquitecturas multimodales, entrenamiento con todas las técnicas anteriores e inferencia clínica en cascada con calibración, *conformal prediction*, incertidumbre y explicabilidad.

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
