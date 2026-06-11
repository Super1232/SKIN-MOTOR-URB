#!/bin/bash
# Deep ensembles calibrados — script de orquestación.
#
# Referencia: Lakshminarayanan et al., "Simple and Scalable Predictive
# Uncertainty Estimation Using Deep Ensembles", NeurIPS 2017.
#
# Entrena 3 réplicas del modelo con seeds diferentes. La inferencia
# (src/inference/predict.py) ya soporta ensemble via `binary_ensemble_paths`
# y `multiclass_ensemble_paths`. Este
# script solo orquesta la generación de los 3 checkpoints; la integración
# es automática al pasar las rutas al constructor de `ClinicalInferenceEngine`.
#
# Uso:
#     bash scripts/train_deep_ensemble.sh binary    # 3 seeds binario
#     bash scripts/train_deep_ensemble.sh multiclass  # 3 seeds multiclase
#
# Coste: 3× el tiempo de un training completo. Ganancia esperada: +0.2-0.3%
# AUC sobre el modelo single-seed. Los 3 checkpoints ocupan ~150-300 MB
# total dependiendo del backbone.
#
# `train.py` acepta `--seed` como argumento de línea de comandos y lo propaga
# a `seed_everything(seed)`. Este script pasa los 3 seeds vía flag explícito.
set -e

MODE="${1:-binary}"
if [[ "$MODE" != "binary" && "$MODE" != "multiclass" ]]; then
    echo "Uso: $0 {binary|multiclass}"
    exit 1
fi

SEEDS=(42 101 202)
CHECKPOINT_DIR="models/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# validar que los seeds son únicos y enteros no-negativos ANTES de
# entrenar. Un ensemble con seeds duplicados tiene diversidad cero — los 3
# checkpoints serían bit-a-bit idénticos y el usuario obtendría 3x latencia
# en inferencia con +0% AUC, sin aviso. También validamos que cada seed sea
# un entero no-negativo para evitar que un typo tipo `SEEDS=(42 "oops" 202)`
# pase a `--seed oops` y produzca un crash críptico en argparse de train.py.
declare -A _SEEN_SEEDS
for seed in "${SEEDS[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "ERROR: seed '$seed' no es un entero no-negativo." >&2
        echo "  SEEDS=(${SEEDS[*]})" >&2
        exit 1
    fi
    if [[ -n "${_SEEN_SEEDS[$seed]:-}" ]]; then
        echo "ERROR: seed $seed duplicado en SEEDS=(${SEEDS[*]})." >&2
        echo "  Un deep ensemble requiere seeds distintos para generar" >&2
        echo "  diversidad (Lakshminarayanan 2017). Con seeds iguales los" >&2
        echo "  checkpoints son idénticos bit-a-bit → +0% AUC, 3x latencia." >&2
        exit 1
    fi
    _SEEN_SEEDS[$seed]=1
done
unset _SEEN_SEEDS

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    idx=$((i + 1))
    echo "============================================================"
    echo "Deep Ensemble $idx/3 — seed=$seed — modo=$MODE"
    echo "============================================================"

    python src/training/train.py --mode "$MODE" \
        --epochs 20 --batch_size 32 --accum_steps 2 --seed "$seed"

    # Renombrar el checkpoint para preservarlo frente al siguiente run.
    mv "${CHECKPOINT_DIR}/best_${MODE}_model.pth" \
       "${CHECKPOINT_DIR}/best_${MODE}_model_seed${seed}.pth"
    mv "${CHECKPOINT_DIR}/best_${MODE}_model_SWA.pth" \
       "${CHECKPOINT_DIR}/best_${MODE}_model_seed${seed}_SWA.pth"
    mv "${CHECKPOINT_DIR}/best_${MODE}_model_EMA.pth" \
       "${CHECKPOINT_DIR}/best_${MODE}_model_seed${seed}_EMA.pth"
done

echo "============================================================"
echo "Deep ensemble completo. Integrar en ClinicalInferenceEngine:"
echo ""
echo "engine = ClinicalInferenceEngine("
echo "    ${MODE}_model_path='${CHECKPOINT_DIR}/best_${MODE}_model_seed42.pth',"
echo "    ${MODE}_ensemble_paths=["
echo "        '${CHECKPOINT_DIR}/best_${MODE}_model_seed101.pth',"
echo "        '${CHECKPOINT_DIR}/best_${MODE}_model_seed202.pth',"
echo "    ],"
echo ")"
echo "============================================================"
