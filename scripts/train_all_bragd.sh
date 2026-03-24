#!/bin/bash

# Training script for all models on BRAGD data (10-fold CV)
# Models: TnT, singlelabel, multilabel (constrained),
#         multilabel (unconstrained-unnormalized), multilabel (unconstrained-normalized)
# Output directory: all_splits_eval_bragd_adamw
# Logs saved per model per fold in: ${OUTPUT_DIR}/logs/

OUTPUT_DIR="results/all_splits_eval_bragd_adamw"
LOG_DIR="${OUTPUT_DIR}/logs"
LEARNING_RATE=2e-5
BATCH_SIZE=8

mkdir -p "${LOG_DIR}"

echo "========================================================================"
echo "BRAGD Training: 5 models x 10 folds (one at a time)"
echo "Optimizer: AdamW | LR: ${LEARNING_RATE} | Batch size: ${BATCH_SIZE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Logs directory:   ${LOG_DIR}"
echo "Started: $(date)"
echo "========================================================================"
echo ""

run_model() {
    local NAME="$1"
    local FOLD="$2"
    shift 2
    local LOG="${LOG_DIR}/${NAME}_fold${FOLD}.log"

    echo "--- ${NAME} (fold ${FOLD}) --- [$(date '+%H:%M:%S')]"
    python POS_tagger.py "$@" 2>&1 | tee "${LOG}"
    local EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: ${NAME} failed for fold ${FOLD} (exit code ${EXIT_CODE})"
        echo "Log: ${LOG}"
        exit 1
    fi

    # Delete model checkpoint after OOD eval is done (results JSON is kept)
    local CKPT_DIR
    CKPT_DIR=$(grep -oP 'Saved to: \K\S*best_model\.pth' "${LOG}" | tail -1)
    if [ -n "${CKPT_DIR}" ] && [ -f "${CKPT_DIR}" ]; then
        rm -f "${CKPT_DIR}"
        echo "  [cleanup] Removed ${CKPT_DIR}"
    fi

    echo ""
}

for FOLD in {0..9}; do
    echo "========================================================================"
    echo "FOLD ${FOLD}  [$(date)]"
    echo "========================================================================"
    echo ""

    # --- TnT (fast, no GPU) ---
    run_model "tnt" ${FOLD} \
        --mode singlelabel \
        --model_type tnt \
        --fold ${FOLD} \
        --output_dir ${OUTPUT_DIR} \
        --evaluate_ood

    # --- Singlelabel ---
    run_model "singlelabel" ${FOLD} \
        --mode singlelabel \
        --fold ${FOLD} \
        --optimizer adamw \
        --learning_rate ${LEARNING_RATE} \
        --batch_size ${BATCH_SIZE} \
        --output_dir ${OUTPUT_DIR} \
        --evaluate_ood

    # --- Multilabel (constrained) ---
    run_model "multilabel" ${FOLD} \
        --mode multilabel \
        --fold ${FOLD} \
        --optimizer adamw \
        --learning_rate ${LEARNING_RATE} \
        --batch_size ${BATCH_SIZE} \
        --output_dir ${OUTPUT_DIR} \
        --evaluate_ood

    # --- Multilabel (unconstrained, unnormalized) ---
    run_model "multilabel_unconstrained_unnormalized" ${FOLD} \
        --mode multilabel \
        --unconstrained_loss unnormalized \
        --fold ${FOLD} \
        --optimizer adamw \
        --learning_rate ${LEARNING_RATE} \
        --batch_size ${BATCH_SIZE} \
        --output_dir ${OUTPUT_DIR} \
        --evaluate_ood

    # --- Multilabel (unconstrained, normalized) ---
    run_model "multilabel_unconstrained_normalized" ${FOLD} \
        --mode multilabel \
        --unconstrained_loss normalized \
        --fold ${FOLD} \
        --optimizer adamw \
        --learning_rate ${LEARNING_RATE} \
        --batch_size ${BATCH_SIZE} \
        --output_dir ${OUTPUT_DIR} \
        --evaluate_ood

done

echo "========================================================================"
echo "All training and OOD evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs saved to:    ${LOG_DIR}"
echo "Finished: $(date)"
echo "========================================================================"
