#!/bin/bash

# Train a single release model on ALL data (in-domain + OOD)
# for uploading to HuggingFace.
#
# Uses fixed 20 epochs (no early stopping). This is justified by
# 10-fold CV showing convergence at mean=11.5 epochs (max=18).
#
# Usage: bash scripts/train_release_model.sh

set -euo pipefail

HF_DIR="release_model"
FIXED_EPOCHS=20
LEARNING_RATE=2e-5
BATCH_SIZE=8

echo "========================================================================"
echo "Training release model (all data, ${FIXED_EPOCHS} epochs)"
echo "Output: ${HF_DIR}"
echo "Started: $(date)"
echo "========================================================================"

python POS_tagger.py \
    --mode multilabel \
    --fold 0 \
    --optimizer adamw \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --output_dir ${HF_DIR} \
    --full_train \
    --include_ood \
    --fixed_epochs ${FIXED_EPOCHS} \
    --save_huggingface ${HF_DIR}/huggingface

echo ""
echo "========================================================================"
echo "Done! Model saved to: ${HF_DIR}/huggingface"
echo "Upload with: huggingface-cli upload Setur/BRAGD ${HF_DIR}/huggingface"
echo "Finished: $(date)"
echo "========================================================================"
