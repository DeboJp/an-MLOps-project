#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Configurable Python interpreter and model settings
PYTHON_EXEC=${PYTHON_EXEC:-".venv/bin/python"}
MODEL_NAME="mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTER_PATH="./adapters"

echo "=========================================================="
# 1. Prepare MLX Dataset
# ==========================================================
echo "[1/3] Preparing Gold + Noisy Padding Training Dataset..."
$PYTHON_EXEC prepare_mlx_data.py

echo "=========================================================="
# 2. Run LoRA Fine-Tuning
# ==========================================================
echo "[2/3] Launching MLX GPU-Accelerated LoRA Training..."
$PYTHON_EXEC -m mlx_lm.lora \
  --model "$MODEL_NAME" \
  --train \
  --data ./mlx_data \
  --iters 600 \
  --steps-per-eval 50 \
  --val-batches 10 \
  --learning-rate 1e-5 \
  --lora-layers 16 \
  --save-every 100 \
  --adapter-path "$ADAPTER_PATH"
