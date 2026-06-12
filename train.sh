#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e
set -x 

# Configurable Python interpreter and model settings
PYTHON_EXEC=${PYTHON_EXEC:-".venv/bin/python"}
MODEL_NAME="mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTER_PATH="./adapters"

echo "=========================================================="
# 1. Prepare MLX Dataset
# ==========================================================
echo "[1/3] Preparing Gold + Noisy Padding Training Dataset..."
$PYTHON_EXEC prepare_mlx_data.py --train-limit 3000

echo "=========================================================="
# 2. Run LoRA Fine-Tuning
# ==========================================================
echo "[2/3] Launching MLX GPU-Accelerated LoRA Training..."
$PYTHON_EXEC -m mlx_lm lora \
  --model "$MODEL_NAME" \
  --train \
  --data ./mlx_data \
  --iters 2500 \
  --batch-size 1 \
  --grad-accumulation-steps 4 \
  --grad-checkpoint \
  --max-seq-length 512 \
  --steps-per-eval 500 \
  --val-batches 2 \
  --learning-rate 1e-5 \
  --save-every 500 \
  --adapter-path "$ADAPTER_PATH"

echo "=========================================================="
# 3. Test generation examlple with fine-tuned adapter
# ==========================================================
echo "[3/3] Running test generation to verify fine-tuning adapters..."

# Aligned system prompt matching exactly what the model was trained on
read -r -d '' SYSTEM_PROMPT << 'EOF' || true
You are a financial calculator assistant. Your task is to output a mathematical program using the FinQA Domain-Specific Language (DSL) to answer the user's question based on the provided text and table.

Allowed operations:
- add(arg1, arg2)
- subtract(arg1, arg2)
- multiply(arg1, arg2)
- divide(arg1, arg2)
- exp(arg1, arg2)
- greater(arg1, arg2)
- table_max(row_name)
- table_min(row_name)
- table_sum(row_name)
- table_average(row_name)

Refer to the results of previous steps using #0, #1, etc.
Format your output exactly as a comma-separated sequence of operations, ending with EOF. Do not include any other explanations, markdown formatting, or conversational filler.

Example 1:
Context:
Table:
| Year | Revenue |
| 2020 | 50000 |
| 2021 | 65000 |
Question: What was the rate of increase of revenue from 2020 to 2021?
Output: subtract(65000, 50000), divide(#0, 50000), EOF

CRITICAL: You must extract the exact literal numbers from the table. Do not use placeholders. Each problem will be unique, and require basic arithmetic decision making. Count digits carefully.
EOF

read -r -d '' USER_PROMPT << 'EOF' || true
Context:
Columns:  | amount ( in millions ) -> Row: 2015 net revenue | $ 5829
Columns:  | amount ( in millions ) -> Row: 2014 net revenue | $ 5735

Question:
what is the net change in net revenue during 2015 for entergy corporation?
EOF

FULL_PROMPT="<|start_header_id|>system<|end_header_id|>

$SYSTEM_PROMPT<|eot_id|><|start_header_id|>user<|end_header_id|>

$USER_PROMPT<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"

$PYTHON_EXEC -m mlx_lm generate \
  --model "$MODEL_NAME" \
  --adapter-path "$ADAPTER_PATH" \
  --prompt "$FULL_PROMPT" \
  --max-tokens 50

