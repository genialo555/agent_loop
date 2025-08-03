#!/bin/bash
# Run Unsloth training for Gemma-3N-E4B model

# Set environment variables
export PYTHONPATH=/home/jerem/agent_loop:$PYTHONPATH
export HF_TOKEN=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

# Model and dataset paths
MODEL_PATH="/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee"
DATASET_PATH="/media/jerem/jeux&travail/datasets/agent_instruct/data"
OUTPUT_DIR="./results/unsloth_gemma3n_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Starting Unsloth training for Gemma-3N-E4B"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"

# Run training with Unsloth (UPDATED AFTER REFACTORING)
python /home/jerem/agent_loop/models/training/qlora/qlora_finetune_unsloth.py \
    --model-config gemma-3n \
    --model-path "$MODEL_PATH" \
    --data "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 1000 \
    --learning-rate 2e-4 \
    --batch-size 2 \
    --wandb-project "gemma3n-agent-instruct" \
    --run-name "unsloth-gemma3n-$(date +%Y%m%d_%H%M%S)" \
    --hf-token "$HF_TOKEN" \
    --export-gguf \
    --quantization-method q4_k_m

echo "✅ Training completed!"