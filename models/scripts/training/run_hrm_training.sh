#!/bin/bash
# Run HRM (Hierarchical Reasoning Model) training with Unsloth

# Set environment variables
export PYTHONPATH=/home/jerem/agent_loop:$PYTHONPATH
export HF_TOKEN=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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate virtual environment
source /home/jerem/agent_loop/.venv/bin/activate

# Model path - UTILISER LE SSD !
MODEL_PATH="/media/jerem/641C8D6C1C8D3A56/MLLMODELS/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee"
OUTPUT_DIR="/home/jerem/agent_loop/models/results/gemma-3n-hrm-test-$(date +%Y%m%d_%H%M%S)"

echo "🧠 Starting HRM training with GSM8K dataset"
echo "Model: Gemma-3N-E4B"
echo "Dataset: GSM8K (Hierarchical Reasoning)"
echo "Output: $OUTPUT_DIR"
echo "Epochs: 1 COMPLÈTE (~7,473 examples)"
echo ""
echo "⚠️  Temps estimé: ~3-4 heures"
echo ""

# Run training with Unsloth on GSM8K for hierarchical reasoning - 1 EPOCH COMPLÈTE
python /home/jerem/agent_loop/models/training/qlora/qlora_finetune_unsloth.py \
    --model-path "$MODEL_PATH" \
    --data "gsm8k" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 1 \
    --learning-rate 2e-4 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --text-column "question" \
    --lora-r 64 \
    --lora-alpha 128

echo "✅ HRM training test completed!"
echo "📁 Results saved to: $OUTPUT_DIR"

# Check if model was saved
if [ -d "$OUTPUT_DIR" ]; then
    echo "📊 Training artifacts:"
    ls -la "$OUTPUT_DIR"
fi