#!/bin/bash
# Script de TEST - 1000 steps pour validation rapide
# RTX 3090 optimisé - Gemma-3N-E4B

echo "🧪 TEST D'ENTRAÎNEMENT - 1000 STEPS"
echo "===================================="

# Variables d'environnement
export PYTHONPATH=/home/jerem/agent_loop:$PYTHONPATH
export HF_TOKEN=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
export HF_HOME=/media/jerem/641C8D6C1C8D3A56/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activer venv
source /home/jerem/agent_loop/.venv/bin/activate

# Configuration
MODEL_PATH="/media/jerem/641C8D6C1C8D3A56/MLLMODELS/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee"
DATASET_PATH="/media/jerem/jeux&travail/datasets/agent_instruct/data"
OUTPUT_DIR="/home/jerem/agent_loop/models/results/gemma3n-test-1000steps-$(date +%Y%m%d_%H%M%S)"

echo "⚡ Configuration rapide:"
echo "  • Steps: 1000 (~1.6 heures)"
echo "  • Batch: 2, Accum: 4 (effective: 8)"
echo "  • Learning rate: 2e-4"
echo ""

# Lancer le test
python /home/jerem/agent_loop/models/training/qlora/qlora_finetune_unsloth.py \
    --model-path "$MODEL_PATH" \
    --data "$DATASET_PATH" \
    --text-column "messages" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 1000 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --lora-r 32 \
    --lora-alpha 64 \
    --seed 42

echo "✅ Test terminé! Résultats dans: $OUTPUT_DIR"