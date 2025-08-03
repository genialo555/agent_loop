#!/bin/bash
# Script de TEST avec GSM8K - Dataset disponible
# Alternative temporaire pendant que jeux&travail est démonté

echo "🧪 TEST D'ENTRAÎNEMENT - GSM8K Dataset"
echo "====================================="

# Variables d'environnement
export PYTHONPATH=/home/jerem/agent_loop:$PYTHONPATH
export HF_TOKEN=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
export HF_HOME=/media/jerem/641C8D6C1C8D3A56/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activer venv
source /home/jerem/agent_loop/.venv/bin/activate

# Configuration
MODEL_PATH="/media/jerem/641C8D6C1C8D3A56/MLLMODELS/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee"
DATASET_NAME="gsm8k"  # Utiliser le nom HuggingFace directement
OUTPUT_DIR="/home/jerem/agent_loop/models/results/gemma3n-gsm8k-test-$(date +%Y%m%d_%H%M%S)"

echo "⚡ Configuration:"
echo "  • Dataset: GSM8K (HuggingFace)"
echo "  • Model: Gemma-3N-E4B"
echo "  • Steps: 100 (test rapide)"
echo ""
echo "⚠️  NOTE: Utilise GSM8K car le drive jeux&travail n'est pas monté"
echo ""

# Lancer le test
python /home/jerem/agent_loop/models/training/qlora/qlora_finetune_unsloth.py \
    --model-path "$MODEL_PATH" \
    --data "$DATASET_NAME" \
    --text-column "question" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps 100 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --lora-r 32 \
    --lora-alpha 64 \
    --seed 42

echo "✅ Test terminé! Résultats dans: $OUTPUT_DIR"