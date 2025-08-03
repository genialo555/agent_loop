#!/bin/bash
# Script d'entraînement optimisé pour RTX 3090 - 1 EPOCH COMPLÈTE
# Configuration spécifique pour Jerem - Gemma-3N-E4B avec Unsloth

echo "🚀 ENTRAÎNEMENT OPTIMISÉ GEMMA-3N-E4B - 1 EPOCH"
echo "================================================"

# Variables d'environnement critiques
export PYTHONPATH=/home/jerem/agent_loop:$PYTHONPATH
export HF_TOKEN=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

# Utiliser les caches SSD pour tout
export HF_HOME=/media/jerem/641C8D6C1C8D3A56/hf_cache
export TRANSFORMERS_CACHE=/media/jerem/641C8D6C1C8D3A56/hf_cache/transformers
export HF_DATASETS_CACHE=/media/jerem/641C8D6C1C8D3A56/hf_cache/datasets
export TORCH_HOME=/media/jerem/641C8D6C1C8D3A56/hf_cache/torch

# Optimisations GPU RTX 3090
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Activer l'environnement virtuel
source /home/jerem/agent_loop/.venv/bin/activate

# Chemins
MODEL_PATH="/media/jerem/641C8D6C1C8D3A56/MLLMODELS/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee"
DATASET_PATH="/media/jerem/jeux&travail/datasets/agent_instruct/data"
OUTPUT_DIR="/home/jerem/agent_loop/models/results/gemma3n-optimized-1epoch-$(date +%Y%m%d_%H%M%S)"

# Créer le dossier de sortie
mkdir -p "$OUTPUT_DIR"

echo "📊 Configuration:"
echo "  • GPU: RTX 3090 (24GB)"
echo "  • Model: Gemma-3N-E4B (4.5B params)"
echo "  • Dataset: Agent Instruct (1.1M examples)"
echo "  • Epochs: 1 complète (~140k steps)"
echo "  • Output: $OUTPUT_DIR"
echo ""
echo "⚠️  Temps estimé: ~226 heures (9.4 jours)"
echo "💡  Conseil: Lancer dans tmux ou screen!"
echo ""
read -p "🔥 Prêt à lancer? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Annulé"
    exit 1
fi

# Lancer l'entraînement avec configuration optimisée
python /home/jerem/agent_loop/models/training/qlora/qlora_finetune_unsloth.py \
    --model-path "$MODEL_PATH" \
    --data "$DATASET_PATH" \
    --text-column "messages" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs 1 \
    --max-seq-length 2048 \
    --batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --lora-r 64 \
    --lora-alpha 128 \
    --seed 42 \
    --use-wandb \
    2>&1 | tee "$OUTPUT_DIR/training.log"

# Sauvegarder l'état final
if [ $? -eq 0 ]; then
    echo "✅ Entraînement terminé avec succès!"
    echo "📁 Modèle sauvegardé dans: $OUTPUT_DIR"
    
    # Statistiques finales
    echo ""
    echo "📊 Statistiques:"
    du -sh "$OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
    
    # Notification (optionnel)
    # notify-send "Entraînement Gemma-3N terminé!" "1 epoch complétée"
else
    echo "❌ Erreur pendant l'entraînement!"
    echo "📝 Vérifier les logs: $OUTPUT_DIR/training.log"
fi