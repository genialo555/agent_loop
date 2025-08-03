#!/bin/bash
echo "🛡️ TRAINING SÉCURISÉ ET OPTIMISÉ - 1 EPOCH 🛡️"

# Environment
source .env
source venv/bin/activate

# Nettoie les processus zombies
pkill -f "python.*qlora" || true
sleep 2

# ========== OPTIMISATIONS SYSTÈME ==========

# 1. CPU Performance (validé)
sudo cpupower frequency-set -g performance 2>/dev/null || true

# 2. RAM - NE PAS drop tous les caches pendant le training !
# Juste nettoyer les caches inutiles
sync  # Flush buffers to disk first
echo 1 | sudo tee /proc/sys/vm/drop_caches  # Free pagecache only, pas les inodes

# 3. Swappiness - 10 est optimal pour 64GB RAM
sudo sysctl -w vm.swappiness=10

# 4. AUGMENTER LES CACHES SYSTÈME
echo "📈 Augmentation des caches système..."
# Garder 20GB minimum pour le cache
echo 20000000 | sudo tee /proc/sys/vm/min_free_kbytes >/dev/null
# Augmenter le readahead pour SSD
echo 8192 | sudo tee /sys/block/nvme0n1/queue/read_ahead_kb >/dev/null 2>&1
echo 8192 | sudo tee /sys/block/nvme1n1/queue/read_ahead_kb >/dev/null 2>&1
# Désactiver les temps d'accès
sudo mount -o remount,noatime /media/jerem/jeux\&travail 2>/dev/null || true
# Précharger le dataset en RAM (en background)
echo "💾 Préchargement du dataset en cache RAM..."
(find /media/jerem/jeux\&travail/datasets/agent_instruct/data -name "*.parquet" -exec cat {} \; > /dev/null 2>&1) &
PRELOAD_PID=$!

# 5. GPU - Reset et config
sudo nvidia-smi -r || true
sleep 3
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 350  # 350W pour éviter thermal throttling

# ========== OPTIMISATIONS PYTORCH ==========

# Configuration CUDA validée pour RTX 3090
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"

# Utiliser jemalloc si disponible (meilleure gestion mémoire)
if [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so:$LD_PRELOAD
fi

# Variables d'environnement sûres
export OMP_NUM_THREADS=8  # Pas tous les cores, évite contention
export MKL_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0  # Async GPU ops
export TOKENIZERS_PARALLELISM=false  # Évite warnings avec multiprocessing

# ========== MONITORING ==========
echo ""
echo "Configuration finale:"
echo "- PYTORCH_CUDA_ALLOC_CONF: max_split_size_mb:512,gc_threshold:0.8"
echo "- Swappiness: 10 (optimal pour 64GB)"
echo "- GPU Power: 350W (safe)"
echo "- CPU threads: 8 (évite contention)"
echo "- Batch size: 2 (testé sans OOM)"
echo ""

# Lance le training
python training/qlora_finetune_unsloth.py \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --num-epochs 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --output-dir ./results/gemma-3n-safe-1epoch \
  --learning-rate 2e-4

# Attendre un peu que le cache se charge
echo "⏳ Attente du préchargement (10s)..."
sleep 10
kill $PRELOAD_PID 2>/dev/null || true

echo ""
echo "✅ Training lancé de manière sécurisée avec cache optimisé !"
echo "📊 Monitor: watch -n 30 'nvidia-smi; free -h'"
echo "💾 Cache: cat /proc/meminfo | grep Cached"