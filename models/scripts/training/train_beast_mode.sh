#!/bin/bash
echo "🔥 BEAST MODE TRAINING - Ryzen 9 + RTX 3090 POWER! 🔥"

# Set CPU performance mode
sudo cpupower frequency-set -g performance 2>/dev/null || echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Optimize system
sudo sysctl -w vm.swappiness=1
sudo sync && sudo sysctl -w vm.drop_caches=3

# GPU max performance
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 370  # Max power for RTX 3090

# Environment setup
source .env
source venv/bin/activate

# Set multi-threading optimizations
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

echo "Starting optimized training with:"
echo "- Batch size: 2 (8x with gradient accumulation)"
echo "- 16 CPU cores for preprocessing"
echo "- 8 DataLoader workers"
echo "- GPU at max performance"

# Run training with optimizations
python training/qlora_finetune_unsloth.py \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --num-epochs 1 \
  --batch-size 2 \
  --output-dir ./results/gemma-3n-beast-mode-1epoch