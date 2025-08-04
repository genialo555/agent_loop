# Commandes d'Entraînement Gemma-3N-E4B

## Configuration des chemins
- **Cache HuggingFace** : `/media/jerem/641C8D6C1C8D3A56/hf_cache`
- **Modèles** : `/media/jerem/jeux&travail/ml_models/`
- **Datasets** : `/media/jerem/jeux&travail/datasets/agent_instruct/data`
- **Token HF** : `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`

## Entraînement complet (100 steps)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
source .env && source venv/bin/activate && python training/qlora_finetune.py \
  --model-config custom \
  --model-name "/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee" \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --text-column messages \
  --max-steps 100 \
  --output-dir ./results/gemma-3n-100steps \
  --batch-size 1 \
  --no-wandb
```

## Test rapide (10 steps, dataset réduit)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
sou1rce .env && source venv/bin/activate && python training/qlora_finetune.py \
  --model-config custom \
  --model-name "/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee" \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data/code_-00000-of-00002.parquet" \
  --text-column messages \
  --max-steps 10 \
  --output-dir ./results/gemma-3n-test \
  --batch-size 1 \
  --no-wandb
```

## Avec Weights & Biases
```bash
source .env && source venv/bin/activate && python training/qlora_finetune.py \
  --model-config custom \
  --model-name "/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee" \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --max-steps 100 \
  --output-dir ./results/gemma-3n-100steps \
  --wandb-project gemma-3n-agent-loop \
  --run-name "gemma-3n-100steps"
```

## Surveillance de l'entraînement
```bash
# Vérifier l'utilisation GPU
nvidia-smi

# Suivre les logs en temps réel
tail -f ./results/gemma-3n-100steps/trainer_state.json

# Vérifier le processus
ps aux | grep python | grep qlora
```

## Notes importantes
- Le dataset contient 1.1M d'exemples, la préparation prend ~10-15 minutes
- Utilisation mémoire GPU : ~20GB avec batch_size=2
- Temps estimé pour 100 steps : ~30-45 minutes après préparation du dataset

## Entraînement Unsloth sans limite de temps (terminal)
```bash
# À lancer dans un terminal séparé sans timeout
source .env && source venv/bin/activate && python training/qlora_finetune_unsloth.py \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --max-steps 100 \
  --output-dir ./results/gemma-3n-unsloth-100steps-fixed
```

## Entraînement Unsloth 2 EPOCHS complets
```bash
# 2 epochs = ~140,000 steps (va prendre plusieurs heures)
source .env && source venv/bin/activate && python training/qlora_finetune_unsloth.py \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --num-epochs 2 \
  --output-dir ./results/gemma-3n-unsloth-2epochs
```

## Test rapide optimisé (1000 steps) - NOUVEAU 01/08/2025
```bash
# Test en ~1.6 heures avec config optimisée RTX 3090
./models/scripts/train_test_1000steps.sh
```

## Entraînement COMPLET 1 EPOCH optimisé - NOUVEAU 01/08/2025
```bash
# 1 epoch complète = ~140k steps = ~226 heures (9.4 jours)
# IMPORTANT: Utiliser tmux ou screen!
tmux new -s training
./models/scripts/train_optimized_1epoch.sh
```

## Script HRM (Hierarchical Reasoning) - GSM8K
```bash
# Entraînement sur GSM8K pour pensée hiérarchique - 1 EPOCH COMPLÈTE
# UTILISE LE BON CHEMIN SSD MAINTENANT !
# Temps estimé: ~3-4 heures pour 7,473 examples
./models/scripts/run_hrm_training.sh
```

## Fusion simple LoRA + Modèle de base - NOUVEAU 01/08/2025
```bash
# Fusionner les poids LoRA HRM avec le modèle de base Gemma-3N
python models/scripts/merge_lora_simple.py

# Le modèle fusionné sera sauvegardé dans:
# /media/jerem/641C8D6C1C8D3A56/MLLMODELS/gemma-3n-hrm-merged/
```

## Conversion des poids LoRA vers Ollama - NOUVEAU 01/08/2025
```bash
# Fusionner LoRA + convertir en GGUF + importer dans Ollama
python models/scripts/merge_and_convert_lora.py \
  --lora-path /home/jerem/agent_loop/models/results/gemma-3n-hrm-test-20250801_015252 \
  --output-name gemma-3n-hrm

# Tester le modèle après import
ollama run gemma-3n-hrm "Explain how to solve 25 + 17 step by step"
```

## Entraînement HRM (Hierarchical Reasoning Model) avec Architecture Complète - NOUVEAU 04/08/2025
```bash
# HRM avec modules hiérarchiques H/L, convergence, gradient O(1) et supervision profonde
# Configurations disponibles: gsm8k, code, agent, full, debug

# 1. GSM8K - Raisonnement mathématique (2 epochs)
python models/training/hrm/hrm_trainer.py \
  --config gsm8k \
  --num-epochs 2 \
  --output-dir ./models/results/gemma-3n-hrm-gsm8k

# 2. Code Generation - Python/SQL (2 epochs)  
python models/training/hrm/hrm_trainer.py \
  --config code \
  --num-epochs 2 \
  --output-dir ./models/results/gemma-3n-hrm-code

# 3. Agent Linux - Navigation OS (2 epochs)
python models/training/hrm/hrm_trainer.py \
  --config agent \
  --num-epochs 2 \
  --output-dir ./models/results/gemma-3n-hrm-agent

# 4. Configuration complète production (2 epochs)
python models/training/hrm/hrm_trainer.py \
  --config full \
  --num-epochs 2 \
  --batch-size 1 \
  --learning-rate 2e-4

# 5. Test rapide debug (100 steps)
python models/training/hrm/hrm_trainer.py \
  --config debug \
  --output-dir ./models/results/gemma-3n-hrm-debug

# NOTES IMPORTANTES HRM:
# - Modules H/L avec LoRA ranks différents (L=32, H=64)
# - Convergence hiérarchique: L converge en ~8-16 steps, H update après
# - Gradient O(1) mémoire au lieu de O(T) - économise ~10GB sur RTX 3090
# - Deep supervision avec 3 segments par défaut
# - ACT (Adaptive Computation Time) activé pour efficacité
# - Sortie en 3 formats: LoRA, 16bit merged, GGUF quantized
```