# Sprint 2 Quick Start - QLoRA Fine-tuning avec Docker

Guide de démarrage rapide pour l'architecture Docker QLoRA du projet Agent Loop.

## 🚀 Démarrage Rapide (5 minutes)

### 1. Vérification des Prérequis
```bash
# Vérifier GPU NVIDIA disponible
nvidia-smi

# Vérifier Docker avec support GPU  
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Vérifier Docker Compose v2
docker compose version
```

### 2. Test de l'Architecture
```bash
# Exécuter suite de tests complète
python scripts/test_docker_training.py

# Résultat attendu: All tests passed! ✅
```

### 3. Lancement Training Développement
```bash
# Environnement interactif complet  
make train-dev

# Accès interfaces:
# 📚 Jupyter: http://localhost:8888 (token: agent-dev-2024)
# 📊 TensorBoard: http://localhost:6006
# 🔍 W&B Monitor: http://localhost:8080
```

### 4. Premier Fine-tuning QLoRA
```bash
# Training Gemma 2B avec données exemple
make train-gemma-2b

# Monitoring en temps réel
make train-status
make train-logs
```

---

## 📋 Commandes Essentielles

### Développement
```bash
# Environnement développement complet
make train-dev                    # Jupyter + TensorBoard + W&B

# Status et monitoring
make train-status                 # Status containers + GPU
make train-logs                   # Logs training temps réel  

# Gestion conteneurs
make train-stop                   # Arrêt propre
make train-clean                  # Nettoyage complet (⚠️ supprime checkpoints)
```

### Training Production
```bash
# Modèles prédéfinis optimisés
make train-gemma-2b              # Gemma 2B - RTX 4090 optimisé
make train-gemma-9b              # Gemma 9B - mémoire réduite

# Training custom
make train-custom MODEL=google/gemma-2-2b DATA=/path/to/data

# Monitoring complet (training + dev + monitoring)
make train-monitor
```

### Configuration Manuelle
```bash
# Docker Compose direct
docker compose -f docker-compose.training.yml --profile training up

# Training interactif  
docker compose -f docker-compose.training.yml --profile training-dev up -d

# Commande training custom
docker compose -f docker-compose.training.yml run --rm qlora-training \
  python training/qlora_finetune.py \
  --model-config gemma-2b \
  --data /app/datasets/processed \
  --output-dir /app/outputs/custom-run
```

---

## 🏗️ Architecture Overview

### Dockerfile Multi-Stage
- **Stage training**: NVIDIA CUDA 12.6 + PyTorch GPU + QLoRA stack
- **Optimisations**: BuildKit cache, non-root user, memory management
- **Size**: ~8GB (optimisé avec BuildKit cache)

### Docker Compose Training
- **qlora-training**: Service principal GPU avec ressources limitées
- **training-dev**: Jupyter Lab + TensorBoard pour développement
- **training-monitor**: W&B Local + métriques GPU temps réel
- **gpu-monitor**: Surveillance GPU avec logs JSON
- **training-backup**: Backup automatique checkpoints (toutes les heures)

### Volumes Optimisés
```
├── training_models_cache/          # Cache HuggingFace (~2-5GB)
├── training_models_gguf/           # Modèles GGUF (~4-8GB)  
├── training_checkpoints/           # Checkpoints training (~1-3GB)
├── training_logs/                  # Logs structurés + métriques GPU
└── training_backup/                # Backup automatique sécurisé
```

---

## ⚙️ Configuration QLoRA

### RTX 4090 Optimisée (24GB VRAM)
```python
# training/qlora_config.py - GEMMA_2B_CONFIG
per_device_train_batch_size: 4      # Batch optimal RTX 4090
gradient_accumulation_steps: 4      # Effective batch = 16
bf16: True                          # Meilleur que fp16 sur Ampere
gradient_checkpointing: True        # Optimisation mémoire
optim: "adamw_8bit"                # Optimizer 8-bit

# QLoRA 4-bit + Double Quantization
load_in_4bit: True
bnb_4bit_use_double_quant: True    # Économie mémoire supplémentaire  
bnb_4bit_quant_type: "nf4"         # Normal Float 4-bit optimal
lora_r: 32                         # Rank augmenté pour performance
target_modules: "all-linear"       # Cible tous les layers linéaires
```

### Configurations Prédéfinies
- **Gemma 2B**: Optimisé RTX 4090 (batch_size=4, 1000 steps)
- **Gemma 9B**: Mémoire réduite (batch_size=2, 800 steps)  
- **Custom**: Support modèles HuggingFace compatibles

---

## 📊 Monitoring et Observabilité

### Interfaces Web
- **Jupyter Lab**: http://localhost:8888 (token: agent-dev-2024)
- **TensorBoard**: http://localhost:6006 (métriques training)
- **W&B Local**: http://localhost:8080 (experiments tracking)

### Monitoring GPU en Temps Réel
```bash
# Métriques GPU structurées (JSON)
tail -f logs/gpu_metrics.jsonl

# nvidia-smi classique
watch -n 1 nvidia-smi

# Status containers
docker stats agent-qlora-training
```

### Logs Structurés
```bash
# Logs training principal
docker logs -f agent-qlora-training

# Logs GPU monitor  
docker logs -f agent-gpu-monitor

# Logs backup automatique
docker logs agent-training-backup
```

---

## 🔧 Troubleshooting Rapide

### GPU Non Détecté
```bash
# Vérifier drivers NVIDIA
nvidia-smi

# Vérifier Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Installer nvidia-container-toolkit si nécessaire
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of Memory (OOM)
1. **Réduire batch size**: Modifier `per_device_train_batch_size` dans `qlora_config.py`
2. **Utiliser Gemma 2B**: Plus petit que Gemma 9B
3. **Vérifier 4-bit quantization**: `load_in_4bit: True`
4. **Gradient checkpointing**: Déjà activé par défaut

### Build Docker Lent
```bash
# Builds subséquents utilisent le cache BuildKit
export DOCKER_BUILDKIT=1

# Build avec cache inline
docker build --cache-from agent-loop:latest .

# Nettoyage si cache corrompu
docker builder prune
```

### Containers Ne Démarrent Pas
```bash
# Vérifier configuration
docker compose -f docker-compose.training.yml config

# Logs détaillés
docker compose -f docker-compose.training.yml logs

# Reset complet
make train-clean
docker system prune -f
```

---

## 📚 Documentation Complète

- **Architecture**: `README_DOCKER.md` (section QLoRA Fine-tuning)
- **Configuration**: `training/qlora_config.py` (paramètres détaillés)  
- **Pipeline**: `training/qlora_finetune.py` (code training)
- **Tests**: `scripts/test_docker_training.py` (validation architecture)

---

## 🎯 Prochaines Étapes

1. **Tester architecture**: `python scripts/test_docker_training.py`
2. **Préparer données**: Placer datasets dans `datasets/processed/`
3. **Télécharger modèle base**: Voir `models/README.md`
4. **Lancer premier training**: `make train-gemma-2b`
5. **Monitoring**: Utiliser interfaces web pour suivre progression

### Training Custom
```bash
# Exemple avec vos données
make train-custom \
  MODEL=google/gemma-2-2b \
  DATA=/path/to/your/training/data
```

### Intégration CI/CD
L'architecture est prête pour intégration dans pipeline MLOps avec tests automatisés et déployment containerisé.

---

**🏁 Sprint 2 Ready!** Architecture Docker QLoRA opérationnelle avec GPU RTX 4090, monitoring avancé et bonnes pratiques ML/DevOps.