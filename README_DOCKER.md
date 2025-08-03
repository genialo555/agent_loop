# Docker Configuration for Agent Loop

Cette configuration Docker suit les meilleures pratiques pour déployer l'application FastAPI Agent Loop.

## Architecture

### Dockerfile Multi-Stage
- **builder**: Stage pour compiler les dépendances Python
- **runtime**: Image de production optimisée 
- **development**: Image de développement avec outils et hot reload
- **test**: Stage pour exécuter les tests

### Optimisations Appliquées

#### DK001: Multi-stage builds
- Séparation entre compilation (builder) et runtime
- Réduction de la taille finale de l'image (~60% plus petite)

#### DK002: Versions épinglées
- `python:3.11.8-slim` pour la reproductibilité
- Images Prometheus/Grafana avec versions spécifiques

#### DK003: Image de base minimale  
- `python:3.11.8-slim` pour la production
- Pas d'outils de développement inutiles

#### DK004: Utilisateur non-root
- Utilisateur `agent` avec UID/GID dédiés
- Principe de moindre privilège

#### DK005: .dockerignore optimisé
- Exclusion des tests, docs, cache Python
- Contexte de build réduit

#### DK006: Cache BuildKit
- Cache pour `apt` et `pip install`  
- Builds incrémentaux plus rapides

#### DK007: Configuration Python
- `PYTHONUNBUFFERED=1` pour les logs temps réel
- `PYTHONDONTWRITEBYTECODE=1` pour éviter les .pyc

## Utilisation

### Développement

```bash
# Lancer l'environnement de développement complet
docker-compose up

# Ou avec Ollama CPU uniquement
docker-compose --profile cpu up

# Initialiser Ollama avec le modèle
docker-compose --profile init up ollama-init

# Hot reload automatique sur changement de code
```

### Production

```bash
# Build optimisé pour production
docker build --target runtime -t agent-loop:prod .

# Déployement production avec monitoring
docker-compose -f docker-compose.prod.yml up -d

# Vérifier les services
docker-compose -f docker-compose.prod.yml ps
```

## Services

### FastAPI App (port 8000)
- Application principale avec API REST
- Intégration Ollama pour LLM
- Métriques Prometheus `/metrics`
- Health checks `/health` et `/ready`

### Ollama (port 11434) 
- Service LLM local avec modèle Gemma 3N
- Support GPU optionnel
- API compatible OpenAI

### Prometheus (port 9090)
- Collecte des métriques applicatives
- Monitoring FastAPI et système  
- Alerting (configuration future)

### Grafana (port 3000)
- Dashboards de monitoring
- Visualisation des métriques
- Credentials: admin/admin123 (dev)

### Production uniquement
- **Nginx**: Load balancer (port 80/443)
- **Node Exporter**: Métriques système (port 9100)  
- **cAdvisor**: Métriques containers (port 8080)

## Volumes

- `ollama_data`: Modèles et cache Ollama
- `prometheus_data`: Données métriques Prometheus  
- `grafana_data`: Configuration et dashboards Grafana
- `prometheus_multiproc`: Métriques multiprocess FastAPI

## Réseaux

- `agent-network`: Bridge network interne (172.20.0.0/16)
- Communication sécurisée entre services

## Commandes Utiles

```bash
# Logs en temps réel
docker-compose logs -f fastapi-app

# Shell dans le container
docker-compose exec fastapi-app bash

# Rebuild complet
docker-compose build --no-cache

# Arrêt propre
docker-compose down -v

# Taille des images
docker images | grep agent-loop

# Utilisation ressources
docker stats
```

## Sécurité

- Utilisateurs non-root dans tous les containers
- Secrets management pour production  
- Network isolation
- Health checks pour tous les services
- Logging centralisé avec rotation

## Performance

- Build cache optimisé (BuildKit)
- Multi-stage builds pour images légères
- Resource limits en production
- Connection pooling HTTP
- Prometheus metrics pour monitoring

## Troubleshooting

### Build lent
Le premier build peut prendre 10-15 minutes à cause de PyTorch et CUDA. Les builds suivants utilisent le cache.

### OOM Errors
Augmenter la RAM Docker (8GB recommandé) ou désactiver PyTorch si non utilisé.

### Ollama model missing
```bash
docker-compose --profile init up ollama-init
```

### Port conflicts
Modifier les ports dans docker-compose.yml si 8000, 3000, 9090 sont occupés.

---

# QLoRA Fine-tuning with Docker - Sprint 2

Architecture Docker GPU optimisée pour le fine-tuning QLoRA avec support RTX 4090.

## Architecture Training

### Dockerfile Stage Training
- **Stage training**: Image NVIDIA CUDA 12.6 avec PyTorch GPU
- **Multi-stage build**: Séparation builder/runtime pour optimisation  
- **GPU Support**: Drivers NVIDIA + CUDA toolkit complet
- **Memory Optimization**: Configuration RTX 3090 24GB

### Docker Compose Training
- **docker-compose.training.yml**: Environnement isolé pour ML
- **GPU Resource Management**: Allocation GPU contrôlée
- **Volume Strategy**: Optimisée pour datasets et modèles volumineux
- **Monitoring**: W&B Local + GPU metrics en temps réel

## Configuration GPU

### Prérequis Système
```bash
# Vérifier support GPU NVIDIA
nvidia-smi

# Docker avec support GPU
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Docker Compose avec GPU (Docker >= 23.0)
docker compose version
```

### Variables d'Environnement
```bash
# Configuration GPU pour training
CUDA_VISIBLE_DEVICES=all
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Optimisation mémoire RTX 4090
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
HF_HOME=/app/models/.cache/huggingface
TRANSFORMERS_CACHE=/app/models/.cache/transformers
```

## Commandes Training

### Développement Interactif
```bash
# Environnement complet avec Jupyter + TensorBoard
make train-dev

# Accès services:
# 📚 Jupyter Lab: http://localhost:8888 (token: agent-dev-2024)
# 📊 TensorBoard: http://localhost:6006  
# 🔍 W&B Monitor: http://localhost:8080
```

### Training en Production
```bash
# Training Gemma 2B avec configuration optimisée
make train-gemma-2b

# Training Gemma 9B (configuration mémoire réduite)  
make train-gemma-9b

# Training avec modèle custom
make train-custom MODEL=google/gemma-2-2b DATA=/path/to/data

# Monitoring complet (training + dev + monitoring)
make train-monitor
```

### Gestion des Conteneurs
```bash
# Status des conteneurs et GPU
make train-status

# Logs training en temps réel
make train-logs

# Arrêt propre
make train-stop

# Nettoyage complet (⚠️ supprime les checkpoints)
make train-clean
```

## Volumes et Données

### Structure des Volumes
```
├── training_models_cache/          # Cache HuggingFace (~2-5GB)
├── training_models_gguf/           # Modèles GGUF (~4-8GB)  
├── training_checkpoints/           # Checkpoints training (~1-3GB)
├── training_logs/                  # Logs structurés + W&B
└── training_backup/                # Backup automatique
```

### Configuration des Modèles GGUF
```bash
# Créer répertoire modèles
mkdir -p models/gguf

# Télécharger modèle de base Gemma 2B (exemple)
cd models/gguf/
wget https://huggingface.co/google/gemma-2-2b-it-gguf/resolve/main/gemma-2-2b-it.gguf

# Vérifier la structure
ls -la models/
```

## Monitoring et Observabilité

### GPU Monitoring
```bash
# Metrics GPU en temps réel  
docker logs agent-gpu-monitor

# Dashboard W&B Local
open http://localhost:8080

# Métriques Prometheus (si activé)
curl http://localhost:9090/metrics | grep gpu
```

### Logs Structurés
```bash
# Logs training principal
docker logs -f agent-qlora-training

# Logs GPU utilization
tail -f logs/gpu_metrics.jsonl

# Logs W&B offline
ls -la logs/wandb/
```

## Configuration QLoRA Avancée

### Modèles Supportés
- **Gemma 2B**: Configuration RTX 4090 optimisée (batch_size=4)
- **Gemma 9B**: Configuration mémoire réduite (batch_size=2)  
- **Custom**: Support modèles HuggingFace compatibles QLoRA

### Paramètres d'Optimisation
```python
# Configuration RTX 4090 (24GB VRAM)
per_device_train_batch_size: 4
gradient_accumulation_steps: 4  # Effective batch = 16
bf16: True                      # Better than fp16 on Ampere
gradient_checkpointing: True    # Memory optimization
optim: "adamw_8bit"            # 8-bit optimizer

# QLoRA 4-bit + Double Quantization  
load_in_4bit: True
bnb_4bit_use_double_quant: True
bnb_4bit_quant_type: "nf4"
```

## Sécurité Training

### Isolation des Conteneurs
- **Network dédié**: training-network (172.25.0.0/16)
- **Utilisateur non-root**: trainer (UID 1000)
- **Volumes read-only**: Datasets protégés en lecture seule
- **Resource limits**: Mémoire container limitée à 20GB

### Backup et Récupération
```bash
# Backup automatique des checkpoints (toutes les heures)
docker logs agent-training-backup

# Restauration manuelle
rsync -av backups/training/ model_checkpoints/
```

## Performance et Optimisation

### Builds Docker Optimisés
- **BuildKit cache**: Cache pip et apt pour builds rapides
- **.dockerignore**: Exclusion fichiers volumineux (*.gguf, *.bin)
- **Multi-stage**: Séparation training builder/runtime
- **Layer caching**: Optimisation ordre instructions

### Memory Management
```bash
# Monitoring mémoire GPU en temps réel
watch -n 1 "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"

# Monitoring mémoire container
docker stats agent-qlora-training
```

## Troubleshooting Training

### Erreurs GPU
```bash
# Vérifier accès GPU dans container
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi

# Logs détaillés CUDA
CUDA_LAUNCH_BLOCKING=1 make train-dev
```

### Out of Memory (OOM)
1. **Réduire batch_size**: Modifier dans qlora_config.py
2. **Gradient checkpointing**: Déjà activé par défaut  
3. **Modèle plus petit**: Utiliser Gemma 2B au lieu de 9B
4. **4-bit quantization**: Vérifier configuration QLoRA

### Slow Training
1. **GPU utilization**: Vérifier >80% dans nvidia-smi
2. **DataLoader**: Augmenter num_workers si CPU disponible
3. **Mixed precision**: bf16 activé par défaut
4. **Storage I/O**: Placer volumes sur SSD NVMe

### Model Loading Issues  
```bash
# Vérifier cache HuggingFace
ls -la models/.cache/huggingface/

# Réinitialiser cache si corrompu
rm -rf models/.cache/huggingface/
docker restart agent-qlora-training
```

## Migration et Intégration

### Vers Production
1. **Image optimisée**: Build avec stage training uniquement
2. **Secrets management**: Variables sensibles via Docker secrets
3. **Orchestration**: Kubernetes ou Docker Swarm pour scalabilité
4. **CI/CD**: Intégration avec pipeline MLOps automatisé

### Intégration MLOps
- **Model Registry**: Intégration W&B/MLflow pour versioning
- **Automated Deployment**: Pipeline CI/CD avec tests modèles
- **Monitoring Production**: Drift detection et performance tracking
- **A/B Testing**: Déployment canary avec métriques business