# 🏗️ Structure du Projet Agent Loop - Guide Architectural Officiel

> **📌 Document de référence pour tous les agents**  
> Maintenu par : `@system-architect`  
> Dernière mise à jour : 2025-07-29

## Vue d'ensemble de l'arborescence

```
agent_loop/
├── 🎓 models/                      # Hub ML complet (Training + Inference)
│   ├── training/                   # Pipelines d'entraînement
│   ├── inference/                  # API de production (FastAPI)
│   ├── datasets/                   # Gestion des données d'entraînement
│   ├── results/                    # Outputs et checkpoints
│   ├── logs/                       # Logs d'entraînement et d'inférence
│   ├── scripts/                    # Scripts opérationnels
│   └── model_cache/               # Cache des modèles téléchargés
│
├── 🤖 agent/                      # Implémentation de l'agent
│   ├── tools/                     # Outils de base
│   ├── plugins/                   # Extensions modulaires
│   └── prompts/                   # Prompts système et exemples
│
├── 🏗️ infrastructure/             # Infrastructure as Code
│   ├── terraform/                 # Définition des ressources cloud  
│   ├── ansible/                   # Configuration management
│   └── docker/                    # Orchestration des conteneurs
│
├── 📊 monitoring/                 # Stack d'observabilité
│   ├── grafana/                   # Dashboards et visualisation
│   ├── prometheus/                # Collecte de métriques
│   └── nginx/                     # Configuration reverse proxy
│
├── 🧪 tests/                      # Assurance qualité
│   ├── unit/                      # Tests unitaires rapides
│   ├── integration/               # Tests d'interaction
│   └── e2e/                       # Tests bout-en-bout
│
├── 📚 docs/                       # Hub de documentation
│   ├── ARCHITECTURE/              # Documents de conception système
│   ├── SECURITY/                  # Analyses et guides de sécurité
│   └── R&D/                       # Documentation recherche
│
├── 🗂️ leboncoin_extension/        # Extension navigateur (exemple)
└── .claude/agents/               # Configuration des agents Claude <think>it's you and your subs<think>
```

---

## 📁 Description détaillée de chaque dossier

### 🤖 `.claude/agents/` - Configuration des Agents Claude
**Quoi mettre ici** : Prompts et configurations des agents spécialisés pour claude code <think its for YOU>
```
.claude/agents/
├── fastapi-async-architect.md    # 🚀 Expert FastAPI
├── system-architect.md           # 🏛️ Garant de l'architecture
├── python-type-guardian.md       # 🐍 Expert typage Python
├── llm-optimization-engineer.md  # 🧠 Expert optimisation ML
├── test-automator.md            # 🧪 Expert tests
├── docker-container-architect.md # 🐳 Expert containers
├── observability-engineer.md     # 📊 Expert monitoring
└── AGENT_COLLABORATION_GUIDE.md  # 🤝 Guide de collaboration
```
**Usage** : Chargé par Claude pour spécialiser les agents

---

### 🎓 `models/` - Le Hub ML Complet
**Quoi mettre ici** : Tout l'écosystème ML (training + inference + data)

#### 📚 `models/training/` - L'École du Modèle
```
models/training/
├── __init__.py
├── qlora/                     # 🎯 Fine-tuning QLoRA
│   ├── qlora_finetune_unsloth.py    # Pipeline Unsloth optimisé
│   ├── qlora_config.py              # Configuration QLoRA
│   └── unsloth_comparison.py        # Benchmarks Unsloth vs vanilla
├── nn/                        # 🧮 Architectures neuronales
│   ├── xnet_head.py          # Tête XNet custom
│   ├── step_hint_loss.py     # Fonction de loss step-hint
│   └── __init__.py
├── security/                  # 🛡️ Sécurité d'entraînement
│   ├── guardrails.py         # Protection données sensibles
│   ├── input_validation.py   # Validation inputs
│   └── secure_data_loader.py # Chargement sécurisé
├── add_step_hints.py         # 📝 Ajout annotations step-hint
├── mix_datasets.py           # 🔄 Fusion sources de données
├── evaluate_toolbench.py     # 📊 Évaluation ToolBench
└── orchestrate_training.py   # 🎼 Orchestration complète
```

#### 🚀 `models/inference/` - La Production (Architecture Moderne)
```
models/inference/
├── app.py                    # 🌐 Application FastAPI principale
├── api.py                    # 🔄 Compatibilité ancienne API
├── groupthink.py            # 🤝 Inférence multi-thread
├── routers/                 # 🛣️ Endpoints organisés par domaine
│   ├── agents.py           # Endpoints /agents/*
│   ├── health.py           # Endpoints /health/*
│   ├── ollama.py           # Endpoints /ollama/*
│   └── training.py         # Endpoints /training/*
├── services/                # 💼 Logique métier
│   ├── ollama.py           # Service Ollama optimisé
│   ├── dependencies.py     # Injection de dépendances
│   ├── external_api.py     # Services externes
│   └── health.py           # Healthchecks avancés
├── models/                  # 📦 Schémas Pydantic
│   └── schemas.py          # Request/Response models
├── middleware/              # 🛡️ Sécurité & observabilité
│   ├── security.py         # Headers, CORS, rate limiting
│   └── __init__.py
└── websockets/              # 🔌 Support WebSocket
    └── __init__.py
```

#### 📊 `models/datasets/` - Gestion des Données
```
models/datasets/
├── README.md               # 📖 Guide des datasets
├── processed/             # 🏭 Données prêtes pour l'entraînement
│   ├── unified_format/    # Format unifié pour tous les datasets
│   ├── train_splits/      # Splits d'entraînement
│   ├── eval_splits/       # Splits d'évaluation
│   └── metadata/          # Métadonnées des datasets
├── toolbench/            # 🛠️ Dataset ToolBench original
├── webarena/             # 🌐 Dataset WebArena
├── browsergym/           # 🏋️ Dataset BrowserGym
├── agent_instruct/       # 🎓 Dataset AgentInstruct
├── react/                # ⚛️ Dataset ReAct
├── miniowb/              # 📖 Dataset MiniOWB
└── camel_agent/          # 🐪 Dataset CAMEL Agent
```

#### 🎯 `models/results/` - Outputs d'Entraînement
```
models/results/
├── gemma-3n-unsloth-100steps-fixed/  # 🏆 Modèle stable 100 steps
├── gemma-3n-beast-mode-1epoch/       # 🦁 Mode intensif 1 époque
├── gemma-3n-safe-1epoch/             # 🛡️ Mode sécurisé 1 époque
├── gemma-3n-unsloth-2epochs/         # 📈 Training long terme
└── gemma3n_agent_v1/                 # 🎯 Modèle de production v1
```

#### 📜 `models/scripts/` - Automatisation Opérationnelle
```
models/scripts/
├── run_unsloth_training.sh      # 🚀 Script training Unsloth
├── train_optimized.py           # 🎯 Training pipeline optimisé
├── model_registry.py            # 📦 Registre des modèles
├── health_check.py              # 🩺 Vérification santé système
├── sync_logs.sh                 # 📡 Synchronisation logs
├── update_model.sh              # 🔄 Mise à jour modèles production
└── secure_dataset_downloader.py # 🔒 Téléchargement sécurisé datasets
```

---

### 🔧 `plugins/` - La Boîte à Outils
**Quoi mettre ici** : Les outils que l'agent peut utiliser
```
plugins/
├── __init__.py
├── browser_tool.py   # 🌐 Navigation web (Playwright)
└── (autres outils à venir)
```
**Pattern** : Outils autonomes utilisables par l'agent

---

### 💬 `prompts/` - Prompts Système
**Quoi mettre ici** : Prompts et exemples pour l'agent
```
prompts/
├── system_prompt.txt        # 📝 Prompt système principal
├── few_shot_examples.json   # 📚 Exemples few-shot
└── agent_rules.xml         # 📋 Règles de l'agent
```
**Usage** : Configuration du comportement de l'agent

---

### 📜 `scripts/` - Les Assistants
**Quoi mettre ici** : Scripts bash/python pour ops
```
scripts/
├── deploy/           # 🚀 Déploiement
│   ├── update_model.sh      # MAJ modèle sur VMs
│   └── rollback.sh          # Retour arrière
├── monitoring/       # 📊 Observabilité
│   ├── sync_logs.sh         # Collecte logs
│   └── check_health.py      # Healthchecks
└── development/      # 🛠️ Dev tools
    ├── setup_dev.sh         # Init environnement
    └── run_local.sh         # Test local
```
**Convention** : Executable (`chmod +x`), shebang en tête

---

### 📋 `rules/` - Le Règlement Interne
**Quoi mettre ici** : Personas et best practices
```
rules/
├── python_development.xml    # 🐍 Standards Python
├── ml_engineering.xml       # 🤖 MLOps rules
├── security.xml            # 🔒 Sécurité
└── README.md              # 📖 Comment utiliser
```
**Usage** : Chargé par l'IDE ou les revues de code

---

### 🧪 `tests/` - Le Laboratoire
**Quoi mettre ici** : Tous les tests (miroir de la structure)
```
tests/
├── unit/             # ⚡ Tests unitaires rapides
│   ├── core/
│   ├── training/
│   └── inference/
├── integration/      # 🔗 Tests d'intégration
│   ├── test_api.py
│   └── test_ollama.py
├── e2e/             # 🎯 Tests bout-en-bout
│   └── test_agent_flow.py
├── fixtures/        # 🎪 Données de test
└── conftest.py      # ⚙️ Config pytest globale
```
**Règle** : 1 fichier de test par module (`test_*.py`)

---

### 📊 `monitoring/` - Observabilité
**Quoi mettre ici** : Configurations Prometheus et Grafana
```
monitoring/
├── prometheus.yml    # ⚙️ Config Prometheus
├── grafana/         # 📊 Dashboards Grafana
│   ├── dashboards/  # JSON dashboards
│   └── datasources/ # Sources de données
└── nginx/           # 🔄 Config reverse proxy
```
**Usage** : Monitoring complet de l'application

---

### 📦 `ansible/` - Configuration Management
**Quoi mettre ici** : Playbooks et configurations Ansible
```
ansible/
├── ansible.cfg      # ⚙️ Configuration Ansible
├── inventory.yml    # 🖥️ Inventaire des serveurs
├── site.yml        # 🎯 Playbook principal
├── playbooks/      # 📋 Playbooks spécifiques
│   ├── base-setup.yml
│   ├── python-setup.yml
│   ├── ollama-setup.yml
│   └── monitoring-setup.yml
└── templates/      # 📄 Templates Jinja2
```
**Principe** : Automatisation du déploiement

---

### 🏗️ `terraform/` - Infrastructure as Code
**Quoi mettre ici** : Définition de l'infrastructure cloud
```
terraform/
├── main.tf          # 🏗️ Configuration principale
├── variables.tf     # 📝 Variables
├── outputs.tf       # 📤 Outputs
├── versions.tf      # 🔢 Versions providers
└── user_data.sh     # 🚀 Script d'init VMs
```
**Usage** : Provisioning automatisé de l'infrastructure
1
---

## 📄 Fichiers à la racine

```
agent_loop/
├── .gitignore          # 🚫 Fichiers à ignorer
├── .env.example        # 🔐 Template variables d'env
├── requirements.txt    # 📦 Dépendances Python
├── pyproject.toml      # 🐍 Config Python moderne
├── Makefile           # 🛠️ Commandes communes
├── README.md          # 📖 Documentation principale
├── LICENSE            # ⚖️ Licence du projet
└── cli.py            # 🖥️ Interface ligne de commande
```

---

## 🎯 Où mettre quoi - Guide rapide

| Type de fichier | Où le mettre | Exemple |
|----------------|--------------|---------|
| Configuration agent Claude | `.claude/agents/` | `new-agent.md` |
| Nouveau modèle Pydantic | `inference/models/schemas.py` | `class GenerateRequest` |
| Nouveau router/endpoint | `inference/routers/` | `predictions.py` |
| Service métier | `inference/services/` | `model_service.py` |
| Nouvelle loss function | `training/nn/` | `contrastive_loss.py` |
| Nouveau tool/plugin | `plugins/` | `email_tool.py` |
| Script Ansible | `ansible/playbooks/` | `deploy-model.yml` |
| Config Terraform | `terraform/` | `gpu-instance.tf` |
| Dashboard Grafana | `monitoring/grafana/dashboards/` | `ml-metrics.json` |
| Test | `tests/` | `test_generate_endpoint.py` |
| Documentation technique | `docs/` | `API_GUIDE.md` |

---

## 🚀 Workflow typique

1. **Développement** : Coder dans `core/`, `training/` ou `inference/`
2. **Test** : Écrire tests dans `tests/` (TDD recommandé)
3. **Lint** : `make lint` vérifie le code
4. **Build** : `make build` crée les images Docker
5. **Deploy** : `scripts/deploy/update_model.sh` pousse en prod

---

## 💡 Bonnes pratiques

- **Imports** : Toujours absolus depuis `agent_loop`
  ```python
  from agent_loop.core.settings import settings
  from agent_loop.inference.groupthink import generate
  ```

- **Logs** : Utiliser structlog partout
  ```python
  from agent_loop.core.utils.logging import get_logger
  logger = get_logger(__name__)
  ```

- **Config** : Jamais de hardcode, tout dans `settings.py`
  ```python
  settings.model_path  # ✅
  "/models/gemma.gguf"  # ❌
  ```

- **Tests** : Un test par fonctionnalité
  ```
  tests/unit/training/nn/test_step_hint_loss.py
  tests/integration/test_browser_tool.py
  ```

Cette structure permet de scaler de 1 à 100 développeurs tout en gardant le code maintenable ! 🚀

---

## 🤝 Agents Garants de l'Architecture

### Responsables par domaine :

| Domaine | Agent Responsable | Rôle |
|---------|------------------|------|
| **Architecture globale** | `@system-architect` | Garant de la structure, patterns, hexagonal architecture |
| **API & Endpoints** | `@fastapi-async-architect` | Structure inference/, routers, async patterns |
| **Types & Schémas** | `@python-type-guardian` | Pydantic models, type safety, mypy compliance |
| **ML & Training** | `@llm-optimization-engineer` | Structure training/, optimisations, modèles |
| **Containers** | `@docker-container-architect` | Dockerfile, docker-compose, volumes |
| **Tests** | `@test-automator` | Structure tests/, fixtures, coverage |
| **Monitoring** | `@observability-engineer` | Métriques, logs, dashboards Grafana |

### Comment demander de l'aide :

```bash
# Avant de créer un fichier :
"@system-architect: Où placer mon nouveau module de [FEATURE] ?"

# Pour un endpoint API :
"@fastapi-async-architect: Dans quel router mettre mon endpoint /predict ?"

# Pour un modèle Pydantic :
"@python-type-guardian: Comment structurer mon modèle de réponse ?"

# Pour un test :
"@test-automator: Où placer les tests pour mon nouveau service ?"
```

### 📚 Documentation de référence :

- Guide de collaboration : `.claude/agents/AGENT_COLLABORATION_GUIDE.md`
- Ce document : `PROJECT_STRUCTURE.md`
- Architecture détaillée : `docs/ARCHITECTURE/`
