# 🏗️ Structure du Projet Agent Loop - Guide Pédagogique

## Vue d'ensemble de l'arborescence

```
agent_loop/
├── core/                    🧠 Le cerveau - Logique métier partagée
├── training/               🎓 L'école - Où le modèle apprend
├── inference/              🚀 La production - Où le modèle travaille
├── plugins/                🔧 La boîte à outils - Extensions modulaires
├── agent/                  🤖 L'ancien agent (deprecated)
├── scripts/                📜 Les assistants - Automatisation
├── rules/                  📋 Le règlement - Bonnes pratiques
├── tests/                  🧪 Le labo - Vérification qualité
├── docs/                   📚 La bibliothèque - Documentation
├── .github/                🔄 L'usine CI/CD - Automatisation GitHub
└── infrastructure/         🏭 Les fondations - Déploiement
```

---

## 📁 Description détaillée de chaque dossier

### 🧠 `core/` - Le Cerveau Central
**Quoi mettre ici** : Tout ce qui est partagé entre training et inference
```
core/
├── __init__.py
├── settings.py         # ⚙️ Configuration globale (Pydantic)
├── models.py          # 📦 Classes de données (Pydantic models)
├── exceptions.py      # ⚠️ Exceptions custom du projet
└── utils/             # 🛠️ Fonctions utilitaires
    ├── __init__.py
    ├── logging.py     # 📝 Configuration des logs structurés
    ├── metrics.py     # 📊 Helpers pour Prometheus
    └── validators.py  # ✅ Validations communes
```
**Principe** : Aucune dépendance vers training/ ou inference/

---

### 🎓 `training/` - L'École du Modèle
**Quoi mettre ici** : Tout ce qui concerne l'entraînement
```
training/
├── __init__.py
├── datasets/          # 📚 Préparation des données
│   ├── __init__.py
│   ├── add_step_hints.py    # Ajout des hints
│   ├── mix_datasets.py      # Fusion des sources
│   └── processors/          # Transformations spécifiques
├── nn/               # 🧮 Architectures neuronales
│   ├── __init__.py
│   ├── xnet_head.py         # Tête XNet custom
│   ├── step_hint_loss.py    # Fonction de loss
│   └── lora_adapters.py     # Configuration LoRA
├── pipelines/        # 🔄 Orchestration
│   ├── __init__.py
│   ├── qlora_finetune.py    # Pipeline principal
│   └── evaluation.py        # Métriques & benchmarks
└── configs/          # 📋 Configurations d'entraînement
    ├── base.yaml
    └── experiments/
```
**À éviter** : Code qui touche à l'API ou au runtime

---

### 🚀 `inference/` - La Production
**Quoi mettre ici** : Le code qui sert le modèle
```
inference/
├── __init__.py
├── api.py            # 🌐 FastAPI endpoints
├── groupthink.py     # 🤝 Inference multi-thread
├── ollama_client.py  # 🦙 Wrapper Ollama
├── middleware/       # 🛡️ Sécurité & monitoring
│   ├── __init__.py
│   ├── auth.py      # JWT validation
│   ├── metrics.py   # Prometheus export
│   └── ratelimit.py # Protection DDoS
└── schemas/          # 📄 Modèles request/response
    ├── __init__.py
    └── agent.py
```
**Focus** : Performance, sécurité, observabilité

---

### 🔧 `plugins/` - La Boîte à Outils
**Quoi mettre ici** : Les outils que l'agent peut utiliser
```
plugins/
├── __init__.py
├── base.py           # 🏗️ Classe abstraite Tool
├── browser_tool.py   # 🌐 Navigation web (Playwright)
├── search_tool.py    # 🔍 Recherche Google/Bing
├── calculator.py     # 🧮 Calculs mathématiques
└── registry.py       # 📋 Découverte automatique
```
**Pattern** : Chaque tool hérite de `BaseTool` et s'enregistre

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

### 🏭 `infrastructure/` - Les Fondations
**Quoi mettre ici** : IaC et configs de déploiement
```
infrastructure/
├── terraform/        # 🏗️ Provisioning cloud
│   ├── main.tf
│   ├── variables.tf
│   └── modules/
├── kubernetes/       # ☸️ Manifests K8s
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── docker/          # 🐳 Conteneurisation
│   ├── Dockerfile.training
│   ├── Dockerfile.inference
│   └── docker-compose.yml
└── ansible/         # 📦 Configuration management
    ├── playbooks/
    └── inventory/
```
**Principe** : Tout est versionné et reproductible

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
| Nouveau modèle Pydantic | `core/models.py` | `class AgentRequest` |
| Nouvelle loss function | `training/nn/` | `contrastive_loss.py` |
| Nouveau endpoint API | `inference/api.py` | `@app.post("/chat")` |
| Nouveau tool/plugin | `plugins/` | `email_tool.py` |
| Script de déploiement | `scripts/deploy/` | `deploy_gpu.sh` |
| Test unitaire | `tests/unit/` | `test_xnet_head.py` |
| Config Kubernetes | `infrastructure/k8s/` | `hpa.yaml` |
| Documentation technique | `docs/` | `architecture.md` |

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
