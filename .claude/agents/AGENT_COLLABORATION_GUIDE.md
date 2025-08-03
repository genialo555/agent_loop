# 🤝 Guide de Collaboration Inter-Agents Claude

## Vue d'ensemble

Ce guide permet aux agents Claude de collaborer efficacement en "chunking" les tâches selon les expertises de chacun.

## 🎯 Matrice de Collaboration Rapide

### Qui demander pour quoi ?

| Si vous devez... | Demandez à... | Questions types |
|------------------|---------------|-----------------|
| **Créer/placer un fichier** | `@system-architect` | "Où placer ce nouveau module?", "Quelle structure suivre?" |
| **Typer du code Python** | `@python-type-guardian` | "Comment typer cette fonction?", "Quelle Pydantic model utiliser?" |
| **Créer un endpoint API** | `@fastapi-async-architect` | "Quel pattern async utiliser?", "Comment structurer cet endpoint?" |
| **Optimiser un modèle ML** | `@llm-optimization-engineer` | "Comment réduire la latence?", "Quelle quantization utiliser?" |
| **Écrire des tests** | `@test-automator` | "Comment tester ce composant?", "Où sont les fixtures?" |
| **Configurer Docker** | `@docker-container-architect` | "Comment optimiser l'image?", "Quels volumes utiliser?" |
| **Ajouter du monitoring** | `@observability-engineer` | "Quelles métriques exposer?", "Comment logger ceci?" |

## 📍 Localisation des Implémentations Clés

```
/home/jerem/agent_loop/
├── inference/              # 🚀 API FastAPI
│   ├── app.py             # Application principale (avec alias /generate)
│   ├── routers/           # Endpoints (/agents/run-agent, etc.)
│   └── services/          # Services (OllamaService = "llm_manager")
├── core/                  # 🏛️ Domaine métier
├── training/              # 🧠 ML/Training
│   └── qlora_finetune.py  # Script de fine-tuning (simulation Sprint 1)
├── tests/                 # 🧪 Tests
└── .claude/agents/        # 📚 Documentation agents
```

## 🔄 Workflow de Collaboration

### 1. Avant de commencer
```
"@system-architect: Je dois implémenter [FEATURE]. Où placer le code?"
```

### 2. Pendant l'implémentation
```
"@python-type-guardian: Comment typer cette structure de données complexe?"
"@fastapi-async-architect: Quel pattern pour cet endpoint qui stream des données?"
```

### 3. Après l'implémentation
```
"@test-automator: J'ai créé [COMPONENT]. Peux-tu m'aider avec les tests?"
"@observability-engineer: Quelles métriques ajouter pour monitorer ceci?"
```

## 🎭 Exemples Concrets de Collaboration

### Cas 1: Créer un nouvel endpoint
```
Agent A: "@system-architect: Je dois créer un endpoint /predict. Où le placer?"
System-Architect: "Crée un nouveau router dans inference/routers/predictions.py"

Agent A: "@fastapi-async-architect: Comment structurer cet endpoint qui appelle Ollama?"
FastAPI-Architect: "Utilise le pattern de OllamaService avec dependency injection..."

Agent A: "@python-type-guardian: Quelle Pydantic model pour la réponse?"
Type-Guardian: "Hérite de BaseModel, regarde RunAgentResponse comme exemple..."
```

### Cas 2: Optimiser les performances
```
Agent B: "@llm-optimization-engineer: L'inférence est lente, comment optimiser?"
LLM-Engineer: "Vérifie d'abord si le batching est activé dans OllamaService..."

Agent B: "@observability-engineer: Comment mesurer les améliorations?"
Observability: "Ajoute ces métriques Prometheus dans inference_duration_seconds..."
```

## 📋 Correspondances Sprint 1

| Spécification | Implémentation Réelle | Agent Responsable |
|---------------|----------------------|-------------------|
| `/generate` | `/agents/run-agent` (alias créé) | fastapi-async-architect |
| `llm_manager.py` | `services/ollama.py` | system-architect |
| Dépendances transformers | Architecture Ollama | llm-optimization-engineer |
| Tests `/generate` | Tests `/agents/run-agent` | test-automator |

## 💡 Conseils Pratiques

1. **Toujours demander avant de créer** - Évite les doublons
2. **Partager les découvertes** - "J'ai trouvé que X est implémenté dans Y"
3. **Chunker les tâches** - Une expertise par agent
4. **Documenter les décisions** - Pour les futurs agents

## 🚀 Commandes Utiles

```bash
# Vérifier où quelque chose est implémenté
@system-architect: "Où se trouve l'implémentation de [FEATURE]?"

# Demander le bon pattern
@[EXPERT]-architect: "Quel pattern utiliser pour [USE_CASE]?"

# Valider une implémentation
@[EXPERT]: "Peux-tu valider cette implémentation de [COMPONENT]?"
```

---

**Remember**: La collaboration entre agents permet de créer un code de meilleure qualité en utilisant l'expertise spécialisée de chaque agent. N'hésitez pas à demander!