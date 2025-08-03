# Contexte : Fine-tuning Gemma-3N-E4B sur Agent Instruct

## Résumé du projet

Je suis en train de fine-tuner le modèle **Gemma-3N-E4B** (4.5B paramètres, architecture expérimentale AltUp de Google) sur le dataset **Agent Instruct** de Microsoft (1.1M d'exemples d'instructions pour agents).

## Objectif

Créer un modèle capable de naviguer et exécuter des tâches dans un OS Linux de manière autonome. L'agent doit comprendre des instructions naturelles comme "trouve tous les logs d'erreur et crée un rapport" et générer les commandes bash/python nécessaires.

## Défis techniques rencontrés

1. **Architecture Gemma-3N incompatible** avec les frameworks standards
   - Erreur "Float can't be cast to unsigned char" 
   - Solution : Migration vers Unsloth (seul framework compatible)

2. **Problèmes de cache** - Les datasets se téléchargeaient sur le mauvais SSD (OOM)
   - Solution : Configuration des variables HF_HOME et TRANSFORMERS_CACHE

3. **Recompilation excessive** avec torch.dynamo
   - Solution : cache_size_limit augmenté + tri des données par longueur

4. **Performance sous-optimale** (single-thread CPU bottleneck)
   - Solution : num_proc=16, batch_size=2, optimisations système

## Configuration matérielle

- **GPU** : RTX 3090 24GB (100% utilisé, 350W, 78-81°C)
- **CPU** : Ryzen 9 7900 (12 cores mais single-thread limité)
- **RAM** : 64GB DDR5 6000MHz (27GB en cache pour le dataset)
- **Coût** : < 2K€ (excellent rapport perf/prix)

## État actuel du training

- **Méthode** : QLoRA 4-bit avec Unsloth
- **Vitesse** : ~5.8s/step
- **Batch size** : 2 (effectif 8 avec gradient accumulation)
- **1 epoch** = 140,177 steps = ~226 heures
- **Loss actuel** : 6.0 → ~1.3-1.4 (très bon apprentissage)

## Commandes importantes

```bash
# Lancer le training optimisé
./train_safe_optimized.sh

# Surveiller
watch -n 60 'nvidia-smi; free -h'

# Tester le modèle (après 5000+ steps)
./test_model_later.py
```

## Prochaines étapes

1. Continuer jusqu'à ~10,000 steps (loss < 1.0)
2. Tester sur des scénarios réels de navigation OS
3. Dockeriser avec API FastAPI
4. Intégrer dans les workflows d'automatisation

---
*Training lancé le 31/07/2025 - Gemma-3N-E4B transformé en agent système Linux*