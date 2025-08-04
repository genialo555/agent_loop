# 🧠 Plan d'Implémentation HRM pour Gemma-3N

## 📋 Résumé Exécutif

**Objectif**: Adapter l'architecture Hierarchical Reasoning Model (HRM) à Gemma-3N (4.5B params) pour améliorer les capacités de raisonnement hiérarchique et la résolution de problèmes complexes.

**Approche**: Implémenter les deux modules récurrents (H et L) de HRM comme des adaptateurs LoRA sur Gemma-3N, en utilisant Unsloth pour l'optimisation et en conservant l'architecture de base.

## 🏗️ Architecture HRM-Gemma-3N

### 1. Structure Modulaire

```python
HRM-Gemma-3N:
├── Module Bas-Niveau (L) - Calculs rapides et détaillés
│   ├── LoRA Adapters sur attention layers (rank=32)
│   ├── Hidden state: 2048 dims
│   └── Update frequency: Every timestep
│
├── Module Haut-Niveau (H) - Planification abstraite
│   ├── LoRA Adapters sur FFN layers (rank=64)
│   ├── Hidden state: 4096 dims
│   └── Update frequency: Every T timesteps
│
└── Gemma-3N Base (frozen ou QLoRA)
    ├── 36 transformer layers
    ├── Hidden dim: 3584
    └── Attention heads: 32
```

### 2. Mécanisme de Convergence Hiérarchique

```python
# Pseudo-code du forward pass
def hrm_forward(x, N=4, T=8):
    # N = nombre de cycles haut-niveau
    # T = timesteps par cycle bas-niveau
    
    x_emb = gemma_embed(x)
    z_L = init_low_state()
    z_H = init_high_state()
    
    for cycle in range(N):
        # Phase bas-niveau (convergence locale)
        for t in range(T):
            z_L = low_level_update(z_L, z_H, x_emb)
        
        # Mise à jour haut-niveau
        z_H = high_level_update(z_H, z_L)
        
        # Reset bas-niveau pour nouveau cycle
        z_L = reset_with_context(z_H)
    
    return decode_output(z_H)
```

## 📁 Structure des Fichiers

```
models/training/hrm/
├── __init__.py
├── hrm_config.py              # Configuration HRM
├── hrm_model.py               # Architecture HRM-Gemma
├── hrm_modules.py             # Modules H et L
├── hrm_trainer.py             # Training avec Unsloth
├── approximate_gradient.py     # 1-step gradient
├── deep_supervision.py        # Deep supervision loop
├── adaptive_compute.py        # ACT avec Q-learning
└── hierarchical_convergence.py # Mécanisme de convergence
```

## 🔧 Implémentation Détaillée

### Phase 1: Architecture de Base (Priorité: HIGH)

1. **Créer les modules H et L comme LoRA adapters**
   ```python
   # hrm_modules.py
   class LowLevelModule(nn.Module):
       def __init__(self, gemma_config):
           self.lora_attn = LoRALayer(rank=32, targets=['q', 'v'])
           self.hidden_dim = 2048
           self.state_projector = nn.Linear(3584, 2048)
   
   class HighLevelModule(nn.Module):
       def __init__(self, gemma_config):
           self.lora_ffn = LoRALayer(rank=64, targets=['gate', 'up'])
           self.hidden_dim = 4096
           self.state_projector = nn.Linear(3584, 4096)
   ```

2. **Implémenter la convergence hiérarchique**
   - Module L converge rapidement (8-16 steps)
   - Module H update seulement après convergence de L
   - Reset de L avec nouveau contexte de H

### Phase 2: Gradient Approximation (Priorité: HIGH)

1. **Implémenter 1-step gradient**
   ```python
   # approximate_gradient.py
   def compute_hrm_gradient(model, loss):
       # Détacher tous sauf derniers états
       with torch.no_grad():
           # Forward pass complet sauf dernière étape
           states = forward_no_grad(model, N*T-1)
       
       # Gradient seulement sur dernière étape
       final_z_L = model.L_step(states['z_L'], states['z_H'])
       final_z_H = model.H_step(states['z_H'], final_z_L)
       output = model.decode(final_z_H)
       
       # Backprop O(1) mémoire
       loss = criterion(output, target)
       loss.backward()
   ```

2. **Avantages pour RTX 3090**
   - Mémoire O(1) au lieu de O(T)
   - Permet batch_size plus large
   - Compatible avec gradient checkpointing

### Phase 3: Deep Supervision (Priorité: HIGH)

1. **Training loop avec supervision multiple**
   ```python
   # deep_supervision.py
   def train_with_deep_supervision(model, data, M=3):
       z = initialize_states()
       total_loss = 0
       
       for segment in range(M):
           # Forward pass segment
           z, output = model.hrm_forward(x, z)
           loss = criterion(output, y)
           
           # Mise à jour immédiate
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           # Détacher pour prochain segment
           z = z.detach()
           total_loss += loss.item()
       
       return total_loss / M
   ```

### Phase 4: Adaptive Computation Time (Priorité: MEDIUM)

1. **Q-Learning pour décision halt/continue**
   ```python
   # adaptive_compute.py
   class QHead(nn.Module):
       def __init__(self, hidden_dim):
           self.q_net = nn.Linear(hidden_dim, 2)  # [halt, continue]
       
       def should_halt(self, z_H, m_current, m_max):
           q_values = torch.sigmoid(self.q_net(z_H))
           q_halt, q_continue = q_values[0], q_values[1]
           
           if m_current >= m_max:
               return True
           return q_halt > q_continue
   ```

## 🚀 Training Strategy avec Unsloth

### Configuration Optimale

```python
# hrm_trainer.py
from unsloth import FastLanguageModel

# Charger Gemma-3N avec Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b",
    max_seq_length=2048,  # Réduit pour HRM
    dtype=torch.float16,
    load_in_4bit=True,
)

# Ajouter modules HRM
model = add_hrm_modules(model, 
    lora_r_low=32,
    lora_r_high=64,
    lora_alpha=16,
    lora_dropout=0.05,
)

# Configuration training
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # RTX 3090 limite
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    output_dir="./results/gemma-3n-hrm",
    # HRM spécifique
    hrm_cycles_N=4,
    hrm_timesteps_T=8,
    deep_supervision_M=3,
)
```

### Datasets HRM pour Gemma-3N

1. **Datasets disponibles** (déjà dans HF cache):
   - GSM8K: Raisonnement mathématique étape par étape
   - CodeAlpaca-20k: Génération de code structuré
   - Python Code Instructions 18k: Solutions hiérarchiques
   - SQL Create Context: Décomposition de requêtes

2. **Format unifié HRM**:
   ```json
   {
     "instruction": "Résoudre étape par étape: ...",
     "high_level_plan": ["Étape 1: Analyser", "Étape 2: Décomposer", ...],
     "low_level_steps": [
       ["sous-étape 1.1", "sous-étape 1.2"],
       ["sous-étape 2.1", "sous-étape 2.2"]
     ],
     "final_output": "Solution complète"
   }
   ```

## 📊 Métriques et Évaluation

### Benchmarks Cibles

1. **Raisonnement Structuré**
   - GSM8K accuracy: >70% (baseline: ~45%)
   - Code generation: >85% syntaxe correcte
   - Tool use accuracy: >95%

2. **Métriques HRM Spécifiques**
   - Convergence L-module: <10 steps
   - Utilisation cycles H: 2-6 selon complexité
   - Temps d'inférence: <2s par requête

### Monitoring

```python
# Métriques à tracker
metrics = {
    'hrm/l_convergence_steps': [],  # Steps avant convergence L
    'hrm/h_cycles_used': [],         # Nombre de cycles H utilisés
    'hrm/forward_residuals': [],     # Activité computationnelle
    'hrm/q_values': [],              # Décisions halt/continue
    'hrm/memory_usage': [],          # GPU memory avec O(1)
}
```

## 🛠️ Optimisations RTX 3090

1. **Mémoire**
   - Gradient approximation O(1): économise ~10GB
   - LoRA ranks adaptés: L=32, H=64
   - Flash Attention 2 activé

2. **Performance**
   - Batch size effectif: 8 (avec accumulation)
   - Mixed precision (fp16)
   - Unsloth kernel optimizations

## 📅 Planning d'Implémentation

### Sprint 1 (1 semaine)
- [x] Analyse architecture HRM
- [ ] Créer structure fichiers
- [ ] Implémenter modules H et L basiques
- [ ] Test forward pass simple

### Sprint 2 (1 semaine)
- [ ] Convergence hiérarchique complète
- [ ] 1-step gradient approximation
- [ ] Integration avec Unsloth
- [ ] Tests unitaires

### Sprint 3 (1 semaine)
- [ ] Deep supervision training
- [ ] Adaptive Computation Time
- [ ] Benchmarks initiaux
- [ ] Optimisations mémoire

### Sprint 4 (1 semaine)
- [ ] Fine-tuning sur datasets HRM
- [ ] Évaluation complète
- [ ] Documentation
- [ ] Intégration API

## 🎯 Résultats Attendus

1. **Performance**
   - Amélioration 25-40% sur raisonnement structuré
   - Réduction latence avec ACT (skip cycles inutiles)
   - Meilleure généralisation avec peu d'exemples

2. **Efficacité**
   - Training 3x plus rapide qu'avec BPTT
   - Inférence adaptative selon complexité
   - Utilisation mémoire constante

3. **Applications**
   - Agent Linux avec planification hiérarchique
   - Décomposition automatique de tâches
   - Raisonnement multi-étapes robuste

## 📝 Notes d'Implémentation

1. **Compatibilité Gemma-3N**
   - Utiliser Unsloth exclusivement
   - Adapter dimensions hidden states
   - Préserver architecture AltUp

2. **Intégration Projet**
   - Suivre structure `models/training/hrm/`
   - Réutiliser `core/settings.py`
   - Logger avec structlog

3. **Tests Critiques**
   - Convergence sur tâches simples
   - Stabilité gradient approximation
   - Performance vs baseline

---

**Prochaine étape**: Commencer par créer `hrm_modules.py` avec les classes LowLevelModule et HighLevelModule.