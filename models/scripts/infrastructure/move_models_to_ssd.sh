#!/bin/bash
# Script pour déplacer les modèles HuggingFace vers SSD et créer liens symboliques
# Usage: ./move_models_to_ssd.sh [destination_path]

set -e

# Configuration
HF_CACHE_DIR="$HOME/.cache/huggingface"
DEFAULT_SSD_PATH="/media/jerem/641C8D6C1C8D3A56/ml_models"
SSD_PATH="${1:-$DEFAULT_SSD_PATH}"
BACKUP_DIR="$HOME/backup_hf_cache_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Migration des modèles HuggingFace vers SSD"
echo "Source: $HF_CACHE_DIR"
echo "Destination: $SSD_PATH"
echo "Backup: $BACKUP_DIR"

# Vérifications préliminaires
if [ ! -d "$HF_CACHE_DIR" ]; then
    echo "❌ Cache HuggingFace introuvable: $HF_CACHE_DIR"
    exit 1
fi

if [ ! -d "$(dirname "$SSD_PATH")" ]; then
    echo "❌ Répertoire SSD parent introuvable: $(dirname "$SSD_PATH")"
    exit 1
fi

# Calcul de l'espace requis
CACHE_SIZE=$(du -sb "$HF_CACHE_DIR" | cut -f1)
CACHE_SIZE_GB=$((CACHE_SIZE / 1024 / 1024 / 1024))
SSD_AVAILABLE=$(df --output=avail -B1 "$(dirname "$SSD_PATH")" | tail -1)
SSD_AVAILABLE_GB=$((SSD_AVAILABLE / 1024 / 1024 / 1024))

echo "📊 Espace requis: ${CACHE_SIZE_GB}GB"
echo "📊 Espace disponible SSD: ${SSD_AVAILABLE_GB}GB"

if [ $CACHE_SIZE -gt $SSD_AVAILABLE ]; then
    echo "❌ Espace insuffisant sur le SSD"
    exit 1
fi

# Confirmation utilisateur
read -p "Continuer la migration ? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Migration annulée"
    exit 0
fi

echo "🔄 Début de la migration..."

# 1. Créer backup du cache actuel (liens seulement)
echo "📦 Création du backup des liens..."
mkdir -p "$BACKUP_DIR"
cp -r "$HF_CACHE_DIR" "$BACKUP_DIR/" 2>/dev/null || true

# 2. Créer le répertoire de destination
echo "📁 Création du répertoire SSD..."
mkdir -p "$SSD_PATH"

# 3. Déplacer les modèles par type
echo "🔄 Migration des modèles Gemma..."
if [ -d "$HF_CACHE_DIR/hub" ]; then
    mkdir -p "$SSD_PATH/hub"
    
    # Déplacer les gros modèles Gemma en premier
    for gemma_dir in "$HF_CACHE_DIR/hub/models--google--gemma"*; do
        if [ -d "$gemma_dir" ]; then
            model_name=$(basename "$gemma_dir")
            echo "  → Déplacement de $model_name..."
            
            # Déplacer vers SSD
            mv "$gemma_dir" "$SSD_PATH/hub/"
            
            # Créer lien symbolique
            ln -sf "$SSD_PATH/hub/$model_name" "$gemma_dir"
        fi
    done
fi

# 4. Déplacer les datasets volumineux
echo "🔄 Migration des datasets..."
if [ -d "$HF_CACHE_DIR/hub" ]; then
    for dataset_dir in "$HF_CACHE_DIR/hub/datasets--"*; do
        if [ -d "$dataset_dir" ]; then
            dataset_size=$(du -sb "$dataset_dir" | cut -f1)
            dataset_size_mb=$((dataset_size / 1024 / 1024))
            
            # Déplacer si > 100MB
            if [ $dataset_size_mb -gt 100 ]; then
                dataset_name=$(basename "$dataset_dir")
                echo "  → Déplacement de $dataset_name (${dataset_size_mb}MB)..."
                
                mv "$dataset_dir" "$SSD_PATH/hub/"
                ln -sf "$SSD_PATH/hub/$dataset_name" "$dataset_dir"
            fi
        fi
    done
fi

# 5. Mettre à jour les variables d'environnement pour agent_loop
echo "⚙️ Configuration de l'environnement agent_loop..."
ENV_FILE="/home/jerem/agent_loop/.env"

# Créer/mettre à jour .env
cat >> "$ENV_FILE" << EOF

# Configuration SSD pour modèles ($(date))
HF_HOME=$SSD_PATH
TRANSFORMERS_CACHE=$SSD_PATH/transformers
HF_DATASETS_CACHE=$SSD_PATH/datasets
TORCH_HOME=$SSD_PATH/torch

# Pour Docker - volumes vers SSD
MODELS_SSD_PATH=$SSD_PATH
EOF

# 6. Mettre à jour docker-compose.training.yml
echo "🐳 Mise à jour de la configuration Docker..."
if [ -f "/home/jerem/agent_loop/docker-compose.training.yml" ]; then
    # Backup du fichier docker-compose
    cp "/home/jerem/agent_loop/docker-compose.training.yml" "/home/jerem/agent_loop/docker-compose.training.yml.backup"
    
    # Ajouter les volumes SSD
    cat >> "/home/jerem/agent_loop/docker-compose.training.yml" << EOF

  # Volumes SSD - Ajouté par move_models_to_ssd.sh
  volumes:
    training_models_ssd:
      driver: local
      driver_opts:
        type: none
        o: bind
        device: $SSD_PATH
EOF
fi

# 7. Vérification finale
echo "✅ Vérification des liens..."
BROKEN_LINKS=0
for link in $(find "$HF_CACHE_DIR" -type l); do
    if [ ! -e "$link" ]; then
        echo "⚠️  Lien cassé: $link"
        BROKEN_LINKS=$((BROKEN_LINKS + 1))
    fi
done

# 8. Rapport final
echo ""
echo "🎉 Migration terminée !"
echo "📊 Rapport final:"
echo "  - Modèles déplacés vers: $SSD_PATH"
echo "  - Espace libéré: ~${CACHE_SIZE_GB}GB"
echo "  - Liens symboliques créés: $(find "$HF_CACHE_DIR" -type l | wc -l)"
echo "  - Liens cassés: $BROKEN_LINKS"
echo "  - Backup disponible: $BACKUP_DIR"

if [ $BROKEN_LINKS -eq 0 ]; then
    echo "✅ Tous les liens sont fonctionnels !"
else
    echo "⚠️  $BROKEN_LINKS liens cassés détectés"
fi

echo ""
echo "📝 Actions suivantes recommandées:"
echo "1. Tester le training: cd /home/jerem/agent_loop && make train-dev"
echo "2. Vérifier les modèles: ls -la $SSD_PATH/hub/"
echo "3. Si tout fonctionne, supprimer le backup: rm -rf $BACKUP_DIR"

echo ""
echo "🚀 Configuration SSD opérationnelle pour Agent Loop !"