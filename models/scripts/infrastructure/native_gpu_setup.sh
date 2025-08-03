#!/bin/bash
# Configuration native pour utiliser le GPU sans Docker

set -e

echo "Configuration de l'environnement natif GPU..."

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Création de l'environnement virtuel..."
    python3.11 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install PyTorch with CUDA support
echo "Installation de PyTorch avec support CUDA..."
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo "Installation des dépendances..."
pip install -r requirements.txt
if [ -f "requirements-training.txt" ]; then
    pip install -r requirements-training.txt
fi

# Create necessary directories
echo "Création des répertoires nécessaires..."
mkdir -p models/{checkpoints,cache,gguf}
mkdir -p datasets logs outputs

# Verify GPU access
echo "Vérification de l'accès GPU..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo "Configuration native terminée!"
echo ""
echo "Pour démarrer l'application:"
echo "  source venv/bin/activate"
echo "  uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "Pour lancer un entraînement:"
echo "  source venv/bin/activate"
echo "  python training/qlora_finetune.py --data datasets/processed --base models/gguf/gemma_base.gguf"