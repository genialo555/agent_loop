#!/usr/bin/env python3
"""
Script d'entraînement optimisé avec gestion intelligente du cache
"""
import os
import sys
import torch
import gc
from pathlib import Path

# Configuration optimisée pour éviter la saturation du cache
os.environ['HF_DATASETS_CACHE'] = '/media/jerem/jeux&travail/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/media/jerem/jeux&travail/ml_models'
os.environ['HF_HOME'] = '/media/jerem/jeux&travail/ml_models'

# Limiter l'utilisation mémoire
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

# Désactiver le cache de datasets pour économiser l'espace
os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE'] = '5000000000'  # 5GB max en RAM

# Forcer le garbage collection
gc.collect()
torch.cuda.empty_cache()

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.qlora_finetune import main as train_main
from training.qlora_config import get_qlora_config

def optimize_config():
    """Optimise la configuration pour l'espace disque limité"""
    config = get_qlora_config()
    
    # Réduire la taille du batch pour économiser la mémoire GPU
    config['training_args'].per_device_train_batch_size = 4  # Réduit de 8 à 4
    config['training_args'].gradient_accumulation_steps = 4  # Pour garder batch effectif à 16
    
    # Optimisations mémoire
    config['training_args'].fp16 = False
    config['training_args'].bf16 = True  # Plus stable
    config['training_args'].optim = "paged_adamw_8bit"  # Optimiseur 8-bit
    config['training_args'].gradient_checkpointing = True
    
    # Sauvegarder moins souvent pour économiser l'espace
    config['training_args'].save_steps = 500  # Au lieu de 100
    config['training_args'].save_total_limit = 2  # Garder seulement 2 checkpoints
    
    # Logging moins fréquent
    config['training_args'].logging_steps = 50
    config['training_args'].eval_steps = 500
    
    # Désactiver le cache de tokenizer
    config['training_args'].use_cache = False
    
    # Utiliser moins de workers pour le data loading
    config['training_args'].dataloader_num_workers = 2
    
    return config

def cleanup_before_training():
    """Nettoie avant l'entraînement"""
    print("🧹 Nettoyage pré-entraînement...")
    
    # Vider le cache PyTorch
    torch.cuda.empty_cache()
    
    # Supprimer les anciens checkpoints temporaires
    checkpoint_dir = Path("/home/jerem/agent_loop/checkpoints")
    if checkpoint_dir.exists():
        for old_checkpoint in checkpoint_dir.glob("checkpoint-*"):
            if old_checkpoint.is_dir():
                print(f"Suppression de {old_checkpoint}")
                import shutil
                shutil.rmtree(old_checkpoint)
    
    # Afficher l'espace disponible
    import subprocess
    print("\n📊 Espace disque disponible:")
    subprocess.run(["df", "-h", "/media/jerem/jeux&travail", "/home/jerem"])
    
    # Afficher l'utilisation GPU
    print("\n🎮 État GPU:")
    subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv"])

def main():
    """Lance l'entraînement optimisé"""
    print("🚀 Lancement de l'entraînement optimisé pour espace limité\n")
    
    # Nettoyer avant de commencer
    cleanup_before_training()
    
    # Obtenir la configuration optimisée
    config = optimize_config()
    
    print("\n⚙️  Configuration optimisée:")
    print(f"- Batch size: {config['training_args'].per_device_train_batch_size}")
    print(f"- Gradient accumulation: {config['training_args'].gradient_accumulation_steps}")
    print(f"- Save steps: {config['training_args'].save_steps}")
    print(f"- Checkpoints limit: {config['training_args'].save_total_limit}")
    print(f"- Optimizer: {config['training_args'].optim}")
    
    try:
        # Lancer l'entraînement
        print("\n🎯 Démarrage de l'entraînement...")
        train_main()
    except torch.cuda.OutOfMemoryError:
        print("\n❌ Erreur mémoire GPU! Tentative avec batch_size=2...")
        config['training_args'].per_device_train_batch_size = 2
        config['training_args'].gradient_accumulation_steps = 8
        gc.collect()
        torch.cuda.empty_cache()
        train_main()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        raise
    finally:
        # Nettoyer après l'entraînement
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()