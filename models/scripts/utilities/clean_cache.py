#!/usr/bin/env python3
"""
Script d'urgence pour nettoyer les caches ML et libérer de l'espace disque
"""
import os
import shutil
from pathlib import Path
import subprocess
from datetime import datetime, timedelta

def get_size_gb(path):
    """Get directory size in GB"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024**3)

def clean_huggingface_cache():
    """Clean old HuggingFace cache files"""
    cache_dir = Path.home() / ".cache" / "huggingface"
    if not cache_dir.exists():
        print("❌ Cache HuggingFace non trouvé")
        return
    
    print(f"📊 Taille du cache HuggingFace: {get_size_gb(cache_dir):.2f} GB")
    
    # Nettoyer les anciens fichiers de plus de 7 jours
    cutoff_date = datetime.now() - timedelta(days=7)
    removed_size = 0
    
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            file_path = Path(root) / file
            try:
                # Vérifier l'âge du fichier
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    size = file_path.stat().st_size
                    file_path.unlink()
                    removed_size += size
            except Exception as e:
                print(f"⚠️  Erreur suppression {file}: {e}")
    
    print(f"✅ Libéré {removed_size / (1024**3):.2f} GB du cache HuggingFace")

def clean_pip_cache():
    """Clean pip cache"""
    try:
        result = subprocess.run(["pip", "cache", "purge"], capture_output=True, text=True)
        print("✅ Cache pip nettoyé")
    except Exception as e:
        print(f"⚠️  Erreur nettoyage pip: {e}")

def clean_torch_cache():
    """Clean PyTorch cache"""
    torch_cache = Path.home() / ".cache" / "torch"
    if torch_cache.exists():
        size_before = get_size_gb(torch_cache)
        try:
            shutil.rmtree(torch_cache / "hub" / "checkpoints", ignore_errors=True)
            print(f"✅ Cache PyTorch nettoyé ({size_before:.2f} GB)")
        except Exception as e:
            print(f"⚠️  Erreur nettoyage PyTorch: {e}")

def setup_cache_limits():
    """Configure cache size limits"""
    config_file = Path.home() / ".cache" / "huggingface" / "config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    config = {
        "cache_dir": "/media/jerem/jeux&travail/ml_models",
        "max_cache_size": "100GB",
        "symlink_to_cache": True
    }
    
    import json
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration des limites de cache mise à jour")

def show_disk_usage():
    """Show current disk usage"""
    print("\n📊 Utilisation disque actuelle:")
    subprocess.run(["df", "-h", "/home/jerem", "/media/jerem/jeux&travail"])

if __name__ == "__main__":
    print("🧹 Nettoyage d'urgence des caches ML\n")
    
    show_disk_usage()
    print("\n🔥 Début du nettoyage...\n")
    
    clean_huggingface_cache()
    clean_pip_cache()
    clean_torch_cache()
    setup_cache_limits()
    
    print("\n📊 Après nettoyage:")
    show_disk_usage()
    
    print("\n✅ Nettoyage terminé!")
    print("💡 Conseil: Exécutez ce script régulièrement pour éviter la saturation")