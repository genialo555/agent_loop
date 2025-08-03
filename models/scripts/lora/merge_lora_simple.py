#!/usr/bin/env python3
"""
Script simple pour fusionner les poids LoRA avec le modèle de base
Crée un modèle complet prêt à l'emploi
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def merge_lora_with_base():
    """Fusionne les poids LoRA HRM avec le modèle Gemma-3N de base"""
    
    # Chemins
    LORA_PATH = "/home/jerem/agent_loop/models/results/gemma-3n-hrm-test-20250801_015252"
    BASE_MODEL = "/media/jerem/641C8D6C1C8D3A56/MLLMODELS/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee"
    OUTPUT_PATH = "/media/jerem/641C8D6C1C8D3A56/MLLMODELS/gemma-3n-hrm-merged"
    
    logger.info("🔄 Fusion des poids LoRA avec le modèle de base")
    logger.info(f"📁 LoRA: {LORA_PATH}")
    logger.info(f"📁 Base: {BASE_MODEL}")
    logger.info(f"📁 Output: {OUTPUT_PATH}")
    
    try:
        # Importer Unsloth
        from unsloth import FastLanguageModel
        import torch
        
        # Vérifier que les chemins existent
        if not Path(LORA_PATH).exists():
            logger.error(f"❌ Chemin LoRA introuvable: {LORA_PATH}")
            return False
            
        if not Path(BASE_MODEL).exists():
            logger.error(f"❌ Modèle de base introuvable: {BASE_MODEL}")
            return False
        
        # Créer le dossier de sortie
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        logger.info("⏳ Chargement du modèle avec LoRA...")
        
        # Charger le modèle + LoRA
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=LORA_PATH,
            dtype=torch.float16,
            load_in_4bit=False,  # Charger en 16bit pour la fusion
        )
        
        logger.info("🔨 Fusion en cours...")
        
        # Fusionner et sauvegarder en 16bit
        model.save_pretrained_merged(
            OUTPUT_PATH,
            tokenizer,
            save_method="merged_16bit",
        )
        
        # Sauvegarder aussi le tokenizer
        tokenizer.save_pretrained(OUTPUT_PATH)
        
        # Créer un fichier info
        info = {
            "base_model": BASE_MODEL,
            "lora_adapter": LORA_PATH,
            "merged_at": datetime.now().isoformat(),
            "merge_method": "merged_16bit",
            "model_type": "gemma-3n-hrm",
            "training_dataset": "gsm8k",
            "training_steps": 935,
            "description": "Gemma-3N with HRM (Hierarchical Reasoning Model) training on GSM8K"
        }
        
        import json
        with open(f"{OUTPUT_PATH}/merge_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        logger.info("✅ Fusion terminée avec succès!")
        logger.info(f"📁 Modèle fusionné sauvegardé dans: {OUTPUT_PATH}")
        logger.info(f"📊 Taille du modèle: ~9GB (float16)")
        
        # Afficher les fichiers créés
        logger.info("\n📂 Fichiers créés:")
        for file in Path(OUTPUT_PATH).glob("*"):
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"   - {file.name}: {size_mb:.1f} MB")
        
        return True
        
    except ImportError:
        logger.error("❌ Unsloth n'est pas installé. Exécutez: pip install unsloth")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur lors de la fusion: {e}")
        return False

def main():
    """Point d'entrée principal"""
    logger.info("=== FUSION LORA + MODÈLE DE BASE ===")
    logger.info("Modèle: Gemma-3N-HRM (Hierarchical Reasoning)")
    
    # Vérifier l'environnement
    if not Path(".venv/bin/activate").exists():
        logger.warning("⚠️  Environnement virtuel non activé")
        logger.info("   Activez avec: source .venv/bin/activate")
    
    # Lancer la fusion
    success = merge_lora_with_base()
    
    if success:
        logger.info("\n🎉 Succès! Prochaines étapes:")
        logger.info("1. Tester le modèle fusionné avec transformers")
        logger.info("2. Convertir en GGUF pour Ollama (optionnel)")
        logger.info("3. Utiliser pour l'inférence ou continuer l'entraînement")
        return 0
    else:
        logger.error("\n❌ La fusion a échoué")
        return 1

if __name__ == "__main__":
    sys.exit(main())