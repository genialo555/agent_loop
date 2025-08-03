#!/usr/bin/env python3
"""
Script pour fusionner les poids LoRA avec le modèle de base et convertir en GGUF pour Ollama
Usage: python merge_and_convert_lora.py --lora-path /path/to/lora --output-name model-name
"""

import argparse
import os
import sys
from pathlib import Path
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append("/home/jerem/agent_loop")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_lora_weights(base_model_path: str, lora_path: str, output_path: str):
    """Fusionne les poids LoRA avec le modèle de base"""
    try:
        from unsloth import FastLanguageModel
        import torch
        
        logger.info(f"Chargement du modèle avec LoRA depuis: {lora_path}")
        
        # Charger le modèle avec les poids LoRA
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        
        logger.info("Fusion des poids LoRA avec le modèle de base...")
        
        # Fusionner et sauvegarder
        model.save_pretrained_merged(
            output_path,
            tokenizer,
            save_method="merged_16bit",  # Sauvegarde en 16bit
        )
        
        logger.info(f"✅ Modèle fusionné sauvegardé dans: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la fusion: {e}")
        return False

def convert_to_gguf(model_path: str, output_gguf: str, quantization: str = "q4_k_m"):
    """Convertit le modèle en format GGUF pour Ollama"""
    logger.info(f"Conversion en GGUF avec quantization {quantization}...")
    
    # Utiliser llama.cpp pour la conversion
    convert_script = "/home/jerem/agent_loop/infrastructure/external_resources/llama.cpp/convert.py"
    quantize_bin = "/home/jerem/agent_loop/infrastructure/external_resources/llama.cpp/quantize"
    
    if not Path(convert_script).exists():
        logger.warning("llama.cpp non trouvé. Installation...")
        os.system("""
        cd /home/jerem/agent_loop/infrastructure/external_resources && \
        git clone https://github.com/ggerganov/llama.cpp && \
        cd llama.cpp && make
        """)
    
    # Conversion en GGUF
    temp_gguf = output_gguf.replace('.gguf', '-f32.gguf')
    
    cmd = f"""
    python {convert_script} {model_path} \
        --outfile {temp_gguf} \
        --outtype f32
    """
    
    logger.info("Étape 1: Conversion en GGUF F32...")
    result = os.system(cmd)
    
    if result == 0:
        # Quantization
        logger.info(f"Étape 2: Quantization en {quantization}...")
        cmd_quant = f"{quantize_bin} {temp_gguf} {output_gguf} {quantization}"
        result = os.system(cmd_quant)
        
        # Nettoyer le fichier temporaire
        if Path(temp_gguf).exists():
            os.remove(temp_gguf)
            
        if result == 0:
            logger.info(f"✅ Modèle GGUF créé: {output_gguf}")
            return True
    
    logger.error("❌ Échec de la conversion GGUF")
    return False

def create_modelfile(model_name: str, gguf_path: str):
    """Crée un Modelfile pour Ollama"""
    modelfile_content = f"""FROM {gguf_path}

# Modèle Gemma-3N avec entraînement HRM
# Entraîné sur GSM8K pour le raisonnement hiérarchique

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

TEMPLATE """{{ .System }}
{{ .Prompt }}"""

SYSTEM """You are an AI assistant trained in hierarchical reasoning. Break down complex problems into clear steps:
1. ANALYZE - Understand the problem
2. PLAN - Break into sub-tasks  
3. EXECUTE - Step-by-step solution
4. VERIFY - Check the result"""
"""
    
    modelfile_path = f"/home/jerem/agent_loop/models/ollama/{model_name}.Modelfile"
    os.makedirs(os.path.dirname(modelfile_path), exist_ok=True)
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    logger.info(f"✅ Modelfile créé: {modelfile_path}")
    return modelfile_path

def import_to_ollama(model_name: str, modelfile_path: str):
    """Importe le modèle dans Ollama"""
    logger.info(f"Import dans Ollama sous le nom: {model_name}")
    
    cmd = f"ollama create {model_name} -f {modelfile_path}"
    result = os.system(cmd)
    
    if result == 0:
        logger.info(f"✅ Modèle importé dans Ollama: {model_name}")
        logger.info(f"   Tester avec: ollama run {model_name}")
        return True
    else:
        logger.error("❌ Échec de l'import dans Ollama")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fusionner LoRA et convertir pour Ollama")
    parser.add_argument("--lora-path", required=True, help="Chemin vers les poids LoRA")
    parser.add_argument("--base-model", default="/media/jerem/641C8D6C1C8D3A56/MLLMODELS/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee", 
                        help="Chemin vers le modèle de base")
    parser.add_argument("--output-name", required=True, help="Nom du modèle de sortie")
    parser.add_argument("--quantization", default="q4_k_m", help="Type de quantization (q4_k_m, q5_k_m, q8_0)")
    parser.add_argument("--skip-merge", action="store_true", help="Passer l'étape de fusion")
    parser.add_argument("--skip-ollama", action="store_true", help="Ne pas importer dans Ollama")
    
    args = parser.parse_args()
    
    # Chemins
    merged_path = f"/media/jerem/641C8D6C1C8D3A56/MLLMODELS/{args.output_name}-merged"
    gguf_path = f"/media/jerem/641C8D6C1C8D3A56/MLLMODELS/{args.output_name}.gguf"
    
    logger.info("=== Merge & Convert LoRA to Ollama ===")
    logger.info(f"LoRA: {args.lora_path}")
    logger.info(f"Output: {args.output_name}")
    
    # 1. Fusionner LoRA avec le modèle de base
    if not args.skip_merge:
        success = merge_lora_weights(args.base_model, args.lora_path, merged_path)
        if not success:
            return 1
    else:
        merged_path = args.lora_path
    
    # 2. Convertir en GGUF
    success = convert_to_gguf(merged_path, gguf_path, args.quantization)
    if not success:
        return 1
    
    # 3. Importer dans Ollama
    if not args.skip_ollama:
        modelfile = create_modelfile(args.output_name, gguf_path)
        success = import_to_ollama(args.output_name, modelfile)
        if not success:
            return 1
    
    # Sauvegarder les infos
    info = {
        "name": args.output_name,
        "lora_source": args.lora_path,
        "base_model": args.base_model,
        "merged_path": merged_path,
        "gguf_path": gguf_path,
        "quantization": args.quantization,
        "created_at": datetime.now().isoformat()
    }
    
    info_path = f"/home/jerem/agent_loop/models/ollama/{args.output_name}_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info("\n✅ Processus terminé avec succès!")
    logger.info(f"📁 Modèle fusionné: {merged_path}")
    logger.info(f"📦 Fichier GGUF: {gguf_path}")
    logger.info(f"🚀 Tester avec: ollama run {args.output_name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())