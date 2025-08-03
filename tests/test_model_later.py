#!/usr/bin/env python3
"""
Script de test pour évaluer le modèle Gemma-3N fine-tuné
À utiliser APRÈS l'entraînement (au moins 5000 steps avec loss < 1.2)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

# Test basique de navigation OS
test_prompts = [
    # Test 1: Commande simple
    "List all Python files in the current directory and subdirectories",
    
    # Test 2: Debug basique  
    "Find and display the last 10 error messages in the system logs",
    
    # Test 3: Analyse de performance
    "Check which process is using the most CPU and memory right now",
    
    # Test 4: Gestion de fichiers
    "Find all files larger than 100MB in /home and sort them by size",
    
    # Test 5: Tâche complexe
    "Create a backup script that archives all .log files older than 7 days to /backup/ with today's date"
]

def test_model(model_path):
    """Test le modèle fine-tuné"""
    print(f"🔍 Loading model from: {model_path}")
    
    # Charger le modèle et tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("✅ Model loaded!\n")
    
    # Tester chaque prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"{'='*60}")
        print(f"Test {i}: {prompt}")
        print(f"{'='*60}")
        
        # Format du prompt système
        full_prompt = f"""You are an AI assistant that helps with Linux system administration tasks.
Provide the exact commands needed to accomplish the following task:

Task: {prompt}

Commands:"""
        
        # Tokenizer et générer
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extraire seulement la partie générée
        response = response.split("Commands:")[-1].strip()
        
        print(f"Response:\n{response}\n")
        
        # Évaluation basique
        has_command = any(cmd in response.lower() for cmd in ['find', 'ls', 'grep', 'ps', 'du', 'tar'])
        print(f"✓ Contains command: {'Yes' if has_command else 'No'}")
        print()

def check_training_status():
    """Vérifie si le modèle est prêt pour les tests"""
    try:
        import json
        with open("results/gemma-3n-safe-1epoch/trainer_state.json", "r") as f:
            state = json.load(f)
        
        last_loss = state['log_history'][-1].get('loss', 999)
        total_steps = state['global_step']
        
        print(f"📊 Training Status:")
        print(f"   Steps: {total_steps}")
        print(f"   Last Loss: {last_loss:.4f}")
        
        if total_steps < 5000:
            print(f"⚠️  Model needs more training! (Current: {total_steps}, Recommended: >5000)")
            return False
        elif last_loss > 1.5:
            print(f"⚠️  Loss still high! (Current: {last_loss:.4f}, Recommended: <1.2)")
            return False
        else:
            print("✅ Model ready for testing!")
            return True
            
    except:
        print("❌ No training state found. Model not ready.")
        return False

if __name__ == "__main__":
    print("🚀 Gemma-3N Agent Testing Script\n")
    
    # Vérifier si le modèle est prêt
    if not check_training_status():
        print("\n⏳ Come back later when the model has trained more!")
        print("💡 Run this command to check progress:")
        print("   tail -f results/gemma-3n-safe-1epoch/trainer_state.json | grep loss")
    else:
        # Chemin du modèle sauvegardé
        model_path = "./results/gemma-3n-safe-1epoch"
        test_model(model_path)
        
    print(f"\n🕐 Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")