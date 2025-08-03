#!/usr/bin/env python3
"""
Comparison script: Standard transformers vs Unsloth for Gemma-3N-E4B training
Shows memory usage, speed, and feature differences
"""

import torch
import psutil
import GPUtil
from tabulate import tabulate


def get_system_info():
    """Get current system resource usage."""
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    # GPU info
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_memory_used = gpu.memoryUsed
        gpu_memory_total = gpu.memoryTotal
        gpu_utilization = gpu.load * 100
    else:
        gpu_memory_used = 0
        gpu_memory_total = 0
        gpu_utilization = 0
    
    return {
        "cpu_percent": cpu_percent,
        "ram_used_gb": memory.used / (1024**3),
        "ram_total_gb": memory.total / (1024**3),
        "gpu_memory_used_mb": gpu_memory_used,
        "gpu_memory_total_mb": gpu_memory_total,
        "gpu_utilization": gpu_utilization,
    }


def compare_implementations():
    """Compare standard transformers vs Unsloth implementation."""
    
    comparison_data = [
        ["Feature", "Standard (transformers + PEFT)", "Unsloth"],
        ["4-bit Quantization", "✓ (bitsandbytes)", "✓ (Optimized)"],
        ["LoRA Support", "✓", "✓ (2-4x faster)"],
        ["Memory Usage", "~15-20GB VRAM", "~10-15GB VRAM"],
        ["Training Speed", "1x (baseline)", "2-4x faster"],
        ["Flash Attention", "Manual setup", "✓ Built-in"],
        ["Gradient Checkpointing", "Standard", "Optimized"],
        ["GGUF Export", "Manual conversion", "✓ Built-in"],
        ["Mixed Precision", "fp16/bf16", "Auto-optimized"],
        ["Long Context", "Limited by memory", "RoPE scaling built-in"],
        ["Multi-GPU", "✓", "✓ (Optimized)"],
    ]
    
    print("\n🔍 Implementation Comparison: Standard vs Unsloth\n")
    print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))
    
    # Code differences
    print("\n📝 Key Code Differences:\n")
    
    print("1. Model Loading:")
    print("   Standard:")
    print("   ```python")
    print("   model = AutoModelForCausalLM.from_pretrained(")
    print("       model_name,")
    print("       quantization_config=bnb_config,")
    print("       device_map='auto'")
    print("   )")
    print("   ```")
    print("\n   Unsloth:")
    print("   ```python")
    print("   model, tokenizer = FastLanguageModel.from_pretrained(")
    print("       model_name,")
    print("       max_seq_length=2048,")
    print("       load_in_4bit=True")
    print("   )")
    print("   ```")
    
    print("\n2. LoRA Setup:")
    print("   Standard:")
    print("   ```python")
    print("   model = prepare_model_for_kbit_training(model)")
    print("   model = get_peft_model(model, lora_config)")
    print("   ```")
    print("\n   Unsloth:")
    print("   ```python")
    print("   model = FastLanguageModel.get_peft_model(")
    print("       model,")
    print("       r=32,")
    print("       target_modules=['q_proj', 'k_proj', ...],")
    print("       use_gradient_checkpointing='unsloth'")
    print("   )")
    print("   ```")
    
    print("\n3. GGUF Export:")
    print("   Standard: Requires external tools (llama.cpp)")
    print("   Unsloth:")
    print("   ```python")
    print("   model.save_pretrained_gguf('model.gguf', tokenizer)")
    print("   ```")


def benchmark_memory_usage():
    """Estimate memory usage for different configurations."""
    
    configs = [
        ["Configuration", "VRAM Usage", "Effective Batch Size", "Training Speed"],
        ["Gemma-3N (4.5B) - Standard", "~15GB", "16", "1x"],
        ["Gemma-3N (4.5B) - Unsloth", "~10GB", "16", "2-3x"],
        ["Gemma-3N + Long Context - Standard", "~20GB", "8", "0.7x"],
        ["Gemma-3N + Long Context - Unsloth", "~14GB", "16", "2x"],
    ]
    
    print("\n💾 Memory Usage Comparison:\n")
    print(tabulate(configs, headers="firstrow", tablefmt="grid"))


def show_migration_guide():
    """Show how to migrate from standard to Unsloth."""
    
    print("\n🔄 Migration Guide:\n")
    print("1. Install Unsloth:")
    print("   ```bash")
    print("   pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
    print("   pip install --no-deps 'xformers<0.0.27' 'trl<0.10.0' 'peft' 'accelerate' 'bitsandbytes'")
    print("   ```")
    
    print("\n2. Update imports:")
    print("   ```python")
    print("   # Add:")
    print("   from unsloth import FastLanguageModel")
    print("   # Remove:")
    print("   # from peft import prepare_model_for_kbit_training")
    print("   ```")
    
    print("\n3. Use the new script:")
    print("   ```bash")
    print("   # Instead of:")
    print("   python training/qlora_finetune.py ...")
    print("   # Use:")
    print("   python training/qlora_finetune_unsloth.py ...")
    print("   ```")
    
    print("\n4. Same command line arguments work!")


def main():
    """Run comparison analysis."""
    print("=" * 70)
    print("Gemma-3N-E4B Training: Standard vs Unsloth Comparison")
    print("=" * 70)
    
    # Show system info
    info = get_system_info()
    print(f"\n💻 System Info:")
    print(f"   CPU Usage: {info['cpu_percent']:.1f}%")
    print(f"   RAM: {info['ram_used_gb']:.1f}/{info['ram_total_gb']:.1f} GB")
    print(f"   GPU Memory: {info['gpu_memory_used_mb']:.0f}/{info['gpu_memory_total_mb']:.0f} MB")
    print(f"   GPU Utilization: {info['gpu_utilization']:.1f}%")
    
    # Run comparisons
    compare_implementations()
    benchmark_memory_usage()
    show_migration_guide()
    
    print("\n✅ Summary:")
    print("   - Unsloth provides 2-4x speedup for QLoRA training")
    print("   - Lower memory usage allows larger batch sizes")
    print("   - Built-in GGUF export simplifies deployment")
    print("   - Drop-in replacement for existing training script")


if __name__ == "__main__":
    main()