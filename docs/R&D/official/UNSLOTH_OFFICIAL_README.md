# Unsloth Official Documentation

Source: https://github.com/unslothai/unsloth

## Overview

Unsloth is an open-source library for efficient machine learning model fine-tuning and reinforcement learning, offering significant performance improvements over standard implementations.

## Mission Statement

"Train your own model with Unsloth, an open-source framework for LLM fine-tuning and reinforcement learning"

Goal: Make AI "as accurate and accessible as possible"

## Key Features

- **2x faster training** compared to standard implementations
- **80% less VRAM usage** enabling larger models on consumer GPUs
- **0% loss in accuracy** - maintains model quality while optimizing performance
- Supports full-finetuning, 4-bit, 8-bit, and 16-bit training
- Compatible with all transformer-style models including TTS, STT, and multimodal models
- All kernels written in OpenAI's Triton language
- Manual backpropagation engine for optimal memory efficiency
- Works on NVIDIA GPUs from 2018 onwards
- Supports Linux and Windows, Google Colab

## Installation

### Linux (Recommended)
```bash
pip install unsloth
```

### Windows
```bash
pip install unsloth
```

### Google Colab
Unsloth is fully compatible with Google Colab's free tier GPUs.

## Supported Models

### Language Models
- **Gemma Family**: Gemma 3n (E2B, E4B), Gemma 3, Gemma 2
- **Llama Family**: Llama 3.1, 3.2, 3.3 (all sizes)
- **Qwen Family**: Qwen3, Qwen2.5
- **Mistral Models**: Mistral v0.3, Mixtral
- **Phi Family**: Phi-3, Phi-4
- **DeepSeek Models**: DeepSeek-R1

### Multimodal Models
- Vision models (Llama 3.2 Vision, Qwen2-VL)
- Text-to-Speech models
- Speech-to-Text models

## Basic Usage Example

```python
from unsloth import FastLanguageModel, FastModel
from trl import SFTTrainer, SFTConfig

# Load model with 4-bit quantization
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4B-it",
    max_seq_length = 2048,
    load_in_4bit = True
)

# Configure LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,  # Optimized to 0
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # 4x longer contexts
    random_state = 3407,
)

# Training configuration continues...
```

## Advanced Features

### 1. Continued Pre-training
- Supports training on `lm_head` and `embed_tokens`
- Allows domain adaptation

### 2. LoRA Adapter Management
- Save and load LoRA adapters separately
- Merge adapters with base model
- Continue training from saved adapters

### 3. Multiple Export Formats
- 16-bit precision models
- GGUF format for llama.cpp/Ollama
- Safetensors format
- Quantized models (4-bit, 8-bit)

### 4. Training Optimizations
- Train on completions/responses only
- Early stopping support
- Gradient checkpointing variants
- Custom chat templates

### 5. Token Management
- Add new tokens to vocabulary
- Resize embeddings automatically
- Handle special tokens properly

## Memory Optimization Techniques

1. **4-bit Quantization**
   - Reduces model size by ~75%
   - Minimal accuracy loss
   - Enables larger models on consumer GPUs

2. **Gradient Checkpointing**
   - `use_gradient_checkpointing = "unsloth"`
   - Enables 4x longer context lengths
   - Trades compute for memory

3. **Paged Optimizers**
   - Use `paged_adamw_8bit` for large models
   - Reduces optimizer memory footprint

## Performance Benchmarks

| Model | Standard | Unsloth | Speedup | VRAM Savings |
|-------|----------|---------|---------|--------------|
| Llama 3 8B | 100% | 200% | 2x | 70% |
| Gemma 3 4B | 100% | 190% | 1.9x | 75% |
| Mistral 7B | 100% | 210% | 2.1x | 80% |

## Wiki Resources

The Unsloth Wiki (https://github.com/unslothai/unsloth/wiki) provides:

- Detailed installation guides
- Hardware-specific optimizations
- Troubleshooting common issues
- Advanced training techniques
- Model conversion guides
- Integration with inference engines

## Documentation Structure

Main documentation available at: https://docs.unsloth.ai/

- **Get Started**: Installation and quickstart
- **Basics**: Core concepts and usage
- **Model Guides**: Specific guides for each model family
- **Fine-tuning**: Detailed fine-tuning tutorials
- **Reinforcement Learning**: RL training guides
- **Deployment**: Model export and deployment

## Community and Support

- **GitHub**: https://github.com/unslothai/unsloth
- **Discord**: Active community for support
- **Documentation**: https://docs.unsloth.ai/
- **Issues**: GitHub issue tracker for bugs and features

## Unique Capabilities

1. **Bug Fixes**: Collaborates with model teams to fix critical bugs
2. **Custom Kernels**: All operations use optimized Triton kernels
3. **Flexible Training**: Supports various training paradigms
4. **Ecosystem Integration**: Works with HuggingFace, Ollama, MLX, etc.

## License

Open-source under Apache 2.0 license. Free for commercial use.

## Citation

If you use Unsloth in your research, please cite:
```
@software{unsloth,
  title = {Unsloth: Efficient LLM Fine-tuning},
  author = {Unsloth AI},
  year = {2024},
  url = {https://github.com/unslothai/unsloth}
}
```