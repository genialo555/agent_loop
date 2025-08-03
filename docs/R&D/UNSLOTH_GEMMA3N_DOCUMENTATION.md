# Unsloth & Gemma 3N Documentation

## Table of Contents
1. [Unsloth Overview](#unsloth-overview)
2. [Gemma 3N Architecture](#gemma-3n-architecture)
3. [Installation & Setup](#installation--setup)
4. [Fine-tuning Guide](#fine-tuning-guide)
5. [Performance Optimizations](#performance-optimizations)
6. [Troubleshooting](#troubleshooting)

## Unsloth Overview

Unsloth is a high-performance fine-tuning and reinforcement learning framework for Large Language Models (LLMs). It's specifically optimized for efficient training with reduced VRAM usage.

### Key Features
- **2x faster training** compared to standard implementations
- **70-80% less VRAM usage** enabling larger models on consumer GPUs
- **0% accuracy loss** - maintains model quality while optimizing performance
- **Float16 support** for Gemma 3 on Tesla T4, RTX 20x series, and V100 GPUs
- **Written in Triton** - All kernels implemented in OpenAI's Triton language
- **Manual backpropagation** engine for optimal memory efficiency

### Supported Models
- Gemma 3n (2B, 4B)
- Gemma 3 (1B, 4B, 12B, 27B)
- Qwen3, Llama 4, Phi-4, Mistral
- Vision models (Llama 3.2 Vision)
- Text-to-Speech models

### Installation

```bash
# For Linux (recommended)
pip install unsloth

# For our project specifically
pip install unsloth
```

## Gemma 3N Architecture

Gemma 3N introduces several groundbreaking architectural innovations specifically designed for edge deployment and mobile-first AI.

### MatFormer (Matryoshka Transformer)

The core innovation of Gemma 3N is the **MatFormer architecture** - a nested transformer that enables elastic inference:

```
┌─────────────────────────────────────┐
│         Gemma 3N E4B Model          │
│  ┌─────────────────────────────┐   │
│  │    Nested E2B Sub-model     │   │
│  │  (Simultaneously trained)   │   │
│  └─────────────────────────────┘   │
│     Additional E4B parameters       │
└─────────────────────────────────────┘
```

#### Key Benefits:
1. **Elastic Execution**: Dynamically switch between E4B and E2B inference paths
2. **Mix-n-Match**: Create custom-sized models between E2B and E4B
3. **Pre-extracted Models**: Both E4B and E2B available for direct use

### AltUp Architecture

AltUp (Alternating Updates) is a parameter-efficient architecture that:
- Reduces computational requirements
- Maintains model quality
- Enables efficient on-device deployment

Paper: https://arxiv.org/abs/2301.13310

### Per-Layer Embeddings (PLE)

PLE dramatically improves model quality without increasing memory footprint:
- Tailored for on-device deployment
- Optimized for GPU/TPU accelerators
- Memory-efficient representation learning

### LAuReL (Learned Augmented Residual Layers)

Specifically using LAuReL-LR (Low Rank) variety:
- Improves model efficiency
- Reduces parameter count
- Maintains performance quality

Paper: https://arxiv.org/abs/2411.07501

### Technical Specifications

| Model | Parameters | Context Length | Languages | Modalities |
|-------|------------|----------------|-----------|------------|
| E2B   | 2B effective | 32K | 140 | Text, Audio, Video, Image |
| E4B   | 4B effective | 32K | 140 | Text, Audio, Video, Image |

## Installation & Setup

### Prerequisites
- Python 3.8+
- NVIDIA GPU (2018 or newer)
- CUDA 11.8+ (for optimal performance)
- Linux or Windows OS

### Project-Specific Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Set Python path
export PYTHONPATH=/home/jerem

# Install Unsloth
pip install unsloth

# For GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Verify Installation

```python
from unsloth import FastLanguageModel, FastVisionModel
print("Unsloth successfully installed!")
```

## Fine-tuning Guide

### Basic Fine-tuning Example

```python
from unsloth import FastLanguageModel
import torch

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3n-e4b-it",
    max_seq_length=2048,
    dtype=torch.float16,  # Use bfloat16 for newer GPUs
    load_in_4bit=True,    # Enable 4-bit quantization
)

# Enable LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,      # 0 is optimized
    bias="none",
    use_gradient_checkpointing="unsloth",  # 4x longer contexts
    random_state=3407,
)
```

### Selective Component Fine-tuning

For Gemma 3N multimodal models:

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,  # Keep vision frozen
    finetune_language_layers   = True,   # Fine-tune language
    finetune_attention_modules = True,   # Fine-tune attention
    finetune_mlp_modules       = True,   # Fine-tune MLPs
)
```

### Training Configuration

```python
from transformers import TrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Start training
trainer.train()
```

### Memory-Efficient Settings for RTX 3090 (24GB)

```python
# For Gemma 3N E4B on RTX 3090
model = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3n-e4b-it",
    max_seq_length=2048,  # Reduce if OOM
    dtype=torch.float16,
    load_in_4bit=True,
)

# Training args optimized for 24GB
args = TrainingArguments(
    per_device_train_batch_size=1,      # Minimal batch size
    gradient_accumulation_steps=8,       # Effective batch = 8
    gradient_checkpointing=True,         # Save memory
    optim="paged_adamw_8bit",           # 8-bit optimizer
    fp16=True,                          # Use float16
    max_grad_norm=0.3,                  # Gradient clipping
)
```

## Performance Optimizations

### Unsloth-Specific Optimizations

1. **Dynamic 4-bit Quantization**
   - Superior accuracy compared to static quantization
   - Automatic during loading with `load_in_4bit=True`

2. **Gradient Checkpointing**
   - Use `use_gradient_checkpointing="unsloth"`
   - Enables 4x longer context lengths

3. **Optimized Kernels**
   - All operations use custom Triton kernels
   - Automatic kernel selection based on hardware

### Hardware-Specific Tips

#### For Float16-only GPUs (T4, RTX 20x, V100)
```python
# Unsloth automatically handles infinity/NaN issues
dtype = torch.float16
# No manual intervention needed!
```

#### For BFloat16 GPUs (RTX 30x+, A100, H100)
```python
dtype = torch.bfloat16  # Preferred for newer hardware
```

### Memory Usage Estimates

| Model | Batch Size | Gradient Accumulation | Est. VRAM |
|-------|------------|----------------------|-----------|
| E2B   | 2          | 4                    | ~8GB      |
| E4B   | 1          | 8                    | ~18GB     |
| E4B   | 2          | 4                    | ~22GB     |

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```python
# Solution 1: Reduce batch size
per_device_train_batch_size = 1

# Solution 2: Enable gradient checkpointing
use_gradient_checkpointing = "unsloth"

# Solution 3: Reduce sequence length
max_seq_length = 1024  # From 2048

# Solution 4: Use 4-bit quantization
load_in_4bit = True
```

#### 2. Infinity/NaN Gradients on Float16 GPUs
```python
# Unsloth handles this automatically!
# If issues persist, ensure you're using latest version:
pip install --upgrade unsloth
```

#### 3. Slow Training Speed
```python
# Enable packing for short sequences
packing = True  # Can be 5x faster

# Use optimized data loading
dataset_num_proc = 4  # Parallel processing

# Ensure compiled mode
torch.compile(model)  # If supported
```

#### 4. Import Errors
```bash
# Ensure Python path is set
export PYTHONPATH=/home/jerem

# Verify Unsloth installation
python -c "import unsloth; print(unsloth.__version__)"
```

### Debugging Commands

```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Monitor training metrics
tensorboard --logdir outputs/runs

# Verify model loading
python -c "from unsloth import FastLanguageModel; print('Success!')"
```

## Resources and Links

### Official Documentation
- [Unsloth Docs](https://docs.unsloth.ai/)
- [Gemma 3N Guide](https://docs.unsloth.ai/basics/gemma-3n-how-to-run-and-fine-tune)
- [GitHub Repository](https://github.com/unslothai/unsloth)

### Research Papers
- [MatFormer Architecture](https://arxiv.org/pdf/2310.07707)
- [AltUp](https://arxiv.org/abs/2301.13310)
- [LAuReL](https://arxiv.org/abs/2411.07501)

### Colab Notebooks
- [Gemma 3N Conversational](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Conversational.ipynb)
- [Gemma 3N Vision](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Vision.ipynb)
- [Gemma 3N Audio](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Audio.ipynb)

### Community
- [Unsloth Discord](https://discord.gg/unsloth)
- [HuggingFace Models](https://huggingface.co/unsloth)

## Project Integration Notes

For the Gemma-3N-Agent-Loop project:

1. **Model Path**: Use `/media/jerem/641C8D6C1C8D3A56/MLLMODELS/` for all model storage
2. **Dataset Path**: Use HF cache at `/media/jerem/641C8D6C1C8D3A56/hf_cache/`
3. **Compiled Cache**: Available at `models/training/unsloth_compiled_cache/`
4. **Batch Size**: Start with 1 for RTX 3090, increase if memory allows
5. **Training Script**: Use `training/qlora_finetune_unsloth.py` for optimized training

Remember to always check GPU memory before starting training:
```bash
nvidia-smi
```

Expected memory usage for E4B model:
- Base model: ~10GB
- Training overhead: ~8-12GB
- Total: ~18-22GB (fits in RTX 3090)