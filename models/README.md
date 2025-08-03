# Models Directory Structure

This directory contains all model-related files for the Agent Loop project, optimized for Docker volume mounting and GPU training workflows.

## Directory Structure

```
models/
├── .cache/                    # HuggingFace model cache (~2-5GB)
│   ├── huggingface/          # HF model cache
│   ├── transformers/         # Transformers library cache  
│   ├── torch/                # PyTorch model cache
│   └── datasets/             # HF datasets cache
├── gguf/                     # Quantized GGUF models (~4-8GB)
│   ├── gemma_2b_base.gguf   # Base Gemma 2B model
│   ├── gemma_2b_finetuned.gguf # Fine-tuned versions
│   └── llama_models/         # Other GGUF models
└── finetuned/               # Fine-tuned model outputs
    ├── checkpoints/         # Training checkpoints
    ├── final_models/        # Completed fine-tuned models
    └── adapters/            # LoRA adapter files
```

## Volume Configuration

### Docker Volume Mounts (docker-compose.training.yml)

- `training_models_cache:/app/models/.cache` - Persistent HF cache
- `training_models_gguf:/app/models/gguf` - GGUF model storage  
- `training_checkpoints:/app/outputs` - Training outputs

### Performance Optimization

1. **SSD Storage**: Place on fastest available storage (NVMe SSD preferred)
2. **Local Binding**: Uses local directory binding for development
3. **Cache Strategy**: Persistent volumes prevent re-downloading large models
4. **Backup Ready**: Structure supports automated backup workflows

## Model Management

### Adding GGUF Models

```bash
# Download base model to GGUF directory
cd models/gguf/
wget https://huggingface.co/google/gemma-2-2b-gguf/resolve/main/model.gguf -O gemma_2b_base.gguf
```

### Training Workflow

1. **Base Model**: Place in `gguf/` directory
2. **Fine-tuning**: Outputs go to `../model_checkpoints/`
3. **Final Model**: Best checkpoint copied to `finetuned/final_models/`
4. **Conversion**: Convert back to GGUF format in `gguf/`

## Docker Integration

### Development Mode
```bash
# Models are mounted with hot-reload
docker-compose -f docker-compose.training.yml --profile training-dev up
```

### Production Training
```bash
# Read-only mounts for security
docker-compose -f docker-compose.training.yml --profile training up
```

## Storage Requirements

- **Base Models**: 2-8GB per GGUF model
- **HF Cache**: 2-5GB for transformers and tokenizers
- **Training Checkpoints**: 1-3GB during training
- **Total Recommended**: 20-50GB free space

## Security Notes

- Models in production are mounted read-only
- Cache directories have appropriate permissions  
- Backup strategy protects against data loss
- No sensitive data should be stored in model files