# Unsloth Official Installation Guide

Source: https://docs.unsloth.ai/get-started/installation

## Installation Methods

### 1. Pip Install (Recommended)

The simplest and most recommended method:

```bash
pip install unsloth
```

This will install Unsloth with all necessary dependencies for your platform.

### 2. Platform-Specific Installation

#### Linux
```bash
pip install unsloth
```
- Fully supported on all Linux distributions
- Best performance and compatibility
- Supports all features

#### Windows
```bash
pip install unsloth
```
- Windows support available
- Some features may have platform-specific considerations
- Refer to Windows-specific documentation for optimization tips

#### macOS
```bash
# For Apple Silicon (M1/M2/M3)
pip install unsloth
```
- Limited support for Apple Silicon
- CPU-only inference available
- Consider using MLX for optimized Apple Silicon performance

### 3. Conda Installation

For Anaconda/Miniconda users:

```bash
conda install -c conda-forge unsloth
```

Note: Conda installation may lag behind pip releases.

### 4. Google Colab

Unsloth is pre-optimized for Google Colab:

```python
# In a Colab notebook
!pip install unsloth

# Import and verify
import unsloth
print(f"Unsloth version: {unsloth.__version__}")
```

### 5. Kaggle Notebooks

Similar to Colab:
```python
!pip install unsloth
```

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU support)
- **PyTorch**: 2.0+ (automatically installed)
- **RAM**: 8GB minimum
- **GPU**: NVIDIA GPU from 2018+ (GTX 10 series or newer)

### Recommended Setup
- **Python**: 3.10 or 3.11
- **CUDA**: 12.1+
- **RAM**: 16GB+
- **GPU**: RTX 30 series or newer with 8GB+ VRAM

### GPU Compatibility
- **Supported**: All NVIDIA GPUs from 2018 onwards
  - GTX 10 series
  - RTX 20 series
  - RTX 30 series
  - RTX 40 series
  - A100, H100, V100
  - Tesla T4, P100
- **Not Supported**: 
  - AMD GPUs (ROCm not supported)
  - Intel GPUs
  - Older NVIDIA GPUs (pre-2018)

## Dependency Management

### Core Dependencies
Automatically installed with pip:
- `torch>=2.0.0`
- `transformers>=4.36.0`
- `datasets`
- `accelerate`
- `peft`
- `trl`
- `bitsandbytes`

### Optional Dependencies

For specific features:
```bash
# For Weights & Biases logging
pip install wandb

# For TensorBoard support
pip install tensorboard

# For advanced tokenizers
pip install sentencepiece
```

## Installation Verification

After installation, verify everything is working:

```python
# test_unsloth.py
import torch
import unsloth
from unsloth import FastLanguageModel

print(f"Unsloth version: {unsloth.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Try loading a small model
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/tinyllama",
        max_seq_length=512,
        load_in_4bit=True
    )
    print("✓ Model loading successful!")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
```

## Troubleshooting Installation

### Common Issues

#### 1. CUDA Not Found
```bash
# Error: CUDA not available
# Solution: Install CUDA toolkit
# Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA website
```

#### 2. PyTorch Version Mismatch
```bash
# Error: PyTorch version incompatible
# Solution: Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Memory Errors During Import
```bash
# Error: Out of memory during import
# Solution: Set environment variable
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

#### 4. Triton Compilation Errors
```bash
# Error: Triton kernel compilation failed
# Solution: Clear Triton cache
rm -rf ~/.triton/cache/
```

## Environment Setup

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv unsloth_env

# Activate environment
# Linux/macOS:
source unsloth_env/bin/activate
# Windows:
unsloth_env\Scripts\activate

# Install Unsloth
pip install unsloth
```

### Docker Installation

For containerized deployment:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Unsloth
RUN pip3 install unsloth

# Set working directory
WORKDIR /app

# Default command
CMD ["python3"]
```

Build and run:
```bash
docker build -t unsloth-container .
docker run --gpus all -it unsloth-container
```

## Development Installation

For contributing or development:

```bash
# Clone repository
git clone https://github.com/unslothai/unsloth.git
cd unsloth

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Platform-Specific Notes

### Google Colab
- Unsloth is optimized for Colab's environment
- T4 GPU fully supported with float16 fixes
- No additional setup required

### AWS SageMaker
```python
# In SageMaker notebook
!pip install unsloth
```

### Azure ML
- Use compute instances with NVIDIA GPUs
- Install via pip in notebook or terminal

### Lambda Labs
- Pre-configured for ML workloads
- Simple pip installation works

## Next Steps

After successful installation:

1. **Quick Start**: Follow the [Getting Started Guide](https://docs.unsloth.ai/get-started/quickstart)
2. **Examples**: Check out example notebooks
3. **Documentation**: Read the full documentation
4. **Community**: Join the Discord server

## Version Management

### Check Current Version
```python
import unsloth
print(unsloth.__version__)
```

### Upgrade to Latest
```bash
pip install --upgrade unsloth
```

### Install Specific Version
```bash
pip install unsloth==0.1.0  # Replace with desired version
```

### Nightly Builds
```bash
pip install --upgrade --pre unsloth
```

## Uninstallation

To completely remove Unsloth:

```bash
pip uninstall unsloth

# Clean cache
rm -rf ~/.cache/unsloth
rm -rf ~/.triton/cache
```

For support, visit:
- Documentation: https://docs.unsloth.ai/
- GitHub Issues: https://github.com/unslothai/unsloth/issues
- Discord Community: Available through GitHub