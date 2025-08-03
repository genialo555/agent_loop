# Models Scripts Organization

This directory contains operational scripts for the Gemma-3N-Agent-Loop MLOps platform, organized by functional domain following our hexagonal architecture principles.

## Directory Structure

### 📁 lora/
Scripts for managing LoRA (Low-Rank Adaptation) weights and model fusion.
- `merge_and_convert_lora.py` - Complete pipeline: merge LoRA → convert to GGUF → import to Ollama
- `merge_lora_simple.py` - Simple LoRA weight merging utility

### 📁 datasets/
Dataset management and validation scripts.
- `DATASET_URLS_UPDATE.md` - Documentation of dataset sources and URLs
- `debug_dataset.py` - Dataset debugging and inspection tool
- `secure_dataset_downloader.py` - Safe dataset downloading with validation
- `test_dataset_loading.py` - Dataset loading verification
- `test_dataset_urls.py` - URL accessibility testing

### 📁 ollama/
Ollama integration and testing scripts.
- `ollama_usage_examples.py` - Example usage patterns for Ollama API
- `test_ollama_integration.py` - Ollama service integration tests

### 📁 docker/
Docker build and deployment scripts.
- `docker-dev.sh` - Development container management
- `docker-prod.sh` - Production container deployment
- `test_docker_training.py` - Docker training environment verification

### 📁 training/
Training execution and monitoring scripts.
- `analyze_training_logs.py` - Training log analysis and metrics extraction
- `monitor_training_example.py` - Real-time training monitoring
- `training_monitor.py` - Training progress tracking utility
- `train_optimized.py` - Optimized training pipeline
- Various shell scripts for different training configurations:
  - `run_hrm_training.sh` - Hierarchical Reasoning Model training
  - `run_training.sh` - Standard training execution
  - `run_unsloth.sh` - Unsloth framework launcher
  - `run_unsloth_training.sh` - Unsloth training pipeline
  - `train_2epochs.sh` - Full 2-epoch training (~280k steps)
  - `train_beast_mode.sh` - Maximum performance training
  - `train_gsm8k_test.sh` - GSM8K dataset training
  - `train_optimized_1epoch.sh` - Single epoch optimized training
  - `train_safe_optimized.sh` - Conservative optimized training
  - `train_test_1000steps.sh` - Quick test training (1000 steps)

### 📁 monitoring/
Model performance and drift monitoring.
- `model_drift_detection.py` - Detect model performance degradation
- `model_registry.py` - Model version tracking and management

### 📁 infrastructure/
System setup and infrastructure management.
- `setup_vm.sh` - Virtual machine initial setup
- `install_nvidia_docker.sh` - NVIDIA Docker runtime installation
- `native_gpu_setup.sh` - Native GPU environment configuration
- `gpu_memory_manager.sh` - GPU memory optimization utilities
- `move_models_to_ssd.sh` - Model storage migration to SSD

### 📁 utilities/
General utility scripts and tools.
- `clean_cache.py` - Cache cleanup utility
- `health_check.py` - System health verification
- `sync_logs.sh` - Log synchronization across nodes
- `update_model.sh` - Model update and deployment
- `test_agent_instruct.py` - Agent instruction testing
- `test_new_architecture.py` - Architecture validation tests
- `demo_download.py` - Download demonstration script
- `start_new_app.py` - Application initialization helper

## Usage Guidelines

1. **Always use absolute paths** when calling scripts from other locations
2. **Check prerequisites** before running infrastructure scripts (GPU, Docker, etc.)
3. **Monitor GPU memory** when running training scripts (use `nvidia-smi`)
4. **Activate virtual environment** before running Python scripts:
   ```bash
   source /home/jerem/agent_loop/.venv/bin/activate
   ```
5. **Set PYTHONPATH** for proper imports:
   ```bash
   export PYTHONPATH=/home/jerem
   ```

## Key Scripts Quick Reference

### Most Used Scripts
- **Merge LoRA weights**: `lora/merge_and_convert_lora.py`
- **Start training**: `training/run_training.sh`
- **Test Ollama**: `ollama/test_ollama_integration.py`
- **Setup new VM**: `infrastructure/setup_vm.sh`
- **Monitor training**: `training/training_monitor.py`

### Critical Maintenance Scripts
- **Clean cache**: `utilities/clean_cache.py`
- **Health check**: `utilities/health_check.py`
- **Sync logs**: `utilities/sync_logs.sh`
- **Model drift detection**: `monitoring/model_drift_detection.py`

## Architecture Alignment

This organization follows the hexagonal architecture principles:
- **Core domain logic** (training, LoRA) is separated from infrastructure
- **External integrations** (Ollama, Docker) have dedicated directories
- **Infrastructure concerns** are isolated in their own directory
- **Monitoring and utilities** provide cross-cutting concerns

Each script should be self-documenting with clear help messages and docstrings.