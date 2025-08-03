# Dataset Audit Report - ML Training Pipeline

**Date**: 2025-07-29  
**Location**: `/media/jerem/jeux&travail/datasets/` (3.8GB total)  
**Project**: `/home/jerem/agent_loop`

## Executive Summary

The datasets are correctly stored on the SSD but the training pipeline needs configuration updates to use them properly. All major datasets are present and accessible.

## 1. Dataset Structure Analysis

### Available Datasets

| Dataset | Size | Format | Files | Status |
|---------|------|--------|-------|--------|
| agent_instruct | 2.3GB | Parquet | 20 files in data/ | ✅ Ready |
| apibench | 42MB | JSON | Multiple train.json | ✅ Ready |
| browsergym | 3.2MB | Unknown | To investigate | ⚠️ Check |
| camel_agent | 1.3GB | Mixed | Multiple subdirs | ✅ Ready |
| miniowb | 18MB | Unknown | To investigate | ⚠️ Check |
| react | 28MB | Unknown | To investigate | ⚠️ Check |
| workarena | 149MB | Unknown | To investigate | ⚠️ Check |

### Key Findings

1. **agent_instruct**: Primary dataset with 20 parquet files containing conversation data
2. **apibench**: Contains separate training files for HuggingFace, TensorFlow, and PyTorch
3. **camel_agent**: Organized by domain (ai_society, biology, code, math, physics)

## 2. Current Configuration Issues

The training script (`training/qlora_finetune.py`) expects either:
- A HuggingFace dataset name
- A single file path

**Problem**: It doesn't handle directories with multiple parquet files correctly.

**Solution Applied**: Updated the script to handle:
- Directory paths with automatic file discovery
- Multiple parquet files
- Prioritization of training files for JSON datasets

## 3. Dataset Loading Methods

### A. Local Loading (Recommended for Development)

```bash
# Agent Instruct (parquet files)
python training/qlora_finetune.py \
  --data /media/jerem/jeux&travail/datasets/agent_instruct/data \
  --text-column conversations \
  --model-config gemma-2b \
  --output-dir ./results/agent_instruct

# APIBench (JSON)
python training/qlora_finetune.py \
  --data /media/jerem/jeux&travail/datasets/apibench/huggingface_train.json \
  --model-config gemma-2b \
  --output-dir ./results/apibench
```

### B. HuggingFace Loading (For CI/CD)

```bash
# Agent Instruct
python training/qlora_finetune.py \
  --data agent-instruct/agent-instruct \
  --model-config gemma-2b \
  --output-dir ./results/agent_instruct

# APIBench
python training/qlora_finetune.py \
  --data gorilla-llm/APIBench \
  --model-config gemma-2b \
  --output-dir ./results/apibench
```

## 4. Performance Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Local SSD | • Instant loading<br>• No network dependency<br>• Full 3.8GB available | • Not version controlled<br>• Manual updates | Development |
| HuggingFace | • Version controlled<br>• Automatic caching<br>• Reproducible | • Network dependency<br>• Initial download | Production/CI |

## 5. Recommendations

### Immediate Actions

1. **Use Local Datasets for Training**
   - Faster iteration during development
   - No network bottlenecks
   - All data already available

2. **Test with agent_instruct First**
   - Largest and most comprehensive dataset
   - Well-structured parquet format
   - Contains diverse instruction-following examples

3. **Memory Optimization**
   ```python
   # In training config:
   gradient_checkpointing = True
   per_device_train_batch_size = 2  # Reduce if OOM
   gradient_accumulation_steps = 8  # Compensate for small batch
   max_seq_length = 512  # Reduce if needed
   ```

### Best Practices

1. **Dataset Preprocessing**
   - Convert conversation format to single text strings
   - Use consistent formatting across datasets
   - Cache preprocessed data for faster loading

2. **Multi-Dataset Training**
   - Mix datasets using `training/mix_datasets.py`
   - Balance dataset sizes to prevent bias
   - Use dataset-specific text columns

3. **Monitoring**
   - Track loading times
   - Monitor memory usage
   - Log dataset statistics

## 6. Test Scripts Created

1. **`scripts/test_dataset_loading.py`** - General dataset testing
2. **`scripts/test_agent_instruct.py`** - Specific agent_instruct validation
3. **`training/dataset_config.yaml`** - Central configuration file

## 7. Next Steps

1. Run test scripts to validate dataset loading
2. Start training with agent_instruct dataset
3. Monitor GPU memory usage and adjust batch sizes
4. Implement dataset mixing for comprehensive training

## Conclusion

The datasets are properly stored and accessible. The training pipeline has been updated to handle local dataset loading correctly. Recommend using local datasets for development (faster) and HuggingFace for production (reproducible).

**Primary Training Command**:
```bash
cd /home/jerem/agent_loop
python training/qlora_finetune.py \
  --data /media/jerem/jeux&travail/datasets/agent_instruct/data \
  --text-column conversations \
  --model-config gemma-2b \
  --output-dir ./results/agent_instruct \
  --max-steps 1000
```