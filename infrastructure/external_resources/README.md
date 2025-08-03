# External Resources - Symbolic Links

This directory contains symbolic links to external storage locations (SSDs) where large ML resources are stored.

## Symbolic Links

- **datasets** → `/media/jerem/jeux&travail/datasets/`
  - Contains all training datasets (3.8GB+)
  - Primary: agent_instruct, apibench, browsergym, camel_agent

- **ml_models** → `/media/jerem/jeux&travail/ml_models/`
  - Stores downloaded models and checkpoints
  - Includes Gemma-3N models and HuggingFace cache

- **hf_cache** → `/media/jerem/641C8D6C1C8D3A56/hf_cache/`
  - HuggingFace cache directory
  - Contains transformers, datasets, hub cache

## Usage

These symbolic links allow the project to access external resources without copying large files:

```python
# Access datasets
datasets_path = "infrastructure/external_resources/datasets/agent_instruct"

# Access models
model_path = "infrastructure/external_resources/ml_models/gemma-3n-e4b"
```

## Important Notes

- These are symbolic links, not actual directories
- The external drives must be mounted for these links to work
- Do not commit large files from these directories to git
- Use these paths in configuration files instead of hardcoding external paths