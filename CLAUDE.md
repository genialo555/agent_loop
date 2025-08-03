# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Version Control (#memorize)
**This project uses BITBUCKET, not GitHub!**
- Repository is hosted on Bitbucket
- Use Bitbucket URLs for issues, wiki, and repository links
- Format: `https://bitbucket.org/your-org/agent-loop`

## Project Overview

**Gemma-3N-Agent-Loop** is a production-ready MLOps platform implementing a continuous learning loop for autonomous agents based on Google's Gemma models. The system follows a hexagonal architecture with clear separation of concerns.

**Main cycle**: Pre-train → Deploy → Log → Fine-tune → Redeploy

### Project Goal
Create a Linux OS navigation agent that can understand natural language instructions like "find all error logs and create a report" and generate the necessary bash/python commands autonomously.

### Technical Stack
- **Backend**: FastAPI with complete async/await patterns
- **ML Framework**: PyTorch + HuggingFace Transformers + Unsloth (for Gemma-3N compatibility)
- **Inference**: Ollama for optimized deployment (GGUF format)
- **Monitoring**: Prometheus + Grafana + Weights&Biases
- **Orchestration**: Prefect for ML pipelines
- **Security**: Multi-layer framework against prompt injections (50+ patterns)
- **Infrastructure**: Docker, Terraform, Ansible

## Key Commands

### Development Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
make install  # Full setup with pre-commit hooks
# or
pip install -r requirements.txt
pip install -e .

# Set Python path for imports
export PYTHONPATH=/home/jerem
```

### Common Development Commands
```bash
make lint        # Run all linters (black, ruff, mypy, bandit)
make format      # Auto-format code with black and isort
make test        # Run full test suite with coverage
make test-unit   # Run unit tests only
make clean       # Clean temporary files
```

### Running the API
```bash
# Start Ollama service first
ollama serve

# Pull Gemma model
ollama pull gemma-3n:e4b

# Start FastAPI development server
cd models/inference
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/agents/run-agent \
  -H "Content-Type: application/json" \
  -d '{"instruction": "What is 2+2?", "use_ollama": true}'
```

### Training Commands
```bash
# Quick test (10 steps)
make train-gemma-2b

# Full training with QLoRA
source .env && source .venv/bin/activate
python training/qlora_finetune.py \
  --model-config custom \
  --model-name "/media/jerem/jeux&travail/ml_models/hub/models--google--gemma-3n-e4b/snapshots/af70430f84ea4d7ac191aaa2bd8e14d2a5e8f6ee" \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --text-column messages \
  --max-steps 100 \
  --output-dir ./results/gemma-3n-100steps \
  --batch-size 1 \
  --no-wandb

# Unsloth optimized training (recommended for Gemma-3N)
source .env && source .venv/bin/activate
python training/qlora_finetune_unsloth.py \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --max-steps 100 \
  --output-dir ./results/gemma-3n-unsloth-100steps-fixed

# Full 2 epochs training (~140k steps, multiple hours)
python training/qlora_finetune_unsloth.py \
  --data "/media/jerem/jeux&travail/datasets/agent_instruct/data" \
  --num-epochs 2 \
  --output-dir ./results/gemma-3n-unsloth-2epochs

# HRM Training with Unsloth (NEW) (#memorize)
# IMPORTANT: Unsloth handles data preparation automatically!
# NOTE: Unsloth compiled cache exists at models/training/unsloth_compiled_cache/
python models/training/qlora/qlora_finetune_unsloth.py \
  --data "gsm8k" \
  --max-steps 50 \
  --output-dir ./models/results/gemma-3n-hrm-test \
  --batch-size 1

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Docker Commands
```bash
make build-dev         # Build development environment
make train-docker      # Run QLoRA training in Docker
make train-monitor     # Start full training with monitoring
make train-logs        # Show training logs
make train-status      # Check training status
```

## Architecture & Code Structure

### Import Pattern
All imports use absolute paths from `agent_loop` root:
```python
from agent_loop.agent.plugins.browser_tool import BrowserTool
from agent_loop.models.core.settings import Settings
from agent_loop.models.inference.services.ollama import OllamaService
from agent_loop.models.training.qlora.qlora_config import QLoRAConfig
```

### Directory Structure
- **`models/`** - ML hub containing all ML-related code
  - `training/` - Training pipelines (QLoRA, Unsloth)
  - `inference/` - FastAPI production API
  - `datasets/` - Training data management
  - `results/` - Model checkpoints and outputs
  - `logs/` - Training and inference logs
  - `scripts/` - Operational scripts
  - `core/` - Core settings and utilities

- **`agent/`** - Agent implementation
  - `tools/` - Base agent tools
  - `plugins/` - Extensible plugins (browser_tool.py)
  - `prompts/` - System prompts

- **`infrastructure/`** - IaC (Terraform, Ansible, Docker)
- **`monitoring/`** - Observability stack (Prometheus, Grafana)
- **`tests/`** - Test suites organized by type

## CRITICAL RULES - NO ERRORS ALLOWED (#memorize)
**À CE STADE, AUCUNE ERREUR N'EST PERMISE**
- Double-check TOUT avant d'exécuter
- Demander confirmation si incertain  
- Alerter quand contexte window atteint 70%
- Vérifier chemins et montages AVANT toute opération
- Tester imports et syntaxe AVANT de lancer

### Key Configuration Paths (#memorize)
- **Model cache**: `/media/jerem/641C8D6C1C8D3A56/MLLMODELS/` ⚠️ **NEW LOCATION - ALL MODELS GO HERE**
- **Datasets**: `/media/jerem/jeux&travail/datasets/`
- **HF cache**: `/media/jerem/641C8D6C1C8D3A56/hf_cache/` ⚠️ **CRITICAL: 58GB+ cache, DO NOT DELETE**
- **Logs**: `models/logs/`
- **Results**: `models/results/`

### CRITICAL MODEL STORAGE (#memorize)
**ALL MODELS MUST BE STORED AND LOADED FROM:**
```
/media/jerem/641C8D6C1C8D3A56/MLLMODELS/
```
- This is on the SSD with HF cache
- Symbolic links created:
  - `models/model_cache` → `/media/jerem/641C8D6C1C8D3A56/MLLMODELS/`
  - `models/training/model_cache` → `/media/jerem/641C8D6C1C8D3A56/MLLMODELS/`
- NEVER store models on the main disk - use the SSD!

### Critical HuggingFace Cache (#memorize)
The HF cache at `/media/jerem/641C8D6C1C8D3A56/hf_cache/` is **EXTREMELY IMPORTANT**:
- Contains 58GB+ of downloaded datasets
- Deleting this would require re-downloading everything (hours/days)
- Always ensure this drive is mounted before training
- The .env file must point to this location for HF_HOME, TRANSFORMERS_CACHE, etc.
- **ALL NEW DATASET DOWNLOADS GO HERE** - Never download datasets elsewhere!
- When using HuggingFace datasets.load_dataset(), it automatically uses this cache
- This prevents duplicate downloads and saves disk space

### Accessing External Resources
External resources are accessible via symbolic links in `infrastructure/external_resources/`:
- `infrastructure/external_resources/datasets/` → External datasets
- `infrastructure/external_resources/ml_models/` → External models
- `infrastructure/external_resources/hf_cache/` → HuggingFace cache

This follows the hexagonal architecture pattern - external resources are accessed through the infrastructure layer.

### API Endpoints
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system status
- `GET /health/ready` - Kubernetes readiness probe
- `GET /metrics` - Prometheus metrics
- `POST /agents/run-agent` - Execute agent instruction
- `POST /ollama/generate` - Direct Ollama generation
- `GET /training/status` - Training pipeline status

## Current Issues & Solutions

### GPU Memory (RTX 3090 24GB)
- OOM with full gemma-3n-e4b model (4.5B parameters)
- Current GPU usage: 5.8GB by other process, leaving ~18GB available
- Solution: Use batch_size=1, enable gradient checkpointing
- Set: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Training stats: ~5.8s/step, 1 epoch = 140,177 steps = ~226 hours

### Unsloth Installation (#memorize)
- Unsloth module not found in current .venv
- Compiled cache exists at `models/training/unsloth_compiled_cache/`
- Scripts exist but need Unsloth installed: `pip install unsloth`
- Alternative: May need specific conda environment or Docker container

### Import Errors
- Ensure `PYTHONPATH=/home/jerem` is set
- Activate virtual environment: `source .venv/bin/activate`
- Settings validation: Added `extra="ignore"` to handle .env fields
- Python version: 3.12.3 (project configured for 3.13)

### Ollama Integration
- `/run-agent` endpoint currently uses regex instead of Ollama
- Replace regex logic with: `result = await ollama_service.generate(instruction)`
- Add JWT authentication: `dependencies=[Depends(verify_jwt)]`
- Ollama models stored at: `/usr/share/ollama/.ollama/models/`

### Docker Issues (from test report)
- Missing NVIDIA CUDA base images (12.6.0-cudnn8)
- Docker compose missing services: qlora-training, training-dev, training-monitor
- GPU not available in Docker (needs --gpus all flag)

## Sprint 2 Priorities

1. **Fix Ollama integration** in `/run-agent` endpoint
2. **Add JWT authentication** to API endpoints
3. **Optimize GPU memory** usage (batch_size, Flash Attention)
4. **Increase test coverage** from 39.4% to 80%
5. **Complete training pipeline** documentation

## Special Commands

### Activate All Specialized Agents
```
@LES_GARS <request>
```

### Important: Complex Problem Solving Approach
**When facing complex problems, always:**
1. Use `<think>` tags to analyze the problem thoroughly
2. Call the `gemma-agent-orchestrator` to determine which specialized agents to involve
3. Let the orchestrator coordinate the work across multiple agents
4. Don't try to solve complex multi-domain problems alone

### Critical: Think Before Acting on Infrastructure
**Before making ANY infrastructure decisions:**
- THINK about where files/directories should be placed according to PROJECT_STRUCTURE.md
- External links and infrastructure-related items belong in `infrastructure/`
- Never create random directories without considering the architecture
- Ask yourself: "Where does this belong in our hexagonal architecture?"
- When in doubt, consult the gemma-agent-orchestrator or system-architect

### Important: Training Scripts Documentation (#memorize)
**When creating ANY training script:**
- ALWAYS add the command to `/home/jerem/agent_loop/docs/TRAINING_COMMANDS_humain.md`
- Include the full command with all parameters
- Add description of what the script does
- Document expected runtime and resource usage

### Important Pattern to Remember (#memorize)
**Always check if external drives are mounted before using them:**
```bash
# Check mounted drives
ls -la /media/jerem/

# If "jeux&travail" is not mounted, the symbolic links won't work
# The drive name contains "&" which requires quotes in commands
```

Example:
```
<think>
This task involves both API changes and training pipeline modifications. 
I should use the gemma-agent-orchestrator to coordinate this work.
</think>

I'll use the gemma-agent-orchestrator to handle this cross-functional task...
```

### Agent Responsibilities
- `@gemma-agent-orchestrator` - **Coordinates complex tasks across multiple agents**
- `@system-architect` - Overall architecture decisions
- `@fastapi-async-architect` - API structure and async patterns
- `@python-type-guardian` - Type safety and Pydantic models
- `@llm-optimization-engineer` - ML training and optimization
- `@docker-container-architect` - Container configuration
- `@test-automator` - Test structure and coverage
- `@observability-engineer` - Monitoring and logging

## Performance Targets
- Inference latency: < 500ms (currently ~350ms)
- Training speed: > 1k tokens/sec
- Test coverage: > 80% (currently 39.4%)
- API uptime: > 99.9%

## Security Considerations
- All user inputs are validated through Pydantic models
- JWT authentication required for protected endpoints
- Sandboxed execution for agent tools
- No hardcoded secrets (use .env file)
- Input sanitization for prompt injection protection
- Multi-layer security framework with 50+ detection patterns
- Security policies: Minimal → Standard → Strict → Maximum
- Real-time monitoring with email/webhook alerts
- Threat intelligence and anomaly detection

## Available Datasets

Located at `/media/jerem/jeux&travail/datasets/` (3.8GB total):

| Dataset | Size | Format | Purpose |
|---------|------|--------|---------|
| agent_instruct | 2.3GB | Parquet (20 files) | Primary training data (1.1M examples) |
| apibench | 42MB | JSON | API usage patterns |
| browsergym | 3.2MB | Unknown | Browser automation |
| camel_agent | 1.3GB | Mixed | Multi-domain (AI, biology, code, math, physics) |
| toolbench | - | - | Tool usage benchmarking |
| webarena | - | - | Web navigation tasks |

### HRM (Hierarchical Reasoning Model) Datasets (#memorize)
Downloaded to HF cache for teaching hierarchical thinking:
- **GSM8K**: 7,473 math problems with step-by-step reasoning
- **CodeAlpaca-20k**: Instructional code generation 
- **Python Code Instructions 18k**: Python with hierarchical solutions
- **SQL Create Context**: Structured query decomposition
- **Custom Linux HRM**: System admin tasks with 4-phase breakdown:
  1. ANALYZE - Understand the problem
  2. PLAN - Break into sub-tasks
  3. EXECUTE - Step-by-step implementation
  4. VERIFY - Check results

### LoRA Weights Management Script (#memorize)
**Script pour fusionner et convertir les poids LoRA:**
```bash
# Script créé: /home/jerem/agent_loop/models/scripts/merge_and_convert_lora.py

# Usage basique (fusion + conversion + import Ollama)
python models/scripts/merge_and_convert_lora.py \
  --lora-path /home/jerem/agent_loop/models/results/gemma-3n-hrm-test-20250801_015252 \
  --output-name gemma-3n-hrm

# Options disponibles:
# --quantization q4_k_m  # Types: q4_k_m (défaut), q5_k_m, q8_0
# --skip-merge          # Si déjà fusionné
# --skip-ollama         # Pour GGUF seulement
```

**Le script fait 3 étapes:**
1. Fusionne LoRA + modèle de base → modèle complet
2. Convertit en GGUF avec quantization → pour Ollama
3. Importe dans Ollama avec Modelfile → `ollama run gemma-3n-hrm`

**Résultats stockés dans:**
- Modèle fusionné: `/media/jerem/641C8D6C1C8D3A56/MLLMODELS/{name}-merged/`
- Fichier GGUF: `/media/jerem/641C8D6C1C8D3A56/MLLMODELS/{name}.gguf`
- Modelfile: `/home/jerem/agent_loop/models/ollama/{name}.Modelfile`

## Hardware Configuration
- **GPU**: RTX 3090 24GB (350W, 78-81°C under load)
- **CPU**: AMD Ryzen 9 7900 (12 cores)
- **RAM**: 64GB DDR5 6000MHz
- **Storage**: External SSDs for models and datasets

## Model Training Progress
- **Current Model**: Gemma-3N-E4B (4.5B params, Google's experimental AltUp architecture)
- **Method**: QLoRA 4-bit with Unsloth framework
- **Speed**: ~5.8s/step, ~1.2k tokens/sec
- **Batch Size**: 2 (effective 8 with gradient accumulation)
- **Loss Progress**: 6.0 → 1.3-1.4 (excellent learning)
- **Target**: < 1.0 loss after ~10,000 steps

## Important File Locations
- **Settings**: `models/core/settings.py` - Central configuration
- **API App**: `models/inference/app.py` - Main FastAPI application
- **Training Script**: `training/qlora_finetune_unsloth.py` - Optimized trainer
- **Agent Router**: `models/inference/routers/agents.py` - Agent endpoints
- **Browser Tool**: `agent/plugins/browser_tool.py` - Web navigation

## Testing Guidelines
- Current coverage: 39.4% (target: 80%)
- Priority modules: training (0%), CLI (0%)
- Run tests: `make test` or `pytest tests/`
- Test markers: unit, integration, property, benchmark, ollama, browser, gpu

## Common Workflows

### Adding a New API Endpoint
1. Create router in `models/inference/routers/`
2. Add Pydantic schemas in `models/inference/models/schemas.py`
3. Implement service logic in `models/inference/services/`
4. Write tests in `tests/integration/test_<endpoint>.py`
5. Update API documentation

### Implementing a New Agent Tool
1. Create tool in `agent/plugins/`
2. Follow the BrowserTool pattern for structure
3. Add tool to agent's available tools list
4. Write unit tests in `tests/unit/`
5. Add integration test with agent

### Training Pipeline Workflow
1. Prepare dataset in parquet format
2. Place in `models/datasets/processed/`
3. Configure training in `training/qlora_config.py`
4. Run training with monitoring
5. Evaluate on test set
6. Export to GGUF for Ollama
7. Deploy via model update script

### Downloading New Datasets (#memorize)
```python
# CORRECT: Downloads to HF cache automatically
from datasets import load_dataset
dataset = load_dataset("agent-instruct/agent-instruct")
# Files go to: /media/jerem/641C8D6C1C8D3A56/hf_cache/datasets/

# WRONG: Don't download manually to other locations
# This wastes space and bypasses the cache
```

## Troubleshooting Tips

### Common Errors
- **ModuleNotFoundError**: Set `PYTHONPATH=/home/jerem`
- **CUDA OOM**: Reduce batch_size or enable gradient_checkpointing
- **Ollama connection**: Ensure `ollama serve` is running
- **Docker GPU**: Add `--gpus all` flag to docker run
- **Import validation**: Check Settings class has `extra="ignore"`

### Quick Checks
```bash
# Check GPU availability
nvidia-smi

# Verify Ollama is running
curl http://localhost:11434/api/version

# Test FastAPI health
curl http://localhost:8000/health

# Check Python path
echo $PYTHONPATH

# Verify virtual env
which python
```

## Project Status Summary
- **Sprint 1**: ✅ Infrastructure, basic API, Ollama integration
- **Sprint 2**: 🚧 JWT auth, GPU optimization, test coverage
- **Current blockers**: GPU memory, Docker CUDA images, test coverage
- **Next milestone**: Production-ready agent with 95% tool accuracy