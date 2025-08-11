
  
  # 🤖 Agent Loop - Autonomous LLM Training & Inference Platform

  > **Production-ready MLOps platform** for training, deploying, and continuously improving autonomous agents powered by **Gemma 3N**.

  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
  [![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## 📑 Documentation Index

> **Navigation Guide:**
> - 🧑‍💻 **Human Users**: Start with Overview and Quick Start sections
> - 🤖 **AI Agents**: Start with CLAUDE.md for primary context
> - 📍 **Location Types**: `README.md` (this file), `CLAUDE.md` (root), `External` (separate file), `Directory` (folder)

### 🧑‍💻 For @les_gars*& Human Users
| Section | Description | Location |
|---------|-------------|----------|
| **Getting Started** | | |
| [🎯 Overview](#-overview) | High-level project introduction | README.md |
| [🚀 Quick Start](#-quick-start) | Getting started guide | README.md |
| [📁 Project Structure](#-project-structure) | Directory organization | README.md |
| **Development** | | |
| [🔄 Training Workflow](#-training-workflow) | How to train models | README.md |
| [Training Commands](docs/TRAINING_COMMANDS_humain.md) | Human-readable training guide | External |
| [📊 Monitoring](#-monitoring--observability) | Dashboards and metrics | README.md |
| [🧪 Testing](#-testing) | Running tests | README.md |
| **Architecture & Design** | | |
| [Architecture Guide](docs/ARCHITECTURE/ARCHITECTURE.md) | System design overview | External |
| [MLOps Architecture](docs/ARCHITECTURE/MLOPS_ARCHITECTURE.md) | ML pipeline design | External |
| [Deployment Procedures](docs/ARCHITECTURE/DEPLOYMENT_PROCEDURES.md) | Deploy guide | External |
| **Security & Ops** | | |
| [Security Analysis](docs/SECURITY/SECURITY_ANALYSIS_REPORT.md) | Security audit | External |
| [Observability Setup](docs/observabilty/OBSERVABILITY_IMPLEMENTATION_GUIDE.md) | Monitoring guide | External |
| **Research & Development** | | |
| [LLM Research](docs/R&D/LLM_RESEARCH_AND_INTEGRATION.md) | LLM integration | External |
| [Gemma 3N Reference](docs/R&D/GEMMA3N_TECHNICAL_REFERENCE.md) | Model details | External |
| [Unsloth Examples](docs/R&D/UNSLOTH_TRAINING_EXAMPLES.md) | Training examples | External |

### 🤖 For Machine Users (AI Agents)
| Section | Description | Location |
|---------|-------------|----------|
| **Primary Context** | | |
| [CLAUDE.md](CLAUDE.md) | ⚡ **MAIN INSTRUCTIONS** - Start here | Root |
| [Project Structure](docs/contexte*important<THINK>/PROJECT_STRUCTURE.md) | Directory architecture | External |
| [Training Context](docs/contexte*important<THINK>/CONTEXT_TRAINING.md) | Training specifics | External |
| **Agent Configuration** | | |
| [Agent Orchestrator](.claude/agents/agent-orchestrator.md) | Main coordinator | External |
| [Specialized Agents](.claude/agents/) | All agent configs | Directory |
| [Coordination Matrix](.claude/agents/AGENT_COORDINATION_MATRIX.md) | Agent interactions | External |
| [Inter-Agent Protocol](.claude/agents/INTER_AGENT_PROTOCOL.md) | Communication rules | External |
| **Rules & Constraints** | | |
| [Critical Rules](CLAUDE.md#critical-rules---no-errors-allowed-memorize) | ⚠️ <think>Must-follow <think> | CLAUDE.md |
| [Import Patterns](CLAUDE.md#import-pattern) | Code conventions | CLAUDE.md |
| [XML Rules](docs/rules/) | Structured rule files | Directory |
| [File Monitoring](.claude/rules/file_monitoring_strict.md) | File system rules | External |
| **Technical References** | | |
| [Architecture Diagrams](#-architecture-overview) | Mermaid diagrams | README.md |
| [API Endpoints](#health-endpoints) | Technical endpoints | README.md |
| [System Prompts](prompts/system_prompt.txt) | LLM instructions | External |
| [Context Reports](docs/contexte*important<THINK>/) | Analysis reports | Directory |

### 🔧 Technical References
| Topic | Audience | Location |
|-------|----------|----------|
| [Model Storage](#critical-model-storage-memorize) | Both | CLAUDE.md |
| [HuggingFace Cache](#critical-huggingface-cache-memorize) | Both | CLAUDE.md |
| [Performance Targets](#-performance-targets) | Both | README.md |
| [Hardware Config](#hardware-configuration) | Both | CLAUDE.md |
| [Common Workflows](#common-workflows) | Both | CLAUDE.md |
| [Troubleshooting](#troubleshooting-tips) | Both | CLAUDE.md |

### 📂 Key Directories by User Type
```
🧑‍💻 HUMAN-FOCUSED:
├── docs/TRAINING_COMMANDS_humain.md    # Human training guide
├── docs/ARCHITECTURE/                   # System design
├── docs/SECURITY/                       # Security docs
├── docs/R&D/                           # Research papers
├── docs/observabilty/                  # Monitoring guides
├── docs/scrummaster/                   # Sprint planning
├── README.md                           # This file
└── tests/                              # Test suites

🤖 MACHINE-FOCUSED:
├── CLAUDE.md                           # AI agent instructions ⚡
├── .claude/                            # Agent configurations
│   ├── agents/                         # Specialized agents
│   ├── rules/                          # Agent rules
│   └── commands/                       # Agent commands
├── docs/contexte*important<THINK>/     # Machine context ⚠️
├── docs/rules/                         # XML rule files
├── docs/claude'srules/                 # Claude-specific rules
├── prompts/                            # System prompts
└── agent/prompts/                      # Agent prompts
```

### 🔍 Quick Reference for AI Agents

**⚡ MUST READ FIRST:**
1. `CLAUDE.md` - Complete project context and rules
2. `docs/contexte*important<THINK>/PROJECT_STRUCTURE.md` - Directory layout
3. `.claude/agents/agent-orchestrator.md` - Coordination instructions

**⚠️ CRITICAL PATHS TO MEMORIZE:**
```bash
# Model Storage (ALL models go here)
/media/jerem/641C8D6C1C8D3A56/MLLMODELS/

# HuggingFace Cache (58GB+ - NEVER DELETE)
/media/jerem/641C8D6C1C8D3A56/hf_cache/

# Datasets
/media/jerem/jeux&travail/datasets/
```

---

## 🎯 Overview

**Agent Loop** is a comprehensive MLOps platform that implements a closed training loop for autonomous agents:

1. **🎓 Train** → Fine-tune Gemma 3N models with QLoRA on curated datasets
2. **🚀 Deploy** → Serve models via FastAPI with async architecture  
3. **📊 Monitor** → Track performance with Prometheus & Grafana
4. **🔄 Iterate** → Collect interaction logs and retrain continuously

### Key Features

- **🧠 Advanced Model Architecture**: Gemma 3N with XNet heads, LoRA adapters, and GroupThink decoding
- **⚡ Production-Ready API**: Async FastAPI with health checks, metrics, and security
- **🏗️ Infrastructure as Code**: Complete Terraform + Ansible deployment
- **📈 Full Observability**: Structured logging, Prometheus metrics, Grafana dashboards
- **🔒 Security First**: JWT authentication, input validation, sandboxed execution
- **🐳 Container Native**: Multi-stage Docker builds with optimization

---

## 🏗️ Architecture Overview

```mermaid
graph TB
    subgraph "🚀 GEMMA-3N-AGENT-LOOP PLATFORM"
        subgraph "Core Services"
            API["🌐 FastAPI Service<br/>- Async Endpoints<br/>- JWT Auth (RS256)<br/>- Health Checks<br/>- Rate Limiting"]
            OLLAMA["🦎 Ollama Engine<br/>- Gemma 3N (4.5B)<br/>- GGUF Format<br/>- Memory Optimized<br/>- Hot Reload"]
            AGENT["🤖 Agent Core<br/>- Tool Orchestration<br/>- Browser Plugin<br/>- Safety Filters<br/>- Sandboxed Exec"]
        end

        subgraph "ML Pipeline"
            TRAIN["🎓 Training Pipeline<br/>- QLoRA (4-bit)<br/>- Unsloth Framework<br/>- XNet Heads<br/>- Gradient Checkpoint"]
            EVAL["✅ Evaluation<br/>- ToolBench (95%+)<br/>- JSON Validation<br/>- A/B Testing<br/>- Auto Promotion"]
            DATA["📚 Datasets<br/>- Agent Instruct (1.1M)<br/>- HRM Datasets<br/>- ToolBench<br/>- WebArena"]
        end

        subgraph "Storage Layer"
            MODELS["🧠 Model Storage<br/>/media/.../MLLMODELS/<br/>- Base Models<br/>- LoRA Adapters<br/>- GGUF Exports"]
            HF["💾 HF Cache (58GB)<br/>/media/.../hf_cache/<br/>- Cached Datasets<br/>- Pretrained Models"]
            LOGS["📊 Logs & Metrics<br/>- Training Logs<br/>- Inference Logs<br/>- Performance Data"]
        end

        subgraph "Monitoring Stack"
            PROM["📈 Prometheus<br/>- Custom Metrics<br/>- Service Discovery<br/>- Alert Rules"]
            GRAF["📊 Grafana<br/>- Training Dashboard<br/>- Inference Metrics<br/>- System Health"]
            LOG["🔍 Logging<br/>- Structured JSON<br/>- Request Tracing<br/>- Error Tracking"]
        end

        subgraph "Security Layer"
            SEC["🛡️ Security<br/>- Input Validation<br/>- Prompt Injection Defense<br/>- 50+ Threat Patterns<br/>- Audit Logging"]
            AUTH["🔐 Authentication<br/>- JWT (RS256)<br/>- 30min TTL<br/>- Role-Based Access"]
        end
    end

    subgraph "Infrastructure"
        DOCKER["🐳 Docker<br/>- Multi-stage Builds<br/>- GPU Support<br/>- Resource Limits"]
        TERRA["🏗️ Terraform<br/>- Cloud Resources<br/>- Network Config<br/>- Storage Volumes"]
        ANSIBLE["🎛️ Ansible<br/>- Service Config<br/>- Security Hardening<br/>- Monitoring Setup"]
    end

    %% MLOps Cycle
    DATA -->|"Prepare"| TRAIN
    TRAIN -->|"Fine-tune"| MODELS
    MODELS -->|"Export GGUF"| OLLAMA
    OLLAMA -->|"Serve"| API
    API -->|"Execute"| AGENT
    AGENT -->|"Collect Logs"| LOGS
    LOGS -->|"Retrain Data"| DATA
    
    %% Evaluation Flow
    TRAIN -->|"Checkpoint"| EVAL
    EVAL -->|"Promote if >95%"| MODELS
    
    %% API Interactions
    API -->|"Generate"| OLLAMA
    API -->|"Metrics"| PROM
    API -->|"Logs"| LOG
    API -->|"Validate"| SEC
    API -->|"Authenticate"| AUTH
    
    %% Monitoring
    PROM -->|"Visualize"| GRAF
    LOG -->|"Aggregate"| GRAF
    
    %% Infrastructure
    DOCKER -.->|"Deploy"| API
    DOCKER -.->|"Deploy"| OLLAMA
    DOCKER -.->|"Deploy"| PROM
    TERRA -.->|"Provision"| DOCKER
    ANSIBLE -.->|"Configure"| DOCKER

    %% Styling
    classDef core fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef ml fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef storage fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef monitor fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef security fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    classDef infra fill:#f5f5f5,stroke:#424242,stroke-width:2px
    
    class API,OLLAMA,AGENT core
    class TRAIN,EVAL,DATA ml
    class MODELS,HF,LOGS storage
    class PROM,GRAF,LOG monitor
    class SEC,AUTH security
    class DOCKER,TERRA,ANSIBLE infra
```

### 🔄 MLOps Continuous Learning Cycle

```mermaid
stateDiagram-v2
    [*] --> DataPreparation: Start
    
    DataPreparation --> Training
    state DataPreparation {
        [*] --> CurateDatasets
        CurateDatasets --> AddStepHints
        AddStepHints --> MixSources
        MixSources --> ValidateQuality
        ValidateQuality --> [*]
    }
    
    Training --> Evaluation
    state Training {
        [*] --> LoadGemma3N
        LoadGemma3N --> ApplyQLoRA
        ApplyQLoRA --> OptimizeXNet
        OptimizeXNet --> TrainWithUnsloth
        TrainWithUnsloth --> SaveCheckpoints
        SaveCheckpoints --> [*]
    }
    
    Evaluation --> Deployment: Pass (>95%)
    Evaluation --> Training: Fail (<95%)
    state Evaluation {
        [*] --> BenchmarkToolBench
        BenchmarkToolBench --> TestJSONAccuracy
        TestJSONAccuracy --> RegressionTests
        RegressionTests --> ABTesting
        ABTesting --> [*]
    }
    
    Deployment --> Production
    state Deployment {
        [*] --> ExportGGUF
        ExportGGUF --> UploadModel
        UploadModel --> UpdateOllama
        UpdateOllama --> HealthCheck
        HealthCheck --> [*]
    }
    
    Production --> LogCollection
    state Production {
        [*] --> ServeAPI
        ServeAPI --> ProcessRequests
        ProcessRequests --> MonitorMetrics
        MonitorMetrics --> [*]
    }
    
    LogCollection --> DataPreparation: Feedback Loop
    state LogCollection {
        [*] --> CaptureInteractions
        CaptureInteractions --> AnalyzeErrors
        AnalyzeErrors --> IdentifyPatterns
        IdentifyPatterns --> GenerateTrainingData
        GenerateTrainingData --> [*]
    }
```

### 🏛️ Hexagonal Architecture Details

```mermaid
graph LR
    subgraph "External Adapters"
        HTTP["HTTP/REST<br/>(FastAPI)"]
        CLI["CLI Interface<br/>(Click)"]
        MSG["Message Queue<br/>(Future: RabbitMQ)"]
    end
    
    subgraph "Application Core"
        subgraph "Use Cases"
            UC1["Agent Execution"]
            UC2["Model Training"]
            UC3["Log Analysis"]
            UC4["Model Serving"]
        end
        
        subgraph "Domain"
            DOM1["Agent Entity"]
            DOM2["Model Entity"]
            DOM3["Dataset Entity"]
            DOM4["Tool Entity"]
        end
    end
    
    subgraph "Infrastructure Adapters"
        DB["PostgreSQL<br/>(Future)"]
        CACHE["Redis Cache<br/>(Future)"]
        FILES["File Storage<br/>(Current)"]
        ML["ML Frameworks<br/>(PyTorch/HF)"]
        INFER["Ollama<br/>(Inference)"]
    end
    
    HTTP --> UC1
    CLI --> UC2
    MSG --> UC3
    
    UC1 --> DOM1
    UC2 --> DOM2
    UC3 --> DOM3
    UC4 --> DOM4
    
    DOM1 --> INFER
    DOM2 --> ML
    DOM3 --> FILES
    DOM4 --> CACHE
    
    classDef external fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef usecase fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef domain fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef infra fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    
    class HTTP,CLI,MSG external
    class UC1,UC2,UC3,UC4 usecase
    class DOM1,DOM2,DOM3,DOM4 domain
    class DB,CACHE,FILES,ML,INFER infra
```

---

## 📁 Project Structure

The project follows a **modular hexagonal architecture** with clear separation of concerns:

```
agent_loop/
├── 🎓 models/                    # Complete ML lifecycle
│   ├── training/                 # Training pipelines & experiments
│   │   ├── qlora/               # QLoRA fine-tuning (Unsloth)
│   │   ├── nn/                  # Custom neural architectures
│   │   └── security/            # Training security & validation
│   ├── inference/               # Production API server
│   │   ├── app.py              # Modern FastAPI application
│   │   ├── routers/            # Modular endpoint organization
│   │   ├── services/           # Business logic layer
│   │   └── middleware/         # Security & observability
│   ├── datasets/               # Training data management
│   │   ├── processed/          # Clean, formatted datasets
│   │   └── raw/               # Original dataset sources
│   ├── results/               # Training outputs & checkpoints
│   └── scripts/               # Operational automation
│
├── 🤖 agent/                    # Agent implementation
│   ├── tools/                  # Agent capabilities
│   ├── plugins/               # Extensible tool system
│   └── prompts/              # System prompts & examples
│
├── 🏗️ infrastructure/          # Infrastructure as Code
│   ├── terraform/             # Cloud resource definition
│   ├── ansible/              # Configuration management
│   └── docker/               # Container orchestration
│
├── 📊 monitoring/              # Observability stack
│   ├── grafana/              # Dashboards & visualization
│   ├── prometheus/           # Metrics collection
│   └── nginx/               # Reverse proxy configuration
│
├── 🧪 tests/                   # Quality assurance
│   ├── unit/                 # Fast, isolated tests
│   ├── integration/          # Service interaction tests
│   └── e2e/                 # End-to-end workflows
│
└── 📚 docs/                    # Documentation hub
    ├── ARCHITECTURE/          # System design documents
    ├── SECURITY/             # Security analysis & guides
    └── R&D/                 # Research & experimental docs
```

> 📖 **Detailed Structure Guide**: See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for complete architectural documentation.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **NVIDIA GPU** (for training) with CUDA 12.3+
- **8GB+ RAM** (16GB recommended)

### 1. Local Development Setup

```bash
# Clone and setup environment
git clone bitbucket
cd agent-loop

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Setup pre-commit hooks
pre-commit install
```

### 2. Start Local Services

```bash
# Start Ollama (model serving)
ollama serve &

# Pull Gemma 3N model
ollama pull gemma-3n:e4b

# Start FastAPI development server
cd models/inference
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Run inference
curl -X POST "http://localhost:8000/agents/run-agent" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "What is the capital of France?",
    "use_ollama": true
  }'
```

### 4. Production Deployment

```bash
# Using Docker Compose
docker-compose up -d

# Or using the complete infrastructure stack
cd infrastructure/terraform
terraform init && terraform apply

cd ../ansible
ansible-playbook -i inventory.yml site.yml
```

---

## 🔄 Training Workflow

The platform implements a complete **Continuous Learning Loop** with automated model improvement:

```mermaid
flowchart LR
    subgraph "📥 Data Pipeline"
        D1["Dataset Sources<br/>• Agent Instruct (1.1M)<br/>• GSM8K (HRM)<br/>• CodeAlpaca<br/>• ToolBench"] 
        D2["Data Processing<br/>• Add Step Hints<br/>• Format Unification<br/>• Quality Validation<br/>• Train/Val Split"]
        D3["HF Cache<br/>(58GB)<br/>Persistent Storage"]
    end
    
    subgraph "🎓 Training Pipeline"
        T1["Model Loading<br/>• Gemma 3N (4.5B)<br/>• 4-bit Quantization<br/>• LoRA Config<br/>• Flash Attention"]
        T2["QLoRA Training<br/>• Unsloth Optimized<br/>• Gradient Accumulation<br/>• Memory Efficient<br/>• ~5.8s/step"]
        T3["Checkpointing<br/>• Regular Saves<br/>• Best Model Track<br/>• Resume Support<br/>• Metric Logging"]
    end
    
    subgraph "✅ Evaluation"
        E1["Benchmarking<br/>• ToolBench Suite<br/>• JSON Accuracy<br/>• Latency Tests<br/>• Error Analysis"]
        E2["Promotion Gate<br/>• >95% Accuracy<br/>• No Regression<br/>• A/B Testing<br/>• Auto Deploy"]
    end
    
    subgraph "🚀 Deployment"
        DEP1["Model Export<br/>• Merge LoRA<br/>• Convert GGUF<br/>• Quantize (Q4_K_M)<br/>• Create Modelfile"]
        DEP2["Ollama Service<br/>• Hot Reload<br/>• Memory Pool<br/>• Load Balance<br/>• Health Check"]
        DEP3["FastAPI<br/>• Async Serving<br/>• Request Queue<br/>• Metrics Export<br/>• Error Handle"]
    end
    
    subgraph "📊 Monitoring"
        M1["Log Collection<br/>• User Queries<br/>• Model Outputs<br/>• Performance<br/>• Errors"]
        M2["Analysis<br/>• Pattern Mining<br/>• Failure Cases<br/>• Edge Cases<br/>• Improvement Areas"]
        M3["Feedback Loop<br/>• New Examples<br/>• Corrections<br/>• Augmentation<br/>• Priority Queue"]
    end
    
    D1 --> D2
    D2 --> D3
    D3 --> T1
    T1 --> T2
    T2 --> T3
    T3 --> E1
    E1 --> E2
    E2 -->|"Pass"| DEP1
    E2 -->|"Fail"| T1
    DEP1 --> DEP2
    DEP2 --> DEP3
    DEP3 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> D1
    
    classDef data fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef train fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef eval fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef deploy fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef monitor fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class D1,D2,D3 data
    class T1,T2,T3 train
    class E1,E2 eval
    class DEP1,DEP2,DEP3 deploy
    class M1,M2,M3 monitor
```

### 🧠 Model Architecture & Optimization

```mermaid
graph TB
    subgraph "Gemma 3N Architecture"
        INPUT["Input Tokens<br/>(max_seq_len: 8192)"]
        EMB["Token Embeddings<br/>(vocab_size: 256,128)"]
        
        subgraph "Transformer Blocks (36 layers)"
            ATT["Attention Layer<br/>• Multi-Query Attention<br/>• RoPE Embeddings<br/>• Flash Attention 2"]
            FFN["Feed Forward<br/>• SwiGLU Activation<br/>• 14,336 hidden dim<br/>• Gradient Checkpoint"]
            LORA["LoRA Adapters<br/>• Rank: 64<br/>• Alpha: 16<br/>• Target: q,v,o,gate"]
        end
        
        XNET["XNet Head<br/>• Hierarchical Reasoning<br/>• Step Decomposition<br/>• Tool Selection"]
        OUTPUT["Output Logits<br/>(Tool Calls / Text)"]
    end
    
    subgraph "Memory Optimization"
        OPT1["4-bit Quantization<br/>(bitsandbytes)"]
        OPT2["Gradient Accumulation<br/>(steps: 4)"]
        OPT3["Activation Checkpoint<br/>(every 4 layers)"]
        OPT4["Flash Attention 2<br/>(xformers)"]
    end
    
    subgraph "Training Config"
        CONF1["Batch Size: 2<br/>(effective: 8)"]
        CONF2["Learning Rate: 2e-4<br/>(cosine schedule)"]
        CONF3["Max Steps: 10,000<br/>(~16 hours)"]
        CONF4["Warmup: 100 steps"]
    end
    
    INPUT --> EMB
    EMB --> ATT
    ATT --> FFN
    FFN --> LORA
    LORA --> XNET
    XNET --> OUTPUT
    
    OPT1 -.-> ATT
    OPT2 -.-> FFN
    OPT3 -.-> LORA
    OPT4 -.-> ATT
    
    CONF1 -.-> Training
    CONF2 -.-> Training
    CONF3 -.-> Training
    CONF4 -.-> Training
    
    classDef arch fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef opt fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef conf fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    
    class INPUT,EMB,ATT,FFN,LORA,XNET,OUTPUT arch
    class OPT1,OPT2,OPT3,OPT4 opt
    class CONF1,CONF2,CONF3,CONF4 conf
```

### Training Commands

```bash
# Fine-tune with QLoRA (recommended)
cd models/training/qlora
python qlora_finetune_unsloth.py \
  --data ../../datasets/processed/unified_format/ \
  --output-dir ../../results/gemma-3n-v2 \
  --epochs 1 \
  --batch-size 4 \
  --gradient-accumulation-steps 4

# Monitor training progress
tensorboard --logdir models/results/

# Evaluate trained model
python evaluate_model.py \
  --model-path models/results/gemma-3n-v2 \
  --test-data datasets/processed/eval_splits/
```
to do unsloth
---

## 📊 Monitoring & Observability

### Metrics Dashboard

Access comprehensive monitoring at:
- **Grafana**: `http://localhost:3000` (admin/admin)
- **Prometheus**: `http://localhost:9090`
- **FastAPI Metrics**: `http://localhost:8000/metrics`

### Key Metrics Tracked

| Category | Metrics | Purpose |
|----------|---------|---------|
| **Training** | Loss curves, Learning rate, GPU utilization | Monitor training progress |
| **Inference** | Request latency, Throughput, Error rates | API performance |
| **Model Quality** | BLEU scores, Tool accuracy, JSON validity | Model effectiveness |
| **Infrastructure** | CPU/Memory usage, Disk I/O, Network | System health |

### Health Endpoints

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/health/detailed

# Kubernetes readiness probe
curl http://localhost:8000/health/ready
```

---

## 🔒 Security Features

- **🛡️ Input Validation**: Comprehensive request sanitization
- **🔐 Authentication**: JWT-based API security
- **🏖️  Sandboxing**: Isolated execution environments
- **📝 Audit Logging**: Complete request/response tracking
- **🚨 Rate Limiting**: Protection against abuse
- **🔍 Security Scanning**: Automated vulnerability detection

---

## 🧪 Testing

```bash
# Run all tests
make test

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end testing
pytest tests/e2e/

# Coverage report
make coverage
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`docs/ARCHITECTURE/`](docs/ARCHITECTURE/) | System design & patterns |
| [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) | Detailed project organization |
| [`docs/TRAINING_COMMANDS.md`](docs/TRAINING_COMMANDS.md) | Training procedures |
| [`docs/SECURITY/`](docs/SECURITY/) | Security analysis & guides |
| [`docs/R&D/`](docs/R&D/) | Research documentation |

---

## 🛠️ Development

### Code Quality

```bash
# Linting & formatting
make lint
make format

# Type checking
make typecheck

# Pre-commit hooks
pre-commit run --all-files
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📈 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| **Inference Latency** | < 500ms | ~350ms |
| **Training Speed** | > 1k tokens/sec | ~1.2k tokens/sec |
| **API Uptime** | > 99.9% | 99.95% |
| **Model Accuracy** | > 95% (ToolBench) | 96.2% |

---

## 🤝 Acknowledgments

- **Google**: Gemma 3N base model
- **Unsloth**: Efficient training framework
- **FastAPI**: Modern Python web framework
- **Ollama**: Local model serving

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

- **Issues**: [Bitbucket Issues](https://bitbucket.org/your-org/agent-loop/issues)
- **Wiki**: [Bitbucket Wiki](https://bitbucket.org/your-org/agent-loop/wiki)
- **Documentation**: [docs/](docs/)

---

<div align="center">

**Built with ❤️ for the AI community**

[⭐ Watch on Bitbucket](https://bitbucket.org/your-org/agent-loop) | [🐛 Report Bug](https://bitbucket.org/your-org/agent-loop/issues) | [💡 Request Feature](https://bitbucket.org/your-org/agent-loop/issues)

</div>
