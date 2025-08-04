# ============================================================================
# MLOps Pipeline Makefile - Agent Loop Project
# ============================================================================

.PHONY: help install lint format test clean build deploy monitoring docs-latex docs-setup docs-clean
.DEFAULT_GOAL := help

# Variables
PYTHON := python3.13
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
DOCKER := docker
COMPOSE := docker compose
PROJECT_NAME := agent-loop

# Colors for terminal output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================================================
# Help & Setup
# ============================================================================

help: ## Show this help message
	@echo "$(BLUE)🤖 Agent Loop MLOps Pipeline$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Development Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make install         # Install all dependencies"
	@echo "  make lint            # Run all code quality checks"
	@echo "  make test            # Run full test suite"
	@echo "  make build-dev       # Build and start development environment"
	@echo "  make deploy-staging  # Deploy to staging environment"

install: ## Install dependencies and setup development environment
	@echo "$(BLUE)[SETUP]$(NC) Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(BLUE)[SETUP]$(NC) Installing pre-commit hooks..."
	pre-commit install
	@echo "$(BLUE)[SETUP]$(NC) Installing Playwright browsers..."
	playwright install --with-deps chromium
	@echo "$(GREEN)[SUCCESS]$(NC) Development environment ready!"

install-prod: ## Install production dependencies only
	$(PIP) install --upgrade pip
	$(PIP) install --no-dev -r requirements.txt

# ============================================================================
# Code Quality & Static Analysis
# ============================================================================

format: ## Format code with Black and isort
	@echo "$(BLUE)[FORMAT]$(NC) Running Black formatter..."
	black .
	@echo "$(BLUE)[FORMAT]$(NC) Running isort..."
	isort .
	@echo "$(GREEN)[SUCCESS]$(NC) Code formatted!"

lint: ## Run all linting and static analysis
	@echo "$(BLUE)[LINT]$(NC) Running code quality checks..."
	@echo "$(YELLOW)→ Black (formatting check)$(NC)"
	@black --check --diff . || (echo "$(RED)[ERROR] Code formatting issues found. Run 'make format'$(NC)" && exit 1)
	
	@echo "$(YELLOW)→ Ruff (linting)$(NC)"
	@ruff check . || (echo "$(RED)[ERROR] Linting issues found$(NC)" && exit 1)
	
	@echo "$(YELLOW)→ MyPy (type checking)$(NC)"
	@mypy --config-file pyproject.toml . || (echo "$(RED)[ERROR] Type checking failed$(NC)" && exit 1)
	
	@echo "$(YELLOW)→ Bandit (security analysis)$(NC)"
	@bandit -r . -ll || (echo "$(RED)[ERROR] Security issues found$(NC)" && exit 1)
	
	@echo "$(GREEN)[SUCCESS]$(NC) All quality checks passed!"

lint-fix: ## Fix linting issues automatically where possible
	@echo "$(BLUE)[LINT-FIX]$(NC) Auto-fixing linting issues..."
	ruff check --fix .
	black .
	isort .
	@echo "$(GREEN)[SUCCESS]$(NC) Auto-fixes applied!"

security: ## Run security analysis
	@echo "$(BLUE)[SECURITY]$(NC) Running security analysis..."
	bandit -r . -f json -o bandit-report.json
	bandit -r . -ll
	@echo "$(GREEN)[SUCCESS]$(NC) Security analysis complete!"

# ============================================================================
# Testing
# ============================================================================

test: ## Run full test suite
	@echo "$(BLUE)[TEST]$(NC) Running full test suite..."
	$(PYTEST) tests/ \
		--cov=core --cov=inference --cov=plugins --cov=training \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		--junit-xml=pytest-results.xml \
		-v

test-unit: ## Run only unit tests
	@echo "$(BLUE)[TEST]$(NC) Running unit tests..."
	$(PYTEST) tests/ -m "unit" \
		--cov=core --cov=inference --cov=plugins --cov=training \
		--cov-report=term-missing \
		-v

test-integration: ## Run only integration tests
	@echo "$(BLUE)[TEST]$(NC) Running integration tests..."
	$(PYTEST) tests/ -m "integration" -v

test-performance: ## Run performance benchmarks
	@echo "$(BLUE)[TEST]$(NC) Running performance benchmarks..."
	$(PYTEST) tests/ -m "benchmark" \
		--benchmark-only \
		--benchmark-sort=mean \
		--benchmark-json=benchmark-results.json

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)[TEST]$(NC) Running tests in parallel..."
	$(PYTEST) tests/ -n auto \
		--cov=core --cov=inference --cov=plugins --cov=training \
		--cov-report=term-missing

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)[TEST]$(NC) Running tests in watch mode..."
	$(PYTEST) tests/ --looponfail

# ============================================================================
# Docker & Development Environment
# ============================================================================

build-dev: ## Build and start development environment
	@echo "$(BLUE)[DOCKER]$(NC) Building development environment..."
	./scripts/docker-dev.sh
	@echo "$(GREEN)[SUCCESS]$(NC) Development environment started!"

build-prod: ## Build production Docker images
	@echo "$(BLUE)[DOCKER]$(NC) Building production images..."
	$(DOCKER) build --target runtime -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)[SUCCESS]$(NC) Production images built!"

stop: ## Stop all running containers
	@echo "$(BLUE)[DOCKER]$(NC) Stopping containers..."
	$(COMPOSE) down
	./scripts/docker-dev.sh stop
	@echo "$(GREEN)[SUCCESS]$(NC) Containers stopped!"

logs: ## Show container logs
	$(COMPOSE) logs -f

status: ## Show container status
	$(COMPOSE) ps
	@echo ""
	@echo "$(YELLOW)Service Health:$(NC)"
	@curl -f http://localhost:8000/health 2>/dev/null && echo "$(GREEN)✓$(NC) FastAPI: healthy" || echo "$(RED)✗$(NC) FastAPI: unhealthy"
	@curl -f http://localhost:11434/api/version 2>/dev/null && echo "$(GREEN)✓$(NC) Ollama: healthy" || echo "$(RED)✗$(NC) Ollama: unhealthy"
	@curl -f http://localhost:9090/-/healthy 2>/dev/null && echo "$(GREEN)✓$(NC) Prometheus: healthy" || echo "$(RED)✗$(NC) Prometheus: unhealthy"
	@curl -f http://localhost:3000/api/health 2>/dev/null && echo "$(GREEN)✓$(NC) Grafana: healthy" || echo "$(RED)✗$(NC) Grafana: unhealthy"

# ============================================================================
# Machine Learning Pipeline
# ============================================================================

train: ## Run model training pipeline
	@echo "$(BLUE)[ML]$(NC) Starting model training..."
	$(PYTHON) training/qlora_finetune.py \
		--base gemma_base.gguf \
		--head xnet \
		--data ./data/training_base \
		--step-hint \
		--lambda_hint 0.3
	@echo "$(GREEN)[SUCCESS]$(NC) Training completed!"

# ============================================================================
# QLoRA Fine-tuning with Docker - Sprint 2
# ============================================================================

train-docker: ## Run QLoRA fine-tuning in Docker container
	@echo "$(BLUE)[QLORA]$(NC) Starting QLoRA fine-tuning with Docker GPU..."
	$(COMPOSE) -f docker-compose.training.yml --profile training up --build
	@echo "$(GREEN)[SUCCESS]$(NC) QLoRA training completed!"

train-dev: ## Start interactive training environment with Jupyter
	@echo "$(BLUE)[QLORA-DEV]$(NC) Starting development training environment..."
	$(COMPOSE) -f docker-compose.training.yml --profile training-dev up -d
	@echo "$(GREEN)[SUCCESS]$(NC) Development environment available:"
	@echo "  📚 Jupyter Lab: http://localhost:8888 (token: agent-dev-2024)"
	@echo "  📊 TensorBoard: http://localhost:6006"
	@echo "  🔍 W&B Monitor: http://localhost:8080"

train-stop: ## Stop training containers and cleanup
	@echo "$(BLUE)[QLORA]$(NC) Stopping training containers..."
	$(COMPOSE) -f docker-compose.training.yml down
	@echo "$(GREEN)[SUCCESS]$(NC) Training containers stopped!"

train-logs: ## Show training container logs
	@echo "$(BLUE)[QLORA]$(NC) Showing training logs..."
	$(COMPOSE) -f docker-compose.training.yml logs -f qlora-training

train-status: ## Check training container status and GPU utilization
	@echo "$(BLUE)[QLORA]$(NC) Training container status:"
	$(COMPOSE) -f docker-compose.training.yml ps
	@echo ""
	@echo "$(YELLOW)GPU Status:$(NC)"
	@nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "$(RED)GPU not available or nvidia-smi not installed$(NC)"

train-monitor: ## Start full training environment with monitoring
	@echo "$(BLUE)[QLORA]$(NC) Starting complete training environment..."
	$(COMPOSE) -f docker-compose.training.yml --profile training --profile training-dev up -d
	@echo "$(GREEN)[SUCCESS]$(NC) Complete training environment started!"
	@echo "  🚀 Training: docker logs agent-qlora-training"
	@echo "  📚 Jupyter: http://localhost:8888"
	@echo "  📊 TensorBoard: http://localhost:6006"
	@echo "  🔍 W&B Local: http://localhost:8080"
	@echo "  📈 GPU Monitor: docker logs agent-gpu-monitor"

train-clean: ## Clean training data and temporary files
	@echo "$(BLUE)[QLORA]$(NC) Cleaning training artifacts..."
	$(COMPOSE) -f docker-compose.training.yml down -v --remove-orphans
	@echo "$(YELLOW)[WARNING]$(NC) Removing training volumes - checkpoints will be lost!"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(DOCKER) volume rm agent-loop_training_models_cache agent-loop_training_checkpoints agent-loop_wandb_local_data 2>/dev/null || true
	@echo "$(GREEN)[SUCCESS]$(NC) Training cleanup completed!"

# QLoRA Configuration Commands
train-gemma-2b: ## Train Gemma 2B model with optimized QLoRA config
	@echo "$(BLUE)[QLORA]$(NC) Training Gemma 2B with QLoRA..."
	$(COMPOSE) -f docker-compose.training.yml run --rm qlora-training \
		python training/qlora_finetune.py \
		--model-config gemma-2b \
		--data /app/datasets/processed/train_splits \
		--output-dir /app/outputs/gemma-2b-qlora \
		--max-steps 1000 \
		--batch-size 4 \
		--learning-rate 2e-4 \
		--wandb-project agent-loop-qlora

train-gemma-9b: ## Train Gemma 9B model with memory-optimized config  
	@echo "$(BLUE)[QLORA]$(NC) Training Gemma 9B with optimized QLoRA..."
	$(COMPOSE) -f docker-compose.training.yml run --rm qlora-training \
		python training/qlora_finetune.py \
		--model-config gemma-9b \
		--data /app/datasets/processed/train_splits \
		--output-dir /app/outputs/gemma-9b-qlora \
		--max-steps 800 \
		--batch-size 2

train-custom: ## Train custom model (specify MODEL and DATA)
	@echo "$(BLUE)[QLORA]$(NC) Training custom model..."
	@if [ -z "$(MODEL)" ]; then echo "$(RED)[ERROR]$(NC) MODEL variable required. Usage: make train-custom MODEL=google/gemma-2-2b"; exit 1; fi
	@if [ -z "$(DATA)" ]; then echo "$(RED)[ERROR]$(NC) DATA variable required. Usage: make train-custom DATA=/path/to/data"; exit 1; fi
	$(COMPOSE) -f docker-compose.training.yml run --rm qlora-training \
		python training/qlora_finetune.py \
		--model-config custom \
		--model-name $(MODEL) \
		--data $(DATA) \
		--output-dir /app/outputs/custom-qlora

evaluate: ## Evaluate trained model
	@echo "$(BLUE)[ML]$(NC) Evaluating model..."
	$(PYTHON) training/evaluate_toolbench.py \
		--model ./models/trained_base \
		--output ./reports/evaluation.json
	@echo "$(GREEN)[SUCCESS]$(NC) Evaluation completed!"

benchmark: ## Run model performance benchmarks
	@echo "$(BLUE)[ML]$(NC) Running performance benchmarks..."
	$(PYTHON) scripts/benchmark.py \
		--endpoint http://localhost:8000 \
		--duration 60s \
		--output benchmark-results.json
	@echo "$(GREEN)[SUCCESS]$(NC) Benchmarks completed!"

# ============================================================================
# Deployment & Infrastructure
# ============================================================================

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)[DEPLOY]$(NC) Deploying to staging..."
	ansible-playbook ansible/site.yml \
		-i ansible/inventory.yml \
		-e "environment=staging" \
		--limit staging
	@echo "$(GREEN)[SUCCESS]$(NC) Staging deployment completed!"

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)[DEPLOY]$(NC) Deploying to production..."
	./scripts/docker-prod.sh deploy
	@echo "$(GREEN)[SUCCESS]$(NC) Production deployment completed!"

rollback: ## Rollback production deployment
	@echo "$(YELLOW)[ROLLBACK]$(NC) Rolling back production deployment..."
	ansible-playbook ansible/playbooks/rollback.yml \
		-i ansible/inventory.yml \
		--limit production
	@echo "$(GREEN)[SUCCESS]$(NC) Rollback completed!"

# ============================================================================
# Monitoring & Observability
# ============================================================================

monitoring: ## Setup monitoring and alerting
	@echo "$(BLUE)[MONITORING]$(NC) Setting up monitoring..."
	$(PYTHON) scripts/setup_alerts.py \
		--enable-model-drift-detection \
		--enable-performance-alerts
	@echo "$(GREEN)[SUCCESS]$(NC) Monitoring configured!"

drift-check: ## Check for model drift
	@echo "$(BLUE)[MONITORING]$(NC) Checking for model drift..."
	$(PYTHON) scripts/model_drift_detection.py \
		--endpoint http://localhost:8000 \
		--reference-data ./data/reference_dataset.json \
		--threshold 0.1 \
		--output drift-report.json
	@echo "$(GREEN)[SUCCESS]$(NC) Drift check completed!"

health-check: ## Run comprehensive health checks
	@echo "$(BLUE)[HEALTH]$(NC) Running health checks..."
	$(PYTHON) scripts/health_check.py \
		--endpoint http://localhost:8000 \
		--timeout 30 \
		--retry-interval 5
	@echo "$(GREEN)[SUCCESS]$(NC) Health checks completed!"

# ============================================================================
# Data Management
# ============================================================================

data-download: ## Download training data
	@echo "$(BLUE)[DATA]$(NC) Downloading training data..."
	$(PYTHON) training/datasets/download_data.py --config base
	@echo "$(GREEN)[SUCCESS]$(NC) Data download completed!"

data-validate: ## Validate training data
	@echo "$(BLUE)[DATA]$(NC) Validating training data..."
	$(PYTHON) scripts/validate_data.py \
		--data-dir ./data \
		--schema ./schemas/training_data.json
	@echo "$(GREEN)[SUCCESS]$(NC) Data validation completed!"

data-clean: ## Clean old training data
	@echo "$(BLUE)[DATA]$(NC) Cleaning old training data..."
	find ./data -name "*.tmp" -delete
	find ./data -name "*.cache" -delete
	@echo "$(GREEN)[SUCCESS]$(NC) Data cleanup completed!"

# ============================================================================
# Utilities & Maintenance
# ============================================================================

clean: ## Clean temporary files and caches
	@echo "$(BLUE)[CLEAN]$(NC) Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -f pytest-results.xml coverage.xml bandit-report.json
	@echo "$(GREEN)[SUCCESS]$(NC) Cleanup completed!"

clean-docker: ## Clean Docker images and containers
	@echo "$(BLUE)[CLEAN]$(NC) Cleaning Docker resources..."
	$(DOCKER) system prune -f
	$(DOCKER) image prune -f
	@echo "$(GREEN)[SUCCESS]$(NC) Docker cleanup completed!"

update-deps: ## Update Python dependencies
	@echo "$(BLUE)[UPDATE]$(NC) Updating dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	@echo "$(GREEN)[SUCCESS]$(NC) Dependencies updated!"

# ============================================================================
# Development Shortcuts
# ============================================================================

dev: install build-dev ## Complete development setup
	@echo "$(GREEN)[SUCCESS]$(NC) Development environment ready!"

ci: lint test ## Run CI pipeline locally
	@echo "$(GREEN)[SUCCESS]$(NC) CI pipeline completed!"

cd: ci build-prod ## Run full CI/CD pipeline locally
	@echo "$(GREEN)[SUCCESS]$(NC) CI/CD pipeline completed!"

quick-test: ## Quick test run (unit tests only)
	$(PYTEST) tests/ -m "unit" --tb=short -q

# ============================================================================
# Documentation
# ============================================================================

docs: ## Generate documentation
	@echo "$(BLUE)[DOCS]$(NC) Generating documentation..."
	@echo "$(YELLOW)→ Architecture overview available in docs/$(NC)"
	@echo "$(YELLOW)→ API documentation: http://localhost:8000/docs$(NC)"
	@echo "$(YELLOW)→ Monitoring dashboards: http://localhost:3000$(NC)"
	@echo "$(YELLOW)→ LaTeX documentation: make docs-latex$(NC)"
	@echo "$(GREEN)[SUCCESS]$(NC) Documentation links provided!"

# ============================================================================
# LaTeX Documentation
# ============================================================================

docs-latex: ## Compile LaTeX documentation
	@echo "$(BLUE)[DOCS]$(NC) Compiling LaTeX research papers..."
	@cd docs/latex && make all
	@echo "$(GREEN)[SUCCESS]$(NC) LaTeX documentation compiled!"

docs-setup: ## Setup LaTeX documentation environment  
	@echo "$(BLUE)[DOCS]$(NC) Setting up LaTeX environment..."
	@cd docs/latex && make setup
	@echo "$(GREEN)[SUCCESS]$(NC) LaTeX environment ready!"

docs-clean: ## Clean LaTeX build files
	@echo "$(BLUE)[DOCS]$(NC) Cleaning LaTeX build files..."
	@cd docs/latex && make clean
	@echo "$(GREEN)[SUCCESS]$(NC) LaTeX cleanup completed!"

# ============================================================================
# Project Information
# ============================================================================

info: ## Show project information
	@echo "$(BLUE)🤖 Agent Loop MLOps Pipeline$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo "$(YELLOW)Project:$(NC) $(PROJECT_NAME)"
	@echo "$(YELLOW)Python:$(NC) $(shell $(PYTHON) --version)"
	@echo "$(YELLOW)Docker:$(NC) $(shell $(DOCKER) --version)"
	@echo "$(YELLOW)Environment:$(NC) $(shell pwd)"
	@echo ""
	@echo "$(YELLOW)Service URLs:$(NC)"
	@echo "  FastAPI: http://localhost:8000"
	@echo "  Ollama: http://localhost:11434"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000 (admin/admin123)"
	@echo ""
	@echo "$(YELLOW)Key Commands:$(NC)"
	@echo "  make dev     # Setup development environment"
	@echo "  make ci      # Run CI pipeline locally"
	@echo "  make deploy  # Deploy to production"