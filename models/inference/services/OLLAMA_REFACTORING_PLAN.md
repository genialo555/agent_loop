# Ollama Service Refactoring Plan

## Overview
This document outlines the refactoring strategy for the Ollama service to improve separation of concerns, testability, and maintainability.

## Current Issues
1. Mixed responsibilities in a single class
2. Hardcoded configuration values
3. Complex error handling mixed with business logic
4. No proper request/response models
5. Health check caching embedded in service
6. Limited dependency injection support

## Target Architecture

### 1. Core Components

#### OllamaConfig (Pydantic Settings)
```python
# models/inference/config/ollama.py
class OllamaConfig(BaseSettings):
    base_url: str = "http://127.0.0.1:11434"
    default_model: str = "gemma3n:e2b"
    timeout: float = 120.0
    health_check_ttl: float = 30.0
    keep_alive: str = "5m"
    # ... other settings
```

#### OllamaClient (HTTP Client Abstraction)
```python
# models/inference/clients/ollama.py
class OllamaClient:
    """Pure HTTP client for Ollama API."""
    async def generate(request: GenerateRequest) -> GenerateResponse
    async def get_version() -> VersionResponse
    async def list_models() -> ModelsResponse
    async def show_model(name: str) -> ModelInfoResponse
```

#### OllamaHealthMonitor (Health Check Logic)
```python
# models/inference/monitors/ollama_health.py
class OllamaHealthMonitor:
    """Dedicated health monitoring with caching."""
    async def check_health(force: bool = False) -> HealthStatus
    async def detailed_health() -> DetailedHealthStatus
```

#### OllamaInferenceService (Business Logic)
```python
# models/inference/services/ollama_inference.py
class OllamaInferenceService:
    """High-level inference operations."""
    async def generate_completion(prompt: str, options: InferenceOptions) -> InferenceResult
    async def generate_with_context(messages: List[Message]) -> InferenceResult
```

#### OllamaModelManager (Model Operations)
```python
# models/inference/services/ollama_models.py
class OllamaModelManager:
    """Model management operations."""
    async def list_models() -> List[ModelInfo]
    async def switch_model(model_name: str) -> None
    async def get_current_model() -> ModelInfo
```

### 2. Schema Models

#### Request/Response Models
```python
# models/inference/models/ollama_schemas.py
class GenerateRequest(BaseModel)
class GenerateResponse(BaseModel)
class InferenceOptions(BaseModel)
class ModelInfo(BaseModel)
class HealthStatus(BaseModel)
# ... etc
```

### 3. Dependency Injection

```python
# models/inference/services/dependencies.py
def get_ollama_config() -> OllamaConfig
def get_ollama_client(config: OllamaConfig, http_client: AsyncClient) -> OllamaClient
def get_ollama_health_monitor(client: OllamaClient) -> OllamaHealthMonitor
def get_ollama_inference_service(client: OllamaClient, config: OllamaConfig) -> OllamaInferenceService
def get_ollama_model_manager(client: OllamaClient) -> OllamaModelManager
```

## Implementation Steps

### Phase 1: Create Schema Models
1. Define all Pydantic models for Ollama API
2. Create configuration model
3. Define inference options and results

### Phase 2: Extract HTTP Client
1. Create pure OllamaClient class
2. Move all HTTP operations to client
3. Add proper error handling and retries

### Phase 3: Separate Health Monitoring
1. Extract health check logic to OllamaHealthMonitor
2. Implement caching mechanism
3. Add detailed health metrics

### Phase 4: Create Service Layer
1. Implement OllamaInferenceService for inference logic
2. Create OllamaModelManager for model operations
3. Add proper logging and metrics

### Phase 5: Update Routers
1. Update existing routers to use new services
2. Implement proper dependency injection
3. Add new endpoints as needed

### Phase 6: Testing
1. Create unit tests for each component
2. Add integration tests
3. Create mock fixtures for Ollama API

## Benefits
1. **Separation of Concerns**: Each component has a single responsibility
2. **Testability**: Components can be tested in isolation
3. **Configurability**: All settings externalized to configuration
4. **Extensibility**: Easy to add new features without modifying existing code
5. **Maintainability**: Clear structure and dependencies

## Backward Compatibility
- Maintain existing API endpoints
- Keep same response formats
- Gradual migration path for consumers

## Agent Assignments

### System Architect
- Review and approve overall architecture
- Ensure hexagonal architecture principles
- Define interfaces and contracts

### FastAPI Async Architect
- Implement async patterns
- Design connection pooling
- Handle dependency injection

### Python Type Guardian
- Create all Pydantic models
- Add comprehensive type hints
- Ensure type safety

### Test Automator
- Design test strategy
- Create test fixtures
- Implement test suites

### Observability Engineer
- Add logging and metrics
- Implement tracing
- Create monitoring dashboards