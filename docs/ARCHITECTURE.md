# Gemma-3N-Agent-Loop Architecture

## Table of Contents

1. [Overview](#overview)
2. [Architectural Principles](#architectural-principles)
3. [System Architecture](#system-architecture)
4. [Component Design](#component-design)
5. [Data Flow](#data-flow)
6. [Deployment Architecture](#deployment-architecture)
7. [Security Architecture](#security-architecture)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Architecture Decision Records](#architecture-decision-records)

## Overview

The Gemma-3N-Agent-Loop system implements a continuous learning AI agent using hexagonal architecture principles. The system is designed to be maintainable, scalable, and testable while supporting real-time inference and asynchronous training cycles.

## Architectural Principles

### Hexagonal Architecture (Ports and Adapters)

The system follows hexagonal architecture to ensure:
- **Domain Independence**: Core business logic is isolated from external dependencies
- **Testability**: All components can be tested in isolation using test doubles
- **Flexibility**: External systems can be swapped without affecting the core

```
┌─────────────────────────────────────────────────────────────┐
│                        External Systems                      │
│  (Users, APIs, Databases, Message Queues, File Systems)    │
└─────────────────────────┬───────────────┬───────────────────┘
                          │               │
                    ┌─────▼─────┐   ┌─────▼─────┐
                    │  Adapters  │   │  Adapters  │
                    │ (Inbound)  │   │ (Outbound) │
                    └─────┬─────┘   └─────┬─────┘
                          │               │
                    ┌─────▼───────────────▼─────┐
                    │         Ports             │
                    │  (Interfaces/Contracts)   │
                    └─────┬───────────────┬─────┘
                          │               │
                    ┌─────▼───────────────▼─────┐
                    │    Application Layer      │
                    │    (Use Cases/Services)   │
                    └─────┬───────────────┬─────┘
                          │               │
                    ┌─────▼───────────────▼─────┐
                    │      Domain Layer         │
                    │  (Entities, Value Objects,│
                    │   Domain Services)        │
                    └───────────────────────────┘
```

### Clean Architecture Layers

1. **Domain Layer** (`core/domain/`)
   - Entities: Agent, Tool, Conversation, TrainingData
   - Value Objects: ModelConfig, ToolResult, AgentResponse
   - Domain Services: InferenceService, TrainingService
   - Domain Events: ToolExecuted, ResponseGenerated, TrainingCompleted

2. **Application Layer** (`core/application/`)
   - Use Cases: ExecuteToolUseCase, GenerateResponseUseCase, StartTrainingUseCase
   - Application Services: AgentOrchestrator, TrainingCoordinator
   - DTOs: Request/Response objects for use cases

3. **Infrastructure Layer** (`infrastructure/`)
   - Repositories: ConversationRepository, TrainingDataRepository
   - External Services: OllamaAdapter, PrometheusAdapter
   - Message Queue: TrainingQueueAdapter

4. **Interface Adapters** (`adapters/`)
   - REST API: FastAPI controllers
   - CLI: Command-line interface
   - WebSocket: Real-time communication

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Applications                     │
│              (Web UI, CLI, API Consumers)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    API Gateway                               │
│              (Authentication, Rate Limiting)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Agent Service                               │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐        │
│  │ Inference  │  │    Tool     │  │   Training   │        │
│  │  Engine    │  │ Orchestrator│  │ Coordinator  │        │
│  └────────────┘  └─────────────┘  └──────────────┘        │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                 Infrastructure Services                      │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐        │
│  │   Ollama   │  │  PostgreSQL │  │   Redis      │        │
│  │   Server   │  │  Database   │  │   Cache      │        │
│  └────────────┘  └─────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

1. **Inference Engine**
   - Manages model loading and inference
   - Handles prompt engineering
   - Implements retry logic and fallbacks

2. **Tool Orchestrator**
   - Manages tool registration and discovery
   - Validates tool inputs/outputs
   - Implements tool execution sandbox

3. **Training Coordinator**
   - Schedules training jobs
   - Manages dataset preparation
   - Handles model versioning

## Component Design

### Domain Entities

```python
# core/domain/entities/agent.py
@dataclass
class Agent:
    id: AgentId
    name: str
    model_config: ModelConfig
    tools: List[Tool]
    created_at: datetime
    updated_at: datetime
    
    def can_use_tool(self, tool: Tool) -> bool:
        """Domain logic for tool authorization"""
        return tool in self.tools

# core/domain/value_objects/model_config.py
@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    temperature: float
    max_tokens: int
    lora_adapter_path: Optional[str]
    
    def validate(self) -> None:
        """Validates model configuration"""
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
```

### Application Services

```python
# core/application/services/agent_orchestrator.py
class AgentOrchestrator:
    def __init__(
        self,
        inference_service: InferenceService,
        tool_executor: ToolExecutor,
        conversation_repo: ConversationRepository
    ):
        self.inference_service = inference_service
        self.tool_executor = tool_executor
        self.conversation_repo = conversation_repo
    
    async def process_message(
        self,
        agent_id: AgentId,
        message: str,
        context: ConversationContext
    ) -> AgentResponse:
        """Orchestrates message processing with tool execution"""
        # Implementation follows use case flow
```

### Infrastructure Adapters

```python
# infrastructure/adapters/ollama_adapter.py
class OllamaAdapter(InferencePort):
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def generate(
        self,
        prompt: str,
        model_config: ModelConfig
    ) -> InferenceResult:
        """Adapts Ollama API to domain interface"""
        # Implementation maps domain objects to API calls
```

## Data Flow

### Request Processing Flow

```
1. Client Request → API Gateway
2. API Gateway → Authentication/Authorization
3. Authenticated Request → Agent Service
4. Agent Service → Load Conversation Context
5. Context + Message → Inference Engine
6. Inference Result → Tool Detection
7. If tools needed → Tool Orchestrator
8. Tool Results → Inference Engine (for final response)
9. Final Response → Save to Database
10. Response → Client
```

### Training Loop Flow

```
1. Scheduled Trigger → Training Coordinator
2. Training Coordinator → Fetch Conversation Logs
3. Logs → Dataset Preparation
4. Prepared Dataset → Training Service
5. Training Service → QLoRA Fine-tuning
6. New Model → Validation Service
7. If validation passes → Model Registry
8. Model Registry → Update Agent Configuration
9. Agent Configuration → Reload Model
```

## Deployment Architecture

### Infrastructure Components

```yaml
# Deployment topology
production:
  load_balancer:
    - nginx (SSL termination, rate limiting)
  
  application_tier:
    - agent_service (2 instances, blue-green deployment)
    - api_gateway (2 instances)
  
  data_tier:
    - postgresql (primary + read replica)
    - redis (cache cluster)
    - ollama_server (GPU-enabled instance)
  
  monitoring:
    - prometheus
    - grafana
    - loki (log aggregation)
```

### Scaling Strategy

1. **Horizontal Scaling**
   - Agent Service: Auto-scale based on request rate
   - API Gateway: Fixed pool with load balancing

2. **Vertical Scaling**
   - Ollama Server: GPU-optimized instances
   - PostgreSQL: Scaled based on storage and IOPS

3. **Caching Strategy**
   - Redis: Conversation context caching
   - Model weights: Cached in GPU memory

## Security Architecture

### Authentication & Authorization

```python
# infrastructure/security/jwt_authenticator.py
class JWTAuthenticator:
    def __init__(self, public_key: str, algorithm: str = "RS256"):
        self.public_key = public_key
        self.algorithm = algorithm
    
    def verify_token(self, token: str) -> TokenPayload:
        """Verifies JWT token and extracts claims"""
        # Implementation with proper validation
```

### Security Measures

1. **Network Security**
   - All internal communication over private network
   - Ollama bound to localhost only
   - Firewall rules for ingress/egress

2. **Application Security**
   - Input sanitization for all user inputs
   - Rate limiting per user/IP
   - Request size limits

3. **Data Security**
   - Encryption at rest for database
   - Encryption in transit (TLS 1.3)
   - Sensitive data masking in logs

## Monitoring and Observability

### Metrics Collection

```python
# infrastructure/monitoring/metrics.py
class MetricsCollector:
    def __init__(self, prometheus_gateway: str):
        self.gateway = prometheus_gateway
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.request_duration = Histogram(
            'agent_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint', 'status']
        )
        
        self.tool_execution_counter = Counter(
            'tool_executions_total',
            'Total number of tool executions',
            ['tool_name', 'status']
        )
```

### Key Metrics

1. **Application Metrics**
   - Request latency (P50, P95, P99)
   - Tool execution success rate
   - Model inference time
   - Cache hit rate

2. **Infrastructure Metrics**
   - CPU/Memory utilization
   - GPU utilization
   - Disk I/O
   - Network throughput

3. **Business Metrics**
   - Active conversations
   - Training frequency
   - Model performance scores

## Architecture Decision Records

### ADR-001: Hexagonal Architecture

**Status**: Accepted  
**Date**: 2025-01-28

**Context**: Need for a maintainable, testable architecture that supports multiple interfaces and external integrations.

**Decision**: Adopt hexagonal architecture (ports and adapters) pattern.

**Consequences**:
- (+) Clear separation of concerns
- (+) Easy to test domain logic in isolation
- (+) Flexible integration with external systems
- (-) Additional abstraction layers
- (-) More initial setup complexity

### ADR-002: Asynchronous Processing

**Status**: Accepted  
**Date**: 2025-01-28

**Context**: Need to handle long-running operations (training, tool execution) without blocking the main request flow.

**Decision**: Use async/await pattern with FastAPI and implement background job processing.

**Consequences**:
- (+) Better resource utilization
- (+) Improved response times
- (+) Natural fit with Python's asyncio
- (-) Complexity in error handling
- (-) Need for proper async patterns throughout

### ADR-003: Event-Driven Training

**Status**: Accepted  
**Date**: 2025-01-28

**Context**: Training should be triggered based on data availability and quality metrics, not just time-based schedules.

**Decision**: Implement event-driven training triggers with configurable thresholds.

**Consequences**:
- (+) More efficient resource usage
- (+) Better model improvement cycles
- (+) Flexible triggering mechanisms
- (-) Complex event handling logic
- (-) Need for robust event store

### ADR-004: Repository Pattern for Persistence

**Status**: Accepted  
**Date**: 2025-01-28

**Context**: Need to abstract data persistence from domain logic and support different storage backends.

**Decision**: Implement repository pattern for all persistence operations.

**Consequences**:
- (+) Storage backend independence
- (+) Easier testing with in-memory implementations
- (+) Clear data access patterns
- (-) Additional abstraction layer
- (-) Potential for leaky abstractions

## Future Considerations

1. **Multi-Model Support**: Architecture supports adding multiple model backends
2. **Federated Learning**: Design allows for distributed training scenarios
3. **A/B Testing**: Infrastructure ready for model experimentation
4. **Real-time Streaming**: WebSocket support for streaming responses
5. **Multi-tenancy**: Isolation boundaries for supporting multiple organizations