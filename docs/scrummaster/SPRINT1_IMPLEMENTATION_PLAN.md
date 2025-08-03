# Sprint 1 Implementation Plan - Hexagonal Architecture Refactoring

## Overview

This document outlines the implementation plan for refactoring the existing agent-loop system to properly implement hexagonal architecture principles, while maintaining the existing functionality and infrastructure.

## Current State Analysis

### Existing Components (Working)
- ✅ **VM Infrastructure**: Terraform + Ansible setup complete
- ✅ **Ollama Service**: Running with Gemma 3N model
- ✅ **FastAPI Application**: Basic endpoints functional
- ✅ **Browser Tool**: Playwright-based web automation
- ✅ **Monitoring**: Prometheus + Grafana setup
- ✅ **CI/CD**: GitHub Actions with 90% test coverage

### Architecture Violations (To Fix)
- ❌ **API-Infrastructure Coupling**: FastAPI directly imports Ollama service
- ❌ **Missing Domain Layer**: No domain entities or business logic separation
- ❌ **No Repository Pattern**: Direct infrastructure access everywhere
- ❌ **Monolithic Controllers**: All logic in FastAPI endpoints
- ❌ **No Event System**: No domain events or event-driven patterns

## Implementation Phases

### Phase 1: Domain Layer Foundation (Days 1-3)

#### 1.1 Create Domain Entities
**Location**: `core/domain/entities/`

```python
# core/domain/entities/agent.py
@dataclass(frozen=True)
class AgentId:
    value: str

@dataclass
class Agent:
    id: AgentId
    name: str 
    model_config: ModelConfig
    tools: List['Tool']
    created_at: datetime
    updated_at: datetime
    
    def can_execute_tool(self, tool_name: str) -> bool:
        """Domain logic: Check if agent can use specific tool"""
        return any(tool.name == tool_name for tool in self.tools)

# core/domain/entities/conversation.py
@dataclass
class Conversation:
    id: ConversationId
    agent_id: AgentId
    messages: List[Message]
    created_at: datetime
    
    def add_message(self, message: Message) -> None:
        """Domain logic: Add message with validation"""
        if not message.content.strip():
            raise DomainError("Message content cannot be empty")
        self.messages.append(message)

# core/domain/entities/tool.py
@dataclass  
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Domain logic: Validate tool parameters"""
        return all(key in params for key in self.parameters.get('required', []))
```

#### 1.2 Create Value Objects
**Location**: `core/domain/value_objects/`

```python
# core/domain/value_objects/model_config.py
@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    temperature: float
    max_tokens: int
    lora_adapter_path: Optional[str] = None
    
    def __post_init__(self) -> None:
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

# core/domain/value_objects/inference_result.py
@dataclass(frozen=True)
class InferenceResult:
    text: str
    tokens_generated: int
    inference_time_ms: int
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 1.3 Create Domain Events
**Location**: `core/domain/events/`

```python
# core/domain/events/base.py
from abc import ABC
from datetime import datetime
from dataclasses import dataclass

@dataclass(frozen=True)
class DomainEvent(ABC):
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

# core/domain/events/agent_events.py
@dataclass(frozen=True)
class AgentExecutionStarted(DomainEvent):
    agent_id: str
    instruction: str
    correlation_id: str

@dataclass(frozen=True)
class ToolExecuted(DomainEvent):
    agent_id: str
    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    execution_time_ms: int
```

### Phase 2: Application Layer (Days 4-6)

#### 2.1 Create Ports (Interfaces)
**Location**: `core/application/ports/`

```python
# core/application/ports/inference_port.py
from abc import ABC, abstractmethod

class InferencePort(ABC):
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        model_config: ModelConfig
    ) -> InferenceResult:
        """Generate text using LLM"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if inference service is available"""
        pass

# core/application/ports/tool_port.py
class ToolPort(ABC):
    @abstractmethod
    async def execute(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool with given parameters"""
        pass
    
    @abstractmethod  
    def list_available_tools(self) -> List[Tool]:
        """Get list of available tools"""
        pass

# core/application/ports/conversation_port.py
class ConversationPort(ABC):
    @abstractmethod
    async def save(self, conversation: Conversation) -> None:
        """Persist conversation"""
        pass
    
    @abstractmethod
    async def get_by_id(self, conversation_id: ConversationId) -> Optional[Conversation]:
        """Retrieve conversation by ID"""
        pass
```

#### 2.2 Create Use Cases
**Location**: `core/application/use_cases/`

```python
# core/application/use_cases/execute_agent_use_case.py
class ExecuteAgentUseCase:
    def __init__(
        self,
        inference_port: InferencePort,
        tool_port: ToolPort,
        conversation_port: ConversationPort,
        event_publisher: EventPublisher
    ):
        self.inference_port = inference_port
        self.tool_port = tool_port
        self.conversation_port = conversation_port
        self.event_publisher = event_publisher
    
    async def execute(self, request: ExecuteAgentRequest) -> ExecuteAgentResponse:
        """Execute agent with proper domain logic separation"""
        
        # 1. Validate request
        if not request.instruction.strip():
            raise ValidationError("Instruction cannot be empty")
        
        # 2. Publish domain event
        await self.event_publisher.publish(
            AgentExecutionStarted(
                agent_id=request.agent_id,
                instruction=request.instruction,
                correlation_id=request.correlation_id
            )
        )
        
        # 3. Check if tools are needed
        tools_needed = self._extract_tools_from_instruction(request.instruction)
        tool_results = {}
        
        for tool_name in tools_needed:
            tool_result = await self.tool_port.execute(
                tool_name, 
                self._extract_tool_parameters(request.instruction, tool_name)
            )
            tool_results[tool_name] = tool_result
            
            await self.event_publisher.publish(
                ToolExecuted(
                    agent_id=request.agent_id,
                    tool_name=tool_name,
                    parameters=tool_result.parameters,
                    result=tool_result.result,
                    execution_time_ms=tool_result.execution_time_ms
                )
            )
        
        # 4. Generate response using LLM
        enhanced_prompt = self._build_prompt_with_tool_results(
            request.instruction, 
            tool_results
        )
        
        inference_result = await self.inference_port.generate(
            enhanced_prompt,
            request.model_config
        )
        
        # 5. Save conversation
        conversation = Conversation(
            id=ConversationId(str(uuid.uuid4())),
            agent_id=AgentId(request.agent_id),
            messages=[
                Message(role="user", content=request.instruction),
                Message(role="assistant", content=inference_result.text)
            ],
            created_at=datetime.utcnow()
        )
        
        await self.conversation_port.save(conversation)
        
        return ExecuteAgentResponse(
            response=inference_result.text,
            conversation_id=conversation.id.value,
            tool_results=tool_results,
            inference_metrics=inference_result.metadata
        )
```

### Phase 3: Infrastructure Adapters (Days 7-9)

#### 3.1 Refactor Ollama Adapter
**Location**: `infrastructure/adapters/ollama_adapter.py`

```python
class OllamaAdapter(InferencePort):
    """Adapter that implements InferencePort using Ollama API"""
    
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        self.client = httpx.AsyncClient()
    
    async def generate(
        self, 
        prompt: str, 
        model_config: ModelConfig
    ) -> InferenceResult:
        """Implementation of domain port using Ollama API"""
        
        payload = {
            "model": model_config.model_name,
            "prompt": prompt,
            "options": {
                "temperature": model_config.temperature,
                "num_predict": model_config.max_tokens,
            },
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            
            result = response.json()
            inference_time = int((time.time() - start_time) * 1000)
            
            return InferenceResult(
                text=result.get("response", ""),
                tokens_generated=result.get("eval_count", 0),
                inference_time_ms=inference_time,
                model_used=model_config.model_name,
                metadata={
                    "eval_duration": result.get("eval_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0)
                }
            )
            
        except httpx.HTTPError as e:
            raise InfrastructureError(f"Ollama API error: {e}")
    
    async def health_check(self) -> bool:
        """Check Ollama service health"""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/version",
                timeout=5.0
            )
            return response.status_code == 200
        except:
            return False
```

#### 3.2 Create Repository Adapters
**Location**: `infrastructure/adapters/conversation_repository.py`

```python
class PostgreSQLConversationRepository(ConversationPort):
    """PostgreSQL implementation of conversation repository"""
    
    def __init__(self, connection_pool: asyncpg.Pool):
        self.pool = connection_pool
    
    async def save(self, conversation: Conversation) -> None:
        """Save conversation to PostgreSQL"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversations (id, agent_id, messages, created_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO UPDATE SET
                messages = $3, updated_at = NOW()
            """, 
            conversation.id.value,
            conversation.agent_id.value, 
            json.dumps([msg.dict() for msg in conversation.messages]),
            conversation.created_at
            )
    
    async def get_by_id(self, conversation_id: ConversationId) -> Optional[Conversation]:
        """Retrieve conversation from PostgreSQL"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, agent_id, messages, created_at 
                FROM conversations 
                WHERE id = $1
            """, conversation_id.value)
            
            if not row:
                return None
            
            messages = [Message(**msg) for msg in json.loads(row['messages'])]
            
            return Conversation(
                id=ConversationId(row['id']),
                agent_id=AgentId(row['agent_id']),
                messages=messages,
                created_at=row['created_at']
            )
```

### Phase 4: Interface Adapters (Days 10-12)

#### 4.1 Refactor FastAPI Controllers
**Location**: `interfaces/rest/agent_controller.py`

```python
class AgentController:
    """Clean REST controller that delegates to use cases"""
    
    def __init__(self, execute_agent_use_case: ExecuteAgentUseCase):
        self.execute_agent_use_case = execute_agent_use_case
    
    async def execute_agent(
        self, 
        request: AgentExecutionRequest,
        correlation_id: str
    ) -> AgentExecutionResponse:
        """REST endpoint that calls domain use case"""
        
        # Map REST request to domain request
        domain_request = ExecuteAgentRequest(
            agent_id=request.agent_id or "default",
            instruction=request.instruction,
            model_config=ModelConfig(
                model_name=request.model_name or "gemma:3n-e2b",
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens or 1024
            ),
            use_tools=request.use_tools,
            correlation_id=correlation_id
        )
        
        try:
            # Execute domain use case
            domain_response = await self.execute_agent_use_case.execute(domain_request)
            
            # Map domain response to REST response
            return AgentExecutionResponse(
                success=True,
                response=domain_response.response,
                conversation_id=domain_response.conversation_id,
                tool_results=domain_response.tool_results,
                execution_time_ms=domain_response.inference_metrics.get("inference_time_ms"),
                model_used=domain_response.inference_metrics.get("model_used")
            )
            
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except DomainError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except InfrastructureError as e:
            raise HTTPException(status_code=503, detail=str(e))
```

#### 4.2 Create Dependency Injection Configuration
**Location**: `infrastructure/dependency_injection.py`

```python
class DIContainer:
    """Dependency injection container for the application"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._instances = {}
    
    async def get_inference_port(self) -> InferencePort:
        """Factory for inference port implementation"""
        if "inference_port" not in self._instances:
            self._instances["inference_port"] = OllamaAdapter(
                base_url=self.settings.ollama_base_url,
                model_name=self.settings.default_model
            )
        return self._instances["inference_port"]
    
    async def get_conversation_port(self) -> ConversationPort:
        """Factory for conversation repository"""
        if "conversation_port" not in self._instances:
            pool = await asyncpg.create_pool(self.settings.database_url)
            self._instances["conversation_port"] = PostgreSQLConversationRepository(pool)
        return self._instances["conversation_port"]
    
    async def get_execute_agent_use_case(self) -> ExecuteAgentUseCase:
        """Factory for execute agent use case"""
        return ExecuteAgentUseCase(
            inference_port=await self.get_inference_port(),
            tool_port=await self.get_tool_port(),
            conversation_port=await self.get_conversation_port(),
            event_publisher=await self.get_event_publisher()
        )
```

### Phase 5: Event System Implementation (Days 13-15)

#### 5.1 Create Event System
**Location**: `core/application/events/`

```python
# core/application/events/event_publisher.py
class EventPublisher(ABC):
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish domain event"""
        pass

# infrastructure/events/in_memory_event_publisher.py
class InMemoryEventPublisher(EventPublisher):
    """Simple in-memory event publisher for Sprint 1"""
    
    def __init__(self):
        self.handlers: Dict[Type[DomainEvent], List[Callable]] = {}
    
    def subscribe(self, event_type: Type[DomainEvent], handler: Callable):
        """Subscribe handler to event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish event to all subscribed handlers"""
        event_type = type(event)
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}", exc_info=True)
```

#### 5.2 Create Event Handlers
**Location**: `core/application/event_handlers/`

```python
# core/application/event_handlers/training_event_handler.py
class TrainingEventHandler:
    """Handles events related to training triggers"""
    
    def __init__(self, training_coordinator: TrainingCoordinator):
        self.training_coordinator = training_coordinator
    
    async def handle_tool_executed(self, event: ToolExecuted) -> None:
        """Handle tool execution event - might trigger training"""
        
        # Business logic: Trigger training after N tool executions
        execution_count = await self._get_recent_tool_execution_count()
        
        if execution_count >= 100:  # Configurable threshold
            await self.training_coordinator.schedule_training(
                trigger_reason="tool_execution_threshold",
                metadata={"execution_count": execution_count}
            )
    
    async def handle_response_generated(self, event: ResponseGenerated) -> None:
        """Handle response generation - collect training data"""
        
        training_sample = TrainingSample(
            input=event.prompt,
            output=event.response,
            metadata={
                "model_used": event.model_used,
                "timestamp": event.occurred_at,
                "inference_time_ms": event.inference_time_ms
            }
        )
        
        await self.training_coordinator.add_training_sample(training_sample)
```

## Migration Strategy

### Backward Compatibility
1. **Keep Existing Endpoints**: Maintain current API contracts during migration
2. **Feature Flags**: Use feature flags to gradually switch to new implementation
3. **Adapter Bridge**: Create bridge adapters for gradual migration

### Testing Strategy
1. **Contract Tests**: Ensure ports and adapters maintain contracts
2. **Integration Tests**: Test full request/response cycles
3. **Domain Tests**: Comprehensive unit tests for domain logic
4. **Performance Tests**: Ensure no regression in performance

### Rollout Plan
1. **Week 1**: Implement domain layer and ports
2. **Week 2**: Create adapters and use cases
3. **Week 3**: Refactor REST layer and add events
4. **Week 4**: Integration testing and performance validation

## Success Criteria

### Technical Metrics
- ✅ All tests pass with 90%+ coverage
- ✅ No performance regression (latency within 10% of baseline)
- ✅ All existing API endpoints continue to work
- ✅ Domain logic is fully isolated and testable

### Architecture Validation
- ✅ Domain layer has zero infrastructure dependencies
- ✅ Business logic can be tested without external services
- ✅ Infrastructure adapters can be swapped without domain changes
- ✅ Event system enables loose coupling between components

### Documentation
- ✅ Updated architecture documentation
- ✅ API documentation reflects new structure
- ✅ Developer onboarding updated
- ✅ Deployment procedures validated

This plan ensures a systematic migration to hexagonal architecture while maintaining system functionality and reliability throughout the Sprint 1 development cycle.