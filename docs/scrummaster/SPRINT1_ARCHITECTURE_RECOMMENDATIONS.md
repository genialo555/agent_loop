# Sprint 1 Architecture Recommendations - System Architect Summary

## Executive Summary

Based on my analysis of the existing `/home/jerem/agent_loop` system, I've identified critical architectural violations that must be addressed in Sprint 1 to ensure long-term maintainability and scalability. The current implementation has solid infrastructure foundations but violates hexagonal architecture principles, creating tight coupling between business logic and infrastructure concerns.

## Key Findings & Recommendations

### 🔴 Critical Issues Identified

1. **Architecture Violations (ARCH001 & ARCH002)**
   - Current `inference/api.py` mixes presentation, business logic, and infrastructure
   - Direct imports: `from .groupthink import generate` and `OllamaService` in controllers
   - **Impact**: Untestable business logic, inflexible to infrastructure changes

2. **Missing Domain Layer (ARCH001)**
   - No domain entities (`Agent`, `Conversation`, `Tool`) 
   - Business rules scattered across infrastructure code
   - **Impact**: Cannot evolve business logic independently

3. **Repository Pattern Absent (ARCH004)**
   - No abstraction for data persistence
   - **Impact**: Cannot test without external dependencies, hard to change storage

### ✅ Strong Foundations to Build On

1. **Infrastructure Excellence**
   - Terraform/Ansible automation ✅
   - Ollama service properly configured ✅  
   - Prometheus/Grafana monitoring ✅
   - 90% test coverage requirement enforced ✅

2. **Development Standards**
   - Modern Python (3.13) with strict typing ✅
   - Black/Ruff/MyPy configured correctly ✅
   - AsyncIO throughout the stack ✅

## Strategic Architecture Decisions

### ADR-005: Hexagonal Architecture Refactoring (RECOMMENDED)

**Decision**: Implement strict hexagonal architecture with complete domain isolation

**Rationale**:
- Current coupling prevents independent testing of business logic
- Ollama could be replaced with OpenAI without domain changes
- Event-driven architecture enables future advanced features
- Aligns with project's stated hexagonal architecture goals

**Implementation Priority**: 🔴 **CRITICAL** - Must be done in Sprint 1

### Key Architectural Patterns to Implement

#### 1. Domain-Driven Design (DDD)
```
core/domain/
├── entities/          # Agent, Conversation, Tool
├── value_objects/     # ModelConfig, InferenceResult  
├── services/          # InferenceService, GroupThinkService
└── events/           # ToolExecuted, ResponseGenerated
```

#### 2. Ports and Adapters (Hexagonal)
```
core/application/ports/     # Interfaces
infrastructure/adapters/    # Implementations
interfaces/rest/           # FastAPI controllers
```

#### 3. Event-Driven Architecture (ARCH005)
```python
# Domain events enable loose coupling
await event_publisher.publish(ToolExecuted(
    agent_id=agent.id,
    tool_name="browser",
    result=browser_result
))
```

## Sprint 1 Implementation Roadmap

### Phase 1: Foundation (Days 1-3) - CRITICAL
- [ ] Create domain entities (`Agent`, `Conversation`, `Tool`)
- [ ] Define value objects (`ModelConfig`, `InferenceResult`)
- [ ] Establish domain events system
- [ ] **Success Criteria**: Domain logic testable without infrastructure

### Phase 2: Application Layer (Days 4-6) - HIGH
- [ ] Define application ports (interfaces)
- [ ] Implement use cases (`ExecuteAgentUseCase`)
- [ ] Create application services
- [ ] **Success Criteria**: Business logic isolated from external concerns

### Phase 3: Infrastructure (Days 7-9) - HIGH  
- [ ] Refactor `OllamaAdapter` to implement `InferencePort`
- [ ] Create repository adapters for persistence
- [ ] Implement tool execution adapters
- [ ] **Success Criteria**: Can swap infrastructure without domain changes

### Phase 4: Interface Layer (Days 10-12) - MEDIUM
- [ ] Refactor FastAPI controllers to use use cases
- [ ] Implement dependency injection
- [ ] Add proper error handling
- [ ] **Success Criteria**: Clean REST layer that delegates to domain

### Phase 5: Events & Testing (Days 13-15) - MEDIUM
- [ ] Implement event publishing system
- [ ] Create event handlers for training triggers
- [ ] Comprehensive test suite
- [ ] **Success Criteria**: Event-driven architecture enables future features

## Non-Functional Requirements Targets

### Performance (Validated against current VM specs)
- **Agent Response Time**: P95 < 2s (simple), P95 < 10s (complex tools)
- **API Latency**: P95 < 100ms (health checks)  
- **Throughput**: 100 RPS sustained, 10 concurrent agent sessions
- **Resource Usage**: < 8GB RAM, < 80% CPU average

### Security (Building on existing foundation)
- **Authentication**: JWT (RS256) with 30min TTL
- **Network**: All services properly bound (Ollama to localhost ✅)
- **Data**: TLS 1.3, database encryption at rest

### Scalability (Architecture enables)
- **Horizontal**: Stateless FastAPI design for load balancing
- **Vertical**: Efficient Ollama GPU utilization
- **Storage**: PostgreSQL with archiving strategy

## Technology Stack Validation

### Recommended (Aligned with existing)
- **Framework**: FastAPI (async, well-documented) ✅
- **LLM**: Ollama with Gemma 3N (already working) ✅  
- **Database**: PostgreSQL (production-ready, ACID compliance)
- **Cache**: Redis (session management, response caching)
- **Monitoring**: Prometheus/Grafana (already configured) ✅

### Justification for Key Choices
1. **PostgreSQL over File-based**: Conversation history requires ACID properties
2. **Async/Await**: Ollama inference is I/O bound, async maximizes throughput
3. **Event System**: Enables training triggers without tight coupling

## Migration Strategy

### Backward Compatibility Approach
1. **Gradual Migration**: Keep existing endpoints during refactoring
2. **Feature Flags**: Toggle between old/new implementations
3. **Bridge Adapters**: Smooth transition for external consumers

### Risk Mitigation
1. **Comprehensive Testing**: Domain logic 100% unit test coverage
2. **Performance Monitoring**: Continuous validation during migration  
3. **Rollback Plan**: Can revert to current implementation if needed

## Success Metrics

### Technical Validation
- [ ] Domain layer has zero infrastructure dependencies
- [ ] All business logic testable without external services  
- [ ] Infrastructure components swappable (Ollama → OpenAI test)
- [ ] Event system enables loose coupling verification
- [ ] Performance within 10% of current baseline

### Business Value
- [ ] Faster feature development (isolated domain logic)
- [ ] Easier testing (mock external dependencies)
- [ ] Future-proof architecture (multi-model support ready)
- [ ] Maintainable codebase (clear separation of concerns)

## Recommended Next Steps

### Immediate Actions (This Week)
1. **Review & Approve** this architecture plan with team
2. **Prioritize** Phase 1 domain layer implementation
3. **Setup** development branch for hexagonal refactoring
4. **Create** acceptance criteria for each phase

### Sprint 1 Focus Areas
1. **Domain Layer**: Foundation for all future development
2. **Use Cases**: Business logic isolation and testability
3. **Adapters**: Infrastructure flexibility and swappability
4. **Events**: Enable advanced features like automated training

## Architectural Compliance Checklist

### Hexagonal Architecture (ARCH001)
- [ ] Core business logic isolated from external concerns
- [ ] Domain entities contain business rules
- [ ] Infrastructure accessed only through ports
- [ ] Adapters implement ports without affecting domain

### Clean Architecture (ARCH002)  
- [ ] Dependencies point inward toward domain
- [ ] Use cases orchestrate domain logic
- [ ] Infrastructure details abstracted away
- [ ] Domain layer has no framework dependencies

### Dependency Injection (ARCH003)
- [ ] Components depend on abstractions (ports)
- [ ] Concrete implementations injected at runtime
- [ ] Easy to swap implementations for testing
- [ ] Clear dependency flow documentation

### Repository Pattern (ARCH004)
- [ ] Data access abstracted from domain logic
- [ ] Can test with in-memory implementations
- [ ] Storage backend independence achieved
- [ ] Query logic separated from business logic

### Event-Driven Architecture (ARCH005)
- [ ] Domain events published by aggregates
- [ ] Loose coupling between bounded contexts
- [ ] Asynchronous processing capability
- [ ] Event sourcing foundation for future

## Conclusion

The current system has excellent infrastructure foundations but requires significant architectural refactoring to achieve the stated hexagonal architecture goals. The proposed Sprint 1 plan addresses critical coupling issues while building on existing strengths.

**Key Message**: We must prioritize architectural quality now to prevent technical debt from hampering future development. The refactoring is complex but essential for long-term success.

**Recommended Decision**: Approve hexagonal architecture refactoring as Sprint 1 primary objective, with performance and functionality preservation as non-negotiable constraints.

---

**Prepared by**: System Architect  
**Date**: 2025-07-28  
**Review Required**: Sprint Planning Team  
**Implementation Target**: Sprint 1 (15 days)