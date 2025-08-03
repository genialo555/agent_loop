"""Pydantic models for API requests and responses."""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "timestamp": "2025-01-28T10:30:00Z",
            "service": "async-fastapi-example",
            "version": "1.0.0",
            "checks": {
                "http_client": True,
                "ollama": True,
                "database": True
            }
        }
    })
    
    status: str = Field(description="Overall service health status")
    timestamp: datetime = Field(description="Health check timestamp")
    service: str = Field(description="Service name")
    version: str = Field(description="Service version")
    checks: Dict[str, bool] = Field(description="Individual component health checks")
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime in seconds")


class ReadinessResponse(BaseModel):
    """Response for readiness probe endpoint."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "ready",
            "checks": {
                "http_client": True,
                "ollama": True,
                "external_apis": True
            },
            "timestamp": "2025-01-28T10:30:00Z"
        }
    })
    
    status: str = Field(description="Readiness status")
    checks: Dict[str, Any] = Field(description="Dependency checks")
    timestamp: str = Field(description="Check timestamp")


class RunRequest(BaseModel):
    """Request for executing the agent."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "instruction": "Analyse ce texte pour moi",
            "use_groupthink": True,
            "timeout_seconds": 30,
            "webhook_url": "https://example.com/webhook"
        }
    })
    
    instruction: str = Field(
        description="Instruction to process by the agent",
        min_length=1,
        max_length=1000
    )
    use_groupthink: bool = Field(
        default=False,
        description="Use groupthink mode for better quality"
    )
    timeout_seconds: Optional[int] = Field(
        default=30,
        ge=1,
        le=300,
        description="Processing timeout in seconds"
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for async notification"
    )


class RunResponse(BaseModel):
    """Response from agent execution."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "processing",
            "answer": "Voici mon analyse...",
            "processing_time": 1.23,
            "correlation_id": "123e4567-e89b-12d3-a456-426614174000"
        }
    })
    
    task_id: str = Field(description="Unique task identifier")
    status: str = Field(description="Processing status")
    answer: Optional[str] = Field(default=None, description="Agent response")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    correlation_id: str = Field(description="Correlation ID for tracking")


class RunAgentRequest(BaseModel):
    """Request for the agent PoC endpoint with Ollama support."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "instruction": "Open https://example.com and return the title",
            "use_ollama": True,
            "temperature": 0.7,
            "max_tokens": 1024,
            "system_prompt": "You are a helpful AI assistant."
        }
    })
    
    instruction: str = Field(
        description="Instruction for the agent",
        min_length=1,
        max_length=2000,
        examples=[
            "Open https://example.com and return the title",
            "Analyze this text and provide insights",
            "Generate a summary of the following content"
        ]
    )
    use_ollama: bool = Field(
        default=True,
        description="Use Ollama for LLM inference instead of simple parsing"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for Ollama generation (0.0-1.0)"
    )
    max_tokens: Optional[int] = Field(
        default=1024,
        ge=10,
        le=4096,
        description="Maximum tokens to generate"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional system prompt for Ollama"
    )


class RunAgentResponse(BaseModel):
    """Response from the agent PoC endpoint with Ollama metrics."""
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
        "example": {
            "success": True,
            "result": {
                "response": "Here is the analysis...",
                "type": "llm_generation"
            },
            "execution_time_ms": 1250.5,
            "model_used": "gemma:3n-e2b",
            "inference_metrics": {
                "inference_time_ms": 1200.0,
                "temperature": 0.7,
                "tokens_per_second": 45.2
            }
        }
    })
    
    success: bool = Field(description="Whether execution was successful")
    result: Dict[str, Any] = Field(description="Execution results")
    execution_time_ms: float = Field(description="Total execution time in milliseconds")
    model_used: Optional[str] = Field(default=None, description="LLM model used for inference")
    inference_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Ollama inference metrics")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ErrorResponse(BaseModel):
    """Standard format for API errors."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error": "Validation failed",
            "detail": {"field": "instruction", "issue": "too short"},
            "correlation_id": "123e4567-e89b-12d3-a456-426614174000"
        }
    })
    
    error: str = Field(description="Error message")
    detail: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    correlation_id: str = Field(description="Correlation ID for tracking")


class OllamaModelInfo(BaseModel):
    """Information about the loaded Ollama model."""
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
        "example": {
            "success": True,
            "model_info": {
                "name": "gemma:3n-e2b",
                "size": 2345678901,
                "modified_at": "2025-01-28T10:30:00Z",
                "digest": "sha256:abc123...",
                "details": {"format": "gguf"}
            },
            "timestamp": "2025-01-28T10:30:00Z"
        }
    })
    
    success: bool = Field(description="Whether request was successful")
    model_info: Dict[str, Any] = Field(description="Model information")
    timestamp: str = Field(description="Request timestamp")


class OllamaHealthResponse(BaseModel):
    """Response from Ollama health check."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "service": "ollama",
            "model": "gemma:3n-e2b",
            "endpoint": "http://127.0.0.1:11434",
            "timestamp": "2025-01-28T10:30:00Z"
        }
    })
    
    status: str = Field(description="Health status")
    service: str = Field(description="Service name")
    model: str = Field(description="Active model")
    endpoint: str = Field(description="Ollama endpoint URL")
    timestamp: str = Field(description="Check timestamp")


# Training API Schemas - Sprint 2

class TrainingRequest(BaseModel):
    """Request for starting a fine-tuning job."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "base_model": "gemma:3n-e2b",
            "dataset_path": "/data/training/my_dataset.jsonl",
            "training_config": {
                "epochs": 3,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "lora_r": 16,
                "lora_alpha": 32,
                "max_seq_length": 2048
            },
            "experiment_name": "experiment_2025_01_28",
            "resume_from_checkpoint": None,
            "webhook_url": "https://example.com/training-webhook"
        }
    })
    
    base_model: str = Field(
        description="Base model to fine-tune (e.g., 'gemma:3n-e2b')",
        examples=["gemma:3n-e2b", "llama2:7b", "mistral:7b"]
    )
    dataset_path: str = Field(
        description="Path to training dataset (JSONL format)",
        min_length=1
    )
    training_config: Dict[str, Any] = Field(
        description="Training hyperparameters and configuration",
        default_factory=lambda: {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            "lora_r": 16,
            "lora_alpha": 32,
            "max_seq_length": 2048
        }
    )
    experiment_name: Optional[str] = Field(
        default=None,
        description="Optional experiment name for tracking"
    )
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to checkpoint to resume from"
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for training completion notification"
    )


class TrainingResponse(BaseModel):
    """Response from training job submission."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "job_id": "train_550e8400_e29b_41d4_a716_446655440000",
            "status": "queued",
            "message": "Training job submitted successfully",
            "estimated_duration_minutes": 45,
            "experiment_name": "experiment_2025_01_28",
            "created_at": "2025-01-28T10:30:00Z",
            "websocket_url": "ws://localhost:8000/ws/training/train_550e8400_e29b_41d4_a716_446655440000"
        }
    })
    
    job_id: str = Field(description="Unique training job identifier")
    status: str = Field(description="Initial job status")
    message: str = Field(description="Status message")
    estimated_duration_minutes: Optional[int] = Field(
        default=None,
        description="Estimated training duration in minutes"
    )
    experiment_name: Optional[str] = Field(
        default=None,
        description="Experiment name if provided"
    )
    created_at: datetime = Field(description="Job creation timestamp")
    websocket_url: Optional[str] = Field(
        default=None,
        description="WebSocket URL for real-time logs"
    )


class TrainingStatusResponse(BaseModel):
    """Response for training job status query."""
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
        "example": {
            "job_id": "train_550e8400_e29b_41d4_a716_446655440000",
            "status": "running",
            "progress": {
                "current_epoch": 2,
                "total_epochs": 3,
                "current_step": 150,
                "total_steps": 300,
                "progress_percentage": 66.7
            },
            "metrics": {
                "train_loss": 0.65,
                "learning_rate": 1.8e-4,
                "tokens_per_second": 245.3,
                "gpu_utilization": 85.2
            },
            "timing": {
                "started_at": "2025-01-28T10:30:00Z",
                "elapsed_seconds": 1800,
                "estimated_remaining_seconds": 900
            },
            "logs_url": "/training/train_550e8400_e29b_41d4_a716_446655440000/logs",
            "checkpoint_path": "/checkpoints/train_550e8400_e29b_41d4_a716_446655440000/epoch_2"
        }
    })
    
    job_id: str = Field(description="Training job identifier")
    status: str = Field(
        description="Current job status",
        examples=["queued", "running", "completed", "failed", "cancelled"]
    )
    progress: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Training progress information"
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Current training metrics"
    )
    timing: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Timing information"
    )
    logs_url: Optional[str] = Field(
        default=None,
        description="URL to fetch training logs"
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Path to latest checkpoint"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if job failed"
    )


class TrainingLogEntry(BaseModel):
    """Single training log entry for WebSocket streaming."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "timestamp": "2025-01-28T10:30:15.123Z",
            "level": "INFO",
            "message": "Epoch 2/3, Step 150/300: Loss=0.65, LR=1.8e-4",
            "step": 150,
            "epoch": 2,
            "metrics": {"loss": 0.65, "lr": 1.8e-4}
        }
    })
    
    timestamp: datetime = Field(description="Log entry timestamp")
    level: str = Field(
        description="Log level",
        examples=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    message: str = Field(description="Log message")
    step: Optional[int] = Field(default=None, description="Training step number")
    epoch: Optional[int] = Field(default=None, description="Training epoch number")
    metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Associated metrics"
    )


class TrainingJobsListResponse(BaseModel):
    """Response for listing training jobs."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "jobs": [
                {
                    "job_id": "train_550e8400_e29b_41d4_a716_446655440000",
                    "status": "completed",
                    "experiment_name": "experiment_2025_01_28",
                    "created_at": "2025-01-28T10:30:00Z",
                    "completed_at": "2025-01-28T11:15:00Z",
                    "base_model": "gemma:3n-e2b"
                }
            ],
            "total": 1,
            "page": 1,
            "per_page": 10
        }
    })
    
    jobs: list[Dict[str, Any]] = Field(description="List of training jobs")
    total: int = Field(description="Total number of jobs")
    page: int = Field(description="Current page number")
    per_page: int = Field(description="Jobs per page")