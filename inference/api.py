from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import asyncio
import logging
import time
import uuid
import os

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, ConfigDict
import httpx
import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Info, generate_latest, 
    CONTENT_TYPE_LATEST, CollectorRegistry, multiprocess, 
    REGISTRY
)

from core.settings import settings  # noqa: F401  # future use
from .groupthink import generate as gt_generate

# Configuration du logging structuré avec structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if os.getenv("ENVIRONMENT") == "development" else structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)

# Métriques Prometheus
# Registre pour les métriques (support multiprocess si nécessaire)
if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
else:
    registry = REGISTRY

# Métriques HTTP
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'handler', 'status'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'handler'],
    registry=registry
)

# Métriques application
inference_requests_total = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['groupthink_enabled', 'status'],
    registry=registry
)

inference_duration_seconds = Histogram(
    'inference_duration_seconds',
    'Inference processing time',
    ['groupthink_enabled'],
    registry=registry
)

active_requests = Gauge(
    'active_requests',
    'Number of active requests',
    registry=registry
)

webhook_notifications_total = Counter(
    'webhook_notifications_total',
    'Total webhook notifications sent',
    ['status'],
    registry=registry
)

# Informations système
app_info = Info(
    'app_info',
    'Application information',
    registry=registry
)

app_info.info({
    'version': '1.0.0',
    'environment': os.getenv('ENVIRONMENT', 'production'),
    'service': 'fastapi-inference-api'
})

# Client HTTP asynchrone réutilisable
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application."""
    # Démarrage
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
    )
    logger.info("Application started", http_client_initialized=True)
    
    yield
    
    # Arrêt
    if http_client:
        await http_client.aclose()
    logger.info("Application shutdown", resources_cleaned=True)


app = FastAPI(
    title="Async FastAPI Best Practices Example",
    description="Exemple d'API asynchrone haute performance avec FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS (adapter selon vos besoins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Jamais "*" en production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Middleware personnalisé pour logging et correlation IDs
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware pour ajouter des correlation IDs et logger les requêtes."""
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id
    
    start_time = time.time()
    
    # Log de la requête entrante
    logger.info(
        f"Request started",
        extra={
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log de la réponse
        logger.info(
            f"Request completed",
            extra={
                "correlation_id": correlation_id,
                "status_code": response.status_code,
                "process_time": process_time
            }
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed",
            extra={
                "correlation_id": correlation_id,
                "error": str(e),
                "process_time": process_time
            },
            exc_info=True
        )
        raise


# Modèles Pydantic avec validation stricte
class HealthResponse(BaseModel):
    """Réponse du endpoint de santé."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "timestamp": "2025-01-28T10:30:00Z",
            "service": "async-fastapi-example"
        }
    })
    
    status: str = Field(description="État de santé du service")
    timestamp: datetime = Field(description="Horodatage de la vérification")
    service: str = Field(description="Nom du service")


class RunRequest(BaseModel):
    """Requête pour exécuter l'agent."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "instruction": "Analyse ce texte pour moi",
            "use_groupthink": True,
            "timeout_seconds": 30,
            "webhook_url": "https://example.com/webhook"
        }
    })
    
    instruction: str = Field(
        description="Instruction à traiter par l'agent",
        min_length=1,
        max_length=1000
    )
    use_groupthink: bool = Field(
        default=False,
        description="Utiliser le mode groupthink pour une meilleure qualité"
    )
    timeout_seconds: Optional[int] = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout en secondes pour le traitement"
    )
    webhook_url: Optional[str] = Field(
        default=None,
        description="URL de webhook pour notification asynchrone"
    )


class RunResponse(BaseModel):
    """Réponse de l'exécution de l'agent."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "processing",
            "answer": "Voici mon analyse...",
            "processing_time": 1.23,
            "correlation_id": "123e4567-e89b-12d3-a456-426614174000"
        }
    })
    
    task_id: str = Field(description="Identifiant unique de la tâche")
    status: str = Field(description="Statut du traitement")
    answer: Optional[str] = Field(default=None, description="Réponse de l'agent")
    processing_time: Optional[float] = Field(default=None, description="Temps de traitement en secondes")
    correlation_id: str = Field(description="ID de corrélation pour le suivi")


class ErrorResponse(BaseModel):
    """Format standard pour les erreurs."""
    error: str = Field(description="Message d'erreur")
    detail: Optional[Dict[str, Any]] = Field(default=None, description="Détails supplémentaires")
    correlation_id: str = Field(description="ID de corrélation pour le suivi")


# Services injectables
class ExternalAPIService:
    """Service pour les appels API externes."""
    
    async def fetch_data(self, url: str) -> Dict[str, Any]:
        """Récupère des données depuis une API externe de manière non-bloquante."""
        if not http_client:
            raise RuntimeError("HTTP client not initialized")
        
        try:
            response = await http_client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"External API error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External service unavailable"
            )


class OllamaService:
    """Service optimisé pour les appels à Ollama avec gestion des performances."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
        self.model = "gemma:3n-e2b"
        self.default_options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": 2048,
            "num_ctx": 8192,
            "stop": ["<|endoftext|>", "</s>", "<|im_end|>"]
        }
    
    async def health_check(self) -> bool:
        """Vérifie la disponibilité d'Ollama."""
        if not http_client:
            return False
        
        try:
            response = await http_client.get(
                f"{self.base_url}/api/version",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Génère une réponse avec Ollama en utilisant les optimisations ML.
        
        Args:
            prompt: Le prompt utilisateur
            system_prompt: Prompt système optionnel
            temperature: Contrôle de la créativité (0.0-1.0)
            max_tokens: Nombre maximum de tokens à générer
            stream: Mode streaming (non implémenté dans cette version)
        
        Returns:
            La réponse générée par le modèle
        """
        if not http_client:
            raise RuntimeError("HTTP client not initialized")
        
        # Optimisation ML001: Préparation des options avec quantification
        options = self.default_options.copy()
        if temperature is not None:
            options["temperature"] = max(0.0, min(1.0, temperature))
        if max_tokens is not None:
            options["num_predict"] = min(max_tokens, 4096)
        
        # Construction du prompt avec system prompt si fourni
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "options": options,
            "stream": False,  # Mode non-streaming pour cette implémentation
            "keep_alive": "5m"  # Optimisation ML004: Cache KV pendant 5 minutes
        }
        
        try:
            start_time = time.time()
            
            # Appel API avec timeout optimisé
            response = await http_client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=httpx.Timeout(120.0)  # Timeout étendu pour l'inférence
            )
            response.raise_for_status()
            
            result = response.json()
            inference_time = time.time() - start_time
            
            # Logging des métriques de performance
            logger.info(
                f"Ollama inference completed",
                extra={
                    "model": self.model,
                    "inference_time": inference_time,
                    "prompt_length": len(prompt),
                    "response_length": len(result.get("response", "")),
                    "eval_count": result.get("eval_count", 0),
                    "eval_duration": result.get("eval_duration", 0) / 1e9,  # Convert to seconds
                    "tokens_per_second": result.get("eval_count", 0) / max(result.get("eval_duration", 1) / 1e9, 0.001)
                }
            )
            
            return result.get("response", "")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model {self.model} not found. Please ensure it's downloaded."
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ollama API error: {e.response.status_code}"
            )
        except httpx.TimeoutException:
            logger.error("Ollama request timeout")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="Ollama inference timeout"
            )
        except Exception as e:
            logger.error(f"Ollama service error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ollama service unavailable: {str(e)}"
            )
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Récupère les informations sur le modèle chargé."""
        if not http_client:
            raise RuntimeError("HTTP client not initialized")
        
        try:
            response = await http_client.get(
                f"{self.base_url}/api/tags",
                timeout=5.0
            )
            response.raise_for_status()
            
            models_data = response.json()
            current_model = None
            
            for model in models_data.get("models", []):
                if model["name"] == self.model:
                    current_model = model
                    break
            
            if current_model:
                return {
                    "name": current_model["name"],
                    "size": current_model.get("size", 0),
                    "modified_at": current_model.get("modified_at"),
                    "digest": current_model.get("digest", ""),
                    "details": current_model.get("details", {})
                }
            else:
                return {"error": f"Model {self.model} not found"}
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}


async def get_external_service() -> ExternalAPIService:
    """Factory pour obtenir le service API externe."""
    return ExternalAPIService()


async def get_ollama_service() -> OllamaService:
    """Factory pour obtenir le service Ollama."""
    return OllamaService()


# Fonction asynchrone pour exécuter l'agent
async def run_agent_async(instruction: str, use_groupthink: bool, timeout: int) -> str:
    """Exécute l'agent de manière asynchrone avec timeout."""
    try:
        # Simulation d'une opération asynchrone
        # Dans la vraie vie, gt_generate devrait être async
        loop = asyncio.get_event_loop()
        
        if use_groupthink:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, gt_generate, instruction),
                timeout=timeout
            )
        else:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, gt_generate, instruction, 1),
                timeout=timeout
            )
        
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=f"Processing timeout after {timeout} seconds"
        )
    except Exception as e:
        logger.error(f"Error in agent processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error"
        )


# Tâche de fond pour notification webhook
async def send_webhook_notification(
    webhook_url: str,
    task_id: str,
    result: str,
    correlation_id: str
):
    """Envoie une notification webhook de manière asynchrone."""
    if not http_client:
        logger.error("HTTP client not available for webhook")
        return
    
    try:
        payload = {
            "task_id": task_id,
            "result": result,
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        response = await http_client.post(
            webhook_url,
            json=payload,
            timeout=10.0
        )
        response.raise_for_status()
        logger.info(f"Webhook sent successfully to {webhook_url}")
    except Exception as e:
        logger.error(f"Failed to send webhook to {webhook_url}: {e}")


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Endpoint de vérification de santé (léger et rapide)."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        service="async-fastapi-example"
    )


@app.get("/ready")
async def readiness_check(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """Endpoint de vérification de disponibilité (vérifie les dépendances)."""
    ollama_healthy = await ollama_service.health_check()
    
    checks = {
        "http_client": http_client is not None,
        "ollama": ollama_healthy,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    if not all(checks.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    return {"status": "ready", "checks": checks}


@app.post(
    "/run",
    response_model=RunResponse,
    responses={
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    }
)
async def run_agent(
    request: Request,
    req: RunRequest,
    background_tasks: BackgroundTasks,
    external_service: ExternalAPIService = Depends(get_external_service)
) -> RunResponse:
    """
    Exécute l'agent avec l'instruction donnée.
    
    Cet endpoint traite l'instruction de manière asynchrone et peut optionnellement
    envoyer le résultat à un webhook.
    """
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Exemple d'appel API externe (commenté pour éviter les erreurs)
        # external_data = await external_service.fetch_data("https://api.example.com/data")
        
        # Traitement principal
        answer = await run_agent_async(
            req.instruction,
            req.use_groupthink,
            req.timeout_seconds or 30
        )
        
        processing_time = time.time() - start_time
        
        # Planifier la notification webhook si demandée
        if req.webhook_url:
            background_tasks.add_task(
                send_webhook_notification,
                req.webhook_url,
                task_id,
                answer,
                correlation_id
            )
        
        return RunResponse(
            task_id=task_id,
            status="completed",
            answer=answer,
            processing_time=processing_time,
            correlation_id=correlation_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run_agent: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error occurred"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Gestionnaire personnalisé pour les exceptions HTTP."""
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail={"status_code": exc.status_code},
            correlation_id=correlation_id
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Gestionnaire pour toutes les autres exceptions."""
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail={"type": type(exc).__name__},
            correlation_id=correlation_id
        ).model_dump()
    )


# Sprint 1 PoC: /run-agent endpoint with Ollama integration
class RunAgentRequest(BaseModel):
    """Request for the agent PoC endpoint with Ollama support."""
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
    success: bool = Field(description="Whether execution was successful")
    result: Dict[str, Any] = Field(description="Execution results")
    execution_time_ms: float = Field(description="Total execution time in milliseconds")
    model_used: Optional[str] = Field(default=None, description="LLM model used for inference")
    inference_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Ollama inference metrics")
    error: Optional[str] = Field(default=None, description="Error message if failed")


@app.post("/run-agent", response_model=RunAgentResponse)
async def run_agent(
    request: Request,
    req: RunAgentRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> RunAgentResponse:
    """
    Enhanced agent endpoint with Ollama LLM integration.
    
    This endpoint can handle both browser tool operations and general LLM queries
    using the optimized Ollama service with Gemma 3N model.
    """
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    start_time = time.time()
    inference_metrics = None
    model_used = None
    
    try:
        # Check if this is a browser-specific instruction
        is_browser_request = ("open" in req.instruction.lower() and 
                            ("http" in req.instruction.lower() or "www." in req.instruction.lower()))
        
        if is_browser_request and not req.use_ollama:
            # Legacy browser tool for direct URL operations
            try:
                from plugins.browser_tool import BrowserTool
                browser = BrowserTool()
                
                # Extract URL from instruction
                import re
                url_match = re.search(r'https?://[^\s]+|www\.[^\s]+', req.instruction)
                if url_match:
                    url = url_match.group(0)
                    if not url.startswith(('http://', 'https://')):
                        url = 'https://' + url
                    
                    result = await browser(url=url)
                    execution_time = (time.time() - start_time) * 1000
                    
                    return RunAgentResponse(
                        success=True,
                        result=result,
                        execution_time_ms=execution_time,
                        model_used="browser_tool"
                    )
            except Exception as browser_error:
                logger.warning(f"Browser tool failed, falling back to Ollama: {browser_error}")
                # Continue to use Ollama instead
        
        # Use Ollama for LLM-powered agent responses
        if req.use_ollama:
            # Check Ollama health first
            if not await ollama_service.health_check():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Ollama service is not available"
                )
            
            # Prepare system prompt for agent context
            system_prompt = req.system_prompt or (
                "You are an intelligent AI agent that can help with various tasks. "
                "For web-related requests, provide detailed analysis and insights. "
                "Be concise but comprehensive in your responses."
            )
            
            # Generate response using Ollama
            inference_start = time.time()
            response_text = await ollama_service.generate(
                prompt=req.instruction,
                system_prompt=system_prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens
            )
            inference_time = (time.time() - inference_start) * 1000
            
            model_used = ollama_service.model
            inference_metrics = {
                "inference_time_ms": inference_time,
                "model": model_used,
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
                "prompt_length": len(req.instruction),
                "response_length": len(response_text)
            }
            
            # For browser requests, try to extract URLs and provide enhanced context
            if is_browser_request:
                import re
                urls = re.findall(r'https?://[^\s]+|www\.[^\s]+', req.instruction)
                if urls:
                    inference_metrics["urls_mentioned"] = urls
                    # Add URL context to the response
                    response_text += f"\n\n[Agent Note: URLs mentioned in request: {', '.join(urls)}]"
            
            execution_time = (time.time() - start_time) * 1000
            
            return RunAgentResponse(
                success=True,
                result={
                    "response": response_text,
                    "type": "llm_generation",
                    "enhanced_context": is_browser_request
                },
                execution_time_ms=execution_time,
                model_used=model_used,
                inference_metrics=inference_metrics
            )
        
        # Fallback for non-Ollama, non-browser requests
        return RunAgentResponse(
            success=False,
            result={},
            execution_time_ms=(time.time() - start_time) * 1000,
            error="No suitable processing method available. Please enable Ollama or provide a valid URL."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in run_agent: {e}", exc_info=True)
        return RunAgentResponse(
            success=False,
            result={},
            execution_time_ms=(time.time() - start_time) * 1000,
            model_used=model_used,
            inference_metrics=inference_metrics,
            error=str(e)
        )


@app.get("/ollama/model-info")
async def get_ollama_model_info(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """Get information about the loaded Ollama model."""
    try:
        model_info = await ollama_service.get_model_info()
        return {
            "success": True,
            "model_info": model_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not retrieve model information: {str(e)}"
        )


@app.get("/ollama/health")
async def ollama_health_check(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """Dedicated Ollama health check endpoint."""
    is_healthy = await ollama_service.health_check()
    
    if not is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is not available"
        )
    
    return {
        "status": "healthy",
        "service": "ollama",
        "model": ollama_service.model,
        "endpoint": ollama_service.base_url,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Endpoint pour les métriques Prometheus."""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )
