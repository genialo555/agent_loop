"""Ollama-specific endpoints for model management and health checking."""
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
import structlog

from ..models.schemas import OllamaModelInfo, OllamaHealthResponse
from ..services.dependencies import get_ollama_service
from ..services.ollama import OllamaService

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/ollama", tags=["ollama"])


@router.get(
    "/health",
    response_model=OllamaHealthResponse,
    responses={
        503: {"description": "Ollama service unavailable"}
    },
    summary="Ollama health check",
    description="Dedicated health check for Ollama service availability"
)
async def ollama_health_check(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> OllamaHealthResponse:
    """
    Dedicated Ollama health check endpoint.
    
    This endpoint specifically checks:
    - Ollama service availability
    - Response time
    - Model accessibility
    
    Returns 503 if Ollama is not available.
    """
    is_healthy = await ollama_service.health_check()
    
    if not is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ollama service is not available"
        )
    
    return OllamaHealthResponse(
        status="healthy",
        service="ollama",
        model=ollama_service.model,
        endpoint=ollama_service.base_url,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get(
    "/health/detailed",
    response_model=Dict[str, Any],
    summary="Detailed Ollama health check",
    description="Comprehensive Ollama health check with performance metrics"
)
async def ollama_detailed_health(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """
    Detailed Ollama health check with comprehensive information.
    
    This endpoint provides:
    - Service availability status
    - Response time metrics
    - Model availability and details
    - Version information
    - Error details if any issues are found
    """
    health_info = await ollama_service.detailed_health_check()
    
    return {
        "service": "ollama",
        "endpoint": ollama_service.base_url,
        "model": ollama_service.model,
        "health": health_info,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get(
    "/model-info",
    response_model=OllamaModelInfo,
    responses={
        503: {"description": "Could not retrieve model information"}
    },
    summary="Get Ollama model information",
    description="Retrieve detailed information about the currently loaded Ollama model"
)
async def get_ollama_model_info(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> OllamaModelInfo:
    """
    Get information about the loaded Ollama model.
    
    This endpoint provides:
    - Model name and version
    - Model size and format details
    - Last modified timestamp
    - Model digest for verification
    - Additional model metadata
    """
    try:
        model_info = await ollama_service.get_model_info()
        return OllamaModelInfo(
            success=True,
            model_info=model_info,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not retrieve model information: {str(e)}"
        )


@router.post(
    "/model/switch",
    response_model=Dict[str, Any],
    summary="Switch Ollama model",
    description="Switch to a different Ollama model (requires model to be downloaded)"
)
async def switch_ollama_model(
    model_name: str,
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """
    Switch to a different Ollama model.
    
    Args:
        model_name: Name of the model to switch to
        
    Returns:
        Confirmation of model switch
        
    Note:
        The model must already be downloaded using `ollama pull <model_name>`
    """
    old_model = ollama_service.model
    
    try:
        # Switch the model
        ollama_service.set_model(model_name)
        
        # Verify the new model is available
        model_info = await ollama_service.get_model_info()
        if "error" in model_info:
            # Revert to old model if new one is not available
            ollama_service.set_model(old_model)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found. Please ensure it's downloaded with 'ollama pull {model_name}'"
            )
        
        return {
            "success": True,
            "message": f"Successfully switched from '{old_model}' to '{model_name}'",
            "old_model": old_model,
            "new_model": model_name,
            "model_info": model_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Revert to old model on any error
        ollama_service.set_model(old_model)
        logger.error(f"Error switching model to {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch model: {str(e)}"
        )


@router.get(
    "/models",
    response_model=Dict[str, Any],
    summary="List available Ollama models",
    description="Get list of all downloaded Ollama models"
)
async def list_ollama_models(
    ollama_service: OllamaService = Depends(get_ollama_service)
) -> Dict[str, Any]:
    """
    List all available Ollama models.
    
    This endpoint provides:
    - List of all downloaded models
    - Model sizes and formats
    - Currently active model
    - Last modified dates
    """
    try:
        if not ollama_service.http_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HTTP client not available"
            )
        
        response = await ollama_service.http_client.get(
            f"{ollama_service.base_url}/api/tags",
            timeout=10.0
        )
        response.raise_for_status()
        
        models_data = response.json()
        
        return {
            "success": True,
            "current_model": ollama_service.model,
            "available_models": models_data.get("models", []),
            "total_count": len(models_data.get("models", [])),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not retrieve model list: {str(e)}"
        )