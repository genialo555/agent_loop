"""Agent execution endpoints with Ollama integration."""
import time
import uuid
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
import structlog

from ..models.schemas import (
    RunRequest, RunResponse, RunAgentRequest, RunAgentResponse, ErrorResponse
)
from ..services.dependencies import get_ollama_service, get_external_api_service
from ..services.ollama import OllamaService
from ..services.external_api import ExternalAPIService
from ..groupthink import generate as gt_generate

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/agents", tags=["agents"])


async def run_agent_async(instruction: str, use_groupthink: bool, timeout: int) -> str:
    """Execute agent asynchronously with timeout.
    
    Args:
        instruction: Instruction to process
        use_groupthink: Whether to use groupthink mode
        timeout: Timeout in seconds
        
    Returns:
        Agent response
        
    Raises:
        HTTPException: For timeout or processing errors
    """
    try:
        # Simulation of async operation
        # In real implementation, gt_generate should be made async
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


async def send_webhook_notification(
    webhook_url: str,
    task_id: str,
    result: str,
    correlation_id: str
) -> None:
    """Send webhook notification asynchronously.
    
    Args:
        webhook_url: URL to send webhook to
        task_id: Task identifier
        result: Processing result
        correlation_id: Correlation ID for tracking
    """
    from datetime import datetime, timezone
    import httpx
    
    # Get HTTP client from dependency (this is a simplified version)
    # In production, you'd want to use the shared HTTP client
    async with httpx.AsyncClient() as client:
        try:
            payload = {
                "task_id": task_id,
                "result": result,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            response = await client.post(
                webhook_url,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            logger.info(f"Webhook sent successfully to {webhook_url}")
        except Exception as e:
            logger.error(f"Failed to send webhook to {webhook_url}: {e}")


@router.post(
    "/run",
    response_model=RunResponse,
    responses={
        408: {"model": ErrorResponse, "description": "Request timeout"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Execute agent with instruction",
    description="Execute the agent with the given instruction, optionally using groupthink mode"
)
async def run_agent(
    request: Request,
    req: RunRequest,
    background_tasks: BackgroundTasks,
    external_service: ExternalAPIService = Depends(get_external_api_service)
) -> RunResponse:
    """
    Execute the agent with the given instruction.
    
    This endpoint processes the instruction asynchronously and can optionally
    send the result to a webhook for fire-and-forget notifications.
    """
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Example of external API call (commented to avoid errors)
        # external_data = await external_service.fetch_data("https://api.example.com/data")
        
        # Main processing
        answer = await run_agent_async(
            req.instruction,
            req.use_groupthink,
            req.timeout_seconds or 30
        )
        
        processing_time = time.time() - start_time
        
        # Schedule webhook notification if requested
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


@router.post(
    "/run-agent",
    response_model=RunAgentResponse,
    summary="Enhanced agent with Ollama LLM integration",
    description="Execute agent with Ollama LLM support for both browser operations and general queries"
)
async def run_agent_with_ollama(
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
        logger.error(f"Error in run_agent_with_ollama: {e}", exc_info=True)
        return RunAgentResponse(
            success=False,
            result={},
            execution_time_ms=(time.time() - start_time) * 1000,
            model_used=model_used,
            inference_metrics=inference_metrics,
            error=str(e)
        )