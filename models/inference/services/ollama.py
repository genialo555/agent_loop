"""Optimized Ollama service with enhanced health checking and error handling."""
import time
from typing import Optional, Dict, Any
import httpx
import structlog
from fastapi import HTTPException, status

logger = structlog.get_logger(__name__)


class OllamaService:
    """Service optimized for Ollama calls with performance monitoring and health checks."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434", http_client: Optional[httpx.AsyncClient] = None):
        """Initialize Ollama service.
        
        Args:
            base_url: Ollama API base URL
            http_client: Shared HTTP client for connection pooling
        """
        self.base_url = base_url
        self.model = "gemma3n:e2b"
        self.http_client = http_client
        self.default_options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": 2048,
            "num_ctx": 8192,
            "stop": ["<|endoftext|>", "</s>", "<|im_end|>"]
        }
        # Track health check cache to avoid hammering Ollama
        self._last_health_check: float = 0.0
        self._last_health_status = False
        self._health_cache_ttl: float = 30.0  # seconds
    
    async def health_check(self, force: bool = False) -> bool:
        """Check Ollama availability with caching.
        
        Args:
            force: Force health check bypassing cache
            
        Returns:
            True if Ollama is available, False otherwise
        """
        current_time = time.time()
        
        # Use cached result if recent and not forced
        if not force and (current_time - self._last_health_check) < self._health_cache_ttl:
            return self._last_health_status
        
        if not self.http_client:
            logger.warning("HTTP client not available for health check")
            self._last_health_status = False
            self._last_health_check = current_time
            return False
        
        try:
            response = await self.http_client.get(
                f"{self.base_url}/api/version",
                timeout=5.0
            )
            is_healthy = response.status_code == 200
            
            # Cache the result
            self._last_health_check = current_time
            self._last_health_status = is_healthy
            
            if is_healthy:
                logger.debug("Ollama health check passed")
            else:
                logger.warning(f"Ollama health check failed with status {response.status_code}")
            
            return is_healthy
            
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            self._last_health_check = current_time
            self._last_health_status = False
            return False
    
    async def detailed_health_check(self) -> Dict[str, Any]:
        """Perform a detailed health check including model availability.
        
        Returns:
            Dictionary with detailed health information
        """
        if not self.http_client:
            return {
                "available": False,
                "error": "HTTP client not initialized",
                "timestamp": time.time()
            }
        
        health_info: Dict[str, Any] = {
            "available": False,
            "version": None,
            "model_available": False,
            "model_name": self.model,
            "response_time_ms": None,
            "error": None,
            "timestamp": time.time()
        }
        
        start_time = time.time()
        
        try:
            # Check version endpoint
            version_response = await self.http_client.get(
                f"{self.base_url}/api/version",
                timeout=5.0
            )
            
            response_time = (time.time() - start_time) * 1000
            health_info["response_time_ms"] = response_time
            
            if version_response.status_code == 200:
                health_info["available"] = True
                version_data = version_response.json()
                health_info["version"] = version_data.get("version", "unknown")
                
                # Check if our model is available
                try:
                    tags_response = await self.http_client.get(
                        f"{self.base_url}/api/tags",
                        timeout=5.0
                    )
                    
                    if tags_response.status_code == 200:
                        models_data = tags_response.json()
                        available_models = [m["name"] for m in models_data.get("models", [])]
                        health_info["model_available"] = self.model in available_models
                        health_info["available_models"] = available_models
                    
                except Exception as model_check_error:
                    logger.warning(f"Model availability check failed: {model_check_error}")
                    health_info["model_check_error"] = str(model_check_error)
            
            else:
                health_info["error"] = f"Version endpoint returned {version_response.status_code}"
                
        except httpx.TimeoutException:
            health_info["error"] = "Timeout connecting to Ollama"
            health_info["response_time_ms"] = (time.time() - start_time) * 1000
        except Exception as e:
            health_info["error"] = str(e)
            health_info["response_time_ms"] = (time.time() - start_time) * 1000
        
        return health_info
    
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """Generate response with Ollama using ML optimizations.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Creativity control (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Streaming mode (not implemented in this version)
        
        Returns:
            Generated response from the model
        
        Raises:
            HTTPException: For various error conditions
        """
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")
        
        # ML001 Optimization: Prepare options with quantization
        options = self.default_options.copy()
        if temperature is not None:
            options["temperature"] = max(0.0, min(1.0, temperature))
        if max_tokens is not None:
            options["num_predict"] = min(max_tokens, 4096)
        
        # Construct prompt with system prompt if provided
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "options": options,
            "stream": False,  # Non-streaming mode for this implementation
            "keep_alive": "5m"  # ML004 Optimization: KV cache for 5 minutes
        }
        
        try:
            start_time = time.time()
            
            # API call with optimized timeout
            response = await self.http_client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=httpx.Timeout(120.0)  # Extended timeout for inference
            )
            response.raise_for_status()
            
            result = response.json()
            inference_time = time.time() - start_time
            
            # Performance metrics logging
            logger.info(
                "Ollama inference completed",
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
            
            return str(result.get("response", ""))
            
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
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model information or error details
        """
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")
        
        try:
            response = await self.http_client.get(
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
    
    def set_model(self, model_name: str) -> None:
        """Change the active model.
        
        Args:
            model_name: Name of the model to use
        """
        self.model = model_name
        logger.info(f"Ollama model changed to: {model_name}")
        # Invalidate health cache when model changes
        self._last_health_check = 0.0