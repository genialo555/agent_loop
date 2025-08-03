"""Service for external API calls with proper async patterns."""
from typing import Dict, Any, Optional
import httpx
import structlog
from fastapi import HTTPException, status

logger = structlog.get_logger(__name__)


class ExternalAPIService:
    """Service for non-blocking external API calls with error handling."""
    
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        """Initialize external API service.
        
        Args:
            http_client: Shared HTTP client for connection pooling
        """
        self.http_client = http_client
    
    async def fetch_data(self, url: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Fetch data from external API in a non-blocking way.
        
        Args:
            url: URL to fetch data from
            timeout: Request timeout in seconds
            
        Returns:
            JSON response data
            
        Raises:
            HTTPException: For various error conditions
        """
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")
        
        try:
            logger.info(f"Fetching data from external API: {url}")
            
            response = await self.http_client.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "FastAPI-Agent/1.0.0",
                    "Accept": "application/json"
                }
            )
            response.raise_for_status()
            
            data: Dict[str, Any] = response.json()
            logger.info(f"Successfully fetched data from {url}")
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e.response.status_code}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"External API error: {e.response.status_code}"
            )
        except httpx.TimeoutException:
            logger.error(f"Timeout fetching data from {url}")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="External API request timeout"
            )
        except httpx.RequestError as e:
            logger.error(f"Request error fetching {url}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External service connection error"
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External service unavailable"
            )
    
    async def post_data(self, url: str, data: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Post data to external API in a non-blocking way.
        
        Args:
            url: URL to post data to
            data: Data to post
            timeout: Request timeout in seconds
            
        Returns:
            JSON response data
            
        Raises:
            HTTPException: For various error conditions
        """
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")
        
        try:
            logger.info(f"Posting data to external API: {url}")
            
            response = await self.http_client.post(
                url,
                json=data,
                timeout=timeout,
                headers={
                    "User-Agent": "FastAPI-Agent/1.0.0",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            response.raise_for_status()
            
            result: Dict[str, Any] = response.json()
            logger.info(f"Successfully posted data to {url}")
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error posting to {url}: {e.response.status_code}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"External API error: {e.response.status_code}"
            )
        except httpx.TimeoutException:
            logger.error(f"Timeout posting data to {url}")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT,
                detail="External API request timeout"
            )
        except httpx.RequestError as e:
            logger.error(f"Request error posting to {url}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External service connection error"
            )
        except Exception as e:
            logger.error(f"Unexpected error posting to {url}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="External service unavailable"
            )
    
    async def health_check(self, url: str) -> bool:
        """Check if an external service is available.
        
        Args:
            url: URL to check
            
        Returns:
            True if service is available, False otherwise
        """
        if not self.http_client:
            return False
        
        try:
            response = await self.http_client.get(
                url,
                timeout=5.0,
                headers={"User-Agent": "FastAPI-Agent/1.0.0"}
            )
            return response.status_code < 500
        except Exception as e:
            logger.warning(f"External service health check failed for {url}: {e}")
            return False