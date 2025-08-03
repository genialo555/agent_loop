"""Playwright-based browser automation tool for Sprint 1 PoC."""
from typing import Dict, Any, Optional
import asyncio
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BrowserTool:
    """Simple browser tool for PoC - captures title and content from a webpage."""
    
    name = "browse"
    description = "Browse a webpage and return its title and content"
    
    def __init__(self, headless: bool = True):
        """Initialize browser tool.
        
        Args:
            headless: Whether to run browser in headless mode
        """
        self.headless = headless
        self.screenshot_dir = Path("/tmp/agent_screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)
    
    async def __call__(self, url: str, screenshot: bool = True) -> Dict[str, Any]:
        """Browse a URL and extract information.
        
        Args:
            url: The URL to browse
            screenshot: Whether to take a screenshot
            
        Returns:
            Dictionary with title, content, and optional screenshot path
        """
        # For Sprint 1 PoC, we'll try to use Playwright if available
        # Otherwise fall back to a simple httpx request
        try:
            from playwright.async_api import async_playwright
            return await self._browse_with_playwright(url, screenshot)
        except ImportError:
            logger.warning("Playwright not installed, using simple HTTP request")
            return await self._browse_with_httpx(url)
    
    async def _browse_with_playwright(self, url: str, screenshot: bool) -> Dict[str, Any]:
        """Browse using Playwright (full browser automation)."""
        from playwright.async_api import async_playwright
        
        start_time = time.time()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            )
            
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle")
                
                # Get title
                title = await page.title()
                
                # Get text content (first 1000 chars)
                content = await page.evaluate(
                    """() => {
                        const text = document.body?.innerText || '';
                        return text.substring(0, 1000);
                    }"""
                )
                
                # Take screenshot if requested
                screenshot_path = None
                if screenshot:
                    timestamp = int(time.time())
                    screenshot_path = self.screenshot_dir / f"screenshot_{timestamp}.png"
                    await page.screenshot(path=str(screenshot_path))
                
                return {
                    "title": title,
                    "content": content,
                    "url": url,
                    "screenshot_path": str(screenshot_path) if screenshot_path else None,
                    "load_time_ms": (time.time() - start_time) * 1000,
                }
                
            finally:
                await browser.close()
    
    async def _browse_with_httpx(self, url: str) -> Dict[str, Any]:
        """Simple fallback using httpx (no JavaScript support)."""
        import httpx
        from html.parser import HTMLParser
        
        class TitleParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.title = ""
                self.in_title = False
                self.text_content = []
            
            def handle_starttag(self, tag, attrs):
                if tag == "title":
                    self.in_title = True
            
            def handle_endtag(self, tag):
                if tag == "title":
                    self.in_title = False
            
            def handle_data(self, data):
                if self.in_title:
                    self.title += data
                else:
                    # Collect text content
                    text = data.strip()
                    if text:
                        self.text_content.append(text)
        
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            # Parse HTML to extract title
            parser = TitleParser()
            parser.feed(response.text)
            
            # Get first 1000 chars of text content
            content = " ".join(parser.text_content)[:1000]
            
            return {
                "title": parser.title.strip(),
                "content": content,
                "url": str(response.url),  # Final URL after redirects
                "screenshot_path": None,  # No screenshot in fallback mode
                "load_time_ms": (time.time() - start_time) * 1000,
                "warning": "Using fallback HTTP mode - no JavaScript support",
            }
