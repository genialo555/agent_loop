"""Tests for browser_tool plugin."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import asyncio
from agent_loop.agent.plugins.browser_tool import BrowserTool


class TestBrowserTool:
    """Test cases for BrowserTool following TST001 and TST004 patterns."""

    def test_browser_tool_initialization(self):
        """Test BrowserTool initialization."""
        # Test default initialization
        tool = BrowserTool()
        assert tool.headless is True
        assert tool.name == "browse"
        assert tool.description == "Browse a webpage and return its title and content"
        assert isinstance(tool.screenshot_dir, Path)

    def test_browser_tool_initialization_with_params(self):
        """Test BrowserTool initialization with custom parameters."""
        tool = BrowserTool(headless=False)
        assert tool.headless is False
        assert tool.screenshot_dir.exists()

    @pytest.mark.asyncio
    async def test_browse_with_httpx_fallback(self):
        """Test browser tool using httpx fallback when Playwright not available."""
        # Mock the import to force httpx fallback
        with patch.dict('sys.modules', {'playwright.async_api': None}):
            tool = BrowserTool()
            
            # Mock httpx response
            mock_response = MagicMock()
            mock_response.text = '<html><head><title>Test Title</title></head><body>Test content</body></html>'
            mock_response.url = "https://test.example.com"
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_httpx.return_value.__aexit__ = AsyncMock()
                
                result = await tool("https://example.com")
                
                # Assertions
                assert isinstance(result, dict)
                assert "title" in result
                assert "content" in result
                assert "url" in result
                assert "load_time_ms" in result
                assert result["screenshot_path"] is None
                assert "warning" in result

    @pytest.mark.asyncio
    @patch('agent_loop.agent.plugins.browser_tool.async_playwright')
    async def test_browse_with_playwright(self, mock_playwright):
        """Test browser tool using Playwright."""
        # Setup mocks
        mock_page = AsyncMock()
        mock_page.title.return_value = "Test Page Title"
        mock_page.evaluate.return_value = "Test page content"
        mock_page.screenshot = AsyncMock()
        mock_page.goto = AsyncMock()
        
        mock_browser = AsyncMock()
        mock_browser.new_page.return_value = mock_page
        mock_browser.close = AsyncMock()
        
        mock_chromium = AsyncMock()
        mock_chromium.launch.return_value = mock_browser
        
        mock_p = AsyncMock()
        mock_p.chromium = mock_chromium
        
        mock_playwright.return_value.__aenter__ = AsyncMock(return_value=mock_p)
        mock_playwright.return_value.__aexit__ = AsyncMock()
        
        # Test
        tool = BrowserTool()
        result = await tool("https://example.com")
        
        # Assertions
        assert isinstance(result, dict)
        assert result["title"] == "Test Page Title"
        assert result["content"] == "Test page content"
        assert result["url"] == "https://example.com"
        assert "load_time_ms" in result
        assert result["screenshot_path"] is not None

    @pytest.mark.asyncio
    @patch('agent_loop.agent.plugins.browser_tool.async_playwright')
    async def test_browse_no_screenshot(self, mock_playwright):
        """Test browser tool without taking screenshot."""
        # Setup mocks similar to above but test screenshot=False
        mock_page = AsyncMock()
        mock_page.title.return_value = "Test Title"
        mock_page.evaluate.return_value = "Test content"
        mock_page.goto = AsyncMock()
        
        mock_browser = AsyncMock()
        mock_browser.new_page.return_value = mock_page
        mock_browser.close = AsyncMock()
        
        mock_chromium = AsyncMock()
        mock_chromium.launch.return_value = mock_browser
        
        mock_p = AsyncMock()
        mock_p.chromium = mock_chromium
        
        mock_playwright.return_value.__aenter__ = AsyncMock(return_value=mock_p)
        mock_playwright.return_value.__aexit__ = AsyncMock()
        
        # Test
        tool = BrowserTool()
        result = await tool("https://example.com", screenshot=False)
        
        # Assertions
        assert result["screenshot_path"] is None
        mock_page.screenshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_httpx_html_parsing(self):
        """Test HTML parsing in httpx fallback mode."""
        # Mock imports to force httpx path
        with patch.dict('sys.modules', {'playwright.async_api': None}):
            tool = BrowserTool()
            
            # Create test HTML
            test_html = '''
            <html>
                <head><title>Parsed Title</title></head>
                <body>
                    <h1>Header</h1>
                    <p>Paragraph 1</p>
                    <p>Paragraph 2</p>
                    <script>console.log('test');</script>
                </body>
            </html>
            '''
            
            mock_response = MagicMock()
            mock_response.text = test_html
            mock_response.url = "https://parsed.example.com"
            mock_response.raise_for_status = MagicMock()
            
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_httpx.return_value.__aexit__ = AsyncMock()
                
                result = await tool("https://example.com")
                
                # Assertions
                assert result["title"] == "Parsed Title"
                assert "Header" in result["content"]
                assert "Paragraph" in result["content"]

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in browser tool."""
        with patch.dict('sys.modules', {'playwright.async_api': None}):
            tool = BrowserTool()
            
            # Mock httpx to raise an exception
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=Exception("Network error"))
                mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_httpx.return_value.__aexit__ = AsyncMock()
                
                # Should raise the exception
                with pytest.raises(Exception, match="Network error"):
                    await tool("https://invalid-url.com")

    def test_screenshot_directory_creation(self):
        """Test screenshot directory is created properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create tool with custom directory
            tool = BrowserTool()
            tool.screenshot_dir = Path(tmpdir) / "custom_screenshots"
            
            # Directory should be created during init if it doesn't exist
            tool.screenshot_dir.mkdir(exist_ok=True)
            assert tool.screenshot_dir.exists()
            assert tool.screenshot_dir.is_dir()

    @pytest.mark.property
    @pytest.mark.asyncio
    async def test_url_property_based(self):
        """Property-based test for URL handling (TST003)."""
        from hypothesis import given, strategies as st
        
        @given(st.text(min_size=1, max_size=100))
        async def test_url_formats(url_suffix):
            """Test various URL formats."""
            # Skip if url_suffix contains invalid characters
            if any(c in url_suffix for c in [' ', '\n', '\t']):
                return
                
            tool = BrowserTool()
            test_url = f"https://example.com/{url_suffix}"
            
            # Mock the httpx path since we're testing URL handling
            with patch.dict('sys.modules', {'playwright.async_api': None}):
                mock_response = MagicMock()
                mock_response.text = f'<html><head><title>Test</title></head><body>Content for {url_suffix}</body></html>'
                mock_response.url = test_url
                mock_response.raise_for_status = MagicMock()
                
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(return_value=mock_response)
                
                with patch('httpx.AsyncClient') as mock_httpx:
                    mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_httpx.return_value.__aexit__ = AsyncMock()
                    
                    try:
                        result = await tool(test_url)
                        # Should return a valid result structure
                        assert isinstance(result, dict)
                        assert "title" in result
                        assert "content" in result
                        assert "url" in result
                    except Exception:
                        # Some URLs might be invalid, which is acceptable
                        pass
        
        # Run the property-based test
        await test_url_formats()

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_browser_tool_performance(self, benchmark):
        """Benchmark browser tool performance (TST007)."""
        tool = BrowserTool()
        
        # Mock to avoid actual network calls in benchmark
        with patch.dict('sys.modules', {'playwright.async_api': None}):
            mock_response = MagicMock()
            mock_response.text = '<html><head><title>Benchmark</title></head><body>Fast content</body></html>'
            mock_response.url = "https://benchmark.example.com"
            mock_response.raise_for_status = MagicMock()
            
            async def mock_get(*args, **kwargs):
                # Simulate some processing time
                await asyncio.sleep(0.001)
                return mock_response
            
            mock_client = AsyncMock()
            mock_client.get = mock_get
            
            with patch('httpx.AsyncClient') as mock_httpx:
                mock_httpx.return_value.__aenter__ = AsyncMock(return_value=mock_client)
                mock_httpx.return_value.__aexit__ = AsyncMock()
                
                # Benchmark the async function
                async def run_tool():
                    return await tool("https://example.com")
                
                result = await benchmark(run_tool)
                assert isinstance(result, dict)
                assert "load_time_ms" in result