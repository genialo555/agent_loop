"""Test core functionality."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.settings import Settings
from core.utils import create_logger


def test_python_version():
    """Test Python version requirement."""
    # For development, accept Python 3.12+ until 3.13 is widely available
    assert sys.version_info >= (3, 12), "Python 3.12+ required (3.13+ preferred)"


def test_settings_import():
    """Test that Settings can be imported."""
    assert Settings is not None
    # Settings class exists and can be referenced


def test_settings_initialization():
    """Test Settings class initialization and default values."""
    settings = Settings()
    
    # Test default values
    assert settings.lambda_hint == 0.3
    assert settings.mu_gt == 0.1
    assert settings.base_model == "gemma_base.gguf"
    assert settings.tools == ["browse", "click", "extract"]
    
    # Test path attributes
    assert settings.data_dir.name == "datasets"
    assert settings.model_dir.name == "models"


def test_settings_with_env_override():
    """Test Settings with environment variable override."""
    import os
    
    # Test with environment variable override (case insensitive)
    original_value = os.environ.get("LAMBDA_HINT")
    os.environ["LAMBDA_HINT"] = "0.5"
    
    try:
        settings = Settings()
        assert settings.lambda_hint == 0.5
    finally:
        # Cleanup
        if original_value is not None:
            os.environ["LAMBDA_HINT"] = original_value
        else:
            os.environ.pop("LAMBDA_HINT", None)


def test_logger_creation():
    """Test logger creation utility."""
    # Test basic logger creation
    logger = create_logger("test_logger")
    assert logger is not None
    
    # Test logger with custom level
    logger_debug = create_logger("test_debug", "DEBUG")
    assert logger_debug is not None
    
    # Test logger with invalid level falls back to INFO
    logger_invalid = create_logger("test_invalid", "INVALID_LEVEL")
    assert logger_invalid is not None


def test_project_structure():
    """Test that expected directories exist."""
    root = Path(__file__).parent.parent
    
    expected_dirs = [
        "agent",
        "agent/tools", 
        "core",
        "core/utils",
        "inference",
        "plugins",
        "scripts",
        "training",
        "training/nn",
        "tests",
        "terraform",
        "ansible"
    ]
    
    for dir_name in expected_dirs:
        dir_path = root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} should exist"
        assert dir_path.is_dir(), f"{dir_name} should be a directory"