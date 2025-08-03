"""
API compatibility layer.

This module provides backward compatibility by importing the app from app.py.
This allows existing references to 'inference.api:app' to continue working
while the main application logic is in app.py following the new modular architecture.
"""

from .app import app

__all__ = ["app"]