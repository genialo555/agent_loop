"""
Hierarchical Reasoning Model (HRM) implementation for Gemma-3N.

This module implements the HRM architecture with hierarchical convergence,
approximate gradients, and deep supervision for enhanced reasoning capabilities.
"""

from agent_loop.models.training.hrm.hrm_config import HRMConfig
from agent_loop.models.training.hrm.hrm_model import HRMGemma3N

__all__ = ["HRMConfig", "HRMGemma3N"]