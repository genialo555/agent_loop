"""
HRM Module implementations: Low-level and High-level modules for Gemma-3N.

This module implements the two core components of HRM:
- Low-level module (L): Fast, detailed computations with rapid convergence
- High-level module (H): Slow, abstract planning with strategic guidance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.activations import ACT2FN


@dataclass
class ModuleOutput:
    """Output from HRM modules."""
    hidden_state: torch.Tensor
    converged: bool = False
    num_iterations: int = 0
    convergence_metric: float = float('inf')
    auxiliary_loss: Optional[torch.Tensor] = None


class HRMModuleBase(nn.Module):
    """Base class for HRM modules with common functionality."""
    
    def __init__(
        self,
        gemma_config,
        module_config,
        module_type: str = "low"
    ):
        super().__init__()
        self.config = module_config
        self.module_type = module_type
        self.gemma_hidden_size = gemma_config.hidden_size  # 3584 for Gemma-3N
        
        # State projectors
        self.input_projector = nn.Linear(
            self.gemma_hidden_size, 
            self.config.hidden_dim
        )
        self.output_projector = nn.Linear(
            self.config.hidden_dim,
            self.gemma_hidden_size
        )
        
        # Layer normalization
        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(
                self.config.hidden_dim, 
                eps=self.config.norm_eps
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize module weights."""
        # Use truncated normal as in HRM paper
        std = 0.02
        nn.init.trunc_normal_(self.input_projector.weight, std=std)
        nn.init.trunc_normal_(self.output_projector.weight, std=std)
        nn.init.zeros_(self.input_projector.bias)
        nn.init.zeros_(self.output_projector.bias)
    
    def compute_convergence_metric(
        self, 
        current_state: torch.Tensor, 
        previous_state: torch.Tensor
    ) -> float:
        """Compute convergence metric between states."""
        with torch.no_grad():
            # L2 norm of the difference
            diff = current_state - previous_state
            metric = torch.norm(diff, p=2, dim=-1).mean().item()
        return metric
    
    def check_convergence(
        self, 
        current_state: torch.Tensor, 
        previous_state: torch.Tensor
    ) -> bool:
        """Check if module has converged."""
        metric = self.compute_convergence_metric(current_state, previous_state)
        return metric < self.config.convergence_threshold


class LowLevelModule(HRMModuleBase):
    """
    Low-level module for fast, detailed computations.
    Updates every timestep and converges quickly within each high-level cycle.
    """
    
    def __init__(self, gemma_config, module_config):
        super().__init__(gemma_config, module_config, "low")
        
        # Core transformation layers
        self.transform_layer = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.intermediate_dim),
            nn.GELU(),
            nn.Dropout(self.config.lora_dropout),
            nn.Linear(self.config.intermediate_dim, self.config.hidden_dim)
        )
        
        # Gating mechanism for state updates
        self.update_gate = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.Sigmoid()
        )
        
        # LoRA configuration for attention layers
        self.lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
    
    def forward(
        self,
        z_l: torch.Tensor,  # Low-level hidden state
        z_h: torch.Tensor,  # High-level hidden state (fixed during cycle)
        x_emb: torch.Tensor,  # Input embeddings
        iteration: int = 0
    ) -> ModuleOutput:
        """
        Forward pass of low-level module.
        
        Args:
            z_l: Current low-level state [batch, seq_len, hidden_dim]
            z_h: Current high-level state [batch, seq_len, hidden_dim]
            x_emb: Input embeddings [batch, seq_len, gemma_hidden_size]
            iteration: Current iteration within cycle
        
        Returns:
            ModuleOutput with updated state and convergence info
        """
        batch_size, seq_len = x_emb.shape[:2]
        
        # Project inputs to module dimension
        x_proj = self.input_projector(x_emb)
        z_h_proj = self.input_projector(z_h) if z_h.shape[-1] != self.config.hidden_dim else z_h
        
        # Store previous state for convergence check
        z_l_prev = z_l.clone()
        
        # Concatenate all inputs
        combined = torch.cat([z_l, z_h_proj, x_proj], dim=-1)
        
        # Compute transformation
        transformed = self.transform_layer(combined)
        
        # Compute update gate
        gate = self.update_gate(combined)
        
        # Gated update
        z_l_new = gate * transformed + (1 - gate) * z_l
        
        # Apply layer norm if enabled
        if self.config.use_layer_norm:
            z_l_new = self.layer_norm(z_l_new)
        
        # Check convergence
        converged = self.check_convergence(z_l_new, z_l_prev)
        convergence_metric = self.compute_convergence_metric(z_l_new, z_l_prev)
        
        return ModuleOutput(
            hidden_state=z_l_new,
            converged=converged,
            num_iterations=iteration + 1,
            convergence_metric=convergence_metric
        )
    
    def reset_with_context(self, z_h: torch.Tensor) -> torch.Tensor:
        """Reset low-level state with high-level context."""
        # Initialize new state influenced by high-level state
        batch_size, seq_len = z_h.shape[:2]
        
        # Project high-level state and add noise for exploration
        z_h_proj = self.input_projector(z_h) if z_h.shape[-1] != self.config.hidden_dim else z_h
        noise = torch.randn_like(z_h_proj) * 0.1
        
        # New initial state is a transformation of high-level state
        z_l_new = self.layer_norm(z_h_proj + noise) if self.config.use_layer_norm else z_h_proj + noise
        
        return z_l_new


class HighLevelModule(HRMModuleBase):
    """
    High-level module for abstract planning and strategic guidance.
    Updates only after low-level module converges.
    """
    
    def __init__(self, gemma_config, module_config):
        super().__init__(gemma_config, module_config, "high")
        
        # Core transformation layers with larger capacity
        self.transform_layer = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, self.config.intermediate_dim),
            nn.GELU(),
            nn.Dropout(self.config.lora_dropout),
            nn.Linear(self.config.intermediate_dim, self.config.intermediate_dim),
            nn.GELU(),
            nn.Dropout(self.config.lora_dropout),
            nn.Linear(self.config.intermediate_dim, self.config.hidden_dim)
        )
        
        # Strategic planning layer
        self.planning_layer = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim, eps=self.config.norm_eps),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )
        
        # LoRA configuration for FFN layers
        self.lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
    
    def forward(
        self,
        z_h: torch.Tensor,  # High-level hidden state
        z_l_final: torch.Tensor,  # Final low-level state after convergence
        cycle: int = 0
    ) -> ModuleOutput:
        """
        Forward pass of high-level module.
        
        Args:
            z_h: Current high-level state [batch, seq_len, hidden_dim]
            z_l_final: Converged low-level state [batch, seq_len, hidden_dim]
            cycle: Current high-level cycle number
        
        Returns:
            ModuleOutput with updated state
        """
        # Project low-level state if needed
        z_l_proj = self.input_projector(z_l_final) if z_l_final.shape[-1] != self.config.hidden_dim else z_l_final
        
        # Store previous state
        z_h_prev = z_h.clone()
        
        # Combine states
        combined = torch.cat([z_h, z_l_proj], dim=-1)
        
        # Main transformation
        transformed = self.transform_layer(combined)
        
        # Strategic planning refinement
        z_h_new = self.planning_layer(transformed)
        
        # Residual connection
        z_h_new = z_h_new + z_h
        
        # Apply layer norm
        if self.config.use_layer_norm:
            z_h_new = self.layer_norm(z_h_new)
        
        # High-level module doesn't check convergence per update
        return ModuleOutput(
            hidden_state=z_h_new,
            converged=False,  # Convergence managed at cycle level
            num_iterations=cycle + 1,
            convergence_metric=self.compute_convergence_metric(z_h_new, z_h_prev)
        )


class HRMStateInitializer(nn.Module):
    """Initialize HRM hidden states from input embeddings."""
    
    def __init__(self, gemma_config, low_config, high_config):
        super().__init__()
        self.gemma_hidden_size = gemma_config.hidden_size
        
        # Separate initializers for each module
        self.low_init = nn.Sequential(
            nn.Linear(self.gemma_hidden_size, low_config.hidden_dim),
            nn.LayerNorm(low_config.hidden_dim),
            nn.GELU(),
            nn.Linear(low_config.hidden_dim, low_config.hidden_dim)
        )
        
        self.high_init = nn.Sequential(
            nn.Linear(self.gemma_hidden_size, high_config.hidden_dim),
            nn.LayerNorm(high_config.hidden_dim),
            nn.GELU(),
            nn.Linear(high_config.hidden_dim, high_config.hidden_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small random weights."""
        for module in [self.low_init, self.high_init]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize both module states from input embeddings.
        
        Args:
            x_emb: Input embeddings [batch, seq_len, gemma_hidden_size]
        
        Returns:
            Tuple of (z_l_init, z_h_init)
        """
        # Use mean pooling over sequence for initialization
        x_pooled = x_emb.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        
        # Initialize states
        z_l_init = self.low_init(x_pooled)
        z_h_init = self.high_init(x_pooled)
        
        # Expand to full sequence length
        seq_len = x_emb.shape[1]
        z_l_init = z_l_init.expand(-1, seq_len, -1)
        z_h_init = z_h_init.expand(-1, seq_len, -1)
        
        return z_l_init, z_h_init


def create_hrm_modules(gemma_config, hrm_config):
    """
    Factory function to create HRM modules.
    
    Args:
        gemma_config: Gemma model configuration
        hrm_config: HRM configuration
    
    Returns:
        Dictionary with initialized modules
    """
    modules = {
        'low_level': LowLevelModule(gemma_config, hrm_config.low_level_config),
        'high_level': HighLevelModule(gemma_config, hrm_config.high_level_config),
        'state_init': HRMStateInitializer(
            gemma_config, 
            hrm_config.low_level_config,
            hrm_config.high_level_config
        )
    }
    
    return modules