"""
Main HRM-Gemma3N model implementation.

This module combines Gemma-3N base model with HRM modules to create
a hierarchical reasoning model with enhanced computational depth.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import get_peft_model, TaskType

from .hrm_config import HRMConfig
from .hrm_modules import (
    create_hrm_modules, 
    ModuleOutput,
    LowLevelModule,
    HighLevelModule,
    HRMStateInitializer
)


@dataclass
class HRMOutput:
    """Output from HRM forward pass."""
    logits: torch.Tensor
    low_level_states: List[torch.Tensor]
    high_level_states: List[torch.Tensor]
    convergence_info: Dict[str, Any]
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    loss: Optional[torch.Tensor] = None


class HierarchicalConvergence:
    """Manages the hierarchical convergence process."""
    
    def __init__(self, config: HRMConfig):
        self.config = config
        self.convergence_history = []
    
    def run_low_level_cycle(
        self,
        low_module: LowLevelModule,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        x_emb: torch.Tensor,
        max_iterations: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run low-level module until convergence or max iterations.
        
        Returns:
            Final low-level state and convergence info
        """
        max_iterations = max_iterations or self.config.timesteps_per_cycle
        convergence_info = {
            'iterations': 0,
            'converged': False,
            'final_metric': float('inf'),
            'metrics_history': []
        }
        
        for t in range(max_iterations):
            # Low-level update
            output = low_module(z_l, z_h, x_emb, iteration=t)
            z_l = output.hidden_state
            
            # Track convergence
            convergence_info['iterations'] = t + 1
            convergence_info['metrics_history'].append(output.convergence_metric)
            
            if output.converged:
                convergence_info['converged'] = True
                convergence_info['final_metric'] = output.convergence_metric
                break
        
        convergence_info['final_metric'] = output.convergence_metric
        return z_l, convergence_info
    
    def step(
        self,
        low_module: LowLevelModule,
        high_module: HighLevelModule,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        x_emb: torch.Tensor,
        cycle: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute one complete hierarchical convergence cycle.
        
        Returns:
            Updated (z_l, z_h) states and convergence info
        """
        # Run low-level to convergence
        z_l_final, low_conv_info = self.run_low_level_cycle(
            low_module, z_l, z_h, x_emb
        )
        
        # High-level update using converged low-level state
        high_output = high_module(z_h, z_l_final, cycle=cycle)
        z_h_new = high_output.hidden_state
        
        # Reset low-level with new high-level context
        z_l_new = low_module.reset_with_context(z_h_new)
        
        # Combine convergence info
        conv_info = {
            'cycle': cycle,
            'low_level': low_conv_info,
            'high_level': {
                'metric': high_output.convergence_metric,
                'iterations': high_output.num_iterations
            }
        }
        
        self.convergence_history.append(conv_info)
        
        return z_l_new, z_h_new, conv_info


class HRMGemma3N(nn.Module):
    """
    Hierarchical Reasoning Model based on Gemma-3N.
    
    Combines Gemma-3N base model with HRM modules for enhanced reasoning.
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        
        # Load base Gemma model
        self.gemma = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            torch_dtype=torch.float16 if config.torch_dtype == "float16" else torch.float32,
            device_map=config.device_map,
            load_in_4bit=config.load_in_4bit,
            trust_remote_code=True,
            cache_dir=config.dataset_cache_dir  # Use HF cache
        )
        
        # Get Gemma config
        self.gemma_config = self.gemma.config
        
        # Create HRM modules
        self.hrm_modules = create_hrm_modules(self.gemma_config, config)
        self.low_module = self.hrm_modules['low_level']
        self.high_module = self.hrm_modules['high_level']
        self.state_initializer = self.hrm_modules['state_init']
        
        # Apply LoRA to Gemma model
        self._apply_lora()
        
        # Hierarchical convergence manager
        self.convergence_manager = HierarchicalConvergence(config)
        
        # Output projection
        self.output_projection = nn.Linear(
            config.high_level_config.hidden_dim,
            self.gemma_config.vocab_size,
            bias=False
        )
        
        # Adaptive Computation Time components (if enabled)
        if config.enable_act:
            self.q_head = nn.Linear(
                config.high_level_config.hidden_dim,
                2  # halt, continue
            )
    
    def _apply_lora(self):
        """Apply LoRA adapters to Gemma model."""
        # Apply LoRA for low-level (attention layers)
        low_lora_config = self.low_module.lora_config
        
        # Apply LoRA for high-level (FFN layers)
        high_lora_config = self.high_module.lora_config
        
        # Note: In practice, we'd use peft.get_peft_model here
        # For now, we'll keep the base model as is
        warnings.warn("LoRA application to be implemented with PEFT integration")
    
    def embed_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings from Gemma model."""
        return self.gemma.model.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_cycles: Optional[int] = None,
        return_dict: bool = True,
        **kwargs
    ) -> HRMOutput:
        """
        Forward pass with hierarchical reasoning.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target labels for loss computation
            num_cycles: Override number of high-level cycles
            return_dict: Whether to return HRMOutput
        
        Returns:
            HRMOutput with logits and state information
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get input embeddings
        x_emb = self.embed_inputs(input_ids)
        
        # Initialize states
        z_l, z_h = self.state_initializer(x_emb)
        
        # Storage for states history
        low_states = []
        high_states = []
        convergence_infos = []
        
        # Number of cycles
        N = num_cycles or self.config.num_high_cycles
        
        # Hierarchical reasoning cycles
        for cycle in range(N):
            # Execute one hierarchical cycle
            z_l, z_h, conv_info = self.convergence_manager.step(
                self.low_module,
                self.high_module,
                z_l, z_h, x_emb,
                cycle
            )
            
            # Store states
            low_states.append(z_l.clone())
            high_states.append(z_h.clone())
            convergence_infos.append(conv_info)
        
        # Project final high-level state to vocabulary
        # Use the high-level state directly for output
        logits = self.output_projection(z_h)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Prepare output
        if not return_dict:
            output = (logits,) + (low_states, high_states, convergence_infos)
            return ((loss,) + output) if loss is not None else output
        
        return HRMOutput(
            logits=logits,
            low_level_states=low_states,
            high_level_states=high_states,
            convergence_info={
                'cycles': convergence_infos,
                'total_low_iterations': sum(c['low_level']['iterations'] for c in convergence_infos),
                'avg_low_convergence': sum(c['low_level']['final_metric'] for c in convergence_infos) / len(convergence_infos)
            },
            loss=loss
        )
    
    def generate_with_hrm(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using HRM reasoning.
        
        This method implements generation with hierarchical reasoning
        for each token prediction.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids
        
        for _ in range(max_length - input_ids.shape[1]):
            # Get HRM output
            with torch.no_grad():
                output = self.forward(generated, return_dict=True)
                logits = output.logits[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS token (assuming 2 is EOS)
            if (next_token == 2).all():
                break
        
        return generated
    
    def save_pretrained(self, save_directory: str):
        """Save HRM model and configuration."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(save_directory, "hrm_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save HRM modules
        hrm_state = {
            'low_module': self.low_module.state_dict(),
            'high_module': self.high_module.state_dict(),
            'state_initializer': self.state_initializer.state_dict(),
            'output_projection': self.output_projection.state_dict(),
        }
        
        if self.config.enable_act:
            hrm_state['q_head'] = self.q_head.state_dict()
        
        hrm_path = os.path.join(save_directory, "hrm_modules.pt")
        torch.save(hrm_state, hrm_path)
        
        # Save base model if it has been modified
        # self.gemma.save_pretrained(save_directory)
        
        print(f"HRM model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[HRMConfig] = None):
        """Load pretrained HRM model."""
        import os
        import json
        
        # Load configuration if not provided
        if config is None:
            config_path = os.path.join(model_path, "hrm_config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = HRMConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load HRM modules
        hrm_path = os.path.join(model_path, "hrm_modules.pt")
        hrm_state = torch.load(hrm_path, map_location='cpu')
        
        model.low_module.load_state_dict(hrm_state['low_module'])
        model.high_module.load_state_dict(hrm_state['high_module'])
        model.state_initializer.load_state_dict(hrm_state['state_initializer'])
        model.output_projection.load_state_dict(hrm_state['output_projection'])
        
        if config.enable_act and 'q_head' in hrm_state:
            model.q_head.load_state_dict(hrm_state['q_head'])
        
        return model