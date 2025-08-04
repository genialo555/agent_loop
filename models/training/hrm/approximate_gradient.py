"""
Approximate Gradient implementation for HRM.

This module implements the 1-step gradient approximation that allows
O(1) memory complexity instead of O(T) for BPTT, making training
more efficient and biologically plausible.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable, Tuple
from contextlib import contextmanager
import warnings


class ApproximateGradient:
    """
    Implements 1-step gradient approximation for HRM training.
    
    Key features:
    - O(1) memory complexity
    - No need for BPTT
    - Based on Deep Equilibrium Models (DEQ) theory
    - Biologically plausible local learning
    """
    
    def __init__(
        self,
        enabled: bool = True,
        gradient_clip: float = 1.0,
        use_gradient_checkpointing: bool = True
    ):
        self.enabled = enabled
        self.gradient_clip = gradient_clip
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Statistics tracking
        self.stats = {
            'forward_passes': 0,
            'backward_passes': 0,
            'memory_saved_gb': 0.0,
            'avg_gradient_norm': 0.0
        }
    
    @contextmanager
    def approximate_gradient_context(self):
        """Context manager for approximate gradient computation."""
        if not self.enabled:
            yield
            return
        
        # Save current gradient computation state
        prev_grad_enabled = torch.is_grad_enabled()
        
        try:
            # Disable gradients for intermediate computations
            torch.set_grad_enabled(False)
            yield
        finally:
            # Restore gradient computation state
            torch.set_grad_enabled(prev_grad_enabled)
    
    def compute_hrm_forward_approx(
        self,
        model,
        input_ids: torch.Tensor,
        N: int,
        T: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute HRM forward pass with approximate gradient.
        
        Only the final step of each module computes gradients.
        All intermediate steps are computed with gradients disabled.
        
        Args:
            model: HRM model
            input_ids: Input token IDs
            N: Number of high-level cycles
            T: Timesteps per cycle
            
        Returns:
            Dictionary with final states and intermediate values
        """
        # Get input embeddings
        x_emb = model.embed_inputs(input_ids)
        
        # Initialize states
        z_l, z_h = model.state_initializer(x_emb)
        
        # Storage for final states (no intermediate storage needed!)
        states = {
            'x_emb': x_emb,
            'z_l_init': z_l.clone(),
            'z_h_init': z_h.clone()
        }
        
        # Forward pass with gradients disabled except for final step
        with self.approximate_gradient_context():
            # Run N*T-1 steps without gradients
            for i in range(N * T - 1):
                # Low-level update
                z_l_output = model.low_module(z_l, z_h, x_emb, iteration=i % T)
                z_l = z_l_output.hidden_state
                
                # High-level update at end of cycle
                if (i + 1) % T == 0:
                    z_h_output = model.high_module(z_h, z_l, cycle=i // T)
                    z_h = z_h_output.hidden_state
                    # Reset low-level
                    z_l = model.low_module.reset_with_context(z_h)
        
        # Now compute final step WITH gradients
        # This is the key to O(1) memory!
        torch.set_grad_enabled(True)
        
        # Final low-level step
        z_l_output = model.low_module(z_l, z_h, x_emb, iteration=(N*T-1) % T)
        z_l = z_l_output.hidden_state
        
        # Final high-level step
        z_h_output = model.high_module(z_h, z_l, cycle=N-1)
        z_h = z_h_output.hidden_state
        
        # Store final states
        states['z_l_final'] = z_l
        states['z_h_final'] = z_h
        
        # Update statistics
        self.stats['forward_passes'] += 1
        
        return states
    
    def compute_loss_and_backward(
        self,
        model,
        states: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Compute loss and backward pass with approximate gradient.
        
        Args:
            model: HRM model
            states: States from forward pass
            labels: Target labels
            loss_fn: Loss function (default: CrossEntropyLoss)
            
        Returns:
            Loss value
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # Get final high-level state
        z_h_final = states['z_h_final']
        
        # Project to vocabulary
        logits = model.output_projection(z_h_final)
        
        # Compute loss
        if labels is not None:
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Backward pass - gradients flow only through final states
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            self._clip_gradients(model)
        
        # Update statistics
        self.stats['backward_passes'] += 1
        self._estimate_memory_saved(states)
        
        return loss
    
    def _clip_gradients(self, model):
        """Clip gradients to prevent explosion."""
        total_norm = 0.0
        parameters = [p for p in model.parameters() if p.grad is not None]
        
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        self.stats['avg_gradient_norm'] = (
            0.9 * self.stats['avg_gradient_norm'] + 
            0.1 * total_norm
        )
        
        clip_coef = self.gradient_clip / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    
    def _estimate_memory_saved(self, states: Dict[str, torch.Tensor]):
        """Estimate memory saved compared to BPTT."""
        # Rough estimation: BPTT would store all intermediate states
        # We only store initial and final states
        
        # Get tensor sizes
        z_l_size = states['z_l_final'].element_size() * states['z_l_final'].nelement()
        z_h_size = states['z_h_final'].element_size() * states['z_h_final'].nelement()
        
        # BPTT would need N*T copies of states
        # We only need 2 (initial and final)
        N, T = 4, 8  # Default values, should be passed as parameters
        memory_bptt = (z_l_size + z_h_size) * N * T
        memory_approx = (z_l_size + z_h_size) * 2
        
        memory_saved_bytes = memory_bptt - memory_approx
        memory_saved_gb = memory_saved_bytes / (1024**3)
        
        self.stats['memory_saved_gb'] += memory_saved_gb
    
    def get_stats_summary(self) -> str:
        """Get summary of gradient approximation statistics."""
        return f"""
Approximate Gradient Statistics:
- Forward passes: {self.stats['forward_passes']}
- Backward passes: {self.stats['backward_passes']}
- Memory saved: {self.stats['memory_saved_gb']:.2f} GB
- Average gradient norm: {self.stats['avg_gradient_norm']:.4f}
- Gradient clipping: {self.gradient_clip}
"""


class HRMGradientApproximation(nn.Module):
    """
    Wrapper module that applies approximate gradient to any HRM model.
    """
    
    def __init__(
        self,
        hrm_model,
        config,
        enable_approximation: bool = True
    ):
        super().__init__()
        self.model = hrm_model
        self.config = config
        self.approximator = ApproximateGradient(
            enabled=enable_approximation,
            gradient_clip=1.0,
            use_gradient_checkpointing=config.gradient_checkpointing
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_approximate: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with optional approximate gradient.
        
        Args:
            input_ids: Input token IDs
            labels: Target labels
            use_approximate: Whether to use approximate gradient
            
        Returns:
            Dictionary with loss and other outputs
        """
        if use_approximate and self.approximator.enabled:
            # Use approximate gradient
            states = self.approximator.compute_hrm_forward_approx(
                self.model,
                input_ids,
                self.config.num_high_cycles,
                self.config.timesteps_per_cycle
            )
            
            # Compute output
            z_h_final = states['z_h_final']
            logits = self.model.output_projection(z_h_final)
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                loss = self.approximator.compute_loss_and_backward(
                    self.model, states, labels
                )
            
            return {
                'loss': loss,
                'logits': logits,
                'states': states,
                'stats': self.approximator.stats
            }
        else:
            # Use standard forward pass
            output = self.model(input_ids, labels=labels, **kwargs)
            return {
                'loss': output.loss,
                'logits': output.logits,
                'states': {
                    'z_l_final': output.low_level_states[-1],
                    'z_h_final': output.high_level_states[-1]
                }
            }
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Single training step with approximate gradient.
        
        Compatible with PyTorch Lightning or custom training loops.
        """
        input_ids = batch['input_ids']
        labels = batch.get('labels', input_ids)  # Use input as label for language modeling
        
        # Forward pass with approximate gradient
        outputs = self.forward(input_ids, labels, use_approximate=True)
        
        return outputs['loss']
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'saved_gb': self.approximator.stats['memory_saved_gb']
            }
        return {}


def test_approximate_gradient():
    """Test approximate gradient implementation."""
    from .hrm_config import get_config_debug
    from .hrm_model import HRMGemma3N
    
    # Create debug config
    config = get_config_debug()
    config.use_approximate_gradient = True
    
    # Create model (this would need actual model loading)
    # model = HRMGemma3N(config)
    
    # Test gradient approximation
    # approx = ApproximateGradient()
    
    print("Approximate gradient test would run here with actual model")
    print("Key benefits:")
    print("- O(1) memory instead of O(N*T)")
    print("- No BPTT required")
    print("- Faster training on limited GPU memory")


if __name__ == "__main__":
    test_approximate_gradient()