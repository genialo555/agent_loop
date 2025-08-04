"""
Deep Supervision implementation for HRM.

This module implements deep supervision training where the model
is supervised at multiple segments, inspired by neural oscillations
that regulate learning in the brain.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .hrm_config import HRMConfig
from .approximate_gradient import HRMGradientApproximation


@dataclass
class SegmentOutput:
    """Output from a single supervision segment."""
    segment_id: int
    loss: torch.Tensor
    logits: torch.Tensor
    hidden_states: Dict[str, torch.Tensor]
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DeepSupervisionOutput:
    """Combined output from deep supervision training."""
    total_loss: torch.Tensor
    segment_outputs: List[SegmentOutput]
    avg_metrics: Dict[str, float]
    convergence_info: Dict[str, Any]


class DeepSupervisionTrainer:
    """
    Implements deep supervision training for HRM.
    
    Key features:
    - Multiple supervision segments per sample
    - State detachment between segments
    - Gradient accumulation across segments
    - Adaptive segment control
    """
    
    def __init__(
        self,
        model: HRMGradientApproximation,
        config: HRMConfig,
        optimizer: Optimizer,
        loss_fn: Optional[Callable] = None
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        
        # Deep supervision parameters
        self.num_segments = config.deep_supervision_segments
        self.supervision_weight = config.supervision_loss_weight
        
        # State management
        self.current_states = None
        self.segment_history = []
        
        # Statistics
        self.training_stats = {
            'total_segments': 0,
            'avg_segment_loss': 0.0,
            'convergence_improvements': 0,
            'early_stops': 0
        }
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> DeepSupervisionOutput:
        """
        Execute one deep supervision training step.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target labels [batch, seq_len]
            attention_mask: Optional attention mask
            
        Returns:
            DeepSupervisionOutput with losses and metrics
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize states for first segment
        if self.current_states is None:
            x_emb = self.model.model.embed_inputs(input_ids)
            z_l, z_h = self.model.model.state_initializer(x_emb)
            self.current_states = {'z_l': z_l, 'z_h': z_h}
        
        segment_outputs = []
        total_loss = 0.0
        
        # Run M supervision segments
        for segment in range(self.num_segments):
            # Get current states (detached from previous segment)
            z_l = self.current_states['z_l'].detach()
            z_h = self.current_states['z_h'].detach()
            
            # Forward pass for this segment
            segment_output = self._run_segment(
                input_ids, labels, z_l, z_h, 
                segment_id=segment,
                attention_mask=attention_mask
            )
            
            segment_outputs.append(segment_output)
            total_loss = total_loss + segment_output.loss * self.supervision_weight
            
            # Update states for next segment
            self.current_states = segment_output.hidden_states
            
            # Gradient step after each segment
            self._gradient_step(segment_output.loss)
            
            # Early stopping based on convergence
            if self._should_early_stop(segment_outputs):
                self.training_stats['early_stops'] += 1
                break
        
        # Compute average metrics
        avg_metrics = self._compute_average_metrics(segment_outputs)
        
        # Convergence analysis
        convergence_info = self._analyze_convergence(segment_outputs)
        
        # Update statistics
        self._update_statistics(segment_outputs)
        
        return DeepSupervisionOutput(
            total_loss=total_loss / len(segment_outputs),
            segment_outputs=segment_outputs,
            avg_metrics=avg_metrics,
            convergence_info=convergence_info
        )
    
    def _run_segment(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        segment_id: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> SegmentOutput:
        """Run a single supervision segment."""
        # Prepare states
        states = {
            'z_l_init': z_l,
            'z_h_init': z_h,
            'x_emb': self.model.model.embed_inputs(input_ids)
        }
        
        # Forward pass with HRM
        if self.config.use_approximate_gradient:
            # Use approximate gradient for this segment
            forward_states = self.model.approximator.compute_hrm_forward_approx(
                self.model.model,
                input_ids,
                self.config.num_high_cycles,
                self.config.timesteps_per_cycle
            )
            
            # Update states with initial values
            forward_states['z_l_init'] = z_l
            forward_states['z_h_init'] = z_h
            
            # Get logits
            logits = self.model.model.output_projection(forward_states['z_h_final'])
            
            # Compute loss
            loss = self._compute_loss(logits, labels)
        else:
            # Standard forward pass
            output = self.model.model(
                input_ids,
                labels=labels,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = output.logits
            loss = output.loss
            forward_states = {
                'z_l_final': output.low_level_states[-1],
                'z_h_final': output.high_level_states[-1]
            }
        
        # Compute segment metrics
        metrics = self._compute_segment_metrics(logits, labels, loss)
        
        return SegmentOutput(
            segment_id=segment_id,
            loss=loss,
            logits=logits,
            hidden_states=forward_states,
            metrics=metrics
        )
    
    def _gradient_step(self, loss: torch.Tensor):
        """Perform gradient step for a segment."""
        # Scale loss by gradient accumulation if needed
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if hasattr(self.config, 'max_grad_norm') and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
        
        # Optimizer step (could be accumulated)
        # Note: In practice, you might want to accumulate across segments
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss for language modeling."""
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def _compute_segment_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for a segment."""
        with torch.no_grad():
            # Perplexity
            perplexity = torch.exp(loss).item()
            
            # Accuracy
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            predictions = shift_logits.argmax(dim=-1)
            accuracy = (predictions == shift_labels).float().mean().item()
            
            # Token-level metrics
            metrics = {
                'loss': loss.item(),
                'perplexity': perplexity,
                'accuracy': accuracy,
                'grad_norm': self._get_gradient_norm()
            }
        
        return metrics
    
    def _get_gradient_norm(self) -> float:
        """Compute current gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _should_early_stop(
        self,
        segment_outputs: List[SegmentOutput]
    ) -> bool:
        """Determine if training should stop early."""
        if len(segment_outputs) < 2:
            return False
        
        # Check if loss is not improving
        recent_losses = [s.metrics['loss'] for s in segment_outputs[-2:]]
        if recent_losses[-1] > recent_losses[-2] * 1.1:  # 10% worse
            return True
        
        # Check if accuracy is very high
        recent_acc = segment_outputs[-1].metrics.get('accuracy', 0)
        if recent_acc > 0.99:
            return True
        
        return False
    
    def _compute_average_metrics(
        self,
        segment_outputs: List[SegmentOutput]
    ) -> Dict[str, float]:
        """Compute average metrics across segments."""
        if not segment_outputs:
            return {}
        
        # Collect all metrics
        all_metrics = {}
        for output in segment_outputs:
            for key, value in output.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Compute averages
        avg_metrics = {}
        for key, values in all_metrics.items():
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
        
        return avg_metrics
    
    def _analyze_convergence(
        self,
        segment_outputs: List[SegmentOutput]
    ) -> Dict[str, Any]:
        """Analyze convergence across segments."""
        if len(segment_outputs) < 2:
            return {'converged': False, 'improvement': 0.0}
        
        # Loss improvement
        first_loss = segment_outputs[0].metrics['loss']
        last_loss = segment_outputs[-1].metrics['loss']
        improvement = (first_loss - last_loss) / first_loss
        
        # Check if improving
        losses = [s.metrics['loss'] for s in segment_outputs]
        is_improving = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
        
        return {
            'converged': is_improving and improvement > 0.01,
            'improvement': improvement,
            'final_loss': last_loss,
            'loss_trajectory': losses
        }
    
    def _update_statistics(self, segment_outputs: List[SegmentOutput]):
        """Update training statistics."""
        self.training_stats['total_segments'] += len(segment_outputs)
        
        # Update average loss
        avg_loss = np.mean([s.metrics['loss'] for s in segment_outputs])
        alpha = 0.1  # Exponential moving average
        self.training_stats['avg_segment_loss'] = (
            (1 - alpha) * self.training_stats['avg_segment_loss'] +
            alpha * avg_loss
        )
        
        # Check for improvements
        convergence_info = self._analyze_convergence(segment_outputs)
        if convergence_info['converged']:
            self.training_stats['convergence_improvements'] += 1
    
    def get_training_summary(self) -> str:
        """Get summary of deep supervision training."""
        return f"""
Deep Supervision Training Summary:
- Total segments trained: {self.training_stats['total_segments']}
- Average segment loss: {self.training_stats['avg_segment_loss']:.4f}
- Convergence improvements: {self.training_stats['convergence_improvements']}
- Early stops: {self.training_stats['early_stops']}
- Segments per sample: {self.num_segments}
"""


class DeepSupervisionScheduler:
    """
    Scheduler for adapting deep supervision parameters during training.
    """
    
    def __init__(
        self,
        initial_segments: int = 3,
        min_segments: int = 1,
        max_segments: int = 8,
        adaptation_rate: float = 0.1
    ):
        self.current_segments = initial_segments
        self.min_segments = min_segments
        self.max_segments = max_segments
        self.adaptation_rate = adaptation_rate
        
        # History for adaptation
        self.improvement_history = []
    
    def update(self, convergence_info: Dict[str, Any]):
        """Update number of segments based on convergence."""
        improvement = convergence_info.get('improvement', 0.0)
        self.improvement_history.append(improvement)
        
        # Only adapt after sufficient history
        if len(self.improvement_history) < 10:
            return
        
        # Calculate recent average improvement
        recent_improvement = np.mean(self.improvement_history[-10:])
        
        # Adapt segments
        if recent_improvement < 0.01:  # Poor improvement
            # Increase segments for more supervision
            self.current_segments = min(
                self.max_segments,
                self.current_segments + 1
            )
        elif recent_improvement > 0.05:  # Good improvement
            # Decrease segments for efficiency
            self.current_segments = max(
                self.min_segments,
                self.current_segments - 1
            )
    
    def get_num_segments(self) -> int:
        """Get current number of segments."""
        return self.current_segments


def create_deep_supervision_trainer(
    model,
    config: HRMConfig,
    optimizer: Optimizer
) -> DeepSupervisionTrainer:
    """Factory function to create deep supervision trainer."""
    # Wrap model with gradient approximation if needed
    if config.use_approximate_gradient:
        model = HRMGradientApproximation(model, config)
    
    return DeepSupervisionTrainer(
        model=model,
        config=config,
        optimizer=optimizer
    )