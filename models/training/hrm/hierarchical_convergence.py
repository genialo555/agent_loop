"""
Hierarchical Convergence mechanism for HRM.

This module implements the core hierarchical convergence process where:
- Low-level module converges rapidly within each cycle
- High-level module updates only after low-level convergence
- Low-level module resets with new high-level context
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field
import numpy as np

from .hrm_modules import (
    LowLevelModule,
    HighLevelModule,
    ModuleOutput
)


@dataclass
class ConvergenceMetrics:
    """Metrics for tracking convergence behavior."""
    cycle: int
    low_level_iterations: int
    low_level_converged: bool
    low_level_final_metric: float
    low_level_metrics_history: List[float] = field(default_factory=list)
    high_level_metric: float = 0.0
    time_elapsed: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'cycle': self.cycle,
            'low_iterations': self.low_level_iterations,
            'low_converged': self.low_level_converged,
            'low_final_metric': self.low_level_final_metric,
            'low_avg_metric': np.mean(self.low_level_metrics_history) if self.low_level_metrics_history else 0.0,
            'high_metric': self.high_level_metric,
            'time_elapsed': self.time_elapsed
        }


class HierarchicalConvergenceManager:
    """
    Manages the hierarchical convergence process between low and high level modules.
    
    Key responsibilities:
    1. Run low-level module to convergence
    2. Update high-level module with converged low-level state
    3. Reset low-level module with new high-level context
    4. Track convergence metrics and patterns
    """
    
    def __init__(
        self,
        num_high_cycles: int = 4,
        timesteps_per_cycle: int = 8,
        convergence_threshold: float = 1e-3,
        early_stopping: bool = True,
        track_metrics: bool = True
    ):
        self.num_high_cycles = num_high_cycles
        self.timesteps_per_cycle = timesteps_per_cycle
        self.convergence_threshold = convergence_threshold
        self.early_stopping = early_stopping
        self.track_metrics = track_metrics
        
        # Metrics tracking
        self.metrics_history: List[ConvergenceMetrics] = []
        self.global_step = 0
        
    def run_low_level_to_convergence(
        self,
        low_module: LowLevelModule,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        x_emb: torch.Tensor,
        max_iterations: Optional[int] = None
    ) -> Tuple[torch.Tensor, ConvergenceMetrics]:
        """
        Run low-level module until convergence or max iterations.
        
        Args:
            low_module: Low-level module
            z_l: Initial low-level state
            z_h: High-level state (fixed during cycle)
            x_emb: Input embeddings
            max_iterations: Maximum iterations (default: timesteps_per_cycle)
            
        Returns:
            Final low-level state and convergence metrics
        """
        max_iterations = max_iterations or self.timesteps_per_cycle
        metrics = ConvergenceMetrics(
            cycle=self.global_step,
            low_level_iterations=0,
            low_level_converged=False,
            low_level_final_metric=float('inf')
        )
        
        # Track computation time
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        # Low-level convergence loop
        for t in range(max_iterations):
            # Low-level update
            output: ModuleOutput = low_module(z_l, z_h, x_emb, iteration=t)
            z_l = output.hidden_state
            
            # Track metrics
            metrics.low_level_iterations = t + 1
            metrics.low_level_metrics_history.append(output.convergence_metric)
            
            # Check convergence
            if output.converged and self.early_stopping:
                metrics.low_level_converged = True
                metrics.low_level_final_metric = output.convergence_metric
                break
        
        # Final metric
        metrics.low_level_final_metric = output.convergence_metric
        
        # Record time
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            metrics.time_elapsed = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        return z_l, metrics
    
    def execute_cycle(
        self,
        low_module: LowLevelModule,
        high_module: HighLevelModule,
        z_l: torch.Tensor,
        z_h: torch.Tensor,
        x_emb: torch.Tensor,
        cycle: int
    ) -> Tuple[torch.Tensor, torch.Tensor, ConvergenceMetrics]:
        """
        Execute one complete hierarchical convergence cycle.
        
        Steps:
        1. Run low-level module to convergence
        2. Update high-level module with converged state
        3. Reset low-level module with new context
        
        Returns:
            Updated (z_l, z_h) states and convergence metrics
        """
        # Step 1: Low-level convergence
        z_l_converged, metrics = self.run_low_level_to_convergence(
            low_module, z_l, z_h, x_emb
        )
        
        # Step 2: High-level update
        high_output: ModuleOutput = high_module(z_h, z_l_converged, cycle=cycle)
        z_h_new = high_output.hidden_state
        metrics.high_level_metric = high_output.convergence_metric
        
        # Step 3: Reset low-level with new context
        z_l_new = low_module.reset_with_context(z_h_new)
        
        # Update global step
        self.global_step += 1
        
        # Track metrics
        if self.track_metrics:
            self.metrics_history.append(metrics)
        
        return z_l_new, z_h_new, metrics
    
    def run_full_hrm_forward(
        self,
        low_module: LowLevelModule,
        high_module: HighLevelModule,
        x_emb: torch.Tensor,
        z_l_init: torch.Tensor,
        z_h_init: torch.Tensor,
        num_cycles: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[ConvergenceMetrics]]:
        """
        Run complete HRM forward pass with N high-level cycles.
        
        Args:
            low_module: Low-level module
            high_module: High-level module
            x_emb: Input embeddings
            z_l_init: Initial low-level state
            z_h_init: Initial high-level state
            num_cycles: Number of high-level cycles (default: num_high_cycles)
            
        Returns:
            Final states and list of metrics for each cycle
        """
        num_cycles = num_cycles or self.num_high_cycles
        z_l, z_h = z_l_init, z_h_init
        cycle_metrics = []
        
        for cycle in range(num_cycles):
            z_l, z_h, metrics = self.execute_cycle(
                low_module, high_module,
                z_l, z_h, x_emb,
                cycle
            )
            cycle_metrics.append(metrics)
        
        return z_l, z_h, cycle_metrics
    
    def analyze_convergence_patterns(self) -> Dict[str, Any]:
        """
        Analyze convergence patterns from metrics history.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.metrics_history:
            return {}
        
        # Extract data
        low_iterations = [m.low_level_iterations for m in self.metrics_history]
        low_converged = [m.low_level_converged for m in self.metrics_history]
        low_final_metrics = [m.low_level_final_metric for m in self.metrics_history]
        high_metrics = [m.high_level_metric for m in self.metrics_history]
        
        analysis = {
            'avg_low_iterations': np.mean(low_iterations),
            'std_low_iterations': np.std(low_iterations),
            'convergence_rate': sum(low_converged) / len(low_converged),
            'avg_low_final_metric': np.mean(low_final_metrics),
            'avg_high_metric': np.mean(high_metrics),
            'total_cycles': len(self.metrics_history),
            'metrics_trend': {
                'low_improving': low_final_metrics[-10:] < low_final_metrics[:10] if len(low_final_metrics) > 20 else None,
                'high_stable': np.std(high_metrics[-10:]) < np.std(high_metrics[:10]) if len(high_metrics) > 20 else None
            }
        }
        
        return analysis
    
    def get_convergence_summary(self) -> str:
        """Get human-readable convergence summary."""
        analysis = self.analyze_convergence_patterns()
        
        if not analysis:
            return "No convergence data available yet."
        
        summary = f"""
Hierarchical Convergence Summary:
- Total cycles completed: {analysis['total_cycles']}
- Low-level convergence:
  - Average iterations: {analysis['avg_low_iterations']:.1f} ± {analysis['std_low_iterations']:.1f}
  - Convergence rate: {analysis['convergence_rate']*100:.1f}%
  - Average final metric: {analysis['avg_low_final_metric']:.4f}
- High-level stability:
  - Average metric: {analysis['avg_high_metric']:.4f}
"""
        return summary
    
    def reset(self):
        """Reset metrics history."""
        self.metrics_history = []
        self.global_step = 0


class AdaptiveConvergence(HierarchicalConvergenceManager):
    """
    Adaptive version of hierarchical convergence that adjusts parameters
    based on observed convergence patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Adaptive parameters
        self.adaptive_timesteps = True
        self.min_timesteps = 4
        self.max_timesteps = 16
        self.adaptation_rate = 0.1
        
        # Running statistics
        self.running_avg_iterations = self.timesteps_per_cycle
        
    def adapt_timesteps(self, recent_metrics: List[ConvergenceMetrics]):
        """Adapt timesteps based on recent convergence behavior."""
        if not self.adaptive_timesteps or len(recent_metrics) < 5:
            return
        
        # Calculate recent average iterations
        recent_iterations = [m.low_level_iterations for m in recent_metrics[-5:]]
        avg_recent = np.mean(recent_iterations)
        
        # Update running average
        self.running_avg_iterations = (
            (1 - self.adaptation_rate) * self.running_avg_iterations +
            self.adaptation_rate * avg_recent
        )
        
        # Adjust timesteps_per_cycle
        if self.running_avg_iterations < self.timesteps_per_cycle * 0.7:
            # Converging faster than expected, reduce timesteps
            self.timesteps_per_cycle = max(
                self.min_timesteps,
                int(self.timesteps_per_cycle * 0.9)
            )
        elif self.running_avg_iterations > self.timesteps_per_cycle * 0.95:
            # Not converging fast enough, increase timesteps
            self.timesteps_per_cycle = min(
                self.max_timesteps,
                int(self.timesteps_per_cycle * 1.1)
            )
    
    def execute_cycle(self, *args, **kwargs):
        """Execute cycle with adaptive timesteps."""
        # Run parent method
        z_l, z_h, metrics = super().execute_cycle(*args, **kwargs)
        
        # Adapt parameters based on recent history
        if self.track_metrics and len(self.metrics_history) > 0:
            self.adapt_timesteps(self.metrics_history)
        
        return z_l, z_h, metrics


def create_convergence_manager(config, adaptive: bool = False):
    """Factory function to create convergence manager."""
    if adaptive:
        return AdaptiveConvergence(
            num_high_cycles=config.num_high_cycles,
            timesteps_per_cycle=config.timesteps_per_cycle,
            convergence_threshold=config.low_level_config.convergence_threshold,
            early_stopping=True,
            track_metrics=True
        )
    else:
        return HierarchicalConvergenceManager(
            num_high_cycles=config.num_high_cycles,
            timesteps_per_cycle=config.timesteps_per_cycle,
            convergence_threshold=config.low_level_config.convergence_threshold,
            early_stopping=True,
            track_metrics=True
        )