"""
Adaptive Computation Time (ACT) implementation for HRM.

This module implements ACT with Q-learning to dynamically adjust
computational resources based on task complexity, inspired by the
brain's ability to switch between fast and slow thinking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import numpy as np
from collections import deque
import random


@dataclass
class ACTDecision:
    """Decision made by ACT module."""
    action: str  # 'halt' or 'continue'
    q_values: torch.Tensor
    confidence: float
    segment: int
    reasoning_depth: int


class QHead(nn.Module):
    """
    Q-value head for halt/continue decisions.
    
    Predicts Q-values for actions based on high-level state.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Q-value prediction network
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)  # [Q_halt, Q_continue]
        )
        
        # Value baseline for variance reduction
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for module in [self.q_network, self.value_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        z_h: torch.Tensor,
        return_value: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to compute Q-values.
        
        Args:
            z_h: High-level state [batch, seq_len, hidden_dim]
            return_value: Whether to return value estimate
            
        Returns:
            Q-values [batch, 2] and optionally value estimate
        """
        # Pool over sequence dimension
        if z_h.dim() == 3:
            z_h_pooled = z_h.mean(dim=1)  # [batch, hidden_dim]
        else:
            z_h_pooled = z_h
        
        # Compute Q-values
        q_values = self.q_network(z_h_pooled)
        q_values = torch.sigmoid(q_values)  # Bound between 0 and 1
        
        # Compute value if requested
        value = None
        if return_value:
            value = self.value_network(z_h_pooled)
        
        return q_values, value


class AdaptiveComputationTime:
    """
    Manages adaptive computation time for HRM using Q-learning.
    
    Features:
    - Dynamic halt/continue decisions
    - Q-learning for policy improvement
    - Exploration vs exploitation balance
    - Computational budget management
    """
    
    def __init__(
        self,
        config,
        q_head: QHead
    ):
        self.config = config
        self.q_head = q_head
        
        # ACT parameters
        self.max_segments = config.act_max_segments
        self.epsilon = config.act_epsilon
        self.q_learning_rate = config.q_learning_rate
        self.discount_factor = config.q_discount_factor
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            'avg_segments_used': 0.0,
            'halt_decisions': 0,
            'continue_decisions': 0,
            'total_decisions': 0,
            'avg_confidence': 0.0,
            'computation_saved': 0.0
        }
        
        # Running averages
        self.segment_history = deque(maxlen=100)
    
    def should_halt(
        self,
        z_h: torch.Tensor,
        segment: int,
        min_segments: Optional[int] = None,
        training: bool = True
    ) -> ACTDecision:
        """
        Decide whether to halt or continue computation.
        
        Args:
            z_h: Current high-level state
            segment: Current segment number
            min_segments: Minimum segments to compute
            training: Whether in training mode
            
        Returns:
            ACTDecision with action and metadata
        """
        # Determine minimum segments
        if min_segments is None:
            if training and random.random() < self.epsilon:
                # Exploration: random minimum segments
                min_segments = random.randint(2, self.max_segments)
            else:
                min_segments = 1
        
        # Force halt at maximum segments
        if segment >= self.max_segments:
            return ACTDecision(
                action='halt',
                q_values=torch.tensor([1.0, 0.0]),
                confidence=1.0,
                segment=segment,
                reasoning_depth=segment
            )
        
        # Get Q-values
        with torch.no_grad():
            q_values, _ = self.q_head(z_h)
            q_halt = q_values[0, 0].item()
            q_continue = q_values[0, 1].item()
        
        # Make decision
        if segment >= min_segments and q_halt > q_continue:
            action = 'halt'
        else:
            action = 'continue'
        
        # Compute confidence
        confidence = abs(q_halt - q_continue) / (q_halt + q_continue + 1e-8)
        
        # Create decision
        decision = ACTDecision(
            action=action,
            q_values=q_values[0],
            confidence=confidence,
            segment=segment,
            reasoning_depth=segment if action == 'halt' else segment + 1
        )
        
        # Update statistics
        self._update_stats(decision)
        
        return decision
    
    def update_q_values(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: Optional[torch.Tensor] = None,
        done: bool = True
    ):
        """
        Update Q-values using Q-learning.
        
        Args:
            state: State when decision was made
            action: Action taken (0=halt, 1=continue)
            reward: Reward received
            next_state: Next state (if any)
            done: Whether episode ended
        """
        # Store experience
        self.replay_buffer.append((state, action, reward, next_state, done))
        
        # Sample batch from replay buffer
        if len(self.replay_buffer) < 32:
            return
        
        batch = random.sample(self.replay_buffer, 32)
        
        # Prepare batch tensors
        states = torch.stack([e[0] for e in batch])
        actions = torch.tensor([e[1] for e in batch])
        rewards = torch.tensor([e[2] for e in batch])
        dones = torch.tensor([e[4] for e in batch])
        
        # Current Q-values
        current_q, _ = self.q_head(states)
        current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Target Q-values
        target_q_values = rewards.clone()
        
        # Add future rewards for non-terminal states
        non_terminal_mask = ~dones
        if non_terminal_mask.any():
            next_states = torch.stack([
                e[3] for e in batch if e[3] is not None
            ])
            with torch.no_grad():
                next_q, _ = self.q_head(next_states)
                max_next_q = next_q.max(dim=1)[0]
                target_q_values[non_terminal_mask] += (
                    self.discount_factor * max_next_q
                )
        
        # Compute Q-learning loss
        q_loss = F.mse_loss(current_q_values, target_q_values)
        
        # Note: In practice, this loss would be added to the main training loss
        # For now, we just compute it for monitoring
        
        return q_loss.item()
    
    def compute_reward(
        self,
        task_success: bool,
        segments_used: int,
        accuracy: float
    ) -> float:
        """
        Compute reward for Q-learning.
        
        Balances task success with computational efficiency.
        """
        # Base reward for task success
        if task_success:
            base_reward = 1.0
        else:
            base_reward = -0.5
        
        # Efficiency bonus (fewer segments is better)
        efficiency_bonus = (self.max_segments - segments_used) / self.max_segments
        efficiency_bonus *= 0.2  # Scale down
        
        # Accuracy component
        accuracy_bonus = accuracy * 0.3
        
        # Total reward
        reward = base_reward + efficiency_bonus + accuracy_bonus
        
        return reward
    
    def _update_stats(self, decision: ACTDecision):
        """Update running statistics."""
        self.stats['total_decisions'] += 1
        
        if decision.action == 'halt':
            self.stats['halt_decisions'] += 1
            self.segment_history.append(decision.segment)
        else:
            self.stats['continue_decisions'] += 1
        
        # Update running averages
        alpha = 0.1
        self.stats['avg_confidence'] = (
            (1 - alpha) * self.stats['avg_confidence'] +
            alpha * decision.confidence
        )
        
        if self.segment_history:
            self.stats['avg_segments_used'] = np.mean(self.segment_history)
            
            # Compute computation saved
            avg_used = self.stats['avg_segments_used']
            self.stats['computation_saved'] = (
                (self.max_segments - avg_used) / self.max_segments * 100
            )
    
    def get_stats_summary(self) -> str:
        """Get summary of ACT statistics."""
        total = self.stats['total_decisions']
        if total == 0:
            return "No ACT decisions made yet."
        
        halt_rate = self.stats['halt_decisions'] / total * 100
        
        return f"""
Adaptive Computation Time Statistics:
- Total decisions: {total}
- Halt rate: {halt_rate:.1f}%
- Average segments used: {self.stats['avg_segments_used']:.2f} / {self.max_segments}
- Computation saved: {self.stats['computation_saved']:.1f}%
- Average confidence: {self.stats['avg_confidence']:.3f}
"""


class ACTWrapper(nn.Module):
    """
    Wrapper that adds ACT to any HRM model.
    """
    
    def __init__(
        self,
        hrm_model,
        config
    ):
        super().__init__()
        self.model = hrm_model
        self.config = config
        
        # Create Q-head
        self.q_head = QHead(
            hidden_dim=config.high_level_config.hidden_dim,
            dropout=config.high_level_config.lora_dropout
        )
        
        # ACT manager
        self.act_manager = AdaptiveComputationTime(config, self.q_head)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_act: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with adaptive computation.
        
        Args:
            input_ids: Input token IDs
            labels: Target labels
            use_act: Whether to use ACT
            
        Returns:
            Model output with ACT information
        """
        if not use_act or not self.config.enable_act:
            # Standard forward without ACT
            return self.model(input_ids, labels=labels, **kwargs)
        
        # Initialize
        x_emb = self.model.embed_inputs(input_ids)
        z_l, z_h = self.model.state_initializer(x_emb)
        
        # Storage
        segment_outputs = []
        total_loss = 0.0
        
        # Adaptive computation loop
        for segment in range(self.act_manager.max_segments):
            # Run one HRM forward pass
            output = self.model(
                input_ids,
                labels=labels,
                num_cycles=self.config.num_high_cycles,
                **kwargs
            )
            
            segment_outputs.append(output)
            
            # Check if should halt
            decision = self.act_manager.should_halt(
                output.high_level_states[-1],
                segment,
                training=self.training
            )
            
            if decision.action == 'halt':
                break
        
        # Combine outputs
        final_output = {
            'logits': segment_outputs[-1].logits,
            'loss': segment_outputs[-1].loss if labels is not None else None,
            'segments_used': len(segment_outputs),
            'act_decisions': decision,
            'computation_saved': (
                (self.act_manager.max_segments - len(segment_outputs)) /
                self.act_manager.max_segments * 100
            )
        }
        
        # Update Q-values if training
        if self.training and labels is not None:
            # Compute accuracy for reward
            with torch.no_grad():
                predictions = final_output['logits'].argmax(dim=-1)
                accuracy = (predictions == labels).float().mean().item()
            
            # Compute reward
            reward = self.act_manager.compute_reward(
                task_success=accuracy > 0.9,
                segments_used=len(segment_outputs),
                accuracy=accuracy
            )
            
            # Update Q-values (simplified - in practice, this would be done properly)
            self.act_manager.update_q_values(
                state=output.high_level_states[-1],
                action=0 if decision.action == 'halt' else 1,
                reward=reward
            )
        
        return final_output
    
    def get_act_summary(self) -> str:
        """Get ACT statistics summary."""
        return self.act_manager.get_stats_summary()


def create_act_model(hrm_model, config):
    """Factory function to create ACT-enabled model."""
    if config.enable_act:
        return ACTWrapper(hrm_model, config)
    return hrm_model