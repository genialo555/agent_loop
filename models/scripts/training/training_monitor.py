#!/usr/bin/env python3
"""
Real-time Training Monitor for MLOps Pipeline

Monitors training logs in real-time with:
- Loss curves visualization
- Learning rate tracking
- Gradient norms monitoring
- Training speed (tokens/sec)
- Checkpoint management
- WebSocket integration for live updates
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Deque
from collections import deque
import threading
from dataclasses import dataclass, field

import httpx
import websockets
import structlog
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, push_to_gateway
import wandb

# Configure logging
logger = structlog.get_logger(__name__)
console = Console()

# Prometheus metrics
registry = CollectorRegistry()

# Training metrics
training_loss = Gauge('training_loss', 'Current training loss', registry=registry)
learning_rate = Gauge('training_learning_rate', 'Current learning rate', registry=registry)
gradient_norm = Gauge('training_gradient_norm', 'Current gradient norm', registry=registry)
training_speed = Gauge('training_tokens_per_second', 'Training speed in tokens/sec', registry=registry)
epoch_progress = Gauge('training_epoch_progress', 'Current epoch progress', registry=registry)
step_counter = Counter('training_steps_total', 'Total training steps', registry=registry)
checkpoint_counter = Counter('training_checkpoints_saved', 'Total checkpoints saved', registry=registry)

# Training performance histograms
step_duration = Histogram('training_step_duration_seconds', 'Training step duration', registry=registry)
loss_histogram = Histogram('training_loss_distribution', 'Distribution of training loss values', registry=registry)


@dataclass
class TrainingMetrics:
    """Container for training metrics with history."""
    loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    tokens_per_second: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_gb: float = 0.0
    
    # History for plotting
    loss_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    lr_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    grad_norm_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    speed_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    # Step tracking
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    
    # Timing
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    # Checkpoints
    last_checkpoint: Optional[str] = None
    checkpoints_saved: int = 0
    
    def update(self, metrics: Dict[str, Any]) -> None:
        """Update metrics from incoming data."""
        self.last_update = datetime.now(timezone.utc)
        
        if 'train_loss' in metrics:
            self.loss = metrics['train_loss']
            self.loss_history.append(self.loss)
            training_loss.set(self.loss)
            loss_histogram.observe(self.loss)
            
        if 'learning_rate' in metrics:
            self.learning_rate = metrics['learning_rate']
            self.lr_history.append(self.learning_rate)
            learning_rate.set(self.learning_rate)
            
        if 'gradient_norm' in metrics:
            self.gradient_norm = metrics['gradient_norm']
            self.grad_norm_history.append(self.gradient_norm)
            gradient_norm.set(self.gradient_norm)
            
        if 'tokens_per_second' in metrics:
            self.tokens_per_second = metrics['tokens_per_second']
            self.speed_history.append(self.tokens_per_second)
            training_speed.set(self.tokens_per_second)
            
        if 'gpu_utilization' in metrics:
            self.gpu_utilization = metrics['gpu_utilization']
            
        if 'gpu_memory_gb' in metrics:
            self.gpu_memory_gb = metrics['gpu_memory_gb']
    
    def get_sparkline(self, data: List[float], width: int = 20) -> str:
        """Generate ASCII sparkline chart."""
        if not data:
            return "─" * width
            
        chars = "▁▂▃▄▅▆▇█"
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return chars[4] * width
            
        # Sample data if too long
        if len(data) > width:
            indices = np.linspace(0, len(data) - 1, width, dtype=int)
            sampled = [data[i] for i in indices]
        else:
            sampled = list(data)
            
        sparkline = ""
        for val in sampled:
            normalized = (val - min_val) / (max_val - min_val)
            idx = int(normalized * (len(chars) - 1))
            sparkline += chars[idx]
            
        return sparkline


class TrainingMonitor:
    """Real-time training monitor with dashboard and alerting."""
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8000",
                 wandb_project: Optional[str] = None,
                 prometheus_gateway: Optional[str] = None):
        self.api_base_url = api_base_url
        self.wandb_project = wandb_project
        self.prometheus_gateway = prometheus_gateway
        
        self.metrics = TrainingMetrics()
        self.active_job_id: Optional[str] = None
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Alert thresholds
        self.loss_spike_threshold = 2.0  # Alert if loss increases by 2x
        self.min_speed_threshold = 100.0  # Alert if speed drops below 100 tokens/sec
        self.gradient_explosion_threshold = 10.0  # Alert if gradient norm > 10
        
        # Initialize wandb if configured
        if self.wandb_project:
            wandb.init(project=self.wandb_project, name="training-monitor")
    
    async def connect_to_job(self, job_id: str) -> None:
        """Connect to a training job via WebSocket."""
        self.active_job_id = job_id
        ws_url = f"ws://localhost:8000/ws/training/{job_id}"
        
        logger.info(f"Connecting to training job: {job_id}")
        
        try:
            self.websocket = await websockets.connect(ws_url)
            logger.info("WebSocket connection established")
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise
    
    async def monitor_logs(self) -> None:
        """Monitor training logs from WebSocket stream."""
        if not self.websocket:
            raise RuntimeError("Not connected to any training job")
            
        async for message in self.websocket:
            try:
                log_entry = json.loads(message)
                
                # Update metrics if present
                if 'metrics' in log_entry and log_entry['metrics']:
                    self.metrics.update(log_entry['metrics'])
                    
                    # Push to Prometheus if configured
                    if self.prometheus_gateway:
                        push_to_gateway(
                            self.prometheus_gateway, 
                            job=f'training_{self.active_job_id}',
                            registry=registry
                        )
                    
                    # Log to wandb if configured
                    if self.wandb_project:
                        wandb.log(log_entry['metrics'])
                
                # Update step/epoch info
                if 'step' in log_entry:
                    self.metrics.current_step = log_entry['step']
                    step_counter.inc()
                    
                if 'epoch' in log_entry:
                    self.metrics.current_epoch = log_entry['epoch']
                    
                # Check for checkpoint saves
                if 'checkpoint' in log_entry.get('message', '').lower():
                    self.metrics.checkpoints_saved += 1
                    checkpoint_counter.inc()
                    
                # Check alerts
                await self._check_alerts()
                
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in WebSocket message: {message}")
            except Exception as e:
                logger.error(f"Error processing log entry: {e}")
    
    async def _check_alerts(self) -> None:
        """Check for alert conditions."""
        # Loss spike detection
        if len(self.metrics.loss_history) > 10:
            recent_avg = np.mean(list(self.metrics.loss_history)[-10:-5])
            current_avg = np.mean(list(self.metrics.loss_history)[-5:])
            
            if current_avg > recent_avg * self.loss_spike_threshold:
                logger.warning(
                    f"Loss spike detected! Recent: {recent_avg:.4f}, Current: {current_avg:.4f}"
                )
        
        # Speed drop detection
        if self.metrics.tokens_per_second < self.min_speed_threshold:
            logger.warning(
                f"Training speed below threshold: {self.metrics.tokens_per_second:.1f} tokens/sec"
            )
        
        # Gradient explosion detection
        if self.metrics.gradient_norm > self.gradient_explosion_threshold:
            logger.error(
                f"Gradient explosion detected! Norm: {self.metrics.gradient_norm:.2f}"
            )
    
    def create_dashboard(self) -> Layout:
        """Create rich dashboard layout."""
        layout = Layout()
        
        # Header
        header = Panel(
            Text("🚀 Training Monitor Dashboard", justify="center", style="bold blue"),
            height=3
        )
        
        # Metrics table
        metrics_table = Table(title="Current Metrics", box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        metrics_table.add_column("History", style="yellow")
        
        # Add rows
        metrics_table.add_row(
            "Loss", 
            f"{self.metrics.loss:.6f}",
            self.metrics.get_sparkline(list(self.metrics.loss_history))
        )
        metrics_table.add_row(
            "Learning Rate", 
            f"{self.metrics.learning_rate:.2e}",
            self.metrics.get_sparkline(list(self.metrics.lr_history))
        )
        metrics_table.add_row(
            "Gradient Norm", 
            f"{self.metrics.gradient_norm:.4f}",
            self.metrics.get_sparkline(list(self.metrics.grad_norm_history))
        )
        metrics_table.add_row(
            "Speed (tokens/s)", 
            f"{self.metrics.tokens_per_second:.1f}",
            self.metrics.get_sparkline(list(self.metrics.speed_history))
        )
        
        # Progress info
        progress_text = f"Epoch {self.metrics.current_epoch}/{self.metrics.total_epochs} | "
        progress_text += f"Step {self.metrics.current_step}/{self.metrics.total_steps}"
        
        if self.metrics.total_steps > 0:
            progress_pct = (self.metrics.current_step / self.metrics.total_steps) * 100
            progress_text += f" ({progress_pct:.1f}%)"
        
        progress_panel = Panel(progress_text, title="Training Progress")
        
        # System info
        system_table = Table(title="System Info", box=None)
        system_table.add_column("Resource", style="cyan")
        system_table.add_column("Usage", style="green")
        
        system_table.add_row("GPU Utilization", f"{self.metrics.gpu_utilization:.1f}%")
        system_table.add_row("GPU Memory", f"{self.metrics.gpu_memory_gb:.2f} GB")
        system_table.add_row("Checkpoints Saved", str(self.metrics.checkpoints_saved))
        
        # Layout assembly
        layout.split_column(
            header,
            Layout(name="main"),
            Layout(name="footer", height=3)
        )
        
        layout["main"].split_row(
            Layout(Panel(metrics_table), name="metrics"),
            Layout(name="right")
        )
        
        layout["right"].split_column(
            Layout(progress_panel, height=5),
            Layout(Panel(system_table))
        )
        
        # Footer with timing info
        if self.metrics.last_update:
            elapsed = (datetime.now(timezone.utc) - self.metrics.start_time).total_seconds() if self.metrics.start_time else 0
            footer_text = f"Last Update: {self.metrics.last_update.strftime('%H:%M:%S')} | "
            footer_text += f"Elapsed: {int(elapsed // 3600):02d}:{int((elapsed % 3600) // 60):02d}:{int(elapsed % 60):02d}"
        else:
            footer_text = "Waiting for data..."
            
        layout["footer"] = Panel(footer_text, style="dim")
        
        return layout
    
    async def run_dashboard(self) -> None:
        """Run interactive dashboard with live updates."""
        with Live(self.create_dashboard(), refresh_per_second=2) as live:
            while True:
                live.update(self.create_dashboard())
                await asyncio.sleep(0.5)
    
    async def monitor_job(self, job_id: str) -> None:
        """Monitor a specific training job."""
        # Get initial job status
        response = await self.http_client.get(f"{self.api_base_url}/training/{job_id}/status")
        if response.status_code != 200:
            raise ValueError(f"Job {job_id} not found")
            
        job_status = response.json()
        logger.info(f"Monitoring job: {job_id}, Status: {job_status['status']}")
        
        # Set initial metrics
        if job_status.get('progress'):
            self.metrics.current_epoch = job_status['progress'].get('current_epoch', 0)
            self.metrics.total_epochs = job_status['progress'].get('total_epochs', 0)
            self.metrics.current_step = job_status['progress'].get('current_step', 0)
            self.metrics.total_steps = job_status['progress'].get('total_steps', 0)
        
        self.metrics.start_time = datetime.now(timezone.utc)
        
        # Connect to WebSocket
        await self.connect_to_job(job_id)
        
        # Run monitoring and dashboard in parallel
        await asyncio.gather(
            self.monitor_logs(),
            self.run_dashboard()
        )
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.websocket:
            await self.websocket.close()
        await self.http_client.aclose()
        
        if self.wandb_project:
            wandb.finish()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Training Monitor")
    parser.add_argument("job_id", help="Training job ID to monitor")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--wandb-project", help="W&B project name")
    parser.add_argument("--prometheus-gateway", help="Prometheus pushgateway URL")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(
        api_base_url=args.api_url,
        wandb_project=args.wandb_project,
        prometheus_gateway=args.prometheus_gateway
    )
    
    try:
        await monitor.monitor_job(args.job_id)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    finally:
        await monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())