#!/usr/bin/env python3
"""
Training Monitor Module
Beautiful and reusable training progress monitoring with rich terminal UI.
"""

import time
from typing import Dict, Any, Optional, List
import psutil
import GPUtil
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from transformers import TrainerCallback


class RichProgressCallback(TrainerCallback):
    """
    Beautiful training progress callback with system metrics.
    
    Features:
    - Real-time progress percentage
    - GPU utilization and memory monitoring
    - CPU and RAM usage tracking
    - Loss tracking with moving average
    - ETA calculation
    - Temperature and power monitoring
    - Beautiful rich terminal UI
    """
    
    def __init__(
        self, 
        total_steps: int,
        model_name: str = "Model",
        show_gpu: bool = True,
        show_cpu: bool = True,
        update_frequency: int = 1,
        theme: str = "default"
    ):
        """
        Initialize the progress callback.
        
        Args:
            total_steps: Total number of training steps
            model_name: Name of the model being trained
            show_gpu: Whether to show GPU metrics
            show_cpu: Whether to show CPU metrics
            update_frequency: Update display every N steps
            theme: Color theme ('default', 'minimal', 'cyberpunk')
        """
        self.total_steps = total_steps
        self.model_name = model_name
        self.show_gpu = show_gpu
        self.show_cpu = show_cpu
        self.update_frequency = update_frequency
        self.theme = theme
        
        self.current_step = 0
        self.start_time = time.time()
        self.console = Console()
        self.last_loss = 0.0
        self.losses: List[float] = []
        self.learning_rates: List[float] = []
        
        # Theme colors
        self.themes = {
            "default": {
                "title": "bold magenta",
                "border": "blue",
                "label": "bold cyan",
                "value": "green",
                "progress": "blue",
                "warning": "yellow",
                "critical": "red"
            },
            "minimal": {
                "title": "bold white",
                "border": "white",
                "label": "white",
                "value": "bright_white",
                "progress": "white",
                "warning": "yellow",
                "critical": "red"
            },
            "cyberpunk": {
                "title": "bold bright_magenta",
                "border": "bright_cyan",
                "label": "bright_cyan",
                "value": "bright_green",
                "progress": "bright_magenta",
                "warning": "bright_yellow",
                "critical": "bright_red"
            }
        }
        self.colors = self.themes.get(theme, self.themes["default"])
    
    def _get_gpu_info(self) -> Dict[str, str]:
        """Get GPU utilization and memory info."""
        if not self.show_gpu:
            return {}
            
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assume first GPU
                
                # Determine temperature color
                temp_color = self.colors["value"]
                if gpu.temperature > 80:
                    temp_color = self.colors["critical"]
                elif gpu.temperature > 70:
                    temp_color = self.colors["warning"]
                
                # Determine memory color
                mem_color = self.colors["value"]
                if gpu.memoryUtil > 0.9:
                    mem_color = self.colors["critical"]
                elif gpu.memoryUtil > 0.8:
                    mem_color = self.colors["warning"]
                
                return {
                    "gpu_name": gpu.name,
                    "gpu_util": f"{gpu.load * 100:.1f}%",
                    "gpu_mem": f"{gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f}GB ({gpu.memoryUtil * 100:.1f}%)",
                    "gpu_mem_color": mem_color,
                    "gpu_temp": f"{gpu.temperature}°C",
                    "gpu_temp_color": temp_color,
                    "gpu_power": f"{gpu.powerDraw:.0f}W" if hasattr(gpu, 'powerDraw') else "N/A"
                }
        except Exception as e:
            return {
                "gpu_name": "GPU Error",
                "gpu_util": "N/A",
                "gpu_mem": "N/A",
                "gpu_mem_color": self.colors["value"],
                "gpu_temp": "N/A",
                "gpu_temp_color": self.colors["value"],
                "gpu_power": "N/A"
            }
    
    def _get_cpu_info(self) -> Dict[str, str]:
        """Get CPU utilization and memory info."""
        if not self.show_cpu:
            return {}
            
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            
            # Determine CPU color
            cpu_color = self.colors["value"]
            if cpu_percent > 90:
                cpu_color = self.colors["critical"]
            elif cpu_percent > 70:
                cpu_color = self.colors["warning"]
            
            # Determine memory color
            mem_color = self.colors["value"]
            if mem.percent > 90:
                mem_color = self.colors["critical"]
            elif mem.percent > 80:
                mem_color = self.colors["warning"]
            
            return {
                "cpu_util": f"{cpu_percent:.1f}%",
                "cpu_color": cpu_color,
                "ram_used": f"{mem.used / 1e9:.1f}/{mem.total / 1e9:.1f}GB ({mem.percent:.1f}%)",
                "ram_color": mem_color
            }
        except:
            return {
                "cpu_util": "N/A",
                "cpu_color": self.colors["value"],
                "ram_used": "N/A",
                "ram_color": self.colors["value"]
            }
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _create_progress_bar(self, progress: float) -> str:
        """Create a custom progress bar."""
        bar_length = 50
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        return f"[{self.colors['progress']}]{bar}[/{self.colors['progress']}]"
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs:
            if "loss" in logs:
                self.last_loss = logs["loss"]
                self.losses.append(self.last_loss)
            if "learning_rate" in logs:
                self.learning_rates.append(logs["learning_rate"])
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        self.current_step = state.global_step
        
        # Only update display based on frequency
        if self.current_step % self.update_frequency != 0:
            return
        
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.current_step / self.total_steps) * 100
        
        # Calculate ETA
        if self.current_step > 0:
            time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta = time_per_step * remaining_steps
        else:
            eta = 0
        
        # Get system metrics
        gpu_info = self._get_gpu_info()
        cpu_info = self._get_cpu_info()
        
        # Calculate metrics
        avg_loss = sum(self.losses[-100:]) / len(self.losses[-100:]) if self.losses else 0
        current_lr = self.learning_rates[-1] if self.learning_rates else 0
        
        # Create the display
        self._create_display(
            progress_percent, 
            elapsed_time, 
            eta, 
            gpu_info, 
            cpu_info, 
            avg_loss,
            current_lr
        )
    
    def _create_display(
        self, 
        progress: float, 
        elapsed: float, 
        eta: float, 
        gpu_info: Dict,
        cpu_info: Dict,
        avg_loss: float,
        learning_rate: float
    ):
        """Create and display the rich UI."""
        # Clear console
        self.console.clear()
        
        # Main layout
        layout = Layout()
        
        # Title
        title = Text(f"Training {self.model_name}", style=self.colors["title"])
        
        # Progress section
        progress_panel = self._create_progress_panel(progress, elapsed, eta)
        
        # Metrics section
        metrics_panel = self._create_metrics_panel(avg_loss, learning_rate, gpu_info, cpu_info)
        
        # Main panel
        main_content = Layout()
        main_content.split_column(
            Layout(progress_panel, size=8),
            Layout(metrics_panel)
        )
        
        # Display
        self.console.print(Panel(
            main_content,
            title=title,
            border_style=self.colors["border"],
            padding=(1, 2)
        ))
    
    def _create_progress_panel(self, progress: float, elapsed: float, eta: float) -> Panel:
        """Create the progress panel."""
        # Progress bar
        progress_bar = self._create_progress_bar(progress)
        
        # Stats table
        stats_table = Table(show_header=False, box=None, padding=(0, 1))
        stats_table.add_column(style=self.colors["label"], width=15)
        stats_table.add_column(style=self.colors["value"], width=20)
        stats_table.add_column(style=self.colors["label"], width=15)
        stats_table.add_column(style=self.colors["value"], width=20)
        
        stats_table.add_row(
            "Step", f"{self.current_step}/{self.total_steps}",
            "Progress", f"{progress:.1f}%"
        )
        stats_table.add_row(
            "Elapsed", self._format_time(elapsed),
            "ETA", self._format_time(eta)
        )
        
        # Speed calculation
        if self.current_step > 0:
            speed = self.current_step / elapsed
            time_per_step = 1 / speed
            stats_table.add_row(
                "Speed", f"{speed:.2f} steps/s",
                "Time/Step", f"{time_per_step:.2f}s"
            )
        
        # Combine elements
        progress_content = Layout()
        progress_content.split_column(
            Layout(Text(progress_bar, justify="center"), size=3),
            Layout(stats_table, size=5)
        )
        
        return Panel(
            progress_content,
            title="Progress",
            border_style=self.colors["border"]
        )
    
    def _create_metrics_panel(
        self, 
        avg_loss: float, 
        learning_rate: float,
        gpu_info: Dict,
        cpu_info: Dict
    ) -> Panel:
        """Create the metrics panel."""
        metrics_table = Table(show_header=False, box=None, padding=(0, 1))
        metrics_table.add_column(style=self.colors["label"], width=20)
        metrics_table.add_column(width=30)
        
        # Training metrics
        metrics_table.add_row("📉 Current Loss", f"[{self.colors['value']}]{self.last_loss:.4f}[/{self.colors['value']}]")
        metrics_table.add_row("📊 Avg Loss (100)", f"[{self.colors['value']}]{avg_loss:.4f}[/{self.colors['value']}]")
        metrics_table.add_row("🎯 Learning Rate", f"[{self.colors['value']}]{learning_rate:.2e}[/{self.colors['value']}]")
        
        # GPU metrics
        if gpu_info:
            metrics_table.add_row("", "")  # Spacer
            metrics_table.add_row("🎮 GPU", f"[{self.colors['value']}]{gpu_info.get('gpu_name', 'N/A')}[/{self.colors['value']}]")
            metrics_table.add_row("   Utilization", f"[{self.colors['value']}]{gpu_info['gpu_util']}[/{self.colors['value']}]")
            metrics_table.add_row("   Memory", f"[{gpu_info['gpu_mem_color']}]{gpu_info['gpu_mem']}[/{gpu_info['gpu_mem_color']}]")
            metrics_table.add_row("   Temperature", f"[{gpu_info['gpu_temp_color']}]{gpu_info['gpu_temp']}[/{gpu_info['gpu_temp_color']}]")
            metrics_table.add_row("   Power", f"[{self.colors['value']}]{gpu_info['gpu_power']}[/{self.colors['value']}]")
        
        # CPU metrics
        if cpu_info:
            metrics_table.add_row("", "")  # Spacer
            metrics_table.add_row("💻 CPU Usage", f"[{cpu_info['cpu_color']}]{cpu_info['cpu_util']}[/{cpu_info['cpu_color']}]")
            metrics_table.add_row("🧠 RAM Usage", f"[{cpu_info['ram_color']}]{cpu_info['ram_used']}[/{cpu_info['ram_color']}]")
        
        return Panel(
            metrics_table,
            title="Metrics",
            border_style=self.colors["border"]
        )


class SimpleProgressCallback(TrainerCallback):
    """Simple progress callback for environments without rich support."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        """Print simple progress update."""
        self.current_step = state.global_step
        progress = (self.current_step / self.total_steps) * 100
        elapsed = time.time() - self.start_time
        
        # Simple progress bar
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = "=" * filled + "-" * (bar_length - filled)
        
        print(f"\rProgress: [{bar}] {progress:.1f}% ({self.current_step}/{self.total_steps} steps)", end="", flush=True)
        
        if self.current_step >= self.total_steps:
            print(f"\nCompleted in {elapsed:.1f} seconds")


def create_training_monitor(
    total_steps: int,
    model_name: str = "Model",
    rich_display: bool = True,
    **kwargs
) -> TrainerCallback:
    """
    Factory function to create appropriate training monitor.
    
    Args:
        total_steps: Total number of training steps
        model_name: Name of the model being trained
        rich_display: Whether to use rich display (falls back to simple if not available)
        **kwargs: Additional arguments for RichProgressCallback
        
    Returns:
        TrainerCallback instance
    """
    if rich_display:
        try:
            return RichProgressCallback(total_steps, model_name, **kwargs)
        except ImportError:
            print("Rich not available, falling back to simple progress display")
            return SimpleProgressCallback(total_steps)
    else:
        return SimpleProgressCallback(total_steps)