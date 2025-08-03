#!/usr/bin/env python3
"""
Real-time GPU Training Dashboard
Interactive terminal dashboard for monitoring GPU and training metrics.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any
import threading

import pynvml
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
import plotext as plt


# Initialize NVML
pynvml.nvmlInit()


class GPUDashboard:
    """Real-time GPU monitoring dashboard."""
    
    def __init__(self, gpu_index: int = 0, update_interval: float = 1.0):
        self.gpu_index = gpu_index
        self.update_interval = update_interval
        self.console = Console()
        
        # Get GPU info
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
        
        # Data history for graphs (keep last 60 samples)
        self.history_size = 60
        self.gpu_util_history = deque(maxlen=self.history_size)
        self.memory_history = deque(maxlen=self.history_size)
        self.temp_history = deque(maxlen=self.history_size)
        self.power_history = deque(maxlen=self.history_size)
        
        # Training metrics history
        self.loss_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        
        # Dashboard state
        self.is_running = False
        self.start_time = None
        self.current_metrics = {}
        self.training_metrics = {}
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics."""
        try:
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_used_gb = mem_info.used / 1024 / 1024 / 1024
            memory_total_gb = mem_info.total / 1024 / 1024 / 1024
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000  # Watts
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000
            
            # Fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(self.gpu_handle)
            except:
                fan_speed = None
            
            # Clock speeds
            try:
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
            except:
                gpu_clock = None
                mem_clock = None
            
            metrics = {
                'utilization': util.gpu,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'memory_percent': memory_percent,
                'temperature': temp,
                'power': power,
                'power_limit': power_limit,
                'fan_speed': fan_speed,
                'gpu_clock': gpu_clock,
                'mem_clock': mem_clock,
            }
            
            # Update history
            self.gpu_util_history.append(util.gpu)
            self.memory_history.append(memory_percent)
            self.temp_history.append(temp)
            self.power_history.append(power)
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                'cpu_percent': cpu_percent,
                'ram_used_gb': memory.used / 1024 / 1024 / 1024,
                'ram_total_gb': memory.total / 1024 / 1024 / 1024,
                'ram_percent': memory.percent,
            }
        except Exception as e:
            return {'error': str(e)}
    
    def create_header(self) -> Panel:
        """Create dashboard header."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        
        header_text = f"[bold cyan]GPU Training Monitor[/bold cyan]\n"
        header_text += f"[yellow]{self.gpu_name}[/yellow] | "
        header_text += f"Index: {self.gpu_index} | "
        header_text += f"Uptime: {elapsed_str}"
        
        return Panel(Align.center(header_text), style="bold blue")
    
    def create_metrics_table(self) -> Table:
        """Create metrics table."""
        table = Table(title="Current Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", justify="right", style="green")
        table.add_column("Graph", width=40)
        
        metrics = self.current_metrics
        
        if 'error' not in metrics:
            # GPU Utilization
            util_bar = self._create_bar(metrics['utilization'], 100, "█")
            table.add_row("GPU Utilization", f"{metrics['utilization']:.1f}%", util_bar)
            
            # Memory
            mem_bar = self._create_bar(metrics['memory_percent'], 100, "▓")
            table.add_row("Memory Usage", 
                         f"{metrics['memory_used_gb']:.1f}/{metrics['memory_total_gb']:.1f} GB ({metrics['memory_percent']:.1f}%)",
                         mem_bar)
            
            # Temperature
            temp_color = "green" if metrics['temperature'] < 70 else "yellow" if metrics['temperature'] < 80 else "red"
            temp_bar = self._create_bar(metrics['temperature'], 90, "▒", color=temp_color)
            table.add_row("Temperature", f"{metrics['temperature']}°C", temp_bar)
            
            # Power
            power_bar = self._create_bar(metrics['power'], metrics['power_limit'], "░")
            table.add_row("Power Draw", f"{metrics['power']:.1f}/{metrics['power_limit']:.1f}W", power_bar)
            
            # Fan Speed
            if metrics['fan_speed'] is not None:
                fan_bar = self._create_bar(metrics['fan_speed'], 100, "◈")
                table.add_row("Fan Speed", f"{metrics['fan_speed']}%", fan_bar)
            
            # Clocks
            if metrics['gpu_clock'] is not None:
                table.add_row("GPU Clock", f"{metrics['gpu_clock']} MHz", "")
            if metrics['mem_clock'] is not None:
                table.add_row("Memory Clock", f"{metrics['mem_clock']} MHz", "")
        
        return table
    
    def create_training_panel(self) -> Panel:
        """Create training metrics panel."""
        if not self.training_metrics:
            content = "[dim]No training metrics available yet...[/dim]"
        else:
            content = ""
            content += f"[cyan]Step:[/cyan] {self.training_metrics.get('step', 'N/A')}\n"
            content += f"[cyan]Epoch:[/cyan] {self.training_metrics.get('epoch', 'N/A'):.2f}\n"
            content += f"[cyan]Loss:[/cyan] {self.training_metrics.get('loss', 'N/A'):.4f}\n"
            content += f"[cyan]Learning Rate:[/cyan] {self.training_metrics.get('lr', 'N/A'):.2e}\n"
            content += f"[cyan]Throughput:[/cyan] {self.training_metrics.get('throughput', 'N/A'):.1f} samples/sec"
        
        return Panel(content, title="Training Metrics", border_style="green")
    
    def create_graphs(self) -> str:
        """Create ASCII graphs for metrics history."""
        if len(self.gpu_util_history) < 2:
            return "[dim]Collecting data for graphs...[/dim]"
        
        # Create mini plots
        plots = []
        
        # GPU Utilization
        plt.clf()
        plt.plot(list(self.gpu_util_history), label="GPU %")
        plt.ylim(0, 100)
        plt.title("GPU Utilization")
        plt.canvas_color("black")
        plt.axes_color("black") 
        plt.ticks_color("white")
        gpu_plot = plt.build()
        
        # Temperature
        plt.clf()
        plt.plot(list(self.temp_history), label="Temp °C", color="red")
        plt.ylim(20, 90)
        plt.title("Temperature")
        plt.canvas_color("black")
        plt.axes_color("black")
        plt.ticks_color("white")
        temp_plot = plt.build()
        
        # Combine plots
        return f"{gpu_plot}\n\n{temp_plot}"
    
    def _create_bar(self, value: float, max_value: float, char: str = "█", 
                   width: int = 40, color: str = "green") -> str:
        """Create a progress bar."""
        if max_value == 0:
            return ""
        
        filled = int((value / max_value) * width)
        bar = char * filled + " " * (width - filled)
        percentage = (value / max_value) * 100
        
        # Color based on percentage
        if color == "auto":
            if percentage > 90:
                color = "red"
            elif percentage > 70:
                color = "yellow"
            else:
                color = "green"
        
        return f"[{color}]{bar}[/{color}]"
    
    def create_system_info(self) -> Panel:
        """Create system info panel."""
        sys_metrics = self.get_system_metrics()
        
        content = ""
        if 'error' not in sys_metrics:
            content += f"[cyan]CPU Usage:[/cyan] {sys_metrics['cpu_percent']:.1f}%\n"
            content += f"[cyan]RAM Usage:[/cyan] {sys_metrics['ram_used_gb']:.1f}/{sys_metrics['ram_total_gb']:.1f} GB ({sys_metrics['ram_percent']:.1f}%)"
        else:
            content = f"[red]Error: {sys_metrics['error']}[/red]"
        
        return Panel(content, title="System Info", border_style="blue")
    
    def update_training_metrics(self, metrics: Dict[str, Any]):
        """Update training metrics from external source."""
        self.training_metrics = metrics
        
        # Update history
        if 'loss' in metrics:
            self.loss_history.append(metrics['loss'])
        if 'lr' in metrics:
            self.lr_history.append(metrics['lr'])
    
    def create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(self.create_header(), size=3),
            Layout(name="main"),
            Layout(name="footer", size=1)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        layout["left"].split_column(
            Layout(self.create_metrics_table(), size=15),
            Layout(Panel(self.create_graphs(), title="Metrics History", border_style="yellow"))
        )
        
        layout["right"].split_column(
            Layout(self.create_training_panel()),
            Layout(self.create_system_info())
        )
        
        layout["footer"].update(
            Panel(
                "[bold]Commands:[/bold] [cyan]q[/cyan] Quit | [cyan]r[/cyan] Reset | [cyan]s[/cyan] Save metrics",
                style="dim"
            )
        )
        
        return layout
    
    def save_metrics(self):
        """Save current metrics to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpu_metrics_{timestamp}.json"
        
        data = {
            'timestamp': timestamp,
            'gpu_name': self.gpu_name,
            'gpu_index': self.gpu_index,
            'current_metrics': self.current_metrics,
            'training_metrics': self.training_metrics,
            'history': {
                'gpu_utilization': list(self.gpu_util_history),
                'memory_percent': list(self.memory_history),
                'temperature': list(self.temp_history),
                'power': list(self.power_history),
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.console.print(f"[green]Metrics saved to {filename}[/green]")
    
    def run(self):
        """Run the dashboard."""
        self.is_running = True
        self.start_time = time.time()
        
        # Start keyboard listener in background
        def keyboard_handler():
            while self.is_running:
                key = input()
                if key.lower() == 'q':
                    self.is_running = False
                elif key.lower() == 's':
                    self.save_metrics()
                elif key.lower() == 'r':
                    # Reset history
                    self.gpu_util_history.clear()
                    self.memory_history.clear()
                    self.temp_history.clear()
                    self.power_history.clear()
        
        keyboard_thread = threading.Thread(target=keyboard_handler, daemon=True)
        keyboard_thread.start()
        
        with Live(self.create_layout(), refresh_per_second=1/self.update_interval, 
                 screen=True) as live:
            while self.is_running:
                try:
                    # Update metrics
                    self.current_metrics = self.get_gpu_metrics()
                    
                    # Update display
                    live.update(self.create_layout())
                    
                    time.sleep(self.update_interval)
                    
                except KeyboardInterrupt:
                    self.is_running = False
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    time.sleep(self.update_interval)
        
        self.console.print("[yellow]Dashboard stopped.[/yellow]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPU Training Dashboard")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to monitor")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    parser.add_argument("--metrics-file", type=str, help="File to read training metrics from")
    
    args = parser.parse_args()
    
    dashboard = GPUDashboard(gpu_index=args.gpu, update_interval=args.interval)
    
    try:
        dashboard.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()