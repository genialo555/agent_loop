#!/usr/bin/env python3
"""
GPU Training Monitor - Real-time GPU and training metrics monitoring
Tracks GPU utilization, VRAM, temperature, power draw, and training metrics.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import threading
from collections import deque

import torch
import pynvml
import psutil
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Initialize NVML
pynvml.nvmlInit()

# Prometheus metrics
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization percentage', ['gpu_index'])
gpu_memory_used = Gauge('gpu_memory_used_mb', 'GPU memory used in MB', ['gpu_index'])
gpu_memory_total = Gauge('gpu_memory_total_mb', 'GPU total memory in MB', ['gpu_index'])
gpu_temperature = Gauge('gpu_temperature_celsius', 'GPU temperature in Celsius', ['gpu_index'])
gpu_power_draw = Gauge('gpu_power_draw_watts', 'GPU power draw in watts', ['gpu_index'])
gpu_fan_speed = Gauge('gpu_fan_speed_percent', 'GPU fan speed percentage', ['gpu_index'])

training_loss = Gauge('training_loss', 'Current training loss')
training_step = Counter('training_steps_total', 'Total training steps')
training_epoch = Gauge('training_epoch', 'Current training epoch')
learning_rate = Gauge('learning_rate', 'Current learning rate')
gradient_norm = Gauge('gradient_norm', 'Gradient norm')

training_step_duration = Histogram('training_step_duration_seconds', 'Training step duration',
                                 buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
memory_allocation = Summary('memory_allocation_mb', 'Memory allocation in MB')


@dataclass
class GPUMetrics:
    """GPU metrics dataclass for structured monitoring."""
    timestamp: str
    gpu_index: int
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature_celsius: float
    power_draw_watts: float
    fan_speed_percent: Optional[float]
    compute_sm_utilization: Optional[float]
    memory_bandwidth_utilization: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Training metrics dataclass."""
    timestamp: str
    step: int
    epoch: float
    loss: float
    learning_rate: float
    gradient_norm: Optional[float]
    throughput_samples_per_sec: Optional[float]
    memory_allocated_mb: float
    memory_reserved_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GPUMonitor:
    """Real-time GPU monitoring for training workloads."""
    
    def __init__(self, 
                 gpu_index: int = 0,
                 sample_interval: float = 1.0,
                 history_size: int = 300,
                 log_dir: Optional[Path] = None):
        """
        Initialize GPU Monitor.
        
        Args:
            gpu_index: GPU device index to monitor
            sample_interval: Sampling interval in seconds
            history_size: Number of samples to keep in memory
            log_dir: Directory to save monitoring logs
        """
        self.gpu_index = gpu_index
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.log_dir = Path(log_dir) if log_dir else Path("./monitoring_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Get GPU handle
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.gpu_history = deque(maxlen=history_size)
        self.training_history = deque(maxlen=history_size)
        
        # Log GPU info
        logger.info("gpu_monitor_initialized",
                   gpu_name=self.gpu_name,
                   gpu_index=gpu_index,
                   sample_interval=sample_interval)
    
    def get_gpu_metrics(self) -> GPUMetrics:
        """Collect current GPU metrics."""
        try:
            # Basic utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_used_mb = mem_info.used / 1024 / 1024
            memory_total_mb = mem_info.total / 1024 / 1024
            memory_percent = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000  # Convert to watts
            
            # Fan speed (may not be available on all GPUs)
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(self.gpu_handle)
            except pynvml.NVMLError:
                fan_speed = None
            
            # Advanced metrics (if available)
            try:
                # Get compute and memory bandwidth utilization
                compute_util = None
                mem_bandwidth_util = None
                
                # Try to get more detailed utilization info
                for i in range(pynvml.nvmlDeviceGetNumGpuCores(self.gpu_handle)):
                    try:
                        core_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        if compute_util is None:
                            compute_util = core_util.gpu
                        break
                    except:
                        pass
            except:
                compute_util = util.gpu
                mem_bandwidth_util = util.memory
            
            metrics = GPUMetrics(
                timestamp=datetime.utcnow().isoformat(),
                gpu_index=self.gpu_index,
                utilization_percent=float(util.gpu),
                memory_used_mb=float(memory_used_mb),
                memory_total_mb=float(memory_total_mb),
                memory_percent=float(memory_percent),
                temperature_celsius=float(temp),
                power_draw_watts=float(power),
                fan_speed_percent=float(fan_speed) if fan_speed else None,
                compute_sm_utilization=float(compute_util) if compute_util else None,
                memory_bandwidth_utilization=float(mem_bandwidth_util) if mem_bandwidth_util else None
            )
            
            # Update Prometheus metrics
            gpu_utilization.labels(gpu_index=self.gpu_index).set(metrics.utilization_percent)
            gpu_memory_used.labels(gpu_index=self.gpu_index).set(metrics.memory_used_mb)
            gpu_memory_total.labels(gpu_index=self.gpu_index).set(metrics.memory_total_mb)
            gpu_temperature.labels(gpu_index=self.gpu_index).set(metrics.temperature_celsius)
            gpu_power_draw.labels(gpu_index=self.gpu_index).set(metrics.power_draw_watts)
            if metrics.fan_speed_percent:
                gpu_fan_speed.labels(gpu_index=self.gpu_index).set(metrics.fan_speed_percent)
            
            return metrics
            
        except Exception as e:
            logger.error("gpu_metrics_collection_failed", error=str(e), gpu_index=self.gpu_index)
            raise
    
    def get_training_metrics(self, 
                           step: int,
                           epoch: float,
                           loss: float,
                           lr: float,
                           grad_norm: Optional[float] = None,
                           throughput: Optional[float] = None) -> TrainingMetrics:
        """Create training metrics from provided values."""
        try:
            # Get PyTorch memory stats
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.gpu_index) / 1024 / 1024
                memory_reserved = torch.cuda.memory_reserved(self.gpu_index) / 1024 / 1024
            else:
                memory_allocated = 0.0
                memory_reserved = 0.0
            
            metrics = TrainingMetrics(
                timestamp=datetime.utcnow().isoformat(),
                step=step,
                epoch=epoch,
                loss=loss,
                learning_rate=lr,
                gradient_norm=grad_norm,
                throughput_samples_per_sec=throughput,
                memory_allocated_mb=memory_allocated,
                memory_reserved_mb=memory_reserved
            )
            
            # Update Prometheus metrics
            training_loss.set(loss)
            training_step.inc()
            training_epoch.set(epoch)
            learning_rate.set(lr)
            if grad_norm:
                gradient_norm.set(grad_norm)
            memory_allocation.observe(memory_allocated)
            
            return metrics
            
        except Exception as e:
            logger.error("training_metrics_collection_failed", error=str(e))
            raise
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        logger.info("monitoring_loop_started", gpu_index=self.gpu_index)
        
        while self.is_monitoring:
            try:
                # Collect GPU metrics
                gpu_metrics = self.get_gpu_metrics()
                self.gpu_history.append(gpu_metrics)
                
                # Log high-level metrics
                if gpu_metrics.temperature_celsius > 80:
                    logger.warning("gpu_temperature_high",
                                 temperature=gpu_metrics.temperature_celsius,
                                 gpu_index=self.gpu_index)
                
                if gpu_metrics.memory_percent > 90:
                    logger.warning("gpu_memory_high",
                                 memory_percent=gpu_metrics.memory_percent,
                                 memory_used_mb=gpu_metrics.memory_used_mb,
                                 gpu_index=self.gpu_index)
                
                # Log to file periodically
                if len(self.gpu_history) % 60 == 0:  # Every minute
                    self._save_metrics_to_file()
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.error("monitoring_loop_error", error=str(e), gpu_index=self.gpu_index)
                time.sleep(self.sample_interval)
    
    def _save_metrics_to_file(self):
        """Save metrics history to file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Save GPU metrics
            gpu_file = self.log_dir / f"gpu_metrics_{timestamp}.jsonl"
            with open(gpu_file, 'w') as f:
                for metric in self.gpu_history:
                    f.write(json.dumps(metric.to_dict()) + '\n')
            
            # Save training metrics if available
            if self.training_history:
                training_file = self.log_dir / f"training_metrics_{timestamp}.jsonl"
                with open(training_file, 'w') as f:
                    for metric in self.training_history:
                        f.write(json.dumps(metric.to_dict()) + '\n')
            
            logger.info("metrics_saved_to_file",
                       gpu_file=str(gpu_file),
                       gpu_samples=len(self.gpu_history),
                       training_samples=len(self.training_history))
                       
        except Exception as e:
            logger.error("metrics_save_failed", error=str(e))
    
    def start_monitoring(self):
        """Start background monitoring."""
        if self.is_monitoring:
            logger.warning("monitoring_already_active")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("monitoring_started", gpu_index=self.gpu_index)
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Save final metrics
        self._save_metrics_to_file()
        
        logger.info("monitoring_stopped", gpu_index=self.gpu_index)
    
    def log_training_step(self, 
                         step: int,
                         epoch: float,
                         loss: float,
                         lr: float,
                         grad_norm: Optional[float] = None,
                         throughput: Optional[float] = None,
                         step_duration: Optional[float] = None):
        """Log a training step with metrics."""
        try:
            # Create training metrics
            metrics = self.get_training_metrics(step, epoch, loss, lr, grad_norm, throughput)
            self.training_history.append(metrics)
            
            # Log step duration if provided
            if step_duration:
                training_step_duration.observe(step_duration)
            
            # Log structured event
            logger.info("training_step",
                       step=step,
                       epoch=epoch,
                       loss=loss,
                       learning_rate=lr,
                       gradient_norm=grad_norm,
                       throughput=throughput,
                       memory_allocated_mb=metrics.memory_allocated_mb)
            
        except Exception as e:
            logger.error("training_step_log_failed", error=str(e))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary statistics."""
        summary = {
            "gpu_name": self.gpu_name,
            "gpu_index": self.gpu_index,
            "monitoring_duration_seconds": len(self.gpu_history) * self.sample_interval,
            "total_samples": len(self.gpu_history),
        }
        
        if self.gpu_history:
            # GPU statistics
            gpu_utils = [m.utilization_percent for m in self.gpu_history]
            gpu_temps = [m.temperature_celsius for m in self.gpu_history]
            gpu_powers = [m.power_draw_watts for m in self.gpu_history]
            gpu_mem_used = [m.memory_used_mb for m in self.gpu_history]
            
            summary["gpu_stats"] = {
                "utilization": {
                    "mean": sum(gpu_utils) / len(gpu_utils),
                    "max": max(gpu_utils),
                    "min": min(gpu_utils)
                },
                "temperature": {
                    "mean": sum(gpu_temps) / len(gpu_temps),
                    "max": max(gpu_temps),
                    "min": min(gpu_temps)
                },
                "power": {
                    "mean": sum(gpu_powers) / len(gpu_powers),
                    "max": max(gpu_powers),
                    "min": min(gpu_powers)
                },
                "memory_used_mb": {
                    "mean": sum(gpu_mem_used) / len(gpu_mem_used),
                    "max": max(gpu_mem_used),
                    "min": min(gpu_mem_used)
                }
            }
        
        if self.training_history:
            # Training statistics
            losses = [m.loss for m in self.training_history]
            lrs = [m.learning_rate for m in self.training_history]
            
            summary["training_stats"] = {
                "total_steps": len(self.training_history),
                "final_epoch": self.training_history[-1].epoch,
                "loss": {
                    "initial": losses[0],
                    "final": losses[-1],
                    "min": min(losses)
                },
                "learning_rate": {
                    "initial": lrs[0],
                    "final": lrs[-1]
                }
            }
        
        return summary
    
    def print_live_stats(self):
        """Print live GPU stats to console."""
        try:
            metrics = self.get_gpu_metrics()
            
            print(f"\n{'='*60}")
            print(f"GPU: {self.gpu_name} (Index: {self.gpu_index})")
            print(f"{'='*60}")
            print(f"Utilization:  {metrics.utilization_percent:6.1f}% {'█' * int(metrics.utilization_percent / 2)}")
            print(f"Memory:       {metrics.memory_used_mb:6.0f} / {metrics.memory_total_mb:6.0f} MB ({metrics.memory_percent:.1f}%)")
            print(f"Temperature:  {metrics.temperature_celsius:6.1f}°C")
            print(f"Power Draw:   {metrics.power_draw_watts:6.1f}W")
            if metrics.fan_speed_percent:
                print(f"Fan Speed:    {metrics.fan_speed_percent:6.1f}%")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error getting GPU stats: {e}")


class TrainingMonitorCallback:
    """Callback for integrating GPU monitoring with training loops."""
    
    def __init__(self, gpu_monitor: GPUMonitor):
        self.gpu_monitor = gpu_monitor
        self.step_start_time = None
    
    def on_train_begin(self):
        """Called at the beginning of training."""
        self.gpu_monitor.start_monitoring()
        logger.info("training_monitor_callback_started")
    
    def on_train_end(self):
        """Called at the end of training."""
        self.gpu_monitor.stop_monitoring()
        
        # Print summary
        summary = self.gpu_monitor.get_summary()
        logger.info("training_completed", summary=summary)
    
    def on_step_begin(self):
        """Called at the beginning of each training step."""
        self.step_start_time = time.time()
    
    def on_step_end(self, step: int, epoch: float, loss: float, lr: float, 
                   grad_norm: Optional[float] = None, throughput: Optional[float] = None):
        """Called at the end of each training step."""
        step_duration = time.time() - self.step_start_time if self.step_start_time else None
        
        self.gpu_monitor.log_training_step(
            step=step,
            epoch=epoch,
            loss=loss,
            lr=lr,
            grad_norm=grad_norm,
            throughput=throughput,
            step_duration=step_duration
        )


async def monitor_training_async(gpu_index: int = 0, 
                               prometheus_port: int = 9090,
                               update_interval: float = 1.0):
    """Async monitoring with Prometheus metrics export."""
    # Start Prometheus HTTP server
    start_http_server(prometheus_port)
    logger.info("prometheus_server_started", port=prometheus_port)
    
    # Initialize GPU monitor
    monitor = GPUMonitor(gpu_index=gpu_index, sample_interval=update_interval)
    monitor.start_monitoring()
    
    try:
        # Keep monitoring until interrupted
        while True:
            await asyncio.sleep(10)
            monitor.print_live_stats()
            
    except KeyboardInterrupt:
        logger.info("monitoring_interrupted")
    finally:
        monitor.stop_monitoring()
        summary = monitor.get_summary()
        print(f"\nMonitoring Summary:\n{json.dumps(summary, indent=2)}")


def main():
    """Main entry point for standalone monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Training Monitor")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to monitor")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--prometheus-port", type=int, default=9090, help="Prometheus metrics port")
    parser.add_argument("--log-dir", type=str, default="./monitoring_logs", help="Directory for log files")
    
    args = parser.parse_args()
    
    # Run async monitoring
    asyncio.run(monitor_training_async(
        gpu_index=args.gpu,
        prometheus_port=args.prometheus_port,
        update_interval=args.interval
    ))


if __name__ == "__main__":
    main()