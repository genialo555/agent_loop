"""
HRM Monitoring Integration.

Connects HRM training with existing Prometheus/Grafana monitoring infrastructure.
"""

import time
import torch
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

# Try to import monitoring components
try:
    from ...monitoring.metrics import MetricsCollector
    from ...monitoring.prometheus_exporter import PrometheusExporter
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

from transformers import TrainerCallback
from transformers.trainer_utils import IntervalStrategy


class HRMMetricsCollector:
    """Collects HRM-specific metrics for monitoring."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics_file = self.output_dir / "hrm_metrics.jsonl"
        self.convergence_file = self.output_dir / "convergence_history.json"
        
        # Metric accumulators
        self.metrics_history = []
        self.convergence_history = []
        
        # Current metrics
        self.current_metrics = {
            'hrm/low_level_iterations': 0,
            'hrm/high_level_cycles': 0,
            'hrm/convergence_rate': 0.0,
            'hrm/gradient_norm': 0.0,
            'hrm/memory_saved_gb': 0.0,
            'hrm/act_halt_rate': 0.0,
            'hrm/deep_supervision_segments': 0,
            'hrm/approximate_gradient_active': False,
        }
    
    def update(self, metrics: Dict[str, Any]):
        """Update current metrics."""
        self.current_metrics.update(metrics)
        
        # Add timestamp
        metrics_with_time = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        # Append to history
        self.metrics_history.append(metrics_with_time)
        
        # Write to file (append mode)
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_with_time) + '\n')
    
    def log_convergence(self, convergence_info: Dict[str, Any]):
        """Log convergence information."""
        self.convergence_history.append({
            'timestamp': datetime.now().isoformat(),
            **convergence_info
        })
        
        # Save convergence history
        with open(self.convergence_file, 'w') as f:
            json.dump(self.convergence_history, f, indent=2)
    
    def get_prometheus_metrics(self) -> Dict[str, float]:
        """Get metrics in Prometheus format."""
        # Convert to Prometheus-compatible format
        prom_metrics = {}
        
        for key, value in self.current_metrics.items():
            if isinstance(value, (int, float)):
                # Replace / with _ for Prometheus
                prom_key = key.replace('/', '_')
                prom_metrics[f"hrm_{prom_key}"] = float(value)
        
        return prom_metrics


class HRMTrainingCallback(TrainerCallback):
    """
    Training callback that integrates HRM metrics with training loop.
    """
    
    def __init__(self, hrm_config, metrics_collector: HRMMetricsCollector):
        self.hrm_config = hrm_config
        self.metrics_collector = metrics_collector
        self.step_count = 0
        
        # Prometheus exporter if available
        if MONITORING_AVAILABLE:
            self.prometheus_exporter = PrometheusExporter(port=9091)
            self.prometheus_exporter.start()
        else:
            self.prometheus_exporter = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when training logs metrics."""
        if logs is None:
            return
        
        # Extract HRM metrics from logs
        hrm_metrics = {
            k: v for k, v in logs.items() 
            if k.startswith('hrm/') or k in ['loss', 'learning_rate', 'epoch']
        }
        
        # Add configuration metrics
        hrm_metrics.update({
            'hrm/num_high_cycles': self.hrm_config.num_high_cycles,
            'hrm/timesteps_per_cycle': self.hrm_config.timesteps_per_cycle,
            'hrm/deep_supervision_segments': self.hrm_config.deep_supervision_segments,
            'hrm/approximate_gradient_active': self.hrm_config.use_approximate_gradient,
            'hrm/act_enabled': self.hrm_config.enable_act,
        })
        
        # Update metrics collector
        self.metrics_collector.update(hrm_metrics)
        
        # Export to Prometheus if available
        if self.prometheus_exporter:
            prom_metrics = self.metrics_collector.get_prometheus_metrics()
            for metric_name, value in prom_metrics.items():
                self.prometheus_exporter.update_metric(metric_name, value)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""
        if metrics:
            eval_metrics = {
                f"hrm/eval_{k}": v for k, v in metrics.items()
            }
            self.metrics_collector.update(eval_metrics)
    
    def on_save(self, args, state, control, **kwargs):
        """Called when checkpoint is saved."""
        # Log checkpoint save
        self.metrics_collector.update({
            'hrm/checkpoint_saved': 1,
            'hrm/global_step': state.global_step,
        })
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at end of training."""
        # Final summary
        summary = {
            'hrm/training_completed': 1,
            'hrm/total_steps': state.global_step,
            'hrm/best_loss': state.best_metric if state.best_metric else -1,
        }
        self.metrics_collector.update(summary)
        
        # Stop Prometheus exporter
        if self.prometheus_exporter:
            self.prometheus_exporter.stop()


class HRMGrafanaDashboard:
    """
    Creates Grafana dashboard configuration for HRM metrics.
    """
    
    @staticmethod
    def generate_dashboard() -> Dict[str, Any]:
        """Generate Grafana dashboard JSON for HRM monitoring."""
        return {
            "dashboard": {
                "title": "HRM Training Monitor",
                "panels": [
                    {
                        "title": "Loss & Learning Rate",
                        "targets": [
                            {"expr": "hrm_loss"},
                            {"expr": "hrm_learning_rate"}
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "title": "Convergence Metrics",
                        "targets": [
                            {"expr": "hrm_hrm_low_level_iterations"},
                            {"expr": "hrm_hrm_convergence_rate"}
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "title": "Memory Usage",
                        "targets": [
                            {"expr": "hrm_hrm_memory_saved_gb"},
                            {"expr": "hrm_hrm_gradient_norm"}
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "title": "ACT Metrics",
                        "targets": [
                            {"expr": "hrm_hrm_act_halt_rate"},
                            {"expr": "hrm_hrm_deep_supervision_segments"}
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    }
                ],
                "refresh": "5s",
                "time": {"from": "now-1h", "to": "now"}
            }
        }
    
    @staticmethod
    def save_dashboard(output_dir: str):
        """Save dashboard configuration to file."""
        dashboard = HRMGrafanaDashboard.generate_dashboard()
        dashboard_path = Path(output_dir) / "hrm_grafana_dashboard.json"
        
        with open(dashboard_path, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        return dashboard_path


def create_hrm_monitoring(hrm_config, output_dir: str):
    """
    Create complete HRM monitoring setup.
    """
    # Create metrics collector
    metrics_collector = HRMMetricsCollector(output_dir)
    
    # Create training callback
    callback = HRMTrainingCallback(hrm_config, metrics_collector)
    
    # Generate Grafana dashboard
    dashboard_path = HRMGrafanaDashboard.save_dashboard(output_dir)
    
    print(f"HRM Monitoring initialized:")
    print(f"  - Metrics file: {metrics_collector.metrics_file}")
    print(f"  - Convergence file: {metrics_collector.convergence_file}")
    print(f"  - Grafana dashboard: {dashboard_path}")
    
    if MONITORING_AVAILABLE:
        print("  - Prometheus exporter: Active on port 9091")
    else:
        print("  - Prometheus exporter: Not available (monitoring package missing)")
    
    return callback, metrics_collector