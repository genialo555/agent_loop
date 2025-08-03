#!/usr/bin/env python3
"""
Training Log Analyzer for MLOps Pipeline

Analyzes training logs from files or streams to extract:
- Loss curves and convergence patterns
- Learning rate schedules
- Gradient statistics
- Performance metrics
- Anomaly detection
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import structlog

# Configure logging
logger = structlog.get_logger(__name__)
console = Console()

# Regex patterns for common training log formats
LOG_PATTERNS = {
    'loss': r'(?:loss|Loss)\s*[=:]\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)',
    'learning_rate': r'(?:lr|learning_rate|LR)\s*[=:]\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)',
    'gradient_norm': r'(?:grad_norm|gradient_norm|grad norm)\s*[=:]\s*([0-9.]+(?:[eE][+-]?[0-9]+)?)',
    'epoch': r'(?:epoch|Epoch)\s*[=:]?\s*([0-9]+)',
    'step': r'(?:step|Step|iter|Iter)\s*[=:]?\s*([0-9]+)',
    'tokens_per_sec': r'(?:tokens?/s(?:ec)?|tok/s)\s*[=:]\s*([0-9.]+)',
    'gpu_memory': r'(?:gpu_mem(?:ory)?|GPU memory)\s*[=:]\s*([0-9.]+)\s*(?:GB|gb)?',
    'accuracy': r'(?:acc(?:uracy)?|Accuracy)\s*[=:]\s*([0-9.]+)',
    'perplexity': r'(?:ppl|perplexity|Perplexity)\s*[=:]\s*([0-9.]+)',
}


class TrainingLogAnalyzer:
    """Analyze training logs and extract insights."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.metrics: Dict[str, List[float]] = {
            'loss': [],
            'learning_rate': [],
            'gradient_norm': [],
            'tokens_per_sec': [],
            'gpu_memory': [],
            'accuracy': [],
            'perplexity': [],
        }
        self.epochs: List[int] = []
        self.steps: List[int] = []
        self.timestamps: List[datetime] = []
        
    def parse_log_file(self) -> None:
        """Parse log file and extract metrics."""
        console.print(f"[cyan]Parsing log file: {self.log_path}[/cyan]")
        
        with open(self.log_path, 'r') as f:
            lines = f.readlines()
            
        for line in track(lines, description="Parsing logs..."):
            self._parse_line(line)
            
        console.print(f"[green]Parsed {len(self.steps)} training steps[/green]")
        
    def _parse_line(self, line: str) -> None:
        """Parse a single log line."""
        # Try to extract timestamp
        timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', line)
        if timestamp_match:
            try:
                timestamp = datetime.strptime(timestamp_match.group(), '%Y-%m-%d %H:%M:%S')
                self.timestamps.append(timestamp)
            except:
                pass
        
        # Extract metrics using regex patterns
        for metric, pattern in LOG_PATTERNS.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    
                    if metric == 'epoch':
                        self.epochs.append(int(value))
                    elif metric == 'step':
                        self.steps.append(int(value))
                    else:
                        self.metrics[metric].append(value)
                except ValueError:
                    pass
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence patterns."""
        if not self.metrics['loss']:
            return {'status': 'No loss data found'}
            
        losses = np.array(self.metrics['loss'])
        
        # Calculate moving average
        window = min(100, len(losses) // 10)
        if window > 1:
            ma = pd.Series(losses).rolling(window=window).mean().dropna()
        else:
            ma = pd.Series(losses)
            
        # Detect convergence
        converged = False
        convergence_step = None
        
        if len(ma) > 20:
            # Check if loss has stabilized (low variance in recent steps)
            recent_variance = np.var(ma.tail(20))
            overall_variance = np.var(ma)
            
            if recent_variance < overall_variance * 0.1:
                converged = True
                # Find approximate convergence point
                for i in range(len(ma) - 20):
                    window_var = np.var(ma.iloc[i:i+20])
                    if window_var < overall_variance * 0.1:
                        convergence_step = i
                        break
        
        # Calculate improvement rate
        if len(losses) > 1:
            initial_loss = np.mean(losses[:10]) if len(losses) > 10 else losses[0]
            final_loss = np.mean(losses[-10:]) if len(losses) > 10 else losses[-1]
            improvement = (initial_loss - final_loss) / initial_loss * 100
        else:
            improvement = 0
            
        return {
            'converged': converged,
            'convergence_step': convergence_step,
            'initial_loss': float(losses[0]) if len(losses) > 0 else None,
            'final_loss': float(losses[-1]) if len(losses) > 0 else None,
            'min_loss': float(np.min(losses)) if len(losses) > 0 else None,
            'improvement_percent': improvement,
            'loss_variance': float(np.var(losses)) if len(losses) > 0 else None,
        }
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in training metrics."""
        anomalies = []
        
        # Check for loss spikes
        if len(self.metrics['loss']) > 10:
            losses = np.array(self.metrics['loss'])
            z_scores = np.abs(stats.zscore(losses))
            spike_indices = np.where(z_scores > 3)[0]
            
            for idx in spike_indices:
                anomalies.append({
                    'type': 'loss_spike',
                    'step': self.steps[idx] if idx < len(self.steps) else idx,
                    'value': float(losses[idx]),
                    'z_score': float(z_scores[idx]),
                })
        
        # Check for gradient explosions
        if len(self.metrics['gradient_norm']) > 0:
            grad_norms = np.array(self.metrics['gradient_norm'])
            high_grads = np.where(grad_norms > 10.0)[0]
            
            for idx in high_grads:
                anomalies.append({
                    'type': 'gradient_explosion',
                    'step': self.steps[idx] if idx < len(self.steps) else idx,
                    'value': float(grad_norms[idx]),
                })
        
        # Check for training speed drops
        if len(self.metrics['tokens_per_sec']) > 10:
            speeds = np.array(self.metrics['tokens_per_sec'])
            avg_speed = np.mean(speeds)
            slow_steps = np.where(speeds < avg_speed * 0.5)[0]
            
            for idx in slow_steps:
                anomalies.append({
                    'type': 'speed_drop',
                    'step': self.steps[idx] if idx < len(self.steps) else idx,
                    'value': float(speeds[idx]),
                    'avg_speed': float(avg_speed),
                })
                
        return anomalies
    
    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each metric."""
        stats_dict = {}
        
        for metric, values in self.metrics.items():
            if values:
                arr = np.array(values)
                stats_dict[metric] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr)),
                    'q25': float(np.percentile(arr, 25)),
                    'q75': float(np.percentile(arr, 75)),
                }
            else:
                stats_dict[metric] = {}
                
        return stats_dict
    
    def generate_report(self) -> None:
        """Generate comprehensive training report."""
        console.print("\n[bold cyan]Training Analysis Report[/bold cyan]\n")
        
        # Convergence analysis
        convergence = self.analyze_convergence()
        conv_table = Table(title="Convergence Analysis")
        conv_table.add_column("Metric", style="cyan")
        conv_table.add_column("Value", style="green")
        
        conv_table.add_row("Converged", "Yes" if convergence['converged'] else "No")
        if convergence['initial_loss']:
            conv_table.add_row("Initial Loss", f"{convergence['initial_loss']:.6f}")
        if convergence['final_loss']:
            conv_table.add_row("Final Loss", f"{convergence['final_loss']:.6f}")
        if convergence['min_loss']:
            conv_table.add_row("Minimum Loss", f"{convergence['min_loss']:.6f}")
        conv_table.add_row("Improvement", f"{convergence['improvement_percent']:.1f}%")
        
        console.print(conv_table)
        
        # Statistics
        stats = self.calculate_statistics()
        stats_table = Table(title="\nMetric Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Mean", style="green")
        stats_table.add_column("Std", style="yellow")
        stats_table.add_column("Min", style="blue")
        stats_table.add_column("Max", style="red")
        
        for metric, values in stats.items():
            if values:
                stats_table.add_row(
                    metric,
                    f"{values['mean']:.4f}",
                    f"{values['std']:.4f}",
                    f"{values['min']:.4f}",
                    f"{values['max']:.4f}",
                )
        
        console.print(stats_table)
        
        # Anomalies
        anomalies = self.detect_anomalies()
        if anomalies:
            console.print("\n[bold red]Anomalies Detected:[/bold red]")
            for anomaly in anomalies[:10]:  # Show top 10
                console.print(f"  - {anomaly['type']} at step {anomaly.get('step', 'unknown')}: {anomaly['value']:.4f}")
        else:
            console.print("\n[green]No significant anomalies detected[/green]")
    
    def plot_metrics(self, output_dir: Optional[Path] = None) -> None:
        """Generate metric plots."""
        if output_dir is None:
            output_dir = self.log_path.parent / 'plots'
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Loss curve
        if self.metrics['loss']:
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 2, 1)
            plt.plot(self.metrics['loss'], alpha=0.7, label='Raw')
            if len(self.metrics['loss']) > 20:
                ma = pd.Series(self.metrics['loss']).rolling(window=20).mean()
                plt.plot(ma, label='MA(20)', linewidth=2)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.yscale('log')
            
            # Learning rate
            if self.metrics['learning_rate']:
                plt.subplot(2, 2, 2)
                plt.plot(self.metrics['learning_rate'])
                plt.xlabel('Step')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule')
                plt.yscale('log')
            
            # Gradient norm
            if self.metrics['gradient_norm']:
                plt.subplot(2, 2, 3)
                plt.plot(self.metrics['gradient_norm'])
                plt.xlabel('Step')
                plt.ylabel('Gradient Norm')
                plt.title('Gradient Norm')
                plt.axhline(y=10, color='r', linestyle='--', label='Explosion threshold')
                plt.legend()
            
            # Training speed
            if self.metrics['tokens_per_sec']:
                plt.subplot(2, 2, 4)
                plt.plot(self.metrics['tokens_per_sec'])
                plt.xlabel('Step')
                plt.ylabel('Tokens/sec')
                plt.title('Training Speed')
                avg_speed = np.mean(self.metrics['tokens_per_sec'])
                plt.axhline(y=avg_speed, color='g', linestyle='--', label=f'Avg: {avg_speed:.1f}')
                plt.legend()
            
            plt.tight_layout()
            plot_path = output_dir / 'training_metrics.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            console.print(f"\n[green]Plots saved to: {plot_path}[/green]")
            plt.close()
    
    def export_to_csv(self, output_path: Optional[Path] = None) -> None:
        """Export metrics to CSV for further analysis."""
        if output_path is None:
            output_path = self.log_path.with_suffix('.csv')
            
        # Create DataFrame
        data = {
            'step': range(len(self.metrics['loss'])),
        }
        
        for metric, values in self.metrics.items():
            if values and len(values) == len(data['step']):
                data[metric] = values
                
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        console.print(f"\n[green]Metrics exported to: {output_path}[/green]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("log_file", type=Path, help="Path to training log file")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--export-csv", action="store_true", help="Export metrics to CSV")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        console.print(f"[red]Error: Log file not found: {args.log_file}[/red]")
        sys.exit(1)
    
    # Analyze logs
    analyzer = TrainingLogAnalyzer(args.log_file)
    analyzer.parse_log_file()
    analyzer.generate_report()
    
    if args.plot:
        analyzer.plot_metrics(args.output_dir)
        
    if args.export_csv:
        analyzer.export_to_csv()


if __name__ == "__main__":
    main()