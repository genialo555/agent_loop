#!/usr/bin/env python3
"""
Example script demonstrating GPU monitoring integration.
Shows how to use the monitoring tools during training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import subprocess
from pathlib import Path
import signal
import threading

from monitoring.grafana.dashboards.gpu_monitor import GPUMonitor, TrainingMonitorCallback


def run_dashboard():
    """Run the GPU dashboard in a separate process."""
    cmd = [sys.executable, "-m", "monitoring.gpu_dashboard", "--interval", "0.5"]
    return subprocess.Popen(cmd)


def run_prometheus_exporter(port: int = 9090):
    """Start Prometheus metrics exporter."""
    from prometheus_client import start_http_server
    start_http_server(port)
    print(f"Prometheus metrics available at http://localhost:{port}")


def simulate_training_with_monitoring():
    """Simulate a training loop with GPU monitoring."""
    # Initialize GPU monitor
    monitor = GPUMonitor(gpu_index=0, sample_interval=1.0)
    callback = TrainingMonitorCallback(monitor)
    
    # Start monitoring
    callback.on_train_begin()
    
    print("Starting simulated training with GPU monitoring...")
    print("=" * 60)
    
    try:
        # Simulate training loop
        for epoch in range(3):
            for step in range(100):
                callback.on_step_begin()
                
                # Simulate work (replace with actual training)
                time.sleep(0.1)
                
                # Simulate metrics
                loss = 2.5 - (epoch * 0.5) - (step * 0.001)
                lr = 0.001 * (0.95 ** (epoch * 100 + step))
                grad_norm = 1.5 + (0.5 * (step % 10) / 10)
                throughput = 32 + (step % 5)
                
                # Log metrics
                callback.on_step_end(
                    step=epoch * 100 + step,
                    epoch=epoch + step/100,
                    loss=loss,
                    lr=lr,
                    grad_norm=grad_norm,
                    throughput=throughput
                )
                
                # Print progress every 10 steps
                if step % 10 == 0:
                    monitor.print_live_stats()
                    print(f"Epoch {epoch}, Step {step}: Loss={loss:.4f}, LR={lr:.2e}")
                    print("-" * 60)
        
    finally:
        # Stop monitoring and get summary
        callback.on_train_end()
        summary = monitor.get_summary()
        
        print("\n" + "=" * 60)
        print("MONITORING SUMMARY")
        print("=" * 60)
        
        if 'gpu_stats' in summary:
            gpu_stats = summary['gpu_stats']
            print(f"GPU Utilization: {gpu_stats['utilization']['mean']:.1f}% (avg), {gpu_stats['utilization']['max']:.1f}% (max)")
            print(f"Temperature: {gpu_stats['temperature']['mean']:.1f}°C (avg), {gpu_stats['temperature']['max']:.1f}°C (max)")
            print(f"Power Draw: {gpu_stats['power']['mean']:.1f}W (avg), {gpu_stats['power']['max']:.1f}W (max)")
            print(f"Memory Used: {gpu_stats['memory_used_mb']['mean']:.0f} MB (avg), {gpu_stats['memory_used_mb']['max']:.0f} MB (max)")
        
        if 'training_stats' in summary:
            train_stats = summary['training_stats']
            print(f"\nTraining Steps: {train_stats['total_steps']}")
            print(f"Final Epoch: {train_stats['final_epoch']:.2f}")
            print(f"Loss: {train_stats['loss']['initial']:.4f} → {train_stats['loss']['final']:.4f}")
        
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPU Monitoring Example")
    parser.add_argument("--mode", choices=["monitor", "dashboard", "train"], 
                       default="monitor",
                       help="Mode: monitor (basic monitoring), dashboard (interactive), train (monitored training)")
    parser.add_argument("--prometheus-port", type=int, default=9090,
                       help="Prometheus metrics port")
    parser.add_argument("--enable-prometheus", action="store_true",
                       help="Enable Prometheus metrics export")
    
    args = parser.parse_args()
    
    # Start Prometheus if requested
    if args.enable_prometheus:
        prometheus_thread = threading.Thread(
            target=run_prometheus_exporter, 
            args=(args.prometheus_port,),
            daemon=True
        )
        prometheus_thread.start()
        time.sleep(1)  # Let server start
    
    if args.mode == "monitor":
        # Basic monitoring example
        monitor = GPUMonitor(gpu_index=0)
        monitor.start_monitoring()
        
        print("GPU monitoring started. Press Ctrl+C to stop.")
        print("Metrics are being collected every second.")
        
        try:
            while True:
                time.sleep(5)
                monitor.print_live_stats()
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            summary = monitor.get_summary()
            print(f"\nMonitoring stopped. Summary:")
            print(f"Duration: {summary['monitoring_duration_seconds']}s")
            print(f"Samples collected: {summary['total_samples']}")
    
    elif args.mode == "dashboard":
        # Run interactive dashboard
        dashboard_proc = run_dashboard()
        print("GPU Dashboard started. Press Ctrl+C to stop.")
        
        try:
            dashboard_proc.wait()
        except KeyboardInterrupt:
            dashboard_proc.terminate()
            dashboard_proc.wait()
    
    elif args.mode == "train":
        # Run simulated training with monitoring
        simulate_training_with_monitoring()
    
    print("\nExample completed.")


if __name__ == "__main__":
    main()