# GPU Training Monitoring System

Real-time GPU monitoring solution for training workloads with comprehensive metrics tracking and visualization.

## Features

### Core Monitoring Capabilities
- **GPU Metrics**: Utilization, VRAM usage, temperature, power draw, fan speed, clock speeds
- **Training Metrics**: Loss, learning rate, gradient norm, throughput, step duration
- **System Metrics**: CPU usage, RAM usage
- **Data Persistence**: Automatic logging to JSONL files for post-analysis
- **Prometheus Integration**: Export metrics for Grafana dashboards
- **Structured Logging**: JSON-formatted logs with correlation IDs for distributed tracing

## Components

### 1. GPU Monitor (`monitoring/gpu_monitor.py`)
Core monitoring engine with:
- Real-time GPU metrics collection via NVML
- Training metrics integration
- Prometheus metrics export
- Structured logging with correlation IDs
- Automatic data persistence

### 2. GPU Dashboard (`monitoring/gpu_dashboard.py`)
Interactive terminal dashboard with:
- Real-time metrics visualization
- ASCII graphs for historical data
- Color-coded alerts for thresholds
- Keyboard shortcuts for control

### 3. Monitored Training (`training/qlora_finetune_monitored.py`)
Enhanced training script with:
- Automatic GPU monitoring integration
- Training callbacks for metrics collection
- Comprehensive summary reports
- Prometheus metrics export

## Usage

### Basic GPU Monitoring
```bash
# Start basic monitoring
python -m monitoring.gpu_monitor --gpu 0 --interval 1.0

# With Prometheus export
python -m monitoring.gpu_monitor --gpu 0 --prometheus-port 9090
```

### Interactive Dashboard
```bash
# Launch interactive dashboard
python -m monitoring.gpu_dashboard --gpu 0 --interval 0.5

# Dashboard controls:
# q - Quit
# s - Save metrics to file
# r - Reset history
```

### Monitored Training
```bash
# Run training with integrated monitoring
python training/qlora_finetune_monitored.py \
    --model-config gemma-3n \
    --data /path/to/dataset \
    --output-dir ./results \
    --gpu-index 0 \
    --monitoring-log-dir ./gpu_logs \
    --enable-prometheus \
    --prometheus-port 9090
```

### Example Script
```bash
# Run example demonstrations
python scripts/monitor_training_example.py --mode monitor  # Basic monitoring
python scripts/monitor_training_example.py --mode dashboard # Interactive dashboard
python scripts/monitor_training_example.py --mode train    # Simulated training
```

## Integration with Training Code

### Using TrainingMonitorCallback
```python
from monitoring.gpu_monitor import GPUMonitor, TrainingMonitorCallback

# Initialize monitor
gpu_monitor = GPUMonitor(gpu_index=0, sample_interval=1.0)
callback = TrainingMonitorCallback(gpu_monitor)

# Start monitoring
callback.on_train_begin()

# During training loop
for step in range(num_steps):
    callback.on_step_begin()
    
    # Your training code here
    loss = train_step()
    
    callback.on_step_end(
        step=step,
        epoch=epoch,
        loss=loss,
        lr=optimizer.param_groups[0]['lr'],
        grad_norm=grad_norm,
        throughput=samples_per_sec
    )

# End monitoring
callback.on_train_end()
```

### Using with Transformers Trainer
```python
from monitoring.gpu_monitor import GPUMonitor
from training.qlora_finetune_monitored import GPUMonitoringCallback

# Create monitoring callback
gpu_monitor = GPUMonitor(gpu_index=0)
monitoring_callback = GPUMonitoringCallback(gpu_monitor, log_interval=10)

# Add to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[monitoring_callback],
    # ... other args
)
```

## Prometheus Metrics

Exported metrics for Grafana:
- `gpu_utilization_percent`: GPU compute utilization
- `gpu_memory_used_mb`: GPU memory usage in MB
- `gpu_memory_total_mb`: Total GPU memory in MB
- `gpu_temperature_celsius`: GPU temperature
- `gpu_power_draw_watts`: Current power draw
- `gpu_fan_speed_percent`: Fan speed (if available)
- `training_loss`: Current training loss
- `training_steps_total`: Total training steps (counter)
- `training_epoch`: Current epoch
- `learning_rate`: Current learning rate
- `gradient_norm`: Gradient L2 norm
- `training_step_duration_seconds`: Step duration histogram
- `memory_allocation_mb`: PyTorch memory allocation summary

## Grafana Dashboard

Import the dashboard from `monitoring/grafana/dashboards/gpu-training-dashboard.json` for:
- Real-time GPU utilization graphs
- Temperature gauges with thresholds
- Power draw monitoring
- Memory usage tracking
- Training loss curves
- Learning rate schedules
- Step duration histograms

## Output Files

### GPU Metrics Log (`gpu_metrics_YYYYMMDD_HHMMSS.jsonl`)
```json
{
  "timestamp": "2025-07-30T12:34:56.789Z",
  "gpu_index": 0,
  "utilization_percent": 95.0,
  "memory_used_mb": 22500.0,
  "memory_total_mb": 24576.0,
  "memory_percent": 91.5,
  "temperature_celsius": 75.0,
  "power_draw_watts": 320.0,
  "fan_speed_percent": 65.0
}
```

### Training Metrics Log (`training_metrics_YYYYMMDD_HHMMSS.jsonl`)
```json
{
  "timestamp": "2025-07-30T12:34:56.789Z",
  "step": 1000,
  "epoch": 2.5,
  "loss": 0.1234,
  "learning_rate": 0.0001,
  "gradient_norm": 1.234,
  "throughput_samples_per_sec": 32.5,
  "memory_allocated_mb": 18000.0,
  "memory_reserved_mb": 20000.0
}
```

## Performance Considerations

1. **Sampling Interval**: Default 1.0s, can be reduced to 0.1s for fine-grained monitoring
2. **History Size**: Default 300 samples (5 minutes at 1s interval), adjustable
3. **Log Rotation**: Metrics saved every 60 samples to prevent memory growth
4. **Prometheus Scrape**: Recommended 15s interval to reduce overhead

## Troubleshooting

### Common Issues

1. **NVML Not Found**
   ```bash
   # Install nvidia-ml-py
   pip install nvidia-ml-py pynvml
   ```

2. **Permission Denied**
   ```bash
   # May need to run with sudo for some GPU metrics
   sudo python -m monitoring.gpu_monitor
   ```

3. **No GPU Detected**
   - Ensure NVIDIA drivers are installed
   - Check `nvidia-smi` works
   - Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

## Best Practices

1. **Correlation IDs**: Use correlation IDs in logs for distributed tracing
2. **Metric Aggregation**: Use Prometheus recording rules for expensive queries
3. **Alert Thresholds**: Set alerts for:
   - GPU temperature > 80°C
   - Memory usage > 90%
   - Training loss not decreasing
   - Gradient explosions (norm > 10)
4. **Data Retention**: Configure log rotation for long training runs
5. **Dashboard Layout**: Customize Grafana dashboard for your specific needs

## Future Enhancements

- [ ] Multi-GPU support with aggregated metrics
- [ ] Distributed training monitoring across nodes
- [ ] Model-specific metrics (attention weights, activation stats)
- [ ] Automatic anomaly detection
- [ ] Integration with MLflow/W&B
- [ ] Mobile app notifications for alerts
- [ ] Cost tracking (cloud GPU pricing)