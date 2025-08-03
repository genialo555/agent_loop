#!/usr/bin/env python3
"""
QLoRA Fine-tuning Pipeline with Real-time GPU Monitoring
Extends the base training script with comprehensive GPU and training metrics monitoring.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlora_finetune import *
from monitoring.grafana.dashboards.gpu_monitor import GPUMonitor, TrainingMonitorCallback
import structlog
from pathlib import Path
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


# Configure structured logging for training
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


class GPUMonitoringCallback(TrainerCallback):
    """Transformers Trainer callback for GPU monitoring integration."""
    
    def __init__(self, gpu_monitor: GPUMonitor, log_interval: int = 10):
        self.gpu_monitor = gpu_monitor
        self.log_interval = log_interval
        self.training_monitor = TrainingMonitorCallback(gpu_monitor)
        self.step_start_time = None
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                      control: TrainerControl, **kwargs):
        """Start GPU monitoring when training begins."""
        self.training_monitor.on_train_begin()
        logger.info("gpu_monitoring_started", 
                   correlation_id=f"training_{int(time.time())}",
                   model=kwargs.get('model', {}).config.name_or_path if 'model' in kwargs else "unknown")
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """Stop monitoring and print summary when training ends."""
        self.training_monitor.on_train_end()
        
        # Get and log summary
        summary = self.gpu_monitor.get_summary()
        logger.info("training_completed", 
                   summary=summary,
                   total_steps=state.global_step,
                   best_metric=state.best_metric,
                   best_model_checkpoint=state.best_model_checkpoint)
        
        # Print summary to console
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Total Steps: {state.global_step}")
        print(f"Final Loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")
        
        if 'gpu_stats' in summary:
            gpu_stats = summary['gpu_stats']
            print(f"\nGPU Statistics:")
            print(f"  Average Utilization: {gpu_stats['utilization']['mean']:.1f}%")
            print(f"  Peak Temperature: {gpu_stats['temperature']['max']:.1f}°C")
            print(f"  Average Power Draw: {gpu_stats['power']['mean']:.1f}W")
            print(f"  Peak Memory Usage: {gpu_stats['memory_used_mb']['max']:.0f} MB")
        
        print("="*80 + "\n")
        
        return control
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        """Track step start time."""
        self.training_monitor.on_step_begin()
        return control
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState,
                   control: TrainerControl, **kwargs):
        """Log training step metrics."""
        # Get current metrics from log history
        if state.log_history:
            latest_log = state.log_history[-1]
            loss = latest_log.get('loss', 0.0)
            lr = latest_log.get('learning_rate', args.learning_rate)
            epoch = latest_log.get('epoch', state.epoch)
            
            # Calculate gradient norm if available
            grad_norm = None
            if 'grad_norm' in latest_log:
                grad_norm = latest_log['grad_norm']
            
            # Calculate throughput
            throughput = None
            if hasattr(self, '_last_step_time') and hasattr(self, '_last_step'):
                time_diff = time.time() - self._last_step_time
                step_diff = state.global_step - self._last_step
                if time_diff > 0 and step_diff > 0:
                    samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
                    throughput = (step_diff * samples_per_step) / time_diff
            
            self._last_step_time = time.time()
            self._last_step = state.global_step
            
            # Log metrics
            self.training_monitor.on_step_end(
                step=state.global_step,
                epoch=epoch,
                loss=loss,
                lr=lr,
                grad_norm=grad_norm,
                throughput=throughput
            )
            
            # Print live stats every N steps
            if state.global_step % self.log_interval == 0:
                self.gpu_monitor.print_live_stats()
                print(f"Step {state.global_step}: Loss={loss:.4f}, LR={lr:.2e}")
        
        return control
    
    def on_log(self, args: TrainingArguments, state: TrainerState,
              control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Hook into trainer logging."""
        # Log with correlation ID for distributed tracing
        logger.info("training_metrics",
                   step=state.global_step,
                   metrics=logs,
                   correlation_id=f"step_{state.global_step}")
        return control


def train_model_with_monitoring(
    model,
    tokenizer,
    dataset: Dataset,
    config: QLoRAConfig,
    output_dir: str,
    resume_from_checkpoint: Optional[str] = None,
    gpu_index: int = 0,
    monitoring_log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Enhanced training function with GPU monitoring."""
    logger.info("starting_monitored_training",
               model=config.model_name,
               output_dir=output_dir,
               gpu_index=gpu_index)
    
    # Initialize GPU monitor
    monitor_log_dir = monitoring_log_dir or Path(output_dir) / "gpu_monitoring"
    gpu_monitor = GPUMonitor(
        gpu_index=gpu_index,
        sample_interval=1.0,  # Sample every second
        history_size=3600,    # Keep 1 hour of history
        log_dir=monitor_log_dir
    )
    
    # Create monitoring callback
    monitoring_callback = GPUMonitoringCallback(gpu_monitor, log_interval=10)
    
    # Training arguments
    training_args = config.get_training_args(output_dir)
    
    # Add monitoring callback to existing callbacks
    callbacks = [monitoring_callback]
    
    # Data collator for completion-only training
    response_template = "\n### Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    # Setup SFTTrainer with monitoring
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field=config.dataset_text_field,
        max_seq_length=config.max_seq_length,
        data_collator=collator,
        tokenizer=tokenizer,
        packing=config.packing,
        callbacks=callbacks,
    )
    
    # Train the model
    start_time = time.time()
    
    try:
        if resume_from_checkpoint:
            logger.info("resuming_from_checkpoint", checkpoint=resume_from_checkpoint)
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
        
        training_time = time.time() - start_time
        
        # Save the final model
        trainer.save_model()
        trainer.save_state()
        
        # Get comprehensive training info
        train_results = trainer.state.log_history
        final_loss = train_results[-1].get("train_loss", 0.0) if train_results else 0.0
        
        # Get monitoring summary
        monitoring_summary = gpu_monitor.get_summary()
        
        training_info = {
            "model_name": config.model_name,
            "training_time": round(training_time, 2),
            "final_loss": final_loss,
            "total_steps": trainer.state.global_step,
            "config": config.get_model_info(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_monitoring": monitoring_summary,
        }
        
        # Save comprehensive training info
        with open(Path(output_dir) / "training_info_monitored.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("training_completed_successfully",
                   training_time=training_time,
                   final_loss=final_loss,
                   monitoring_summary=monitoring_summary)
        
        return training_info
        
    except Exception as e:
        logger.error("training_failed", error=str(e), traceback=True)
        raise
    finally:
        # Ensure monitoring is stopped
        gpu_monitor.stop_monitoring()


def main():
    """Main training pipeline with GPU monitoring."""
    # Extend base parser
    parser = build_parser()
    parser.add_argument("--gpu-index", type=int, default=0,
                       help="GPU index to monitor")
    parser.add_argument("--monitoring-log-dir", type=str,
                       help="Directory for GPU monitoring logs")
    parser.add_argument("--enable-prometheus", action="store_true",
                       help="Enable Prometheus metrics export")
    parser.add_argument("--prometheus-port", type=int, default=9090,
                       help="Prometheus metrics port")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Start Prometheus server if requested
    if args.enable_prometheus:
        from prometheus_client import start_http_server
        start_http_server(args.prometheus_port)
        logger.info("prometheus_metrics_server_started", port=args.prometheus_port)
    
    # Get configuration
    config = get_model_config(args.model_config, args.model_name)
    
    # Override config with CLI arguments
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    
    # Dataset configuration
    dataset_config = DatasetConfig(
        dataset_name=args.data if not os.path.exists(args.data) else None,
        dataset_path=args.data if os.path.exists(args.data) else None,
        text_column=args.text_column,
        max_seq_length=args.max_seq_length,
    )
    
    dataset_config.validate()
    
    # Setup power management
    config.setup_power_management()
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Initialize W&B
    if not args.no_wandb and not args.dry_run:
        setup_wandb(config, args.wandb_project, args.run_name)
    
    logger.info("monitored_training_pipeline_started",
               model=config.model_name,
               dataset=args.data,
               output=args.output_dir,
               gpu_index=args.gpu_index)
    
    print("=" * 80)
    print("QLoRA Fine-tuning Pipeline with GPU Monitoring")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"GPU: {args.gpu_index}")
    print(f"Monitoring Logs: {args.monitoring_log_dir or os.path.join(args.output_dir, 'gpu_monitoring')}")
    print("=" * 80)
    
    if args.dry_run:
        logger.info("dry_run_mode_exiting")
        return
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Load dataset
        dataset = load_and_prepare_dataset(dataset_config, tokenizer)
        
        # Train model with monitoring
        training_info = train_model_with_monitoring(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume,
            gpu_index=args.gpu_index,
            monitoring_log_dir=args.monitoring_log_dir,
        )
        
        print("\n" + "🎉" * 40)
        print("Training completed successfully!")
        print(f"Model saved to: {args.output_dir}")
        print(f"Monitoring logs saved to: {args.monitoring_log_dir or os.path.join(args.output_dir, 'gpu_monitoring')}")
        print("🎉" * 40 + "\n")
        
        # Log to W&B
        if not args.no_wandb:
            wandb.log({"final_training_info": training_info})
            wandb.finish()
            
    except Exception as e:
        logger.error("training_pipeline_failed", error=str(e), traceback=True)
        if not args.no_wandb:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()