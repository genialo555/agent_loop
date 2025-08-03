"""Typer CLI entry-point regroupant les commandes courantes.

Exemples :
    python cli.py training mix-datasets *.jsonl --out-dir data_mix
    python cli.py training finetune --data train.jsonl
    python cli.py inference serve --host 0.0.0.0 --port 8000
"""
import subprocess
import logging
from pathlib import Path
from typing import List, Optional
from __future__ import annotations

import typer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="agent-loop",
    help="Agent Loop CLI - Modern ML pipeline management",
    no_args_is_help=True
)
training_app = typer.Typer(help="Training pipeline commands")
inference_app = typer.Typer(help="Inference server commands")
app.add_typer(training_app, name="training")
app.add_typer(inference_app, name="inference")


@training_app.command("mix-datasets")
def mix_datasets_cmd(datasets: List[Path], out_dir: Path) -> None:
    """Thin wrapper around training.mix_datasets script."""
    script_path = Path("/home/jerem/agent_loop/models/training/mix_datasets.py")
    if not script_path.exists():
        typer.echo(f"Error: Script not found at {script_path}", err=True)
        raise typer.Exit(1)
    
    subprocess.run(["python", str(script_path), "--out-dir", str(out_dir), *map(str, datasets)], check=True)


@training_app.command("finetune")
def finetune(
    data: Path, 
    resume: Optional[Path] = typer.Option(None, help="Resume from checkpoint path")
) -> None:
    """Start QLoRA fine-tuning with the specified dataset."""
    script_path = Path("/home/jerem/agent_loop/models/training/qlora/qlora_finetune.py")
    if not script_path.exists():
        typer.echo(f"Error: Training script not found at {script_path}", err=True)
        raise typer.Exit(1)
    
    if not data.exists():
        typer.echo(f"Error: Dataset file not found at {data}", err=True)
        raise typer.Exit(1)
    
    cmd = ["python", str(script_path), "--data", str(data)]
    if resume:
        if not resume.exists():
            typer.echo(f"Error: Resume checkpoint not found at {resume}", err=True)
            raise typer.Exit(1)
        cmd += ["--resume", str(resume)]
    
    typer.echo(f"Starting fine-tuning with data: {data}")
    subprocess.run(cmd, check=True)


@inference_app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
    workers: int = typer.Option(1, help="Number of worker processes")
) -> None:
    """Start the FastAPI inference server."""
    app_path = "models.inference.app:app"
    
    # Setup logging directory
    log_dir = Path("/home/jerem/agent_loop/models/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["uvicorn", app_path, "--host", host, "--port", str(port)]
    
    if reload:
        cmd.append("--reload")
    
    if workers > 1 and not reload:
        cmd.extend(["--workers", str(workers)])
    
    typer.echo(f"Starting server at http://{host}:{port}")
    typer.echo(f"Logs will be stored in: {log_dir}")
    subprocess.run(cmd, check=True)


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo("Agent Loop CLI v2.0.0")
    typer.echo("Enhanced ML pipeline with modern Python standards")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled by user.")
        raise typer.Exit(130)
    except Exception as e:
        logger.error(f"CLI error: {e}")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
