"""Typer CLI entry-point regroupant les commandes courantes.

Exemples :
    python -m agent_loop.cli training mix-datasets *.jsonl --out-dir data_mix
    python -m agent_loop.cli training finetune --data train.jsonl
"""
import subprocess
from pathlib import Path
from typing import List

import typer

app = typer.Typer(name="agent-loop")
training_app = typer.Typer()
inference_app = typer.Typer()
app.add_typer(training_app, name="training")
app.add_typer(inference_app, name="inference")


@training_app.command("mix-datasets")
def mix_datasets_cmd(datasets: List[Path], out_dir: Path):  # noqa: D401
    """Thin wrapper around training.mix_datasets script."""
    subprocess.run(["python", "-m", "agent_loop.training.mix_datasets", "--out-dir", str(out_dir), *map(str, datasets)], check=True)


@training_app.command("finetune")
def finetune(data: Path, resume: Path = typer.Option(None)):  # noqa: D401
    cmd = ["python", "-m", "agent_loop.training.qlora_finetune", "--data", str(data)]
    if resume:
        cmd += ["--resume", str(resume)]
    subprocess.run(cmd, check=True)


@inference_app.command("serve")
def serve(host: str = "0.0.0.0", port: int = 8000):  # noqa: D401
    subprocess.run(["uvicorn", "agent_loop.inference.api:app", "--host", host, "--port", str(port)], check=True)


if __name__ == "__main__":
    app()
