#!/usr/bin/env python3
"""QLoRA fine-tuning entry-point.

This is *not* a full implementation – it only parses the CLI arguments and
prints them back so you can iterate.
"""
import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="gemma_base.gguf")
    p.add_argument("--head", default="xnet")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--gt-regularizer", type=Path)
    p.add_argument("--step-hint", action="store_true")
    p.add_argument("--lambda_hint", type=float, default=0.3)
    return p


def main():
    args = build_parser().parse_args()
    print("[Stub] Starting QLoRA fine-tuning with args:", vars(args))
    # TODO: integrate PEFT / bitsandbytes / transformers training loop


if __name__ == "__main__":
    main()
