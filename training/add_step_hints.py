#!/usr/bin/env python3
"""Insert <step_hint> blocks into ToolBench-style JSONL datasets.

Usage:
    python add_step_hints.py input.jsonl output.jsonl

Each record is expected to contain a `prompt` field.  We prepend a synthetic
plan between <step_hint> … </step_hint> tags before the first CALL_TOOL.
"""
import argparse
import json
from pathlib import Path


STEP_HINT_TOKEN = "<step_hint>"


def generate_hint(prompt: str) -> str:
    """Very naive heuristic to create a placeholder hint from the prompt."""
    return f"Plan: {prompt.split('.')[0].strip()} …"


def process_file(src: Path, dst: Path) -> None:
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            record = json.loads(line)
            prompt = record.get("prompt", "")
            hint = generate_hint(prompt)
            record["prompt"] = f"{STEP_HINT_TOKEN} {hint} {STEP_HINT_TOKEN.replace('<', '</')}\n" + prompt
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert Step-Hints into JSONL dataset")
    parser.add_argument("src", type=Path, help="Input JSONL file")
    parser.add_argument("dst", type=Path, help="Output JSONL file with hints")
    args = parser.parse_args()
    process_file(args.src, args.dst)
