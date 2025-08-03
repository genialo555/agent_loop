#!/usr/bin/env python3
"""Merge, dedupe and split multiple JSONL datasets into train/val sets.

Example:
    python mix_datasets.py --out-dir datasets_mixed \
        ToolBench.jsonl AgenticAI.jsonl synthetic.jsonl
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_all(paths: List[Path]) -> List[Dict[str, Any]]:
    buf = []
    for p in paths:
        with p.open() as f:
            buf.extend(json.loads(line) for line in f)
    return buf


def dedupe(records):
    seen = set()
    uniq = []
    for rec in records:
        h = hash(rec.get("prompt", ""))
        if h not in seen:
            seen.add(h)
            uniq.append(rec)
    return uniq


def split(records, val_ratio=0.2):
    random.shuffle(records)
    n_val = int(len(records) * val_ratio)
    return records[n_val:], records[:n_val]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    records = dedupe(load_all(args.datasets))
    train, val = split(records)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "train.jsonl").open("w") as ftrain:
        for r in train:
            ftrain.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (args.out_dir / "val.jsonl").open("w") as fval:
        for r in val:
            fval.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
