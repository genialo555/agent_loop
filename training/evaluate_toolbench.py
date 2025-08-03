#!/usr/bin/env python3
"""Offline evaluation script against ToolBench.
This is currently a placeholder – implement metric computation later.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print("[Stub] Evaluating", args.model)


if __name__ == "__main__":
    main()
