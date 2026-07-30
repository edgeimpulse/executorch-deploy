"""Standalone ExecuTorch runtime harness.

Loads a `.pte` program and runs a single forward pass. Use this to smoke-test the
deployment package on a Linux/ARM target that has the ExecuTorch Python runtime
installed.

Usage:
    python app/run_pte.py model.pte --labels labels.txt
"""

from __future__ import annotations

import argparse

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ExecuTorch .pte program")
    parser.add_argument("model", help="Path to the .pte program")
    parser.add_argument("--labels", default=None, help="Optional labels.txt file")
    parser.add_argument("--shape", default="1,3,32,32",
                        help="Comma-separated input shape (NCHW)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from executorch.runtime import Runtime

    shape = tuple(int(x) for x in args.shape.split(","))
    example = torch.randn(*shape)

    runtime = Runtime.get()
    program = runtime.load_program(args.model)
    method = program.load_method("forward")
    outputs = method.execute([example])

    logits = outputs[0]
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    top = int(torch.argmax(probs))

    labels = None
    if args.labels:
        with open(args.labels) as f:
            labels = [line.strip() for line in f if line.strip()]

    label = labels[top] if labels and top < len(labels) else str(top)
    print(f"Predicted class: {label} (index {top}, p={probs[top]:.4f})")


if __name__ == "__main__":
    main()
