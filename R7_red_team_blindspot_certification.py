#!/usr/bin/env python3
"""
R7 Red Team Blind Spot Certification

Purpose:
Attack the R6 PINN residual blind spot claim using:
- frequency sweep
- sampling comparison
- precision comparison

This is a certification experiment, not a proof.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="quick", choices=["quick", "full"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="r7_outputs")
    return p.parse_args()


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def hidden_mode(points, m, a=0.005):
    x = points[:, 0]
    y = points[:, 1]
    return a * torch.sin(m * np.pi * x) * torch.sin(m * np.pi * y)


def laplacian_hidden(points, m, a=0.005):
    u = hidden_mode(points, m, a)
    return -2.0 * (m * np.pi) ** 2 * u


def make_grid(n, device, dtype):
    x = torch.linspace(0, 1, n, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=1)


def make_random(n, device, dtype):
    return torch.rand((n, 2), device=device, dtype=dtype)


def make_sobol(n, device, dtype):
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=0)
    return engine.draw(n).to(device=device, dtype=dtype)


def rms(x):
    return torch.sqrt(torch.mean(x * x)).item()


def evaluate(points, m):
    return rms(laplacian_hidden(points, m))


def run(args):
    set_seed()

    device = get_device(args.device)
    dtype = torch.float64

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    n = 32 if args.mode == "quick" else 64
    points = n * n

    freqs = [1, 5, 9, 17, 33, 49] if args.mode == "quick" else [1,5,9,17,33,49,65,81]

    summary = {
        "device": str(device),
        "dtype": str(dtype),
        "grid": n,
        "frequency_results": [],
        "sampling_results": [],
    }

    t0 = time.time()

    for m in freqs:
        structured = evaluate(make_grid(n, device, dtype), m)
        random = evaluate(make_random(points, device, dtype), m)
        sobol = evaluate(make_sobol(points, device, dtype), m)

        summary["frequency_results"].append({
            "mode": m,
            "structured": structured,
            "random": random,
            "sobol": sobol,
            "random_ratio": random / max(structured, 1e-30),
            "sobol_ratio": sobol / max(structured, 1e-30),
        })

    for dtype_test in [torch.float32, torch.float64]:
        grid = make_grid(n, device, dtype_test)
        summary["sampling_results"].append({
            "dtype": str(dtype_test),
            "structured": evaluate(grid, 33)
        })

    summary["runtime_seconds"] = time.time() - t0

    with open(out / "r7_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run(parse_args())
