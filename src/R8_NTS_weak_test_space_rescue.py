#!/usr/bin/env python3
"""
R8_NTS_weak_test_space_rescue.py

Prototype experiment:
Point collocation vs weak test spaces for detecting a hidden residual mode.

Methods:
1) Point residual loss
2) Fixed random weak probes
3) Adaptive compact RBF weak probes

Outputs:
- JSON metrics
- comparison plots

This is an experimental prototype, not a theorem.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="quick", choices=["quick", "full"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="r8_outputs")
    return p.parse_args()


def get_device(x):
    if x == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(x)


def seed():
    torch.manual_seed(0)
    np.random.seed(0)


def grid(n, device):
    x = torch.linspace(0, 1, n, device=device, dtype=torch.float64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.flatten(), Y.flatten()], dim=1)


def residual(points, m=33, amp=0.005):
    x = points[:, 0]
    y = points[:, 1]
    hidden = amp * torch.sin(m*np.pi*x)*torch.sin(m*np.pi*y)
    return 2*(m*np.pi)**2*hidden


def rbf_tests(points, centers, radius):
    d = torch.cdist(points, centers)
    return torch.exp(-(d/radius)**2)


def rms(x):
    return torch.sqrt(torch.mean(x*x)).item()


def run(cfg):
    seed()
    device = get_device(cfg.device)

    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)

    n = 32 if cfg.mode == "quick" else 64
    points = grid(n, device)

    residual_field = residual(points)

    # Method 1: point collocation
    point_measure = rms(residual_field)

    # Method 2: random weak probes
    k = 32 if cfg.mode == "quick" else 128
    random_centers = torch.rand((k,2), device=device, dtype=torch.float64)
    random_tests = rbf_tests(points, random_centers, 0.15)

    random_projection = torch.mean(
        random_tests * residual_field[:,None],
        dim=0
    )

    random_weak = torch.norm(random_projection).item()

    # Method 3: compact adaptive RBF test space
    # centers concentrated around high residual magnitude
    weights = residual_field.abs()
    idx = torch.topk(
        weights,
        min(k, len(weights))
    ).indices

    adaptive_centers = points[idx]

    adaptive_tests = rbf_tests(
        points,
        adaptive_centers,
        0.12
    )

    adaptive_projection = torch.mean(
        adaptive_tests * residual_field[:,None],
        dim=0
    )

    adaptive_weak = torch.norm(adaptive_projection).item()

    result = {
        "device": str(device),
        "grid": n,
        "mode_frequency": 33,
        "point_residual": point_measure,
        "random_weak_residual": random_weak,
        "adaptive_rbf_weak_residual": adaptive_weak,
        "ratios": {
            "point_over_random": point_measure/max(random_weak,1e-30),
            "point_over_adaptive": point_measure/max(adaptive_weak,1e-30)
        },
    }

    with open(out/"r8_summary.json","w") as f:
        json.dump(result,f,indent=2)

    labels=[
        "Point\ncollocation",
        "Random\nweak",
        "Adaptive RBF\nweak"
    ]

    values=[
        point_measure,
        random_weak,
        adaptive_weak
    ]

    plt.figure(figsize=(7,4),dpi=200)
    plt.bar(labels,values)
    plt.yscale("log")
    plt.ylabel("Residual measurement")
    plt.title("R8: Weak test space rescue experiment")
    plt.tight_layout()
    plt.savefig(out/"r8_comparison.png",dpi=300)
    plt.close()

    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    t=time.time()
    run(parse_args())
    print("runtime_seconds:",time.time()-t)
