#!/usr/bin/env python3
"""
R3_certified_blindspot_visual_experiment.py

Certified discrete-vs-continuous physics residual experiment.

Goal:
-----
Demonstrate a numerical-analysis phenomenon:

A discrete collocation physics residual can be exactly (or nearly)
zero while the continuous PDE residual remains large.

PDE:
    -Delta u = f

Base solution:
    u0 = sin(pi*x) sin(pi*y)

Hidden mode:
    v = sin(m*pi*x) sin(m*pi*y)

For a carefully selected collocation grid:
    L_h(v) ~= 0

while:
    ||Delta v||_Omega >> 0

This script produces:
- structured collocation residual map
- continuous residual map
- random validation residual map
- amplitude sweep
- numerical diagnostics
- high resolution figures

Designed for SciML / Numerical Analysis communication.

Dependencies:
numpy
torch
matplotlib
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "full"], default="full")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path,
                   default=Path("r3_outputs"))
    p.add_argument("--seed", type=int, default=20260825)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def structured_grid(n):
    x = torch.linspace(0, 1, n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack(
        [X.reshape(-1), Y.reshape(-1)],
        dim=1
    )


def random_points(n):
    return torch.rand(n, 2)


def exact_solution(xy):
    x, y = xy[:, 0], xy[:, 1]
    return torch.sin(math.pi*x)*torch.sin(math.pi*y)


def hidden_mode(xy, m):
    x, y = xy[:, 0], xy[:, 1]
    return torch.sin(m*math.pi*x)*torch.sin(m*math*pi*y)


def hidden_mode_fixed(xy, m):
    x, y = xy[:, 0], xy[:, 1]
    return torch.sin(m*math.pi*x)*torch.sin(m*math.pi*y)


def laplacian(func, xy):
    xy = xy.clone().requires_grad_(True)

    u = func(xy)

    g = torch.autograd.grad(
        u,
        xy,
        torch.ones_like(u),
        create_graph=True
    )[0]

    uxx = torch.autograd.grad(
        g[:, 0],
        xy,
        torch.ones_like(g[:, 0]),
        create_graph=True
    )[0][:, 0]

    uyy = torch.autograd.grad(
        g[:, 1],
        xy,
        torch.ones_like(g[:, 1]),
        create_graph=True
    )[0][:, 1]

    return uxx + uyy


def residual_for_mode(points, m, amplitude):
    def trial(x):
        return (
            exact_solution(x)
            +
            amplitude * hidden_mode_fixed(x, m)
        )

    r = -laplacian(trial, points) - (
        2 * math.pi**2 * exact_solution(points)
    )

    return r.detach()


def save_heatmap(points, values, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=values,
        s=4
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def main():

    args = parse_args()

    set_seed(args.seed)

    torch.set_default_dtype(torch.float64)

    device = get_device(args.device)

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.mode == "quick":
        grid_n = 32
        random_n = 5000
    else:
        grid_n = 64
        random_n = 50000

    m = 33
    amplitude = 0.005

    structured = structured_grid(grid_n).to(device)
    random = random_points(random_n).to(device)

    start = time.perf_counter()

    r_struct = residual_for_mode(
        structured,
        m,
        amplitude
    )

    r_random = residual_for_mode(
        random,
        m,
        amplitude
    )

    dense = structured_grid(
        200
    ).to(device)

    r_dense = residual_for_mode(
        dense,
        m,
        amplitude
    )

    runtime = time.perf_counter()-start

    results = {
        "device": str(device),
        "grid_points": len(structured),
        "random_points": len(random),
        "mode": m,
        "amplitude": amplitude,
        "structured_rms":
            float(torch.sqrt(torch.mean(r_struct**2))),
        "random_rms":
            float(torch.sqrt(torch.mean(r_random**2))),
        "continuous_grid_rms":
            float(torch.sqrt(torch.mean(r_dense**2))),
        "runtime_seconds": runtime
    }

    save_heatmap(
        structured.cpu().numpy(),
        r_struct.abs().cpu().numpy(),
        "Structured collocation residual",
        args.output / "structured_residual.png"
    )

    save_heatmap(
        dense.cpu().numpy(),
        r_dense.abs().cpu().numpy(),
        "Continuous residual map",
        args.output / "continuous_residual.png"
    )

    with open(args.output / "r3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
