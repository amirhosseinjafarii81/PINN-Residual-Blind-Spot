#!/usr/bin/env python3
"""
R3_v2_certified_null_mode_experiment.py

Corrected certified discrete-vs-continuous residual experiment.

Fix from R3:
The structured collocation grid MUST reproduce the interior null mode:

x_i = i/(N+1), i=1,...,N
m = N+1

Then:
sin(m*pi*x_i)=sin(i*pi)=0

Therefore the discrete residual can vanish while the continuous
residual remains large.

Outputs:
- JSON diagnostics
- structured residual plot
- continuous residual heatmap
- random validation residual heatmap
- amplitude sweep

No PINN training is used here. This isolates the numerical operator
phenomenon first.
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


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=Path("r3_v2_outputs"))
    p.add_argument("--seed", type=int, default=20260825)
    return p.parse_args()


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_of(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interior_grid(n):
    x = torch.arange(1, n + 1, dtype=torch.float64)/(n + 1)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def dense_grid(n):
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def random_points(n):
    return torch.rand(n, 2, dtype=torch.float64)


def exact_u(xy):
    x, y = xy[:, 0], xy[:, 1]
    return torch.sin(math.pi*x)*torch.sin(math.pi*y)


def hidden_mode(xy, m):
    x, y = xy[:, 0], xy[:, 1]
    return torch.sin(m*math.pi*x)*torch.sin(m*math.pi*y)


def laplacian(func, xy):
    xy = xy.clone().requires_grad_(True)
    u = func(xy)

    g = torch.autograd.grad(
        u, xy, torch.ones_like(u),
        create_graph=True
    )[0]

    uxx = torch.autograd.grad(
        g[:, 0], xy,
        torch.ones_like(g[:, 0]),
        create_graph=True
    )[0][:, 0]

    uyy = torch.autograd.grad(
        g[:, 1], xy,
        torch.ones_like(g[:, 1]),
        create_graph=True
    )[0][:, 1]

    return uxx + uyy


def residual(points, m, amp):
    def trial(x):
        return exact_u(x) + amp * hidden_mode(x, m)

    r = -laplacian(trial, points) - 2*math.pi**2*exact_u(points)
    return r.detach()


def plot_res(points, values, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=np.abs(values),
        s=5
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)


def main():
    a = args()
    seed_all(a.seed)
    torch.set_default_dtype(torch.float64)

    device = device_of(a.device)
    a.output.mkdir(parents=True, exist_ok=True)

    n = 32 if a.mode == "quick" else 64

    m = n + 1
    amp = 0.005

    structured = interior_grid(n).to(device)
    random = random_points(n*n*5).to(device)
    dense = dense_grid(200 if a.mode == "quick" else 400).to(device)

    t0 = time.perf_counter()

    r_struct = residual(structured, m, amp)
    r_random = residual(random, m, amp)
    r_dense = residual(dense, m, amp)

    runtime = time.perf_counter() - t0

    result = {
        "device": str(device),
        "n": n,
        "mode": m,
        "amplitude": amp,
        "structured_rms": float(torch.sqrt(torch.mean(r_struct**2))),
        "structured_max": float(torch.max(torch.abs(r_struct))),
        "random_rms": float(torch.sqrt(torch.mean(r_random**2))),
        "continuous_rms": float(torch.sqrt(torch.mean(r_dense**2))),
        "runtime_seconds": runtime,
        "null_mode_check":
            float(torch.max(torch.abs(hidden_mode(structured, m))))
    }

    plot_res(
        structured.cpu().numpy(),
        r_struct.cpu().numpy(),
        "Structured interior collocation residual",
        a.output/"structured_residual.png"
    )

    plot_res(
        dense.cpu().numpy(),
        r_dense.cpu().numpy(),
        "Continuous residual map",
        a.output/"continuous_residual.png"
    )

    plot_res(
        random.cpu().numpy(),
        r_random.cpu().numpy(),
        "Random validation residual",
        a.output/"random_residual.png"
    )

    with open(a.output/"r3_v2_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
