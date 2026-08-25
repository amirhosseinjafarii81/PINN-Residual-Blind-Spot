#!/usr/bin/env python3
"""
R4_PINN_null_mode_training_probe.py

Goal:
-----
Connect the certified numerical null-mode phenomenon (R3-v2)
to an actual optimization experiment.

Question:
Can a PINN reduce its empirical collocation residual while hiding
error in a discrete null mode?

Model:
    u_theta(x,y) = u_exact(x,y) + a_theta * v_null(x,y)

where:
    v_null = sin((N+1)pi*x) sin((N+1)pi*y)

The coefficient a_theta is learned by minimizing only the structured
collocation physics loss.

Measurements:
- structured residual
- random validation residual
- continuous residual
- learned hidden-mode amplitude
- optimization trajectory

This isolates the sampling issue from neural-network representation.

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
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=Path("r4_outputs"))
    p.add_argument("--seed", type=int, default=20260825)
    return p.parse_args()


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interior_grid(n):
    x = torch.arange(1, n + 1, dtype=torch.float64)/(n+1)
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


def null_mode(xy, m):
    x, y = xy[:, 0], xy[:, 1]
    return torch.sin(m*math.pi*x)*torch.sin(m*math.pi*y)


def laplacian(func, xy):
    xy = xy.clone().requires_grad_(True)
    u = func(xy)

    g = torch.autograd.grad(
        u, xy,
        torch.ones_like(u),
        create_graph=True
    )[0]

    uxx = torch.autograd.grad(
        g[:,0], xy,
        torch.ones_like(g[:,0]),
        create_graph=True
    )[0][:,0]

    uyy = torch.autograd.grad(
        g[:,1], xy,
        torch.ones_like(g[:,1]),
        create_graph=True
    )[0][:,1]

    return uxx + uyy


def make_model(m, device):
    # only unknown is null-mode amplitude
    a = torch.nn.Parameter(
        torch.tensor(0.0, dtype=torch.float64, device=device)
    )
    return a


def evaluate(a, points, m):
    def trial(x):
        return exact_u(x) + a*null_mode(x, m)

    r = -laplacian(trial, points) - 2*math.pi**2*exact_u(points)

    return float(torch.sqrt(torch.mean(r*r)).detach().cpu())


def main():

    args = parse_args()
    seed_all(args.seed)

    torch.set_default_dtype(torch.float64)

    device = get_device(args.device)

    args.output.mkdir(parents=True, exist_ok=True)

    n = 32 if args.mode == "quick" else 64
    m = n + 1

    train_points = interior_grid(n).to(device)
    random = random_points(n*n*10).to(device)
    dense = dense_grid(300).to(device)

    a = make_model(m, device)

    opt = torch.optim.Adam([a], lr=1e-2)

    epochs = 1000 if args.mode == "quick" else 5000

    history = []

    start = time.perf_counter()

    for epoch in range(epochs):

        def trial(x):
            return exact_u(x) + a*null_mode(x, m)

        r = -laplacian(trial, train_points) - (
            2*math.pi**2*exact_u(train_points)
        )

        loss = torch.mean(r*r)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 50 == 0:
            history.append({
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "a": float(a.detach().cpu()),
                "random_residual": evaluate(a, random, m),
                "continuous_residual": evaluate(a, dense, m)
            })

    runtime = time.perf_counter()-start

    result = {
        "device": str(device),
        "n": n,
        "mode": m,
        "epochs": epochs,
        "learned_amplitude": float(a.detach().cpu()),
        "structured_residual": evaluate(a, train_points, m),
        "random_residual": evaluate(a, random, m),
        "continuous_residual": evaluate(a, dense, m),
        "runtime_seconds": runtime,
        "history": history
    }

    with open(args.output/"r4_results.json","w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
