#!/usr/bin/env python3
"""
R4_v2_null_mode_injection_gradient_probe.py

Purpose
-------
Test the actual optimization consequence of a discrete collocation null mode.

R3-v2 established:
    ||L_h(v_null)|| ~= 0
    ||L(v_null)||_Omega >> 0

R4 failed because:
    a_init = 0
    dL_h/da = 0

Therefore the optimizer never entered the invisible direction.

R4-v2 fixes this by injecting controlled hidden-mode amplitudes.

Experiment:
    u_theta = u_exact + a_theta * v_null

where:
    v_null = sin((N+1) pi x) sin((N+1) pi y)

We measure:
- evolution of hidden amplitude
- structured physics loss
- continuous residual
- random validation residual
- gradient in hidden direction

The scientific question:
Can a collocation-only physics objective detect and remove
a component that lies in its discrete null space?

"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=Path("r4_v2_outputs"))
    p.add_argument("--init_amplitude", type=float, default=0.005)
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


def interior_grid(n, device):
    x = torch.arange(
        1, n + 1,
        dtype=torch.float64,
        device=device
    ) / (n + 1)

    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack(
        [X.reshape(-1), Y.reshape(-1)],
        dim=1
    )


def dense_grid(n, device):
    x = torch.linspace(
        0, 1, n,
        dtype=torch.float64,
        device=device
    )

    X, Y = torch.meshgrid(x, x, indexing="ij")

    return torch.stack(
        [X.reshape(-1), Y.reshape(-1)],
        dim=1
    )


def random_points(n, device):
    return torch.rand(
        n, 2,
        dtype=torch.float64,
        device=device
    )


def exact_u(x):
    return (
        torch.sin(math.pi*x[:, 0])
        *
        torch.sin(math.pi*x[:, 1])
    )


def null_mode(x, m):
    return (
        torch.sin(m*math.pi*x[:, 0])
        *
        torch.sin(m*math.pi*x[:, 1])
    )


def laplacian(func, xy):

    xy = xy.clone().requires_grad_(True)

    u = func(xy)

    grad = torch.autograd.grad(
        u,
        xy,
        torch.ones_like(u),
        create_graph=True
    )[0]

    uxx = torch.autograd.grad(
        grad[:, 0],
        xy,
        torch.ones_like(grad[:, 0]),
        create_graph=True
    )[0][:, 0]

    uyy = torch.autograd.grad(
        grad[:, 1],
        xy,
        torch.ones_like(grad[:, 1]),
        create_graph=True
    )[0][:, 1]

    return uxx + uyy


def residual(a, points, m):

    def trial(x):
        return exact_u(x) + a * null_mode(x, m)

    return (
        -laplacian(trial, points)
        -
        2*math.pi**2*exact_u(points)
    )


def rms(x):
    return torch.sqrt(torch.mean(x*x))


def evaluate(a, points, m):
    with torch.no_grad():
        pass

    r = residual(a, points, m)

    return float(rms(r).detach().cpu())


def main():

    args = parse_args()

    seed_all(args.seed)
    torch.set_default_dtype(torch.float64)

    device = get_device(args.device)

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    n = 32 if args.mode == "quick" else 64
    epochs = 1000 if args.mode == "quick" else 5000

    m = n + 1

    train = interior_grid(n, device)
    random = random_points(n*n*10, device)
    dense = dense_grid(
        300 if args.mode == "quick" else 600,
        device
    )

    a = torch.nn.Parameter(
        torch.tensor(
            args.init_amplitude,
            dtype=torch.float64,
            device=device
        )
    )

    optimizer = torch.optim.Adam(
        [a],
        lr=1e-2
    )

    history = []

    start = time.perf_counter()

    for epoch in range(epochs):

        r = residual(a, train, m)

        loss = torch.mean(r*r)

        optimizer.zero_grad()
        loss.backward()

        grad_a = float(
            a.grad.detach().cpu()
        )

        optimizer.step()

        if epoch % 50 == 0:
            history.append({
                "epoch": epoch,
                "a": float(a.detach().cpu()),
                "structured_rms": evaluate(a, train, m),
                "random_rms": evaluate(a, random, m),
                "continuous_rms": evaluate(a, dense, m),
                "grad_a": grad_a
            })

    runtime = time.perf_counter() - start

    result = {
        "device": str(device),
        "n": n,
        "mode_frequency": m,
        "epochs": epochs,
        "initial_amplitude": args.init_amplitude,
        "final_amplitude": float(a.detach().cpu()),
        "structured_final": evaluate(a, train, m),
        "random_final": evaluate(a, random, m),
        "continuous_final": evaluate(a, dense, m),
        "runtime_seconds": runtime,
        "history": history
    }

    with open(args.output/"r4_v2_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
