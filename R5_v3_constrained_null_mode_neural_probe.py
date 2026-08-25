#!/usr/bin/env python3
"""
R5_v3_constrained_null_mode_neural_probe.py

Purpose:
--------
A controlled neural experiment after R5-v2.

R5-v2 showed:
- a free Fourier PINN does not automatically discover the null direction.

R5-v3 asks a cleaner question:

If a neural parameterization explicitly contains a discrete null-space
direction, can collocation physics loss detect and remove it?

Model:

    u_theta =
        u_exact
        + a_theta * v_null
        + eps * NN_theta

where:

    v_null = sin((N+1)pi*x) sin((N+1)pi*y)

Only the null amplitude is initialized away from zero.

Measurements:
- evolution of a_theta
- gradient dL/da
- structured residual
- random residual
- continuous residual
- neural correction magnitude

This isolates:
representation capability vs observability.
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
    p.add_argument("--output", type=Path, default=Path("r5_v3_outputs"))
    p.add_argument("--init_amplitude", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=20260825)
    return p.parse_args()


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interior_grid(n, device):
    x = torch.arange(1, n + 1, dtype=torch.float64, device=device)/(n+1)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def random_points(n, device):
    return torch.rand(n, 2, dtype=torch.float64, device=device)


def dense_grid(n, device):
    x = torch.linspace(0, 1, n, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def exact_u(x):
    return torch.sin(math.pi*x[:, 0])*torch.sin(math.pi*x[:, 1])


def null_mode(x, m):
    return torch.sin(m*math.pi*x[:, 0])*torch.sin(m*math.pi*x[:, 1])


class CorrectionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def laplacian(func, x):
    x = x.clone().requires_grad_(True)
    u = func(x)

    g = torch.autograd.grad(
        u, x, torch.ones_like(u),
        create_graph=True
    )[0]

    uxx = torch.autograd.grad(
        g[:, 0], x,
        torch.ones_like(g[:, 0]),
        create_graph=True
    )[0][:, 0]

    uyy = torch.autograd.grad(
        g[:, 1], x,
        torch.ones_like(g[:, 1]),
        create_graph=True
    )[0][:, 1]

    return uxx + uyy


def rms(x):
    return torch.sqrt(torch.mean(x*x))


def main():
    args = parse_args()
    seed_all(args.seed)
    torch.set_default_dtype(torch.float64)

    device = get_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    n = 32 if args.mode == "quick" else 64
    m = n + 1
    epochs = 1500 if args.mode == "quick" else 5000

    train = interior_grid(n, device)
    random = random_points(n*n*10, device)
    dense = dense_grid(300 if args.mode == "quick" else 600, device)

    a = torch.nn.Parameter(
        torch.tensor(args.init_amplitude,
                     dtype=torch.float64,
                     device=device)
    )

    net = CorrectionNet().to(device)

    optimizer = torch.optim.Adam(
        [a] + list(net.parameters()),
        lr=1e-3
    )

    history = []

    start = time.perf_counter()

    def solution(x):
        return exact_u(x) + a*null_mode(x, m) + 0.001*net(x)

    for epoch in range(epochs):

        residual = -laplacian(solution, train) - 2*math.pi**2*exact_u(train)

        loss = torch.mean(residual**2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_a = float(a.grad.detach().cpu())

        optimizer.step()

        if epoch % 100 == 0:
            history.append({
                "epoch": epoch,
                "a": float(a.detach().cpu()),
                "grad_a": grad_a,
                "structured": float(rms(residual).detach().cpu()),
                "random": float(rms(
                    -laplacian(solution, random)
                    - 2*math.pi**2*exact_u(random)
                ).detach().cpu()),
                "continuous": float(rms(
                    -laplacian(solution, dense)
                    - 2*math.pi**2*exact_u(dense)
                ).detach().cpu())
            })

    runtime = time.perf_counter() - start

    def eval_res(points):
        r = -laplacian(solution, points)-2*math.pi**2*exact_u(points)
        return float(rms(r).detach().cpu())

    result = {
        "device": str(device),
        "n": n,
        "frequency": m,
        "epochs": epochs,
        "initial_amplitude": args.init_amplitude,
        "final_amplitude": float(a.detach().cpu()),
        "structured_residual": eval_res(train),
        "random_residual": eval_res(random),
        "continuous_residual": eval_res(dense),
        "runtime_seconds": runtime,
        "history": history
    }

    with open(args.output/"r5_v3_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
