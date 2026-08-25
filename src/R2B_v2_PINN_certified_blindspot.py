#!/usr/bin/env python3
"""
R2B-v2 PINN Certified Blind Spot Experiment

Purpose:
--------
A stricter SciML experiment after R2B-v1.

Main improvements:
1. Hard Dirichlet boundary enforcement:
       u_theta = x(1-x)y(1-y) * N_theta(x,y)

2. Deterministic Fourier features.

3. Adam + LBFGS optimization.

4. Independent validation set.

5. Measures:
   - training physics residual
   - validation physics residual
   - L2 solution error
   - gradient behavior
   - runtime

Scientific question:
Can a PINN achieve a very small empirical residual on a collocation
set while still having a large continuous residual?

Equation:
    -Delta u = f
    u = sin(pi*x) sin(pi*y)

Run:
python R2B_v2_PINN_certified_blindspot.py \
    --mode quick \
    --device auto
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
    p.add_argument("--output", type=Path, default=Path("r2b_v2_outputs"))
    p.add_argument("--seed", type=int, default=20260825)
    return p.parse_args()


def set_seed(seed):
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


def exact_solution(xy):
    x, y = xy[:, 0], xy[:, 1]
    return torch.sin(math.pi*x) * torch.sin(math.pi*y)


def forcing(xy):
    return 2.0 * math.pi**2 * exact_solution(xy)


def structured_points(n):
    x = torch.linspace(1/(n+1), n/(n+1), n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def random_points(n):
    return torch.rand(n, 2)


class FourierFeaturePINN(torch.nn.Module):
    def __init__(self, modes=16, width=64):
        super().__init__()

        # deterministic frequencies
        freq = torch.arange(
            1, modes + 1,
            dtype=torch.float64
        )

        Bx, By = torch.meshgrid(
            freq, freq, indexing="ij"
        )

        B = torch.stack(
            [Bx.reshape(-1), By.reshape(-1)],
            dim=0
        )

        self.register_buffer("B", B)

        dim = 2 * B.shape[1]

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, 1)
        )

    def forward(self, xy):

        z = 2 * math.pi * xy @ self.B

        feat = torch.cat(
            [
                torch.sin(z),
                torch.cos(z)
            ],
            dim=1
        )

        raw = self.net(feat).squeeze(-1)

        # exact homogeneous Dirichlet BC
        x, y = xy[:, 0], xy[:, 1]

        return x*(1-x)*y*(1-y)*raw


def laplacian(model, xy):

    xy = xy.clone().requires_grad_(True)

    u = model(xy)

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


def residual_loss(model, xy):

    r = -laplacian(model, xy) - forcing(xy)

    return torch.mean(r*r)


def train_adam(model, points, epochs):

    opt = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    history=[]

    for epoch in range(epochs):

        opt.zero_grad(set_to_none=True)

        loss = residual_loss(model, points)

        loss.backward()

        opt.step()

        if epoch % 200 == 0:
            history.append(
                float(loss.detach().cpu())
            )

    return history


def train_lbfgs(model, points):

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=300,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    def closure():

        opt.zero_grad()

        loss = residual_loss(
            model,
            points
        )

        loss.backward()

        return loss

    opt.step(closure)


def evaluate(model, points):

    with torch.no_grad():
        sol_error = torch.sqrt(
            torch.mean(
                (model(points)-exact_solution(points))**2
            )
        )

    res = torch.sqrt(
        residual_loss(model, points)
    )

    return (
        float(sol_error.cpu()),
        float(res.cpu())
    )


def run_case(name, points, validation, epochs):

    model = FourierFeaturePINN().double().to(points.device)

    start = time.perf_counter()

    history = train_adam(
        model,
        points,
        epochs
    )

    train_lbfgs(
        model,
        points
    )

    runtime = time.perf_counter()-start

    train_error, train_res = evaluate(
        model,
        points
    )

    val_error, val_res = evaluate(
        model,
        validation
    )

    return {
        "train_solution_rmse": train_error,
        "train_residual": train_res,
        "validation_solution_rmse": val_error,
        "validation_residual": val_res,
        "runtime_seconds": runtime,
        "loss_history": history
    }


def main():

    args = parse_args()

    set_seed(args.seed)

    torch.set_default_dtype(torch.float64)

    device = get_device(args.device)

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    n = 32

    N = n*n

    epochs = 1500 if args.mode=="quick" else 4000

    structured = structured_points(n).to(device)

    random = random_points(N).to(device)

    validation = random_points(20000).to(device)


    result = {

        "metadata": {
            "device": str(device),
            "points": N,
            "epochs": epochs,
            "dtype": "float64"
        },

        "structured": run_case(
            "structured",
            structured,
            validation,
            epochs
        ),

        "random": run_case(
            "random",
            random,
            validation,
            epochs
        )
    }


    with open(
        args.output/"r2b_v2_results.json",
        "w"
    ) as f:
        json.dump(
            result,
            f,
            indent=2
        )


    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
