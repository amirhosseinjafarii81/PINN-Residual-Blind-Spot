#!/usr/bin/env python3
"""
R2B PINN blind-spot training experiment.

Purpose:
--------
A real PINN experiment after R2A representability audit.

Compares:
1) Structured collocation PINN
2) Random collocation PINN

for the 2D Poisson equation:

    -Delta u = f

with:
    u_exact = sin(pi*x) sin(pi*y)

The goal is NOT to prove PINNs fail.
The goal is to test whether the empirical physics loss on a fixed
collocation set is a reliable certificate for the continuous residual.

Scientific safeguards:
- deterministic Fourier features
- fixed seed
- float64
- independent validation points
- solution error measurement
- residual maps
- runtime reporting

Run:
python R2B_PINN_blindspot_training.py --mode quick
python R2B_PINN_blindspot_training.py --mode full
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=Path("r2b_outputs"))
    p.add_argument("--seed", type=int, default=20260825)
    return p.parse_args()


def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def device_select(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exact_u(xy):
    x, y = xy[:, 0], xy[:, 1]
    return torch.sin(math.pi*x)*torch.sin(math.pi*y)


def forcing(xy):
    return 2*math.pi**2*exact_u(xy)


def grid_points(n):
    x = torch.linspace(1/(n+1), n/(n+1), n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def random_points(n):
    return torch.rand(n, 2)


class FourierPINN(torch.nn.Module):
    def __init__(self, modes=32, width=64):
        super().__init__()
        freqs = torch.arange(1, modes+1, dtype=torch.float64)
        Bx, By = torch.meshgrid(freqs, freqs, indexing="ij")
        B = torch.stack([Bx.reshape(-1), By.reshape(-1)], dim=0)
        self.register_buffer("B", B)

        d = 2*B.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, 1)
        )

    def forward(self, xy):
        z = 2*math.pi*xy @ self.B
        feat = torch.cat([torch.sin(z), torch.cos(z)], dim=1)
        return self.net(feat).squeeze(-1)


def laplacian(model, xy):
    xy = xy.clone().requires_grad_(True)
    u = model(xy)

    g = torch.autograd.grad(
        u, xy, torch.ones_like(u), create_graph=True
    )[0]

    uxx = torch.autograd.grad(
        g[:,0], xy, torch.ones_like(g[:,0]), create_graph=True
    )[0][:,0]

    uyy = torch.autograd.grad(
        g[:,1], xy, torch.ones_like(g[:,1]), create_graph=True
    )[0][:,1]

    return uxx + uyy


def train(model, points, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history=[]

    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)

        r = -laplacian(model, points)-forcing(points)
        loss = torch.mean(r*r)

        loss.backward()
        opt.step()

        if epoch % 100 == 0:
            history.append(float(loss.detach().cpu()))

    return history


def evaluate(model, points):
    with torch.no_grad():
        uerr = torch.sqrt(torch.mean((model(points)-exact_u(points))**2))

    r = -laplacian(model, points)-forcing(points)
    residual = torch.sqrt(torch.mean(r*r))

    return float(uerr.cpu()), float(residual.cpu())


def main():
    a=args()
    seed_all(a.seed)
    torch.set_default_dtype(torch.float64)

    dev=device_select(a.device)
    out=a.output
    out.mkdir(exist_ok=True)

    n=32
    N=n*n

    epochs=1000 if a.mode=="quick" else 5000

    train_grid=grid_points(n).to(dev)
    train_rand=random_points(N).to(dev)

    validation=random_points(10000).to(dev)

    results={}

    for name, pts in [
        ("structured", train_grid),
        ("random", train_rand)
    ]:
        model=FourierPINN().to(dev).double()

        t=time.perf_counter()

        hist=train(model, pts, epochs, 1e-3)

        runtime=time.perf_counter()-t

        train_err, train_res=evaluate(model, pts)
        val_err, val_res=evaluate(model, validation)

        results[name]={
            "train_solution_rmse":train_err,
            "train_residual":train_res,
            "validation_solution_rmse":val_err,
            "validation_residual":val_res,
            "runtime_seconds":runtime,
            "loss_history":hist
        }

    results["metadata"]={
        "device":str(dev),
        "epochs":epochs,
        "points":N,
        "dtype":"float64"
    }

    with open(out/"r2b_results.json","w") as f:
        json.dump(results,f,indent=2)

    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
