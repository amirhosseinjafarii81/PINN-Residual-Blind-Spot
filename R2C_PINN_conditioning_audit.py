#!/usr/bin/env python3
"""
R2C_PINN_conditioning_audit.py

Purpose
-------
Cheap conditioning audit before building a final PINN blind-spot experiment.

Questions:
1) Is the PINN failure caused by representation or optimization conditioning?
2) What does the loss landscape look like around the exact solution?
3) Does the learned error contain high-frequency components?

This stage does NOT train a full PINN.

Experiments:
A) Exact-solution perturbation landscape
B) Frequency sensitivity of PDE residual
C) Near-solution optimization stability
D) Hessian/Laplacian computational cost

Equation:
    -Delta u = f
    u = sin(pi*x)sin(pi*y)

Run:
python R2C_PINN_conditioning_audit.py --mode quick --device auto
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
    p.add_argument("--output", type=Path, default=Path("r2c_outputs"))
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


def exact_solution(xy):
    x, y = xy[:, 0], xy[:, 1]
    return torch.sin(math.pi*x)*torch.sin(math.pi*y)


def forcing(xy):
    return 2*math.pi**2*exact_solution(xy)


def random_points(n):
    return torch.rand(n, 2)


def structured_points(n):
    x = torch.linspace(1/(n+1), n/(n+1), n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def laplacian_of_function(func, xy):
    xy = xy.clone().requires_grad_(True)

    u = func(xy)

    grad = torch.autograd.grad(
        u, xy,
        torch.ones_like(u),
        create_graph=True
    )[0]

    uxx = torch.autograd.grad(
        grad[:,0],
        xy,
        torch.ones_like(grad[:,0]),
        create_graph=True
    )[0][:,0]

    uyy = torch.autograd.grad(
        grad[:,1],
        xy,
        torch.ones_like(grad[:,1]),
        create_graph=True
    )[0][:,1]

    return uxx + uyy


def residual_from_function(func, xy):
    return -laplacian_of_function(func, xy)-forcing(xy)


def make_mode(m):
    def mode(xy):
        x, y = xy[:,0], xy[:,1]
        return torch.sin(m*math.pi*x)*torch.sin(m*math.pi*y)
    return mode


def perturbation_landscape(points, modes, eps_values):
    base = exact_solution(points)

    out = []

    for m in modes:

        v = make_mode(m)

        values=[]

        for eps in eps_values:

            def trial(xy, eps=eps, v=v):
                return exact_solution(xy)+eps*v(xy)

            r = residual_from_function(trial, points)

            values.append(
                float(torch.sqrt(torch.mean(r*r)).cpu())
            )

        out.append({
            "mode": m,
            "residual_rms": values
        })

    return out


class TinyPINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(
            torch.nn.Linear(2,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,1)
        )

    def forward(self,x):
        return self.net(x).squeeze(-1)


def near_solution_test(points, steps):

    model=TinyPINN().double().to(points.device)

    # initialize output close to exact amplitude
    torch.nn.init.zeros_(model.net[-1].weight)
    torch.nn.init.zeros_(model.net[-1].bias)

    opt=torch.optim.Adam(model.parameters(),lr=1e-3)

    history=[]

    for _ in range(steps):

        opt.zero_grad(set_to_none=True)

        def trial(x):
            return exact_solution(x)+0.01*model(x)

        loss=torch.mean(
            residual_from_function(trial,points)**2
        )

        loss.backward()
        opt.step()

        history.append(float(loss.detach().cpu()))

    return history


def main():

    args=parse_args()

    seed_all(args.seed)

    torch.set_default_dtype(torch.float64)

    device=get_device(args.device)

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    n=32 if args.mode=="quick" else 64

    points=structured_points(n).to(device)

    eps_values=np.logspace(-6,-1,12)

    modes=[1,5,17,33,49]

    start=time.perf_counter()

    landscape=perturbation_landscape(
        points,
        modes,
        eps_values
    )

    landscape_time=time.perf_counter()-start

    history=near_solution_test(
        points,
        200 if args.mode=="quick" else 1000
    )

    result={
        "metadata":{
            "device":str(device),
            "points":len(points),
            "dtype":"float64"
        },
        "perturbation_modes":modes,
        "eps_values":eps_values.tolist(),
        "landscape":landscape,
        "landscape_seconds":landscape_time,
        "near_solution_initial_loss":history[0],
        "near_solution_final_loss":history[-1],
        "near_solution_history":history
    }

    with open(args.output/"r2c_results.json","w") as f:
        json.dump(result,f,indent=2)

    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
