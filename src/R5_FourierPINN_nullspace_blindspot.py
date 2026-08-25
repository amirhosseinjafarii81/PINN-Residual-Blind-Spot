#!/usr/bin/env python3
"""
R5_FourierPINN_nullspace_blindspot.py

Purpose
-------
Bridge the certified null-mode phenomenon to an actual neural
parameterization.

Scientific question:
Can a neural PDE solver represent a hidden high-frequency mode,
while a structured collocation residual remains blind to it?

This experiment separates:
1. representation capability
2. residual observability

Model:
    u_theta(x,y)=u_exact(x,y)+alpha*NN_theta(x,y)

The network is a small Fourier-feature MLP.
It is NOT asked to discover the solution from scratch.
It only represents a controlled perturbation.

Training objective:
    minimize only structured collocation PDE residual.

Measurements:
- representation error of target null mode
- structured residual
- random residual
- continuous residual
- hidden coefficient correlation
- gradient norm

This is designed as a scientific experiment, not a tutorial.
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
    p.add_argument("--output", type=Path,
                   default=Path("r5_outputs"))
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

    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def interior_grid(n, device):
    x = torch.arange(
        1, n+1,
        dtype=torch.float64,
        device=device
    )/(n+1)

    X,Y=torch.meshgrid(x,x,indexing="ij")

    return torch.stack(
        [X.reshape(-1),Y.reshape(-1)],
        dim=1
    )


def dense_grid(n, device):
    x=torch.linspace(
        0,1,n,
        dtype=torch.float64,
        device=device
    )

    X,Y=torch.meshgrid(x,x,indexing="ij")

    return torch.stack(
        [X.reshape(-1),Y.reshape(-1)],
        dim=1
    )


def random_points(n, device):
    return torch.rand(
        n,2,
        dtype=torch.float64,
        device=device
    )


def exact_u(x):
    return (
        torch.sin(math.pi*x[:,0])
        *
        torch.sin(math.pi*x[:,1])
    )


def null_mode(x,m):
    return (
        torch.sin(m*math.pi*x[:,0])
        *
        torch.sin(m*math.pi*x[:,1])
    )


def laplacian(func,x):

    x=x.clone().requires_grad_(True)

    u=func(x)

    g=torch.autograd.grad(
        u,x,
        torch.ones_like(u),
        create_graph=True
    )[0]

    uxx=torch.autograd.grad(
        g[:,0],x,
        torch.ones_like(g[:,0]),
        create_graph=True
    )[0][:,0]

    uyy=torch.autograd.grad(
        g[:,1],x,
        torch.ones_like(g[:,1]),
        create_graph=True
    )[0][:,1]

    return uxx+uyy


class FourierMLP(torch.nn.Module):

    def __init__(self, modes=33):
        super().__init__()

        self.freq=torch.tensor(
            [[1.,0.],
             [0.,1.],
             [modes,0.],
             [0.,modes],
             [modes,modes]],
            dtype=torch.float64
        )

        self.net=torch.nn.Sequential(
            torch.nn.Linear(10,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,64),
            torch.nn.Tanh(),
            torch.nn.Linear(64,1)
        )

    def features(self,x):

        z=2*math.pi*x@self.freq.T

        return torch.cat(
            [torch.sin(z),torch.cos(z)],
            dim=1
        )

    def forward(self,x):

        return self.net(
            self.features(x)
        ).squeeze(-1)


def residual(model,x,m):

    def trial(z):
        return exact_u(z)+model(z)

    return (
        -laplacian(trial,x)
        -
        2*math.pi**2*exact_u(x)
    )


def rms(v):
    return torch.sqrt(torch.mean(v*v))


def correlation(a,b):
    a=a-a.mean()
    b=b-b.mean()

    return (
        torch.sum(a*b)
        /
        (
            torch.sqrt(torch.sum(a*a))
            *
            torch.sqrt(torch.sum(b*b))
            +1e-30
        )
    )


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

    m=n+1

    train=interior_grid(n,device)

    random=random_points(n*n*10,device)

    dense=dense_grid(
        300 if args.mode=="quick" else 600,
        device
    )

    model=FourierMLP(m).to(device)

    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs=1500 if args.mode=="quick" else 5000

    history=[]

    start=time.perf_counter()

    for epoch in range(epochs):

        r=residual(model,train,m)

        loss=torch.mean(r*r)

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        grad_norm=float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1e9
            )
            .detach()
            .cpu()
        )

        optimizer.step()

        if epoch%100==0:

            with torch.no_grad():

                pred=model(dense)

                target=null_mode(
                    dense,m
                )

                corr=float(
                    correlation(
                        pred,
                        target
                    )
                    .cpu()
                )

            history.append({
                "epoch":epoch,
                "loss":float(loss.detach().cpu()),
                "corr_with_null_mode":corr,
                "grad_norm":grad_norm
            })


    def eval_res(points):

        return float(
            rms(
                residual(model,points,m)
            )
            .detach()
            .cpu()
        )


    runtime=time.perf_counter()-start

    result={
        "device":str(device),
        "n":n,
        "frequency":m,
        "epochs":epochs,
        "structured_residual":eval_res(train),
        "random_residual":eval_res(random),
        "continuous_residual":eval_res(dense),
        "null_mode_correlation":
            float(
                correlation(
                    model(dense),
                    null_mode(dense,m)
                )
                .detach()
                .cpu()
            ),
        "runtime_seconds":runtime,
        "history":history
    }


    with open(
        args.output/"r5_results.json",
        "w"
    ) as f:
        json.dump(result,f,indent=2)

    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
