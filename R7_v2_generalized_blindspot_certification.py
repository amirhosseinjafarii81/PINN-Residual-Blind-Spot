#!/usr/bin/env python3
"""
R7_v2 Generalized Blind Spot Certification

Purpose:
Generalize the R6 observation under controlled conditions.

Tests:
1) frequency sweep under exact base+hidden formulation
2) grid resolution scaling
3) point residual vs Monte Carlo residual vs weak residual

This is an empirical certification experiment.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="quick", choices=["quick", "full"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="r7_v2_outputs")
    return p.parse_args()


def device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed():
    torch.manual_seed(0)
    np.random.seed(0)


def grid(n, dev):
    x = torch.linspace(0, 1, n, device=dev, dtype=torch.float64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], 1)


def random_points(n, dev):
    return torch.rand((n, 2), device=dev, dtype=torch.float64)


def hidden(points, m, amp=0.005):
    x, y = points[:,0], points[:,1]
    return amp * torch.sin(m*np.pi*x)*torch.sin(m*np.pi*y)


def residual(points, m, amp=0.005):
    # -laplacian of hidden mode
    return 2*(m*np.pi)**2 * hidden(points, m, amp)


def rms(x):
    return torch.sqrt(torch.mean(x*x)).item()


def weak_residual(points, m, k=5):
    r = residual(points, m)
    test = torch.cos(k*np.pi*points[:,0])*torch.cos(k*np.pi*points[:,1])
    return torch.mean(r*test).abs().item()


def run(cfg):
    seed()
    dev = device(cfg.device)
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)

    freqs = [5,17,33,49] if cfg.mode=="quick" else [1,5,9,17,33,49,65,81]
    grids = [16,32,64] if cfg.mode=="quick" else [16,32,64,128]

    result = {
        "device": str(dev),
        "frequency_sweep": [],
        "resolution_scaling": []
    }

    t0=time.time()

    for m in freqs:
        for n in grids[:2 if cfg.mode=="quick" else None]:
            gp=grid(n,dev)
            rp=random_points(n*n,dev)

            result["frequency_sweep"].append({
                "mode":m,
                "grid":n,
                "point_rms":rms(residual(gp,m)),
                "random_rms":rms(residual(rp,m)),
                "weak_projection":weak_residual(gp,m),
                "blind_ratio":rms(residual(rp,m))/max(weak_residual(gp,m),1e-30)
            })

    for n in grids:
        gp=grid(n,dev)
        result["resolution_scaling"].append({
            "grid":n,
            "points":n*n,
            "residual":rms(residual(gp,33)),
            "weak":weak_residual(gp,33)
        })

    result["runtime_seconds"]=time.time()-t0

    with open(out/"r7_v2_summary.json","w") as f:
        json.dump(result,f,indent=2)

    print(json.dumps(result,indent=2))


if __name__=="__main__":
    run(args())
