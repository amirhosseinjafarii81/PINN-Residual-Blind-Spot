#!/usr/bin/env python3
"""
R9 Leakage-Free Weak Test Space Benchmark

Goal:
Remove residual-dependent leakage from R8 and test whether
fixed weak test spaces can observe hidden residual components.

Compared methods:
1) Point collocation RMS
2) Fixed random RBF weak space
3) Fixed compact RBF grid weak space
4) Fourier weak basis

No test function is selected using the residual.

Metrics:
- measurement magnitude
- robustness over resolutions
- runtime
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="quick",
                   choices=["quick", "full"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default="r9_outputs")
    return p.parse_args()


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed():
    torch.manual_seed(0)
    np.random.seed(0)


def grid(n, device):
    x = torch.linspace(0, 1, n, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def hidden_residual(points, m=33, amp=0.005):
    x = points[:, 0]
    y = points[:, 1]
    u = amp * torch.sin(m*np.pi*x) * torch.sin(m*np.pi*y)
    return 2*(m*np.pi)**2*u


def rbf(points, centers, sigma):
    d = torch.cdist(points, centers)
    return torch.exp(-(d/sigma)**2)


def fourier_tests(points, modes):
    x = points[:, 0:1]
    y = points[:, 1:2]

    basis = []
    for mx, my in modes:
        basis.append(
            torch.sin(mx*np.pi*x) *
            torch.sin(my*np.pi*y)
        )

    return torch.cat(basis, dim=1)


def weak_measure(test_matrix, residual):
    proj = torch.mean(
        test_matrix * residual[:, None],
        dim=0
    )
    return torch.linalg.norm(proj).item()


def evaluate(n, device, k):
    points = grid(n, device)
    residual = hidden_residual(points)

    point_value = torch.sqrt(
        torch.mean(residual**2)
    ).item()

    # leakage-free random RBF
    random_centers = torch.rand(
        (k,2),
        dtype=torch.float64,
        device=device
    )

    random_value = weak_measure(
        rbf(points, random_centers, 0.15),
        residual
    )

    # fixed regular RBF grid
    side = int(np.sqrt(k))
    c = grid(side, device)
    grid_value = weak_measure(
        rbf(points, c, 0.15),
        residual
    )

    # Fourier weak basis
    modes = []
    for i in range(1, 6):
        for j in range(1, 6):
            modes.append((i,j))

    fourier_value = weak_measure(
        fourier_tests(points, modes),
        residual
    )

    return {
        "grid": n,
        "points": n*n,
        "point_residual": point_value,
        "random_rbf": random_value,
        "grid_rbf": grid_value,
        "fourier": fourier_value
    }


def main():
    cfg = parse_args()
    seed()

    device = get_device(cfg.device)

    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)

    resolutions = [16,32,64] if cfg.mode=="quick" else [16,32,64,128]

    k = 32 if cfg.mode=="quick" else 128

    start = time.time()

    results = []
    for n in resolutions:
        results.append(
            evaluate(n, device, k)
        )

    summary = {
        "device": str(device),
        "dtype": "float64",
        "mode_frequency": 33,
        "leakage_free": True,
        "results": results,
        "runtime_seconds": time.time()-start
    }

    with open(out/"r9_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
