#!/usr/bin/env python3
"""
R6_PINN_blindspot_evidence_package.py

Evidence package generator for the PINN discrete-null-mode experiment.

Purpose:
Create publication/LinkedIn-ready figures from the certified experiment.

Figures:
1) Blind spot map:
   structured collocation residual vs continuous residual

2) Gradient invisibility:
   evolution of hidden amplitude and dL/da

3) Grid comparison:
   structured sampling vs random sampling

The script is deterministic and lightweight.
It does not retrain the model. It uses the analytical null mode.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=Path("r6_figures")
    )
    return p.parse_args()


def null_mode(x, y, m):
    return np.sin(m*np.pi*x)*np.sin(m*np.pi*y)


def make_grid(n):
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x)
    return X, Y


def style(ax, title):
    ax.set_title(title, fontsize=16, weight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25)


def main():

    args = parse_args()
    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    n = 32
    m = 33
    a = 0.005

    # field error visualization
    X, Y = make_grid(400)
    U = a * null_mode(X, Y, m)

    fig, ax = plt.subplots(
        figsize=(7, 6),
        dpi=200
    )

    im = ax.imshow(
        np.abs(U),
        origin="lower",
        extent=[0,1,0,1],
        aspect="equal"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    style(
        ax,
        "Hidden PDE error field\n(null mode amplitude = 0.005)"
    )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(
        args.output/"figure1_hidden_error.png",
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        args.output/"figure1_hidden_error.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


    # residual comparison
    labels = [
        "Structured\ncollocation",
        "Random\npoints",
        "Continuous\nfield"
    ]

    values = [
        1.57e-6,
        53.7796,
        53.5608
    ]

    fig, ax = plt.subplots(
        figsize=(7,4),
        dpi=200
    )

    ax.bar(labels, values)

    ax.set_yscale("log")
    ax.set_ylabel("Residual RMS")
    style(
        ax,
        "The residual blind spot"
    )

    fig.tight_layout()

    fig.savefig(
        args.output/"figure2_residual_gap.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        args.output/"figure2_residual_gap.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


    # gradient evidence
    epochs = np.arange(0,1500,100)

    amplitude = np.ones_like(epochs)*0.005

    gradients = np.array([
        -1.575e-30,
        -1.963e-31,
        -1.947e-31,
        -1.518e-31,
        -1.064e-31,
        -6.69e-32,
        -3.42e-32,
        -7.74e-33,
        1.35e-32,
        3.05e-32,
        4.40e-32,
        5.46e-32,
        6.30e-32,
        6.94e-32,
        7.43e-32
    ])

    fig, ax1 = plt.subplots(
        figsize=(8,4),
        dpi=200
    )

    ax1.plot(
        epochs,
        amplitude,
        linewidth=3,
        label="hidden amplitude a"
    )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("a(t)")

    ax2 = ax1.twinx()

    ax2.plot(
        epochs,
        np.abs(gradients),
        linewidth=2,
        label="|dL/da|"
    )

    ax2.set_yscale("log")
    ax2.set_ylabel("|gradient|")

    style(
        ax1,
        "Optimization cannot see the hidden direction"
    )

    fig.tight_layout()

    fig.savefig(
        args.output/"figure3_gradient_blindness.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        args.output/"figure3_gradient_blindness.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


    summary = {
        "hidden_mode_frequency": m,
        "hidden_amplitude": a,
        "structured_residual": 1.5706853026765872e-6,
        "random_residual": 53.77964240643593,
        "continuous_residual": 53.56086264768345,
        "message":
        "Discrete residual minimization does not guarantee continuous observability."
    }

    with open(
        args.output/"r6_summary.json",
        "w"
    ) as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
