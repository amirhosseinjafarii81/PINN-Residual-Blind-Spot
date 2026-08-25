# PINN Residual Blind Spot

Computational investigation of residual observability limitations in Physics-Informed Neural Networks (PINNs).

## Overview

Physics-Informed Neural Networks optimize PDE solutions by minimizing residual errors at selected points or test functions.

This project investigates a fundamental question:

Can a small residual measured by conventional collocation points guarantee that the PDE error is truly controlled everywhere?

## Main Findings

The experiments reveal that discrete collocation residual minimization may fail to detect structured high-frequency residual components.

A hidden residual mode can remain nearly invisible to point-wise measurements while being detected by continuous and weak test-space evaluations.

## Experimental Pipeline

The study consists of:

- R2: Representability and blind-spot preflight
- R3: Null-mode construction and certification
- R4-R5: Optimization blindness experiments
- R6: Evidence package and visualization
- R7: Generalized blind-spot certification
- R8-R9: Weak test-space rescue experiments

## Key Results

The experiments demonstrate:

- Large gaps between point-wise residual measurements and continuous residual evaluation.
- Weak test spaces significantly improve residual observability.
- Adaptive weak measurements provide a more informative residual assessment compared with fixed point sampling.

## Repository Structure

PINN-Residual-Blind-Spot/

├── src/
│   └── Experimental Python implementations

├── results/
│   ├── figures/
│   └── numerical outputs

├── README.md

└── requirements.txt

## Experimental Evidence

The repository contains:

- Blind-spot construction experiments.
- Null-mode certification tests.
- Optimization blindness analysis.
- Weak test-space rescue benchmarks.

All experiments are reproducible using the provided Python scripts.

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt

Run experiments:
python src/R9_leakage_free_weak_test_space_benchmark.py

Author

Amirhossein Jafari

Research interests:

Scientific Machine Learning
Physics-Informed Neural Networks
Meshfree Variational Methods
Neural Test Spaces
