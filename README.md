# PINN Residual Blind Spot

Computational investigation of residual observability limitations in Physics-Informed Neural Networks (PINNs).

## Overview

Physics-Informed Neural Networks optimize PDE solutions by minimizing residual errors at selected points or test functions.

This project investigates a fundamental question:

Can a small residual measured by conventional collocation points guarantee that the PDE error is truly controlled everywhere?

## Main Findings

The experiments demonstrate that discrete residual minimization can miss structured high-frequency error components.

A hidden residual mode can remain almost invisible to point-wise collocation while being clearly detected by continuous or weak measurements.

## Experimental Pipeline

The study consists of:

- R2: Representability and blind-spot preflight
- R3: Null-mode construction and certification
- R4-R5: Optimization blindness experiments
- R6: Evidence package and visualization
- R7: Generalized blind-spot certification
- R8-R9: Weak test-space rescue experiments

## Key Results

The experiments show:

- Very small structured collocation residuals can coexist with large continuous residuals.
- Weak test spaces provide significantly improved observability.
- Adaptive weak measurements can detect hidden residual components missed by point sampling.

## Repository Structure
src/
Experimental Python implementations

results/
Figures and numerical outputs

requirements.txt
Python dependencies

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
