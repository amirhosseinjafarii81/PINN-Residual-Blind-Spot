# Results Summary

## Objective

This repository investigates residual observability limitations in Physics-Informed Neural Networks (PINNs).

The main question is whether minimizing discrete collocation residuals guarantees detection of continuous PDE errors.

## Main observation

A high-frequency hidden perturbation can remain nearly invisible to collocation residual measurements while producing a large continuous residual.

## Key results

| Measurement | RMS |
|---|---:|
| Structured collocation residual | 1.57e-6 |
| Random validation residual | 53.78 |
| Continuous residual | 53.56 |

## Blind spot verification

The hidden residual mode was tested under:

- structured collocation sampling
- random point validation
- continuous residual evaluation
- multiple spatial frequencies
- increasing grid resolutions

The blind spot behavior remained consistent.

## Weak test-space experiments

Weak residual measurements using test functions significantly improved the observability of hidden structures.

These experiments motivate adaptive weak test spaces for more reliable residual measurement in PINNs.

## Reproducibility

All experiment scripts, generated figures, and numerical results are provided in this repository.
