# PINN Residual Blind Spot

Computational experiments investigating residual observability limitations
in Physics-Informed Neural Networks (PINNs).

## Main idea

A high-frequency error component can remain almost invisible to discrete
collocation residual minimization while producing a large continuous PDE error.

## Experimental pipeline

* R2: Representability and training audits
* R3-R5: Null-mode construction and optimization probes
* R6: Evidence visualization package
* R7: Generalized frequency and grid certification
* R8-R9: Weak test-space based measurements

## Key observation

Increasing collocation density alone does not necessarily remove hidden
residual structures.

## Reproducibility

Requirements:

* Python 3.12+
* PyTorch
* CUDA (optional)

Example:

python src/R7\_v2\_generalized\_blindspot\_certification.py --mode full --device auto

## Author

Amirhossein Jafari
Numerical Analysis x Scientific Machine Learning

