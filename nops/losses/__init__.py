"""
nops.losses — Physics-Informed Losses for Neural Operators
===========================================================

This module provides a comprehensive library of physics-informed loss
functions designed to train neural operators (FNO, DeepONet, GNO, …) on
PDE-constrained problems in 1D, 2D, and 3D.

Submodules
----------
derivatives
    Autograd-based differential operators (gradient, laplacian, divergence, …).
base
    Abstract ``PhysicsLoss`` base class and ``CombinedLoss`` combiner.
data
    ``DataLoss`` — supervised MSE / relative-L2 / MAE.
pde
    Per-PDE residual losses (Burgers, Navier-Stokes, Darcy, Heat, Wave, Poisson,
    and a generic user-defined residual).
boundary
    Boundary condition losses (Dirichlet, Neumann, Periodic, Robin).
initial
    Initial condition losses (value and velocity).
energy
    Variational energy losses (Dirichlet energy, Poisson, Darcy, linear
    elasticity).

Quick start
-----------
>>> from nops.losses import (
...     CombinedLoss, DataLoss,
...     BurgersResidual, DirichletLoss, InitialConditionLoss,
... )
>>>
>>> loss_fn = CombinedLoss({
...     "data":     (1.0,  DataLoss()),
...     "pde":      (1e-3, BurgersResidual(nu=1e-3, dim=1)),
...     "bc":       (1e-2, DirichletLoss()),
...     "ic":       (1e-2, InitialConditionLoss()),
... })
>>>
>>> total, terms = loss_fn(
...     data=(u_pred, u_true),
...     pde=(model, x, t),
...     bc=(model, x_bc, g),
...     ic=(model, x, t0, u0),
... )
"""

# Core infrastructure
from .base import CombinedLoss, PhysicsLoss
from .data import DataLoss

# Differential operators (re-exported for convenience)
from . import derivatives

# PDE residual losses
from .pde import (
    BurgersResidual,
    DarcyResidual,
    GenericPDELoss,
    HeatResidual,
    NavierStokesResidual,
    PoissonResidual,
    WaveResidual,
)

# Boundary condition losses
from .boundary import DirichletLoss, NeumannLoss, PeriodicLoss, RobinLoss

# Initial condition losses
from .initial import InitialConditionLoss, InitialVelocityLoss

# Energy / variational losses
from .energy import (
    DarcyEnergyLoss,
    DirichletEnergyLoss,
    ElasticEnergyLoss,
    PoissonEnergyLoss,
)

__all__ = [
    # Core
    "PhysicsLoss",
    "CombinedLoss",
    "DataLoss",
    # Derivatives module
    "derivatives",
    # PDE residuals
    "BurgersResidual",
    "DarcyResidual",
    "GenericPDELoss",
    "HeatResidual",
    "NavierStokesResidual",
    "PoissonResidual",
    "WaveResidual",
    # Boundary conditions
    "DirichletLoss",
    "NeumannLoss",
    "PeriodicLoss",
    "RobinLoss",
    # Initial conditions
    "InitialConditionLoss",
    "InitialVelocityLoss",
    # Energy / variational
    "DirichletEnergyLoss",
    "PoissonEnergyLoss",
    "DarcyEnergyLoss",
    "ElasticEnergyLoss",
]
