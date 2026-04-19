"""
nops.data — PDE Data Generators and Datasets
=============================================

Pseudo-spectral data generators and :class:`~torch.utils.data.Dataset`
wrappers for six canonical PDE benchmarks used in neural operator research.

Generators
----------
Each generator uses a spectrally-accurate solver and returns a
``dict[str, Tensor]``.

.. list-table::
   :header-rows: 1

   * - Class
     - Equation
     - Solver
   * - :class:`~nops.data.generators.BurgersGenerator`
     - 1D viscous Burgers
     - ETDRK4 (integrating factor)
   * - :class:`~nops.data.generators.NavierStokesGenerator`
     - 2D incompressible NS (vorticity)
     - CN-Heun (IMEX-RK2)
   * - :class:`~nops.data.generators.DarcyGenerator`
     - 2D steady Darcy flow
     - Spectral Picard iteration or file loader
   * - :class:`~nops.data.generators.HeatGenerator`
     - 1D/2D heat equation
     - Exact integrating factor
   * - :class:`~nops.data.generators.WaveGenerator`
     - 1D/2D (damped) wave equation
     - Exact matrix exponential
   * - :class:`~nops.data.generators.PoissonGenerator`
     - 2D Poisson equation
     - Direct spectral solve

Datasets
--------
Each dataset wraps a generator (or pre-built data) and exposes a standard
PyTorch ``Dataset`` returning ``(input, target)`` pairs.

.. list-table::
   :header-rows: 1

   * - Class
     - input
     - target
   * - :class:`~nops.data.datasets.BurgersDataset`
     - ``u(x, 0)``
     - ``u(x, T)``
   * - :class:`~nops.data.datasets.NavierStokesDataset`
     - ``ω(x, 0)``
     - ``ω(x, T)``
   * - :class:`~nops.data.datasets.DarcyDataset`
     - ``a(x)`` (permeability)
     - ``u(x)`` (pressure)
   * - :class:`~nops.data.datasets.HeatDataset`
     - ``u(x, 0)``
     - ``u(x, T)``
   * - :class:`~nops.data.datasets.WaveDataset`
     - ``(u(x,0), ∂u/∂t(x,0))``
     - ``u(x, T)``
   * - :class:`~nops.data.datasets.PoissonDataset`
     - ``f(x)`` (source)
     - ``u(x)`` (potential)

Utility
-------
:class:`~nops.data.utils.GaussianRF`
    Spectral Gaussian Random Field sampler (1D/2D) used by all generators
    for initial conditions and forcing.

Quick start
-----------
>>> from nops.data import BurgersGenerator, BurgersDataset
>>> gen = BurgersGenerator(N=1024, nu=0.1, T=1.0, record_steps=1)
>>> ds = BurgersDataset(generator=gen, n_samples=200)
>>> loader = ds.get_dataloader(batch_size=32)
>>> x, y = next(iter(loader))   # x: (32, 1024),  y: (32, 1024)

>>> from nops.data import NavierStokesGenerator, NavierStokesDataset
>>> gen = NavierStokesGenerator(N=64, nu=1e-3, T=1.0, record_steps=10)
>>> ds = NavierStokesDataset(generator=gen, n_samples=100)
>>> ds.save("ns_data.pt")       # persist to disk
>>> ds2 = NavierStokesDataset.load("ns_data.pt")  # reload
"""

# Generators
from nops.data.generators import (
    BaseGenerator,
    BurgersGenerator,
    DarcyGenerator,
    HeatGenerator,
    NavierStokesGenerator,
    PoissonGenerator,
    WaveGenerator,
)

# Datasets
from nops.data.datasets import (
    BasePDEDataset,
    BurgersDataset,
    DarcyDataset,
    HeatDataset,
    NavierStokesDataset,
    PoissonDataset,
    WaveDataset,
)

# Utilities
from nops.data.utils import GaussianRF

__all__ = [
    # Generators
    "BaseGenerator",
    "BurgersGenerator",
    "DarcyGenerator",
    "HeatGenerator",
    "NavierStokesGenerator",
    "PoissonGenerator",
    "WaveGenerator",
    # Datasets
    "BasePDEDataset",
    "BurgersDataset",
    "DarcyDataset",
    "HeatDataset",
    "NavierStokesDataset",
    "PoissonDataset",
    "WaveDataset",
    # Utilities
    "GaussianRF",
]
