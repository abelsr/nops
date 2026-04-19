"""
Abstract base class for PDE data generators.

All generators follow a uniform interface: a single :meth:`generate` method
that returns a ``dict[str, Tensor]`` with consistent key names.  Subclasses
are responsible for implementing the solver and declaring which keys their
output dict contains.

Common output keys
------------------
``"ic"``
    Initial condition field, shape ``(n, *spatial)``.
    Present for all time-dependent equations.
``"solution"``
    Solution snapshots, shape ``(n, *spatial, T_steps)``.
    For steady-state equations (Darcy, Poisson) this is the full solution
    without a time dimension.
``"t_grid"``
    Time coordinates of recorded snapshots, shape ``(T_steps,)``.
    Only present for time-dependent equations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseGenerator(ABC):
    """Abstract base class for pseudo-spectral PDE data generators.

    Parameters
    ----------
    N : int
        Number of grid points per spatial dimension.
    L : float
        Physical domain length (same in all dimensions).  The domain is
        ``[0, L]^dim`` with periodic boundary conditions.  Default ``2π``.
    device : str or torch.device
        Target device for generation.  Use ``"cuda"`` for GPU-accelerated
        solvers.
    dtype : torch.dtype
        Output dtype.  All internal FFT computations use ``float64``; the
        result is cast to ``dtype`` before being returned.  Default
        ``torch.float32``.
    """

    def __init__(
        self,
        N: int,
        L: float = 2.0 * math.pi,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.N = N
        self.L = L
        self.device = torch.device(device)
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> dict[str, Tensor]:
        """Generate ``n_samples`` PDE trajectories.

        Parameters
        ----------
        n_samples : int
            Number of independent samples to generate.
        **kwargs
            Equation-specific overrides (e.g., ``T``, ``dt``).

        Returns
        -------
        dict[str, Tensor]
            Dictionary of output tensors.  All spatial tensors have leading
            batch dimension ``n_samples``.  dtype matches ``self.dtype``
            for real-valued tensors.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @property
    def domain_length(self) -> float:
        """Physical length of the (cubic) domain."""
        return self.L

    @property
    def resolution(self) -> int:
        """Number of grid points per spatial dimension."""
        return self.N

    def _wavenumbers_1d(self) -> Tensor:
        """Return rfft wavenumbers in physical units ``2π n / L``."""
        k = torch.fft.rfftfreq(self.N, d=1.0 / self.N, device=self.device).double()
        return k * (2.0 * math.pi / self.L)

    def _wavenumbers_2d(self) -> tuple[Tensor, Tensor]:
        """Return (k0_phys, k1_phys) rfft2 wavenumber grids.

        ``k0`` uses the full fft convention (negative frequencies included);
        ``k1`` uses the rfft half (non-negative only).  Both are in physical
        units ``2π n / L``.

        Returns
        -------
        k0 : Tensor  shape (N, N//2+1)
        k1 : Tensor  shape (N, N//2+1)
        """
        scale = 2.0 * math.pi / self.L
        k0_int = torch.fft.fftfreq(self.N, d=1.0 / self.N, device=self.device).double()
        k1_int = torch.fft.rfftfreq(self.N, d=1.0 / self.N, device=self.device).double()
        K0, K1 = torch.meshgrid(k0_int, k1_int, indexing="ij")  # (N, N//2+1)
        return scale * K0, scale * K1

    def _spatial_grid_1d(self) -> Tensor:
        """Return the physical grid ``x ∈ [0, L)``, shape ``(N,)``."""
        return torch.linspace(0.0, self.L, self.N + 1, device=self.device)[:-1]

    def _spatial_grid_2d(self) -> tuple[Tensor, Tensor]:
        """Return 2D meshgrid ``(X, Y)`` each of shape ``(N, N)``."""
        x = self._spatial_grid_1d()
        return torch.meshgrid(x, x, indexing="ij")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"N={self.N}, L={self.L:.4g}, "
            f"device={self.device}, dtype={self.dtype})"
        )
