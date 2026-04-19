"""
Data generator for the 2D Poisson equation.

PDE
---
    -∇²u = f(x)    in Ω = [0, L]²

with periodic boundary conditions.

Expanding in Fourier series:

    (2π/L)² |k|² û_k = f̂_k   ⟹   û_k = f̂_k / ((2π/L)² |k|²)

This is an **exact, instantaneous** solve — no time integration required.
The only special case is ``k = 0`` (zero mode), which is indeterminate for
the pure-Neumann/periodic Poisson problem.  We set ``û_0 = 0`` (zero-mean
solution), which is the standard convention for periodic Poisson.

Steady-state operator learning problem
---------------------------------------
    Input:   f(x)  — source term (from a GRF)
    Output:  u(x)  — potential / pressure

This is the same operator learning setup as the FNO Darcy benchmark, but
with constant permeability ``a ≡ 1``.

References
----------
- Canuto et al., "Spectral Methods: Fundamentals in Single Domains", Springer 2006.
- Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from nops.data.generators.base import BaseGenerator
from nops.data.utils.gaussian_rf import GaussianRF


class PoissonGenerator(BaseGenerator):
    """Direct spectral solver for the 2D periodic Poisson equation.

    Generates pairs ``(f, u)`` where ``-∇²u = f`` is solved exactly in
    Fourier space.

    Parameters
    ----------
    N : int
        Spatial resolution (grid is N×N).  Default ``128``.
    L : float
        Domain length ``[0, L]²``.  Default ``1.0`` (standard convention).
    alpha_grf : float
        GRF smoothness for source term ``f``.  Default ``2.0``.
    tau_grf : float
        GRF length-scale for source term ``f``.  Default ``3.0``.
    device : str or torch.device
        Default ``"cpu"``.
    dtype : torch.dtype
        Default ``torch.float32``.

    Examples
    --------
    >>> gen = PoissonGenerator(N=128, L=1.0)
    >>> data = gen.generate(500)
    >>> data["source"].shape    # (500, 128, 128)
    >>> data["solution"].shape  # (500, 128, 128)
    """

    def __init__(
        self,
        N: int = 128,
        L: float = 1.0,
        alpha_grf: float = 2.0,
        tau_grf: float = 3.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(N=N, L=L, device=device, dtype=dtype)

        self._grf = GaussianRF(
            dim=2, N=N, alpha=alpha_grf, tau=tau_grf, L=L,
            device=device, dtype=dtype,
        )
        self._build_spectral_ops()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_samples: int, **kwargs) -> dict[str, Tensor]:
        """Generate ``n_samples`` Poisson source-solution pairs.

        Returns
        -------
        dict with keys:
            ``"source"``   — source term ``f(x)``,     shape ``(n, N, N)``
            ``"solution"`` — potential ``u(x)``,        shape ``(n, N, N)``
        """
        f = self._grf.sample(n_samples)  # (n, N, N), float32
        u = self._solve(f.to(torch.float64))  # float64
        return {
            "source": f,
            "solution": u.to(self.dtype),
        }

    # ------------------------------------------------------------------
    # Spectral solver
    # ------------------------------------------------------------------

    def _build_spectral_ops(self) -> None:
        """Precompute the inverse negative Laplacian ``1 / ((2π/L)² |k|²)``."""
        N = self.N
        scale = 2.0 * math.pi / self.L

        k0_int = torch.fft.fftfreq(N, d=1.0 / N, device=self.device).double()
        k1_int = torch.fft.rfftfreq(N, d=1.0 / N, device=self.device).double()
        K0, K1 = torch.meshgrid(k0_int, k1_int, indexing="ij")

        k2_phys = scale ** 2 * (K0.pow(2) + K1.pow(2))  # (N, N//2+1)

        # Inverse negative Laplacian: 1 / |k|²_phys
        inv_neg_lap = k2_phys.clone()
        inv_neg_lap[0, 0] = 1.0          # avoid division by zero
        self._inv_neg_lap = 1.0 / inv_neg_lap
        self._inv_neg_lap[0, 0] = 0.0    # zero mean solution

    @torch.no_grad()
    def _solve(self, f: Tensor) -> Tensor:
        """Solve ``-∇²u = f`` exactly.

        Parameters
        ----------
        f : Tensor  shape ``(n, N, N)``, float64

        Returns
        -------
        u : Tensor  shape ``(n, N, N)``, float64
        """
        f_hat = torch.fft.rfft2(f, dim=(-2, -1))  # (n, N, N//2+1)
        u_hat = self._inv_neg_lap * f_hat           # exact division
        u = torch.fft.irfft2(u_hat, s=(self.N, self.N), dim=(-2, -1))
        return u.real
