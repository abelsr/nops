"""
Data generator for the 1D viscous Burgers equation.

PDE
---
    ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²

Domain
------
    x ∈ [0, L],  periodic BC,  t ∈ [0, T]

Solver
------
Exponential Time Differencing RK4 (ETDRK4) in Fourier space.

The key idea is to split the PDE into a *stiff linear* part (diffusion,
handled exactly) and a *nonlinear* advection part (handled with RK4).

In Fourier space:

    dû_k/dt = -ν k² û_k  -  ik F[u ∂u/∂x]_k
              ^^^^^^^^         ^^^^^^^^^^^^^^
              linear (stiff)   nonlinear

Defining the integrating factor ``E_k = exp(-ν k² Δt)`` eliminates all
stiffness, allowing large time steps.  The ``(E-1)/(-νk²)`` coefficient
that appears in the RK4 stages is evaluated with :func:`torch.special.expm1`
to remain numerically stable at ``k=0``.

Dealiasing
----------
The 2/3 rule: wavenumbers ``|k| > N/3`` are zeroed before every nonlinear
evaluation to prevent aliasing errors from the quadratic advection term.

References
----------
- Cox & Matthews, "Exponential Time Differencing for Stiff Systems",
  J. Comput. Phys. 176 (2002).
- Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from nops.data.generators.base import BaseGenerator
from nops.data.utils.gaussian_rf import GaussianRF


class BurgersGenerator(BaseGenerator):
    """Pseudo-spectral ETDRK4 generator for the 1D viscous Burgers equation.

    Parameters
    ----------
    N : int
        Spatial resolution (number of grid points).  Default 1024.
    L : float
        Domain length ``[0, L]``.  Default ``2π``.
    nu : float
        Kinematic viscosity ν ≥ 0.  Set to 0 for the inviscid case (note:
        shock formation will eventually alias — use small but nonzero ν).
        Default ``0.1``.
    T : float
        Final time.  Default ``1.0``.
    dt : float
        Time step.  With ETDRK4 the diffusion is handled exactly, so ``dt``
        only needs to satisfy the *advection* CFL:
        ``dt ≲ dx / max|u|``.  Default ``1e-4``.
    record_steps : int
        Number of solution snapshots to record (equally spaced in time).
        Default ``200``.
    ic_alpha : float
        GRF smoothness exponent for the initial condition.  Default ``2.5``.
    ic_tau : float
        GRF length-scale parameter for the initial condition.  Default ``7.0``.
    device : str or torch.device
        Computation device.  Default ``"cpu"``.
    dtype : torch.dtype
        Output dtype.  Default ``torch.float32``.

    Examples
    --------
    >>> gen = BurgersGenerator(N=1024, nu=0.1, T=1.0, record_steps=1)
    >>> data = gen.generate(100)
    >>> data["ic"].shape        # (100, 1024)
    >>> data["solution"].shape  # (100, 1024, 1)
    >>> data["t_grid"].shape    # (1,)
    """

    def __init__(
        self,
        N: int = 1024,
        L: float = 2.0 * math.pi,
        nu: float = 0.1,
        T: float = 1.0,
        dt: float = 1e-4,
        record_steps: int = 200,
        ic_alpha: float = 2.5,
        ic_tau: float = 7.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(N=N, L=L, device=device, dtype=dtype)
        self.nu = nu
        self.T = T
        self.dt = dt
        self.record_steps = record_steps

        self._grf = GaussianRF(
            dim=1, N=N, alpha=ic_alpha, tau=ic_tau, L=L, device=device, dtype=dtype
        )
        self._precompute(nu, dt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_samples: int, **kwargs) -> dict[str, Tensor]:
        """Generate ``n_samples`` Burgers trajectories.

        Parameters
        ----------
        n_samples : int
            Number of independent trajectories.

        Returns
        -------
        dict with keys:
            ``"ic"``        — shape ``(n, N)``
            ``"solution"``  — shape ``(n, N, record_steps)``
            ``"t_grid"``    — shape ``(record_steps,)``
        """
        u0 = self._grf.sample(n_samples)  # (n, N), float32

        # Run solver in float64 for accuracy
        ic_f64 = u0.to(torch.float64)
        sol_f64, t_grid = self._solve(ic_f64)

        return {
            "ic": u0,  # (n, N)
            "solution": sol_f64.to(self.dtype),  # (n, N, T_steps)
            "t_grid": t_grid.to(self.dtype),  # (T_steps,)
        }

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def _precompute(self, nu: float, dt: float) -> None:
        """Precompute ETDRK4 coefficients (integrating factors)."""
        k = torch.fft.rfftfreq(self.N, d=1.0 / self.N, device=self.device).double()
        # Physical wavenumbers: k_phys = 2π k / L → for convolution the
        # Fourier derivative is  ∂/∂x → i k_phys.  For the Laplacian:
        # ∂²/∂x² → -(k_phys)².
        k_phys = k * (2.0 * math.pi / self.L)

        # Dealias mask: zero top-1/3 wavenumbers (2/3 rule)
        # Physical cut-off at |k_int| ≤ N//3
        self._dealias = (k <= self.N // 3).double()  # (N//2+1,)
        self._ik = 1j * k_phys  # (N//2+1,)  derivative operator

        # Eigenvalues of the linear operator: λ_k = -ν k²
        lam = -nu * k_phys.pow(2)  # (N//2+1,)

        # Full- and half-step integrating factors
        self._E = torch.exp(lam * dt)   # exp(λ dt)
        self._E2 = torch.exp(lam * dt / 2.0)  # exp(λ dt/2)

        # φ₁(z) = (exp(z) - 1) / z,  stable at z=0 (limit = 1)
        # Used in ETDRK4 stage coefficients:
        #   stage coeff = Δt · φ₁(λ Δt)  (or half-step variant)
        self._phi1_full = self._phi1(lam * dt) * dt   # (N//2+1,)
        self._phi1_half = self._phi1(lam * dt / 2.0) * (dt / 2.0)

    @staticmethod
    def _phi1(z: Tensor) -> Tensor:
        """Compute φ₁(z) = (exp(z) - 1) / z, stable near z=0."""
        # torch.special.expm1(z) = exp(z) - 1
        return torch.where(
            z.abs() < 1e-8,
            torch.ones_like(z),
            torch.expm1(z) / z,
        )

    def _nonlinear(self, u_hat: Tensor) -> Tensor:
        """Compute FFT of the nonlinear term ``-u ∂u/∂x`` in Fourier space.

        Applies the 2/3 dealiasing mask before the pointwise multiplication.
        """
        u_hat_d = u_hat * self._dealias  # zero top modes
        # u in physical space
        u = torch.fft.irfft(u_hat_d, n=self.N, dim=-1)
        # ∂u/∂x in physical space
        du_dx = torch.fft.irfft(self._ik * u_hat_d, n=self.N, dim=-1)
        # Nonlinear term: -u · ∂u/∂x, back to Fourier space
        return -torch.fft.rfft(u * du_dx, dim=-1)

    @torch.no_grad()
    def _solve(self, u0: Tensor) -> tuple[Tensor, Tensor]:
        """Run the ETDRK4 solver from initial condition ``u0``.

        Parameters
        ----------
        u0 : Tensor  shape ``(n, N)``, float64

        Returns
        -------
        solution : Tensor  shape ``(n, N, record_steps)``
        t_grid   : Tensor  shape ``(record_steps,)``
        """
        n_steps = int(math.ceil(self.T / self.dt))
        # Evenly space record indices across the total steps
        record_every = max(1, n_steps // self.record_steps)

        u_hat = torch.fft.rfft(u0, dim=-1)  # (n, N//2+1), complex128
        snapshots: list[Tensor] = []
        times: list[float] = []

        E = self._E
        E2 = self._E2
        phi1_full = self._phi1_full
        phi1_half = self._phi1_half

        for step in range(n_steps):
            t = step * self.dt
            N1 = self._nonlinear(u_hat)

            # Stage a: half-step with N1
            a = E2 * u_hat + phi1_half * N1
            N2 = self._nonlinear(a)

            # Stage b: half-step with N2
            b = E2 * u_hat + phi1_half * N2
            N3 = self._nonlinear(b)

            # Stage c: full step with corrected b
            c = E2 * a + phi1_half * (2.0 * N3 - N1)
            N4 = self._nonlinear(c)

            # ETDRK4 update (Cox-Matthews coefficients)
            u_hat = (
                E * u_hat
                + phi1_full * (N1 + 2.0 * (N2 + N3) + N4) / 6.0
            )

            # Pin mean to zero (conserved by Burgers with periodic BC)
            u_hat[..., 0] = 0.0

            if (step + 1) % record_every == 0:
                snapshots.append(torch.fft.irfft(u_hat, n=self.N, dim=-1))
                times.append(t + self.dt)

            if len(snapshots) == self.record_steps:
                break

        # Pad if fewer snapshots than requested (edge case: T very small)
        while len(snapshots) < self.record_steps:
            snapshots.append(snapshots[-1])
            times.append(times[-1])

        sol = torch.stack(snapshots, dim=-1)  # (n, N, T_steps)
        t_grid = torch.tensor(times, device=self.device, dtype=torch.float64)
        return sol, t_grid
