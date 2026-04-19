"""
Data generator for the 2D incompressible Navier-Stokes equations.

PDE (vorticity-stream function formulation)
-------------------------------------------
    ∂ω/∂t + u·∇ω = ν∇²ω + f
    -∇²ψ = ω               (stream function)
    u = ∂ψ/∂y,  v = -∂ψ/∂x  (velocity from stream function)

where:
  ω  — vorticity scalar field (= ∂v/∂x - ∂u/∂y)
  ψ  — stream function
  u  — velocity field (u, v)
  ν  — kinematic viscosity
  f  — external forcing (optional)

Domain
------
    (x, y) ∈ [0, L]²,  periodic BC,  t ∈ [0, T]

Solver
------
Implicit-Explicit Runge-Kutta, 2nd order (IMEX-RK2 / CN-Heun):

  - Diffusion (linear): handled implicitly via Crank-Nicolson in Fourier
    space → exact in the spectral basis.
  - Advection (nonlinear): handled explicitly with Heun's predictor-corrector
    (2-stage explicit RK).

This exactly matches the scheme used to generate the canonical FNO
Navier-Stokes benchmarks (Li et al. 2021, ``NavierStokes_V1e-3_N5000_T50.mat``).

Dealiasing
----------
Circular 2/3 rule: zero modes with ``|k|² > (N/3)²``.  This is more
isotropic than the rectangular 2/3 rule and preferred for turbulence.
The (0,0) mode is also zeroed to enforce zero-mean vorticity.

Forcing modes
-------------
``"kolmogorov"`` (default)
    Deterministic Kolmogorov-type:
    ``f = A * (sin(2π(x+y)/L) + cos(2π(x+y)/L))``

``"grf"``
    Stochastic: a new GRF sample is drawn each time step.

``None``
    No forcing (decaying turbulence).

``Tensor``
    User-supplied static forcing field of shape ``(N, N)``.

References
----------
- Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
- Gottlieb & Shu, "Total Variation Diminishing Runge-Kutta Schemes",
  Math. Comp. 67 (1998) — Heun method.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch
from torch import Tensor

from nops.data.generators.base import BaseGenerator
from nops.data.utils.gaussian_rf import GaussianRF


_ForcingType = Literal["kolmogorov", "grf"] | None | Tensor


class NavierStokesGenerator(BaseGenerator):
    """Pseudo-spectral CN-Heun generator for 2D incompressible Navier-Stokes.

    Parameters
    ----------
    N : int
        Spatial resolution (grid is N×N).  Default ``64``.
    L : float
        Domain length ``[0, L]²``.  Default ``2π``.
    nu : float
        Kinematic viscosity ν.  Typical FNO benchmarks use ``1e-3`` or ``1e-4``.
        Default ``1e-3``.
    T : float
        Final time.  Default ``1.0``.
    dt : float
        Time step.  Default ``1e-4``.
    record_steps : int
        Number of vorticity snapshots to record.  Default ``10``.
    forcing : {"kolmogorov", "grf", None} or Tensor
        External forcing ``f``.  See module docstring.  Default ``"kolmogorov"``.
    forcing_amplitude : float
        Amplitude ``A`` for Kolmogorov or GRF forcing.  Default ``0.1``.
    ic_alpha : float
        GRF smoothness for initial vorticity field.  Default ``2.5``.
    ic_tau : float
        GRF length-scale for initial vorticity field.  Default ``7.0``.
    device : str or torch.device
        Computation device.  Default ``"cpu"``.
    dtype : torch.dtype
        Output dtype.  Default ``torch.float32``.

    Examples
    --------
    >>> gen = NavierStokesGenerator(N=64, nu=1e-3, T=1.0, record_steps=1)
    >>> data = gen.generate(20)
    >>> data["ic"].shape         # (20, 64, 64)
    >>> data["vorticity"].shape  # (20, 64, 64, 1)
    >>> data["t_grid"].shape     # (1,)
    """

    def __init__(
        self,
        N: int = 64,
        L: float = 2.0 * math.pi,
        nu: float = 1e-3,
        T: float = 1.0,
        dt: float = 1e-4,
        record_steps: int = 10,
        forcing: _ForcingType = "kolmogorov",
        forcing_amplitude: float = 0.1,
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
        self.forcing_amplitude = forcing_amplitude

        self._grf_ic = GaussianRF(
            dim=2, N=N, alpha=ic_alpha, tau=ic_tau, L=L, device=device, dtype=dtype
        )

        # Precompute spectral operators
        self._build_spectral_ops()

        # Set up forcing
        self._forcing_mode = forcing if not isinstance(forcing, Tensor) else "tensor"
        if isinstance(forcing, Tensor):
            self._forcing_field = forcing.to(device=device, dtype=torch.float64)
        elif forcing == "grf":
            self._grf_forcing = GaussianRF(
                dim=2, N=N, alpha=2.5, tau=7.0, L=L, device=device, dtype=dtype
            )
        elif forcing == "kolmogorov":
            self._forcing_field = self._kolmogorov_forcing()
        elif forcing is None:
            self._forcing_field = None
        else:
            raise ValueError(
                f"forcing must be 'kolmogorov', 'grf', None, or a Tensor; got {forcing!r}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_samples: int, **kwargs) -> dict[str, Tensor]:
        """Generate ``n_samples`` Navier-Stokes vorticity trajectories.

        Parameters
        ----------
        n_samples : int
            Number of independent trajectories.

        Returns
        -------
        dict with keys:
            ``"ic"``         — initial vorticity, shape ``(n, N, N)``
            ``"vorticity"``  — snapshots, shape ``(n, N, N, record_steps)``
            ``"t_grid"``     — shape ``(record_steps,)``
        """
        w0 = self._grf_ic.sample(n_samples)  # (n, N, N), float32
        w0_f64 = w0.to(torch.float64)

        vort, t_grid = self._solve(w0_f64, n_samples)

        return {
            "ic": w0,
            "vorticity": vort.to(self.dtype),
            "t_grid": t_grid.to(self.dtype),
        }

    # ------------------------------------------------------------------
    # Spectral setup
    # ------------------------------------------------------------------

    def _build_spectral_ops(self) -> None:
        """Precompute wavenumber arrays and spectral operators."""
        N = self.N
        scale = 2.0 * math.pi / self.L

        # rfft2 wavenumbers
        # dim-0: full fft convention (0..N/2, -N/2+1..-1)
        # dim-1: rfft half (0..N/2)
        k0_int = torch.fft.fftfreq(N, d=1.0 / N, device=self.device).double()
        k1_int = torch.fft.rfftfreq(N, d=1.0 / N, device=self.device).double()
        K0, K1 = torch.meshgrid(k0_int, k1_int, indexing="ij")  # (N, N//2+1)

        # Physical wavenumbers
        self._k0 = scale * K0  # "y" direction
        self._k1 = scale * K1  # "x" direction (rfft half)

        # Negative Laplacian eigenvalues: |k|² = k0² + k1²
        k2 = self._k0.pow(2) + self._k1.pow(2)  # (N, N//2+1)

        # Inverse Laplacian (for ψ = ω / |k|²); zero at k=(0,0) → mean-free ψ
        inv_lap = k2.clone()
        inv_lap[0, 0] = 1.0
        self._inv_lap = 1.0 / inv_lap
        self._inv_lap[0, 0] = 0.0  # (N, N//2+1)

        # Viscous eigenvalues ν |k|²
        self._nu_k2 = self.nu * k2  # (N, N//2+1)

        # Circular dealiasing mask: |k|² ≤ (N/3)²
        # Use integer wavenumbers for the mask (resolution-independent)
        k2_int = K0.pow(2) + K1.pow(2)  # integer wavenumber magnitudes squared
        self._dealias = (k2_int <= (N / 3.0) ** 2).double()
        self._dealias[0, 0] = 0.0  # enforce zero-mean vorticity

    def _kolmogorov_forcing(self) -> Tensor:
        """Build deterministic Kolmogorov forcing field."""
        X, Y = self._spatial_grid_2d()
        X = X.double()
        Y = Y.double()
        arg = 2.0 * math.pi * (X + Y) / self.L
        f = self.forcing_amplitude * (torch.sin(arg) + torch.cos(arg))
        return f  # (N, N), float64

    def _get_forcing_hat(self, n_samples: int) -> Tensor | None:
        """Return the forcing field in Fourier space, shape (n, N, N//2+1)."""
        if self._forcing_mode is None:
            return None

        if self._forcing_mode in ("kolmogorov", "tensor"):
            f_hat = torch.fft.rfft2(self._forcing_field, dim=(-2, -1))
            return f_hat.unsqueeze(0).expand(n_samples, -1, -1)

        if self._forcing_mode == "grf":
            # Draw a new static GRF per trajectory (not per time step)
            f = self._grf_forcing.sample(n_samples).to(torch.float64)
            f = f * self.forcing_amplitude
            return torch.fft.rfft2(f, dim=(-2, -1))

        return None

    # ------------------------------------------------------------------
    # Solver (CN-Heun / IMEX-RK2)
    # ------------------------------------------------------------------

    def _nonlinear(self, w_hat: Tensor) -> Tensor:
        """Compute the nonlinear advection term ``-u·∇ω`` in Fourier space.

        Uses the divergence form ``-∂(uω)/∂x - ∂(vω)/∂y`` which exploits
        incompressibility (∇·u = 0) and avoids an extra FFT.
        """
        N = self.N
        # Stream function: ψ̂ = ω̂ / |k|²
        psi_hat = self._inv_lap * w_hat  # (n, N, N//2+1)

        # Velocity: u = ∂ψ/∂y → û = ik0·ψ̂;  v = -∂ψ/∂x → v̂ = -ik1·ψ̂
        u = torch.fft.irfft2(1j * self._k0 * psi_hat, s=(N, N), dim=(-2, -1))
        v = torch.fft.irfft2(-1j * self._k1 * psi_hat, s=(N, N), dim=(-2, -1))
        w = torch.fft.irfft2(w_hat, s=(N, N), dim=(-2, -1))

        # Advection in divergence form: ∂(uω)/∂x + ∂(vω)/∂y
        adv = (
            1j * self._k1 * torch.fft.rfft2(u * w, dim=(-2, -1))
            + 1j * self._k0 * torch.fft.rfft2(v * w, dim=(-2, -1))
        )
        return -adv  # shape (n, N, N//2+1)

    @torch.no_grad()
    def _solve(self, w0: Tensor, n_samples: int) -> tuple[Tensor, Tensor]:
        """Run the CN-Heun solver.

        Parameters
        ----------
        w0 : Tensor  shape ``(n, N, N)``, float64

        Returns
        -------
        vorticity : Tensor  shape ``(n, N, N, record_steps)``
        t_grid    : Tensor  shape ``(record_steps,)``
        """
        N = self.N
        n_steps = int(math.ceil(self.T / self.dt))
        record_every = max(1, n_steps // self.record_steps)

        w_hat = torch.fft.rfft2(w0, dim=(-2, -1))  # (n, N, N//2+1)
        f_hat = self._get_forcing_hat(n_samples)     # (n, N, N//2+1) or None

        # CN-Heun coefficients: diffusion handled implicitly
        # denominator for CN: 1 + 0.5 * dt * ν|k|²
        half_dt_nu_k2 = 0.5 * self.dt * self._nu_k2  # (N, N//2+1)
        denom = 1.0 + half_dt_nu_k2                   # (N, N//2+1)

        snapshots: list[Tensor] = []
        times: list[float] = []

        for step in range(n_steps):
            t = step * self.dt

            # Stage 1: nonlinear at current state
            N1 = self._nonlinear(w_hat)
            if f_hat is not None:
                N1 = N1 + f_hat

            # Predictor (inner Heun)
            w_star = (w_hat + self.dt * (N1 - half_dt_nu_k2 * w_hat)) / denom

            # Stage 2: nonlinear at predicted state
            N2 = self._nonlinear(w_star)
            if f_hat is not None:
                N2 = N2 + f_hat

            # Corrector (Crank-Nicolson + averaged nonlinear)
            w_hat = (
                w_hat
                + self.dt * (0.5 * (N1 + N2) - half_dt_nu_k2 * w_hat)
            ) / denom

            # Dealias
            w_hat = w_hat * self._dealias

            if (step + 1) % record_every == 0:
                w_phys = torch.fft.irfft2(w_hat, s=(N, N), dim=(-2, -1))
                snapshots.append(w_phys)
                times.append(t + self.dt)

            if len(snapshots) == self.record_steps:
                break

        while len(snapshots) < self.record_steps:
            snapshots.append(snapshots[-1])
            times.append(times[-1])

        vort = torch.stack(snapshots, dim=-1)  # (n, N, N, T_steps)
        t_grid = torch.tensor(times, device=self.device, dtype=torch.float64)
        return vort, t_grid
