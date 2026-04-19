"""
Data generator for the 1D/2D heat (diffusion) equation.

PDE
---
    ∂u/∂t = α ∇²u + f(x, t)

Domain
------
    x ∈ [0, L]^d,  periodic BC,  t ∈ [0, T]

Solver
------
Exact integrating-factor method in Fourier space.

Each Fourier mode decouples into a simple ODE:

    dû_k/dt = -α|k|² û_k + f̂_k(t)

For a step ``[t, t + Δt]`` with frozen forcing:

    û_k(t + Δt) = û_k(t) · exp(-α|k|² Δt)
                + f̂_k(t) · (1 - exp(-α|k|² Δt)) / (α|k|²)

The first term is **exact** (zero time-discretization error on the diffusion
operator).  The second term introduces O(Δt²) error only if ``f`` varies in
time, and zero error for static forcing.

Because the heat equation is linear, **no dealiasing is required**.

Forcing modes
-------------
``None`` (default)
    Homogeneous heat equation: ``∂u/∂t = α∇²u``.

``"static_grf"``
    One GRF sample drawn at generation time, held constant throughout.

callable
    Evaluated as ``f(t) → Tensor`` at each recorded time step.

Diffusivity
-----------
``alpha`` may be a scalar or the string ``"random"`` (in which case each
sample gets an independent ``α ~ Uniform(alpha_min, alpha_max)``).  The
per-sample values are returned in the ``"alpha"`` output key.

References
----------
- Cox & Matthews, "Exponential Time Differencing for Stiff Systems",
  J. Comput. Phys. 176 (2002).
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch
from torch import Tensor

from nops.data.generators.base import BaseGenerator
from nops.data.utils.gaussian_rf import GaussianRF


_ForcingType = Literal["static_grf"] | Callable | None


class HeatGenerator(BaseGenerator):
    """Exact integrating-factor generator for the 1D/2D heat equation.

    Parameters
    ----------
    N : int
        Grid points per dimension.  Default ``64``.
    dim : {1, 2}
        Spatial dimensionality.  Default ``2``.
    L : float
        Domain length ``[0, L]^d``.  Default ``2π``.
    alpha : float or "random"
        Thermal diffusivity.  If ``"random"``, each sample draws
        ``α ~ Uniform(alpha_min, alpha_max)``.  Default ``0.01``.
    alpha_min : float
        Lower bound when ``alpha="random"``.  Default ``0.001``.
    alpha_max : float
        Upper bound when ``alpha="random"``.  Default ``0.1``.
    T : float
        Final time.  Default ``1.0``.
    dt : float
        Time step.  Any value works (no CFL constraint for the linear part).
        Default ``0.01``.
    record_steps : int
        Number of snapshots to record.  Default ``10``.
    forcing : None, "static_grf", or callable
        External heat source.  See module docstring.  Default ``None``.
    forcing_alpha : float
        GRF smoothness for static forcing.  Default ``2.0``.
    forcing_tau : float
        GRF length-scale for static forcing.  Default ``3.0``.
    forcing_amplitude : float
        Scaling factor applied to GRF forcing.  Default ``0.1``.
    ic_alpha : float
        GRF smoothness for initial condition.  Default ``2.5``.
    ic_tau : float
        GRF length-scale for initial condition.  Default ``7.0``.
    device : str or torch.device
        Computation device.  Default ``"cpu"``.
    dtype : torch.dtype
        Output dtype.  Default ``torch.float32``.

    Examples
    --------
    >>> gen = HeatGenerator(N=64, dim=2, alpha=0.01, T=1.0, record_steps=1)
    >>> data = gen.generate(100)
    >>> data["ic"].shape         # (100, 64, 64)
    >>> data["solution"].shape   # (100, 64, 64, 1)
    >>> data["alpha"].shape      # (100,)

    >>> # Random diffusivity
    >>> gen = HeatGenerator(alpha="random", alpha_min=0.001, alpha_max=0.1)
    >>> data = gen.generate(50)
    """

    def __init__(
        self,
        N: int = 64,
        dim: int = 2,
        L: float = 2.0 * math.pi,
        alpha: float | Literal["random"] = 0.01,
        alpha_min: float = 0.001,
        alpha_max: float = 0.1,
        T: float = 1.0,
        dt: float = 0.01,
        record_steps: int = 10,
        forcing: _ForcingType = None,
        forcing_alpha: float = 2.0,
        forcing_tau: float = 3.0,
        forcing_amplitude: float = 0.1,
        ic_alpha: float = 2.5,
        ic_tau: float = 7.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim not in (1, 2):
            raise ValueError(f"dim must be 1 or 2, got {dim}")
        super().__init__(N=N, L=L, device=device, dtype=dtype)

        self.dim = dim
        self.alpha_val = alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.T = T
        self.dt = dt
        self.record_steps = record_steps
        self.forcing_amplitude = forcing_amplitude

        self._grf_ic = GaussianRF(
            dim=dim, N=N, alpha=ic_alpha, tau=ic_tau, L=L, device=device, dtype=dtype
        )

        # Precompute |k|² once
        self._k2 = self._build_k2()

        # Forcing setup
        if forcing == "static_grf":
            self._forcing_mode = "static_grf"
            self._grf_forcing = GaussianRF(
                dim=dim, N=N, alpha=forcing_alpha, tau=forcing_tau,
                L=L, device=device, dtype=dtype,
            )
        elif callable(forcing):
            self._forcing_mode = "callable"
            self._forcing_fn: Callable = forcing
        elif forcing is None:
            self._forcing_mode = None
        else:
            raise ValueError(
                f"forcing must be None, 'static_grf', or a callable; got {forcing!r}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_samples: int, **kwargs) -> dict[str, Tensor]:
        """Generate ``n_samples`` heat equation trajectories.

        Returns
        -------
        dict with keys:
            ``"ic"``       — shape ``(n, N)`` or ``(n, N, N)``
            ``"solution"`` — shape ``(n, N, T_steps)`` or ``(n, N, N, T_steps)``
            ``"t_grid"``   — shape ``(T_steps,)``
            ``"alpha"``    — diffusivity per sample, shape ``(n,)``
        """
        u0 = self._grf_ic.sample(n_samples)  # float32 spatial IC

        # Resolve per-sample diffusivity
        if self.alpha_val == "random":
            alphas = (
                torch.rand(n_samples, device=self.device, dtype=torch.float64)
                * (self.alpha_max - self.alpha_min)
                + self.alpha_min
            )
        else:
            alphas = torch.full(
                (n_samples,), float(self.alpha_val),
                device=self.device, dtype=torch.float64
            )

        # Resolve static forcing
        if self._forcing_mode == "static_grf":
            f_static = (
                self._grf_forcing.sample(n_samples).to(torch.float64)
                * self.forcing_amplitude
            )  # (n, *spatial)
        else:
            f_static = None

        sol_f64, t_grid = self._solve(
            u0.to(torch.float64), alphas, f_static
        )

        return {
            "ic": u0,
            "solution": sol_f64.to(self.dtype),
            "t_grid": t_grid.to(self.dtype),
            "alpha": alphas.to(self.dtype),
        }

    # ------------------------------------------------------------------
    # Spectral helpers
    # ------------------------------------------------------------------

    def _build_k2(self) -> Tensor:
        """Return ``|k|²`` in physical units, shape ``(N//2+1,)`` or ``(N, N//2+1)``."""
        scale = 2.0 * math.pi / self.L
        if self.dim == 1:
            k = torch.fft.rfftfreq(self.N, d=1.0 / self.N, device=self.device).double()
            return (scale * k).pow(2)  # (N//2+1,)
        else:
            k0 = torch.fft.fftfreq(self.N, d=1.0 / self.N, device=self.device).double()
            k1 = torch.fft.rfftfreq(self.N, d=1.0 / self.N, device=self.device).double()
            K0, K1 = torch.meshgrid(k0, k1, indexing="ij")
            return (scale * K0).pow(2) + (scale * K1).pow(2)  # (N, N//2+1)

    @staticmethod
    def _phi1_stable(z: Tensor) -> Tensor:
        """Compute φ₁(z) = (exp(z) - 1) / z  stable at z → 0."""
        return torch.where(
            z.abs() < 1e-10,
            torch.ones_like(z),
            torch.expm1(z) / z,
        )

    def _rfft(self, u: Tensor) -> Tensor:
        if self.dim == 1:
            return torch.fft.rfft(u, dim=-1)
        return torch.fft.rfft2(u, dim=(-2, -1))

    def _irfft(self, u_hat: Tensor) -> Tensor:
        if self.dim == 1:
            return torch.fft.irfft(u_hat, n=self.N, dim=-1)
        return torch.fft.irfft2(u_hat, s=(self.N, self.N), dim=(-2, -1))

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _solve(
        self,
        u0: Tensor,
        alphas: Tensor,
        f_static: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run the exact IF solver.

        Parameters
        ----------
        u0     : (n, *spatial), float64
        alphas : (n,),           float64
        f_static : (n, *spatial) or None, float64

        Returns
        -------
        solution : (n, *spatial, T_steps)
        t_grid   : (T_steps,)
        """
        n_steps = int(math.ceil(self.T / self.dt))
        record_every = max(1, n_steps // self.record_steps)

        # Per-sample alpha shapes: (n, 1) for 1D, (n, 1, 1) for 2D
        extra_dims = (1,) * self.dim
        a = alphas.view(n_steps if False else alphas.shape[0], *extra_dims)
        # Actually: shape (n, 1) or (n, 1, 1)
        a = alphas.reshape(-1, *extra_dims)

        # |k|² broadcast: (1, *k2_shape)
        k2 = self._k2.unsqueeze(0)  # (1, N//2+1) or (1, N, N//2+1)

        # Precompute decay and forcing factor for this dt
        z = -a * k2 * self.dt  # (n, *k2_shape), float64
        decay = torch.exp(z)                                # exp(-α|k|²Δt)
        forcing_factor = self._phi1_stable(z) * self.dt    # Δt · φ₁(z)

        # Fourier-transform IC and static forcing
        u_hat = self._rfft(u0)  # (n, *k_shape)
        f_hat = self._rfft(f_static) if f_static is not None else None

        snapshots: list[Tensor] = []
        times: list[float] = []

        for step in range(n_steps):
            t = step * self.dt

            # Exact diffusion step
            u_hat = decay * u_hat

            # Add forcing contribution
            if self._forcing_mode == "static_grf" and f_hat is not None:
                u_hat = u_hat + forcing_factor * f_hat
            elif self._forcing_mode == "callable":
                f_t = self._forcing_fn(t)
                f_hat_t = self._rfft(f_t.to(torch.float64))
                u_hat = u_hat + forcing_factor * f_hat_t

            if (step + 1) % record_every == 0:
                snapshots.append(self._irfft(u_hat))
                times.append(t + self.dt)

            if len(snapshots) == self.record_steps:
                break

        while len(snapshots) < self.record_steps:
            snapshots.append(snapshots[-1])
            times.append(times[-1])

        sol = torch.stack(snapshots, dim=-1)  # (n, *spatial, T_steps)
        t_grid = torch.tensor(times, device=self.device, dtype=torch.float64)
        return sol, t_grid
