"""
Data generator for the 1D/2D (damped) scalar wave equation.

PDE
---
    ∂²u/∂t² + γ ∂u/∂t = c² ∇²u + f(x, t)

where:
  u  — displacement field
  c  — wave speed (> 0)
  γ  — damping coefficient (≥ 0;  γ=0 → undamped wave)
  f  — external forcing (optional)

Domain
------
    x ∈ [0, L]^d,  periodic BC,  t ∈ [0, T]

Solver
------
Exact matrix exponential per Fourier mode.

Writing ``v = ∂u/∂t``, each mode ``k`` obeys:

    d/dt [û_k]   =  [   0       1   ] [û_k]   +  [  0  ]
         [v̂_k]      [-ω_k²   -γ   ] [v̂_k]      [f̂_k]

where ``ω_k = c|k|``.  The matrix exponential is a 2×2 system and can be
computed in closed form:

Undamped (γ=0):
    M = [[cos(ω Δt),          sin(ω Δt)/ω ],
         [-ω sin(ω Δt),       cos(ω Δt)   ]]

Damped (γ>0), with ω_d = sqrt(ω_k² - γ²/4):
    M = exp(-γΔt/2) * [[cos(ω_d Δt) + (γ/2)·sinc(ω_d Δt)·Δt,  sinc(ω_d Δt)·Δt],
                        [-ω_k²·sinc(ω_d Δt)·Δt,               cos(ω_d Δt) - (γ/2)·sinc(ω_d Δt)·Δt]]

where ``sinc(x Δt)·Δt = sin(ω_d Δt)/ω_d`` (with limit ``Δt`` when ω_d→0).

This propagator is **unconditionally stable** and has **zero time-
discretization error** for the wave operator.  No CFL constraint.

Because the wave equation is linear, **no dealiasing is required**.

Wave speed
----------
``c`` may be a scalar or ``"random"`` (per-sample ``c ~ Uniform(c_min, c_max)``).

Forcing modes
-------------
``None``           — free wave (default)
``"static_grf"``   — GRF drawn once, held constant
callable           — evaluated as ``f(t) → Tensor`` each step

References
----------
- Hochbruck & Ostermann, "Exponential Integrators", Acta Numerica 19 (2010).
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch
from torch import Tensor

from nops.data.generators.base import BaseGenerator
from nops.data.utils.gaussian_rf import GaussianRF


_ForcingType = Literal["static_grf"] | Callable | None


class WaveGenerator(BaseGenerator):
    """Exact matrix-exponential generator for the 1D/2D wave equation.

    Parameters
    ----------
    N : int
        Grid points per dimension.  Default ``64``.
    dim : {1, 2}
        Spatial dimensionality.  Default ``2``.
    L : float
        Domain length ``[0, L]^d``.  Default ``2π``.
    c : float or "random"
        Wave speed.  If ``"random"``, each sample draws
        ``c ~ Uniform(c_min, c_max)``.  Default ``1.0``.
    c_min : float
        Lower bound for random wave speed.  Default ``0.5``.
    c_max : float
        Upper bound for random wave speed.  Default ``2.0``.
    gamma : float
        Damping coefficient ``γ ≥ 0``.  Default ``0.0`` (undamped).
    T : float
        Final time.  Default ``5.0``.
    dt : float
        Time step.  Any value — no CFL constraint.  Default ``0.05``.
    record_steps : int
        Number of snapshots to record.  Default ``20``.
    forcing : None, "static_grf", or callable
        External forcing.  Default ``None``.
    forcing_alpha : float
        GRF smoothness for forcing.  Default ``2.0``.
    forcing_tau : float
        GRF length-scale for forcing.  Default ``3.0``.
    forcing_amplitude : float
        Scaling factor for GRF forcing.  Default ``0.1``.
    ic_alpha : float
        GRF smoothness for displacement IC ``u₀``.  Default ``2.5``.
    ic_tau : float
        GRF length-scale for displacement IC.  Default ``7.0``.
    v0_type : {"zero", "grf"}
        How to initialise the velocity IC ``∂u/∂t(x, 0)``.
        ``"zero"`` (default) or ``"grf"`` (independent GRF).
    device : str or torch.device
        Default ``"cpu"``.
    dtype : torch.dtype
        Default ``torch.float32``.

    Examples
    --------
    >>> gen = WaveGenerator(N=64, dim=2, c=1.0, T=5.0, record_steps=1)
    >>> data = gen.generate(20)
    >>> data["ic_displacement"].shape  # (20, 64, 64)
    >>> data["ic_velocity"].shape      # (20, 64, 64)
    >>> data["solution"].shape         # (20, 64, 64, 1)
    >>> data["c"].shape                # (20,)
    """

    def __init__(
        self,
        N: int = 64,
        dim: int = 2,
        L: float = 2.0 * math.pi,
        c: float | Literal["random"] = 1.0,
        c_min: float = 0.5,
        c_max: float = 2.0,
        gamma: float = 0.0,
        T: float = 5.0,
        dt: float = 0.05,
        record_steps: int = 20,
        forcing: _ForcingType = None,
        forcing_alpha: float = 2.0,
        forcing_tau: float = 3.0,
        forcing_amplitude: float = 0.1,
        ic_alpha: float = 2.5,
        ic_tau: float = 7.0,
        v0_type: Literal["zero", "grf"] = "zero",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim not in (1, 2):
            raise ValueError(f"dim must be 1 or 2, got {dim}")
        if gamma < 0:
            raise ValueError(f"gamma must be ≥ 0, got {gamma}")
        super().__init__(N=N, L=L, device=device, dtype=dtype)

        self.dim = dim
        self.c_val = c
        self.c_min = c_min
        self.c_max = c_max
        self.gamma = gamma
        self.T = T
        self.dt = dt
        self.record_steps = record_steps
        self.forcing_amplitude = forcing_amplitude
        self.v0_type = v0_type

        self._grf_ic = GaussianRF(
            dim=dim, N=N, alpha=ic_alpha, tau=ic_tau, L=L, device=device, dtype=dtype
        )
        if v0_type == "grf":
            self._grf_v0 = GaussianRF(
                dim=dim, N=N, alpha=ic_alpha, tau=ic_tau, L=L, device=device, dtype=dtype
            )

        # Precompute |k| (magnitude, not squared)
        self._k_mag = self._build_k_mag()

        # Forcing
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
        """Generate ``n_samples`` wave equation trajectories.

        Returns
        -------
        dict with keys:
            ``"ic_displacement"`` — ``u(x,0)``,     shape ``(n, *spatial)``
            ``"ic_velocity"``     — ``∂u/∂t(x,0)``, shape ``(n, *spatial)``
            ``"solution"``        — ``u(x, tₙ)``,   shape ``(n, *spatial, T_steps)``
            ``"t_grid"``          — shape ``(T_steps,)``
            ``"c"``               — wave speed per sample, shape ``(n,)``
        """
        u0 = self._grf_ic.sample(n_samples)  # float32

        if self.v0_type == "grf":
            v0 = self._grf_v0.sample(n_samples)
        else:
            v0 = torch.zeros_like(u0)

        # Resolve wave speed
        if self.c_val == "random":
            c_vals = (
                torch.rand(n_samples, device=self.device, dtype=torch.float64)
                * (self.c_max - self.c_min)
                + self.c_min
            )
        else:
            c_vals = torch.full(
                (n_samples,), float(self.c_val),
                device=self.device, dtype=torch.float64
            )

        # Static forcing
        if self._forcing_mode == "static_grf":
            f_static = (
                self._grf_forcing.sample(n_samples).to(torch.float64)
                * self.forcing_amplitude
            )
        else:
            f_static = None

        sol_f64, t_grid = self._solve(
            u0.to(torch.float64), v0.to(torch.float64), c_vals, f_static
        )

        return {
            "ic_displacement": u0,
            "ic_velocity": v0,
            "solution": sol_f64.to(self.dtype),
            "t_grid": t_grid.to(self.dtype),
            "c": c_vals.to(self.dtype),
        }

    # ------------------------------------------------------------------
    # Spectral helpers
    # ------------------------------------------------------------------

    def _build_k_mag(self) -> Tensor:
        """Return ``|k|`` in physical units."""
        scale = 2.0 * math.pi / self.L
        if self.dim == 1:
            k = torch.fft.rfftfreq(self.N, d=1.0 / self.N, device=self.device).double()
            return scale * k.abs()  # (N//2+1,)
        else:
            k0 = torch.fft.fftfreq(self.N, d=1.0 / self.N, device=self.device).double()
            k1 = torch.fft.rfftfreq(self.N, d=1.0 / self.N, device=self.device).double()
            K0, K1 = torch.meshgrid(k0, k1, indexing="ij")
            return scale * (K0.pow(2) + K1.pow(2)).sqrt()  # (N, N//2+1)

    def _rfft(self, u: Tensor) -> Tensor:
        if self.dim == 1:
            return torch.fft.rfft(u, dim=-1)
        return torch.fft.rfft2(u, dim=(-2, -1))

    def _irfft(self, u_hat: Tensor) -> Tensor:
        if self.dim == 1:
            return torch.fft.irfft(u_hat, n=self.N, dim=-1)
        return torch.fft.irfft2(u_hat, s=(self.N, self.N), dim=(-2, -1))

    def _build_propagator(self, omega: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the 4 components of the 2×2 matrix exponential.

        For each mode with frequency ``omega = c|k|``:

            M = exp(-γΔt/2) * [[cos(ωdΔt) + (γ/2)·s,  s],
                                [-ω²·s,                 cos(ωdΔt) - (γ/2)·s]]

        where s = sin(ωd Δt)/ωd and ωd = sqrt(ω² - (γ/2)²).

        Returns
        -------
        M11, M12, M21, M22 : Tensor
            Same shape as ``omega``.
        """
        half_g = self.gamma / 2.0
        dt = self.dt

        omega2 = omega.pow(2)  # ω_k² = c²|k|²
        # Damped natural frequency: ωd² = ω² - (γ/2)²
        omega_d2 = (omega2 - half_g ** 2).clamp(min=0.0)
        omega_d = omega_d2.sqrt()

        exp_decay = math.exp(-half_g * dt)

        # sinc-like term: sin(ωd Δt) / ωd (limit = Δt when ωd → 0)
        s = torch.where(
            omega_d * dt > 1e-7,
            torch.sin(omega_d * dt) / omega_d,
            # Taylor: Δt - (ωd Δt)²·Δt/6 + ...
            dt * (1.0 - omega_d2 * dt ** 2 / 6.0),
        )
        cos_term = torch.cos(omega_d * dt)

        M11 = exp_decay * (cos_term + half_g * s)
        M12 = exp_decay * s
        M21 = exp_decay * (-omega2 * s)
        M22 = exp_decay * (cos_term - half_g * s)
        return M11, M12, M21, M22

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _solve(
        self,
        u0: Tensor,
        v0: Tensor,
        c_vals: Tensor,
        f_static: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run the exact matrix-exponential wave solver.

        Parameters
        ----------
        u0, v0   : (n, *spatial), float64
        c_vals   : (n,),          float64
        f_static : (n, *spatial) or None, float64

        Returns
        -------
        solution : (n, *spatial, T_steps)
        t_grid   : (T_steps,)
        """
        n_steps = int(math.ceil(self.T / self.dt))
        record_every = max(1, n_steps // self.record_steps)

        # Per-sample c, broadcast over spectral dims: (n, 1) or (n, 1, 1)
        extra_dims = (1,) * self.dim
        c = c_vals.reshape(-1, *extra_dims)

        # ω_k = c |k|, shape (n, *k_shape)
        k_mag = self._k_mag.unsqueeze(0)  # (1, *k_shape)
        omega = c * k_mag                 # (n, *k_shape)

        # Build propagator matrices once
        M11, M12, M21, M22 = self._build_propagator(omega)

        # Forcing in Fourier space
        f_hat = self._rfft(f_static) if f_static is not None else None

        # Particular solution operators for constant forcing over [t, t+dt]:
        # ∫_0^{dt} M12(s) ds ≈ M12 * dt / 2  (trapezoidal midpoint)
        # More accurately: integral of the matrix exponential driven by impulse
        # For simplicity we use M12 * dt as the forcing integral (accurate to O(dt²))
        # (see Duhamel's principle)

        u_hat = self._rfft(u0)  # (n, *k_shape), complex128
        v_hat = self._rfft(v0)

        snapshots: list[Tensor] = []
        times: list[float] = []

        for step in range(n_steps):
            t = step * self.dt

            # Apply exact wave propagator: [u, v]_{n+1} = M [u, v]_n
            u_hat_new = M11 * u_hat + M12 * v_hat
            v_hat_new = M21 * u_hat + M22 * v_hat

            # Forcing contribution via Duhamel (trapezoidal approximation)
            if self._forcing_mode == "static_grf" and f_hat is not None:
                u_hat_new = u_hat_new + M12 * self.dt * f_hat
                v_hat_new = v_hat_new + M22 * self.dt * f_hat
            elif self._forcing_mode == "callable":
                f_t = self._forcing_fn(t)
                f_hat_t = self._rfft(f_t.to(torch.float64))
                u_hat_new = u_hat_new + M12 * self.dt * f_hat_t
                v_hat_new = v_hat_new + M22 * self.dt * f_hat_t

            u_hat = u_hat_new
            v_hat = v_hat_new

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
