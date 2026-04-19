"""
Wave equation residual loss — 1D, 2D, 3D.

PDE (scalar wave / acoustic)
------------------------------
  ∂²u/∂t² = c² ∇²u + f(x, t)

where:
  u   — displacement / pressure scalar field
  c   — wave speed (> 0)
  f   — optional forcing term

The damped wave equation is also supported via an optional damping
coefficient γ:

  ∂²u/∂t² + γ ∂u/∂t = c² ∇²u + f
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from nops.losses.base import PhysicsLoss
from nops.losses.derivatives import laplacian, second_time_derivative, time_derivative


class WaveResidual(PhysicsLoss):
    """Physics-informed residual loss for the (damped) scalar wave equation.

    Parameters
    ----------
    c : float
        Wave speed c > 0.
    dim : {1, 2, 3}
        Spatial dimensionality.
    gamma : float
        Damping coefficient γ ≥ 0.  Set to 0 (default) for the undamped case.

    Calling convention
    ------------------
    ``forward(model, x, t, forcing=None)``

    - ``model`` : callable, ``(x, t) -> u``
      - ``x`` : shape ``(N, dim)``, ``requires_grad=True``
      - ``t`` : shape ``(N, 1)``,   ``requires_grad=True``
      - output ``u`` : shape ``(N, 1)`` or ``(N,)``
    - ``forcing`` : optional Tensor ``(N,)`` or callable ``(x, t) -> (N,)``

    Returns
    -------
    Tensor
        Scalar mean-squared PDE residual.

    Examples
    --------
    >>> loss_fn = WaveResidual(c=1.0, dim=2)
    >>> loss = loss_fn(model, x, t)

    >>> # Damped wave:
    >>> loss_fn = WaveResidual(c=1.0, dim=1, gamma=0.1)
    """

    def __init__(self, c: float = 1.0, dim: int = 1, gamma: float = 0.0) -> None:
        super().__init__()
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")
        if c <= 0:
            raise ValueError(f"c must be positive, got {c}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        self.c = c
        self.dim = dim
        self.gamma = gamma

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x: Tensor,
        t: Tensor,
        forcing: Optional[Tensor | Callable] = None,
    ) -> Tensor:
        """Compute the wave equation residual loss.

        Parameters
        ----------
        model : callable
            ``(x, t) -> u``.
        x : Tensor
            Spatial collocation points ``(N, dim)``, ``requires_grad=True``.
        t : Tensor
            Time collocation points ``(N, 1)``, ``requires_grad=True``.
        forcing : Tensor or callable, optional
            Source term ``f(x, t)``.

        Returns
        -------
        Tensor
            Scalar residual loss.
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)
        if not t.requires_grad:
            t = t.requires_grad_(True)

        u = model(x, t)
        if u.dim() == 2 and u.shape[-1] == 1:
            u = u.squeeze(-1)

        d2u_dt2 = second_time_derivative(u, t)   # (N,)
        lap_u   = laplacian(u, x)                # (N,)

        # Source term
        if forcing is None:
            f = torch.zeros_like(d2u_dt2)
        elif callable(forcing):
            f = forcing(x, t)
            if f.dim() == 2:
                f = f.squeeze(-1)
        else:
            f = forcing
            if f.dim() == 2:
                f = f.squeeze(-1)

        residual = d2u_dt2 - self.c**2 * lap_u - f

        # Optional damping term: + γ ∂u/∂t
        if self.gamma != 0.0:
            du_dt = time_derivative(u, t)
            residual = residual + self.gamma * du_dt

        return residual.pow(2).mean()

    def extra_repr(self) -> str:
        return f"c={self.c}, dim={self.dim}, gamma={self.gamma}"
