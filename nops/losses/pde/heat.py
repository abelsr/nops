"""
Heat / diffusion equation residual loss — 1D, 2D, 3D.

PDE (parabolic diffusion)
--------------------------
  ∂u/∂t = α ∇²u + f(x, t)

where:
  u   — temperature / concentration scalar field
  α   — thermal diffusivity (> 0)
  f   — optional heat source / sink term

Steady-state limit (f = 0, time-independent): reduces to Poisson ∇²u = 0.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from nops.losses.base import PhysicsLoss
from nops.losses.derivatives import laplacian, time_derivative


class HeatResidual(PhysicsLoss):
    """Physics-informed residual loss for the heat / diffusion equation.

    Parameters
    ----------
    alpha : float
        Thermal diffusivity α > 0.
    dim : {1, 2, 3}
        Spatial dimensionality.

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
    >>> loss_fn = HeatResidual(alpha=0.01, dim=2)
    >>> loss = loss_fn(model, x, t)
    """

    def __init__(self, alpha: float = 1.0, dim: int = 2) -> None:
        super().__init__()
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha
        self.dim = dim

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x: Tensor,
        t: Tensor,
        forcing: Optional[Tensor | Callable] = None,
    ) -> Tensor:
        """Compute the heat equation residual loss.

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

        du_dt = time_derivative(u, t)    # (N,)
        lap_u = laplacian(u, x)          # (N,)

        # Source term
        if forcing is None:
            f = torch.zeros_like(du_dt)
        elif callable(forcing):
            f = forcing(x, t)
            if f.dim() == 2:
                f = f.squeeze(-1)
        else:
            f = forcing
            if f.dim() == 2:
                f = f.squeeze(-1)

        residual = du_dt - self.alpha * lap_u - f
        return residual.pow(2).mean()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, dim={self.dim}"
