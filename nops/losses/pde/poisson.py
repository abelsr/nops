"""
Poisson equation residual loss — 1D, 2D, 3D.

PDE (elliptic)
--------------
  ∇²u = f(x)   (or equivalently  -∇²u = f  with sign convention)

where:
  u   — scalar field (e.g. electrostatic potential, pressure)
  f   — source / right-hand-side term

This is the steady-state limit of the heat equation (α → ∞, t → ∞)
and also a special case of the Darcy equation with constant permeability.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from nops.losses.base import PhysicsLoss
from nops.losses.derivatives import laplacian


class PoissonResidual(PhysicsLoss):
    """Physics-informed residual loss for the Poisson equation.

    Parameters
    ----------
    dim : {1, 2, 3}
        Spatial dimensionality.
    sign : {1, -1}
        Convention sign:
        - ``1``  : residual form ``∇²u - f = 0``
        - ``-1`` : residual form ``-∇²u - f = 0``  (most common in physics)

    Calling convention
    ------------------
    ``forward(model, x, f=None)``

    - ``model`` : callable, ``x -> u``
      - ``x``    : shape ``(N, dim)``, ``requires_grad=True``
      - output   : shape ``(N, 1)`` or ``(N,)``
    - ``f`` : Tensor ``(N,)`` or callable ``x -> (N,)``, optional.
      Defaults to 0 (Laplace equation).

    Returns
    -------
    Tensor
        Scalar mean-squared PDE residual.

    Examples
    --------
    >>> loss_fn = PoissonResidual(dim=2)
    >>> loss = loss_fn(model, x, f=rhs)

    >>> # Laplace equation (f=0):
    >>> loss = loss_fn(model, x)
    """

    def __init__(self, dim: int = 2, sign: int = -1) -> None:
        super().__init__()
        if dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3, got {dim}")
        if sign not in (1, -1):
            raise ValueError(f"sign must be 1 or -1, got {sign}")
        self.dim = dim
        self.sign = sign

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x: Tensor,
        f: Optional[Tensor | Callable] = None,
    ) -> Tensor:
        """Compute the Poisson residual loss.

        Parameters
        ----------
        model : callable
            ``x -> u``.
        x : Tensor
            Spatial collocation points ``(N, dim)``, ``requires_grad=True``.
        f : Tensor or callable, optional
            Source term.

        Returns
        -------
        Tensor
            Scalar residual loss.
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)

        u = model(x)
        if u.dim() == 2 and u.shape[-1] == 1:
            u = u.squeeze(-1)

        lap_u = laplacian(u, x)   # (N,)

        # Source term
        if f is None:
            rhs = torch.zeros_like(lap_u)
        elif callable(f):
            rhs = f(x)
            if rhs.dim() == 2:
                rhs = rhs.squeeze(-1)
        else:
            rhs = f
            if rhs.dim() == 2:
                rhs = rhs.squeeze(-1)

        # Residual: sign * ∇²u - f = 0
        residual = self.sign * lap_u - rhs
        return residual.pow(2).mean()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, sign={self.sign}"
