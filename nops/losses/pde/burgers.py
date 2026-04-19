"""
Burgers equation residual loss — 1D and 2D.

PDE (viscous Burgers)
---------------------
1D:  ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
2D:  ∂u/∂t + u ∂u/∂x + v ∂u/∂y = ν (∂²u/∂x² + ∂²u/∂y²)
     ∂v/∂t + u ∂v/∂x + v ∂v/∂y = ν (∂²v/∂x² + ∂²v/∂y²)

For inviscid Burgers set ``nu=0``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nops.losses.base import PhysicsLoss
from nops.losses.derivatives import gradient, laplacian, time_derivative


class BurgersResidual(PhysicsLoss):
    """Physics-informed residual loss for the viscous Burgers equation.

    Parameters
    ----------
    nu : float
        Kinematic viscosity ν ≥ 0.  Set to 0 for the inviscid case.
    dim : {1, 2}
        Spatial dimensionality.

    Calling convention
    ------------------
    The loss expects the model to map ``(x, t)`` to the solution field.

    **1D** — ``forward(model, x, t)``

    - ``model`` : callable, ``(x, t) -> u``
      - ``x``  : shape ``(N, 1)``, ``requires_grad=True``
      - ``t``  : shape ``(N, 1)``, ``requires_grad=True``
      - output ``u`` : shape ``(N, 1)`` or ``(N,)``

    **2D** — ``forward(model, x, t)``

    - ``model`` : callable, ``(x, t) -> (u, v)``
      - ``x``  : shape ``(N, 2)``, ``requires_grad=True``
      - ``t``  : shape ``(N, 1)``, ``requires_grad=True``
      - output : shape ``(N, 2)`` — first channel is ``u``, second is ``v``

    Returns
    -------
    Tensor
        Scalar mean-squared PDE residual.

    Examples
    --------
    >>> loss_fn = BurgersResidual(nu=1e-3, dim=1)
    >>> loss = loss_fn(model, x, t)
    """

    def __init__(self, nu: float = 1e-3, dim: int = 1) -> None:
        super().__init__()
        if dim not in (1, 2):
            raise ValueError(f"dim must be 1 or 2, got {dim}")
        self.nu = nu
        self.dim = dim

    def forward(self, model, x: Tensor, t: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the Burgers residual loss.

        Parameters
        ----------
        model : callable
            Neural operator / network, ``(x, t) -> u`` (or ``(u, v)`` in 2D).
        x : Tensor
            Spatial collocation points, shape ``(N, dim)``, ``requires_grad=True``.
        t : Tensor
            Time collocation points, shape ``(N, 1)``, ``requires_grad=True``.

        Returns
        -------
        Tensor
            Scalar residual loss.
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)
        if not t.requires_grad:
            t = t.requires_grad_(True)

        output = model(x, t)

        if self.dim == 1:
            return self._residual_1d(output, x, t)
        return self._residual_2d(output, x, t)

    def _residual_1d(self, u: Tensor, x: Tensor, t: Tensor) -> Tensor:
        if u.dim() == 2 and u.shape[-1] == 1:
            u = u.squeeze(-1)

        du_dt = time_derivative(u, t)               # (N,)
        du_dx = gradient(u, x)[..., 0]              # (N,)
        lap_u = laplacian(u, x)                     # (N,)

        residual = du_dt + u * du_dx - self.nu * lap_u
        return residual.pow(2).mean()

    def _residual_2d(self, output: Tensor, x: Tensor, t: Tensor) -> Tensor:
        # output shape: (N, 2)
        u = output[..., 0]  # x-velocity
        v = output[..., 1]  # y-velocity

        du_dt = time_derivative(u, t)
        dv_dt = time_derivative(v, t)

        grad_u = gradient(u, x)   # (N, 2)
        grad_v = gradient(v, x)

        lap_u = laplacian(u, x)
        lap_v = laplacian(v, x)

        # Advection: (u·∇)u,  (u·∇)v
        adv_u = u * grad_u[..., 0] + v * grad_u[..., 1]
        adv_v = u * grad_v[..., 0] + v * grad_v[..., 1]

        res_u = du_dt + adv_u - self.nu * lap_u
        res_v = dv_dt + adv_v - self.nu * lap_v

        return (res_u.pow(2) + res_v.pow(2)).mean()

    def extra_repr(self) -> str:
        return f"nu={self.nu}, dim={self.dim}"
