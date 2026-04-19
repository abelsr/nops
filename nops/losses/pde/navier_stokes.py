"""
Incompressible Navier-Stokes residual loss — 2D and 3D.

PDE (incompressible, constant density ρ=1)
------------------------------------------
Momentum:   ∂u/∂t + (u·∇)u + ∇p - ν∇²u = f
Continuity: ∇·u = 0

where:
  u  — velocity vector  (2D: (u,v); 3D: (u,v,w))
  p  — pressure scalar
  ν  — kinematic viscosity
  f  — body force (optional, defaults to 0)

The model is expected to output the concatenated field ``[u₁, ..., uₐ, p]``
of shape ``(N, d+1)`` where ``d`` is the spatial dimension.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from nops.losses.base import PhysicsLoss
from nops.losses.derivatives import (
    divergence,
    gradient,
    laplacian,
    time_derivative,
    vector_laplacian,
)


class NavierStokesResidual(PhysicsLoss):
    """Physics-informed residual loss for the incompressible Navier-Stokes equations.

    Parameters
    ----------
    nu : float
        Kinematic viscosity ν > 0.
    dim : {2, 3}
        Spatial dimensionality.
    momentum_weight : float
        Weight on the momentum residual (default 1.0).
    continuity_weight : float
        Weight on the divergence-free continuity constraint (default 1.0).

    Calling convention
    ------------------
    ``forward(model, x, t, forcing=None)``

    - ``model`` : callable, ``(x, t) -> output``
      - ``x``     : shape ``(N, dim)``, ``requires_grad=True``
      - ``t``     : shape ``(N, 1)``,   ``requires_grad=True``
      - output    : shape ``(N, dim+1)`` — first ``dim`` channels are velocity,
                    last channel is pressure.
    - ``forcing`` : optional tensor or callable
      - If a Tensor of shape ``(N, dim)``, treated as a constant body force.
      - If callable, called as ``forcing(x, t)`` → Tensor ``(N, dim)``.

    Returns
    -------
    Tensor
        Scalar combined (weighted) residual loss.

    Examples
    --------
    >>> loss_fn = NavierStokesResidual(nu=1e-3, dim=2)
    >>> loss = loss_fn(model, x, t)
    """

    def __init__(
        self,
        nu: float = 1e-3,
        dim: int = 2,
        momentum_weight: float = 1.0,
        continuity_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        self.nu = nu
        self.dim = dim
        self.momentum_weight = momentum_weight
        self.continuity_weight = continuity_weight

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x: Tensor,
        t: Tensor,
        forcing: Optional[Tensor | Callable] = None,
    ) -> Tensor:
        """Compute the Navier-Stokes residual loss.

        Parameters
        ----------
        model : callable
            ``(x, t) -> Tensor`` of shape ``(N, dim+1)``.
        x : Tensor
            Spatial collocation points ``(N, dim)``, ``requires_grad=True``.
        t : Tensor
            Time collocation points ``(N, 1)``, ``requires_grad=True``.
        forcing : Tensor or callable, optional
            Body force field ``f``.

        Returns
        -------
        Tensor
            Scalar residual loss.
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)
        if not t.requires_grad:
            t = t.requires_grad_(True)

        output = model(x, t)  # (N, dim+1)
        vel = output[..., : self.dim]     # (N, dim)
        p   = output[..., self.dim]       # (N,)

        # --- Continuity: ∇·u = 0 ---
        div_u = divergence(vel, x)        # (N,)
        loss_cont = div_u.pow(2).mean()

        # --- Momentum: ∂u/∂t + (u·∇)u + ∇p - ν∇²u = f ---
        du_dt = time_derivative(vel, t)                     # (N, dim)
        lap_u = vector_laplacian(vel, x)                    # (N, dim)
        grad_p = gradient(p, x)                             # (N, dim)

        # Advection (u·∇)uᵢ = Σⱼ uⱼ ∂uᵢ/∂xⱼ
        from nops.losses.derivatives import jacobian
        J = jacobian(vel, x)                                # (N, dim, dim)
        # J[:, i, j] = ∂uᵢ/∂xⱼ
        # (u·∇)u[:, i] = Σⱼ vel[:, j] * J[:, i, j]
        adv = (vel.unsqueeze(1) * J).sum(-1)                # (N, dim)

        # Body force
        if forcing is None:
            f = torch.zeros_like(vel)
        elif callable(forcing):
            f = forcing(x, t)
        else:
            f = forcing

        momentum_res = du_dt + adv + grad_p - self.nu * lap_u - f  # (N, dim)
        loss_mom = momentum_res.pow(2).mean()

        return self.momentum_weight * loss_mom + self.continuity_weight * loss_cont

    def extra_repr(self) -> str:
        return (
            f"nu={self.nu}, dim={self.dim}, "
            f"momentum_weight={self.momentum_weight}, "
            f"continuity_weight={self.continuity_weight}"
        )
