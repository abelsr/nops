"""
Boundary condition losses for physics-informed training.

Classes
-------
DirichletLoss
    Penalizes ``u(x_bc) ≠ g(x_bc)`` (essential BC).
NeumannLoss
    Penalizes ``∂u/∂n(x_bc) ≠ h(x_bc)`` (natural / flux BC).
PeriodicLoss
    Penalizes ``u(x_left) ≠ u(x_right)`` and optionally
    ``∂u/∂x(x_left) ≠ ∂u/∂x(x_right)`` for periodic domains.
RobinLoss
    Penalizes ``α u(x_bc) + β ∂u/∂n(x_bc) ≠ g(x_bc)`` (Robin / mixed BC).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .base import PhysicsLoss
from .derivatives import gradient


class DirichletLoss(PhysicsLoss):
    """Dirichlet (essential) boundary condition loss.

    Penalizes the squared difference between the model prediction and the
    prescribed boundary value:

        L = ‖u(x_bc) - g‖²

    Parameters
    ----------
    reduction : {"mean", "sum"}
        Reduction over the boundary points.

    Calling convention
    ------------------
    ``forward(model, x_bc, g)``

    - ``model`` : callable, ``x_bc -> u_bc``
    - ``x_bc``  : boundary collocation points, shape ``(M, d)``
    - ``g``     : prescribed values, shape ``(M,)`` or ``(M, out)``
                  or a callable ``g(x_bc) -> Tensor``

    Examples
    --------
    >>> loss_fn = DirichletLoss()
    >>> loss = loss_fn(model, x_bc, g=torch.zeros(M))     # homogeneous BC
    >>> loss = loss_fn(model, x_bc, g=lambda x: x[...,0]) # inhomogeneous BC
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x_bc: Tensor,
        g: Tensor | Callable,
    ) -> Tensor:
        u_bc = model(x_bc)

        if callable(g):
            g_val = g(x_bc)
        else:
            g_val = g

        diff = (u_bc - g_val).pow(2)
        return diff.mean() if self.reduction == "mean" else diff.sum()

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class NeumannLoss(PhysicsLoss):
    """Neumann (natural / flux) boundary condition loss.

    Penalizes the squared difference between the outward normal derivative of
    the model prediction and the prescribed flux:

        L = ‖(∇u · n)(x_bc) - h‖²

    Parameters
    ----------
    reduction : {"mean", "sum"}
        Reduction over the boundary points.

    Calling convention
    ------------------
    ``forward(model, x_bc, normals, h=0)``

    - ``model``   : callable, ``x_bc -> u_bc``
    - ``x_bc``    : boundary collocation points ``(M, d)``, ``requires_grad=True``
    - ``normals`` : outward unit normals ``(M, d)``
    - ``h``       : prescribed normal flux ``(M,)`` or scalar (default 0)
                    or callable ``h(x_bc) -> Tensor``

    Examples
    --------
    >>> loss_fn = NeumannLoss()
    >>> # Zero-flux (insulation) on the boundary:
    >>> loss = loss_fn(model, x_bc, normals=normals, h=0.0)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x_bc: Tensor,
        normals: Tensor,
        h: float | Tensor | Callable = 0.0,
    ) -> Tensor:
        if not x_bc.requires_grad:
            x_bc = x_bc.requires_grad_(True)

        u_bc = model(x_bc)
        if u_bc.dim() == 2 and u_bc.shape[-1] == 1:
            u_bc = u_bc.squeeze(-1)

        grad_u = gradient(u_bc, x_bc)               # (M, d)
        normal_deriv = (grad_u * normals).sum(-1)    # (M,)  ∇u · n

        if callable(h):
            h_val = h(x_bc)
            if h_val.dim() == 2:
                h_val = h_val.squeeze(-1)
        elif isinstance(h, Tensor):
            h_val = h
        else:
            h_val = torch.full_like(normal_deriv, float(h))

        diff = (normal_deriv - h_val).pow(2)
        return diff.mean() if self.reduction == "mean" else diff.sum()

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class PeriodicLoss(PhysicsLoss):
    """Periodic boundary condition loss.

    Enforces that the field (and optionally its gradient) is periodic:

        L = ‖u(x_left) - u(x_right)‖²
          + λ ‖∇u(x_left) - ∇u(x_right)‖²   (if ``match_gradient=True``)

    Parameters
    ----------
    match_gradient : bool
        Whether to also enforce gradient periodicity.  Default ``True``.
    gradient_weight : float
        Weight λ on the gradient-matching term.  Default 1.0.
    reduction : {"mean", "sum"}
        Reduction over boundary pairs.

    Calling convention
    ------------------
    ``forward(model, x_left, x_right)``

    - ``model``   : callable, ``x -> u``
    - ``x_left``  : "left" boundary points, shape ``(M, d)``
    - ``x_right`` : "right" boundary points, shape ``(M, d)``
      (paired by row with ``x_left``)

    Examples
    --------
    >>> loss_fn = PeriodicLoss()
    >>> loss = loss_fn(model, x_left, x_right)
    """

    def __init__(
        self,
        match_gradient: bool = True,
        gradient_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.match_gradient = match_gradient
        self.gradient_weight = gradient_weight
        self.reduction = reduction

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x_left: Tensor,
        x_right: Tensor,
    ) -> Tensor:
        if self.match_gradient and not x_left.requires_grad:
            x_left = x_left.requires_grad_(True)
        if self.match_gradient and not x_right.requires_grad:
            x_right = x_right.requires_grad_(True)

        u_left  = model(x_left)
        u_right = model(x_right)

        diff_u = (u_left - u_right).pow(2)
        loss = diff_u.mean() if self.reduction == "mean" else diff_u.sum()

        if self.match_gradient:
            if u_left.dim() == 2 and u_left.shape[-1] == 1:
                u_left  = u_left.squeeze(-1)
                u_right = u_right.squeeze(-1)

            g_left  = gradient(u_left,  x_left)
            g_right = gradient(u_right, x_right)
            diff_g  = (g_left - g_right).pow(2)
            grad_loss = diff_g.mean() if self.reduction == "mean" else diff_g.sum()
            loss = loss + self.gradient_weight * grad_loss

        return loss

    def extra_repr(self) -> str:
        return (
            f"match_gradient={self.match_gradient}, "
            f"gradient_weight={self.gradient_weight}, "
            f"reduction={self.reduction!r}"
        )


class RobinLoss(PhysicsLoss):
    """Robin (mixed) boundary condition loss.

    Penalizes:  ``α u(x_bc) + β (∇u · n)(x_bc) - g(x_bc) = 0``

    Parameters
    ----------
    alpha : float
        Coefficient on the Dirichlet part.
    beta : float
        Coefficient on the Neumann part.
    reduction : {"mean", "sum"}

    Calling convention
    ------------------
    ``forward(model, x_bc, normals, g)``

    - ``model``   : callable, ``x_bc -> u``
    - ``x_bc``    : boundary points, shape ``(M, d)``, ``requires_grad=True``
    - ``normals`` : outward unit normals ``(M, d)``
    - ``g``       : right-hand side ``(M,)`` or scalar or callable

    Examples
    --------
    >>> loss_fn = RobinLoss(alpha=1.0, beta=0.5)
    >>> loss = loss_fn(model, x_bc, normals, g=g_values)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x_bc: Tensor,
        normals: Tensor,
        g: float | Tensor | Callable = 0.0,
    ) -> Tensor:
        if not x_bc.requires_grad:
            x_bc = x_bc.requires_grad_(True)

        u_bc = model(x_bc)
        if u_bc.dim() == 2 and u_bc.shape[-1] == 1:
            u_bc = u_bc.squeeze(-1)

        grad_u = gradient(u_bc, x_bc)               # (M, d)
        normal_deriv = (grad_u * normals).sum(-1)    # (M,)

        if callable(g):
            g_val = g(x_bc).squeeze(-1) if g(x_bc).dim() == 2 else g(x_bc)
        elif isinstance(g, Tensor):
            g_val = g
        else:
            g_val = torch.full_like(u_bc, float(g))

        residual = self.alpha * u_bc + self.beta * normal_deriv - g_val
        diff = residual.pow(2)
        return diff.mean() if self.reduction == "mean" else diff.sum()

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, beta={self.beta}, reduction={self.reduction!r}"
