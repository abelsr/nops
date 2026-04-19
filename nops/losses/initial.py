"""
Initial condition losses for time-dependent physics-informed training.

Classes
-------
InitialConditionLoss
    Penalizes ``u(x, t=0) - u0(x)`` for scalar or vector fields.
InitialVelocityLoss
    Penalizes ``∂u/∂t(x, t=0) - v0(x)`` (for second-order-in-time PDEs
    such as the wave equation).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .base import PhysicsLoss
from .derivatives import time_derivative


class InitialConditionLoss(PhysicsLoss):
    """Loss enforcing the initial condition ``u(x, t=0) = u0(x)``.

    Parameters
    ----------
    reduction : {"mean", "sum"}
        Reduction over the collocation points.

    Calling convention
    ------------------
    ``forward(model, x, t0, u0)``

    - ``model`` : callable, ``(x, t) -> u``
    - ``x``     : spatial collocation points, shape ``(M, d)``
    - ``t0``    : initial time tensor, shape ``(M, 1)`` filled with
                  ``t_start`` (typically 0).  Can also be a scalar float.
    - ``u0``    : prescribed initial condition.
                  A Tensor of shape ``(M,)`` or ``(M, out)``,
                  or a callable ``u0(x) -> Tensor``.

    Examples
    --------
    >>> loss_fn = InitialConditionLoss()
    >>> t0 = torch.zeros(M, 1)
    >>> loss = loss_fn(model, x, t0, u0=torch.sin(x[..., 0]))
    >>> loss = loss_fn(model, x, t0, u0=lambda x: torch.sin(x[..., 0]))
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x: Tensor,
        t0: Tensor | float,
        u0: Tensor | Callable,
    ) -> Tensor:
        # Build t0 tensor if a scalar is given
        if not isinstance(t0, Tensor):
            t0 = torch.full((x.shape[0], 1), float(t0), device=x.device, dtype=x.dtype)

        u_pred = model(x, t0)

        if callable(u0):
            u0_val = u0(x)
        else:
            u0_val = u0

        diff = (u_pred - u0_val).pow(2)
        return diff.mean() if self.reduction == "mean" else diff.sum()

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class InitialVelocityLoss(PhysicsLoss):
    """Loss enforcing the initial velocity ``∂u/∂t(x, t=0) = v0(x)``.

    This is needed for second-order-in-time PDEs (e.g. wave equation)
    where both the initial displacement and the initial velocity must be
    specified.

    Parameters
    ----------
    reduction : {"mean", "sum"}

    Calling convention
    ------------------
    ``forward(model, x, t0, v0)``

    - ``model`` : callable, ``(x, t) -> u``
    - ``x``     : spatial collocation points, shape ``(M, d)``
    - ``t0``    : initial time tensor ``(M, 1)`` or scalar float, with
                  ``requires_grad=True`` (needed for time derivative).
    - ``v0``    : prescribed initial velocity.
                  Tensor ``(M,)`` / ``(M, out)`` or callable ``v0(x)``.

    Examples
    --------
    >>> loss_fn = InitialVelocityLoss()
    >>> t0 = torch.zeros(M, 1, requires_grad=True)
    >>> loss = loss_fn(model, x, t0, v0=torch.zeros(M))
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x: Tensor,
        t0: Tensor | float,
        v0: Tensor | Callable,
    ) -> Tensor:
        if not isinstance(t0, Tensor):
            t0 = torch.full(
                (x.shape[0], 1), float(t0),
                device=x.device, dtype=x.dtype, requires_grad=True,
            )
        elif not t0.requires_grad:
            t0 = t0.requires_grad_(True)

        u_pred = model(x, t0)
        if u_pred.dim() == 2 and u_pred.shape[-1] == 1:
            u_pred = u_pred.squeeze(-1)

        du_dt = time_derivative(u_pred, t0)   # ∂u/∂t at t=t0

        if callable(v0):
            v0_val = v0(x)
            if v0_val.dim() == 2:
                v0_val = v0_val.squeeze(-1)
        else:
            v0_val = v0

        diff = (du_dt - v0_val).pow(2)
        return diff.mean() if self.reduction == "mean" else diff.sum()

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"
