"""
Generic PDE residual loss — user-supplied residual callable.

This module provides a flexible wrapper that lets users plug in any
PDE residual without writing a new class.  The residual callable receives
the model output and the differentiation utilities needed to construct
the strong-form residual.
"""

from __future__ import annotations

from typing import Any, Callable

from torch import Tensor

from nops.losses.base import PhysicsLoss
import nops.losses.derivatives as D


class GenericPDELoss(PhysicsLoss):
    """Physics-informed loss for a user-defined PDE residual.

    The user supplies a ``residual_fn`` that encodes the PDE.  This function
    receives:

    1. The model prediction (output of ``model(*inputs)``)
    2. The input tensors (the same ones passed to ``forward``)
    3. The ``nops.losses.derivatives`` module as a convenience import

    and must return a Tensor whose mean-squared value becomes the loss.

    Parameters
    ----------
    residual_fn : callable
        Signature: ``residual_fn(u, *inputs, derivatives) -> Tensor``

        - ``u``          : model output tensor
        - ``*inputs``    : the same positional inputs passed to ``forward``
        - ``derivatives``: the :mod:`nops.losses.derivatives` module, so users
          can call ``derivatives.laplacian(...)``, etc. directly
        - returns        : residual tensor of any shape (loss = mean of squares)

    reduction : {"mean", "sum"}
        How to reduce the squared residual.  Default ``"mean"``.

    Examples
    --------
    Define the 1D heat equation ∂u/∂t - α ∂²u/∂x² = 0:

    >>> def heat_residual(u, x, t, derivatives):
    ...     du_dt = derivatives.time_derivative(u.squeeze(-1), t)
    ...     lap_u = derivatives.laplacian(u.squeeze(-1), x)
    ...     return du_dt - 0.01 * lap_u
    ...
    >>> loss_fn = GenericPDELoss(residual_fn=heat_residual)
    >>> loss = loss_fn(model, x, t)   # x and t have requires_grad=True

    Define the 2D Burgers equation in one shot:

    >>> def burgers2d(u, x, t, derivatives):
    ...     vel = u[..., :2]
    ...     du_dt = derivatives.time_derivative(vel, t)
    ...     J = derivatives.jacobian(vel, x)
    ...     adv = (vel.unsqueeze(1) * J).sum(-1)
    ...     lap = derivatives.vector_laplacian(vel, x)
    ...     return du_dt + adv - 1e-3 * lap
    ...
    >>> loss_fn = GenericPDELoss(burgers2d)
    >>> loss = loss_fn(model, x, t)
    """

    def __init__(
        self,
        residual_fn: Callable,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.residual_fn = residual_fn
        self.reduction = reduction

    def forward(self, model: Callable, *inputs: Any) -> Tensor:  # type: ignore[override]
        """Compute the user-defined PDE residual loss.

        Parameters
        ----------
        model : callable
            The neural operator / network.  Called as ``model(*inputs)``.
        *inputs : Tensor
            Input tensors forwarded to the model and to ``residual_fn``.
            Any tensor that needs gradients should already have
            ``requires_grad=True`` set by the caller.

        Returns
        -------
        Tensor
            Scalar residual loss.
        """
        u = model(*inputs)
        residual = self.residual_fn(u, *inputs, derivatives=D)
        sq = residual.pow(2)
        return sq.mean() if self.reduction == "mean" else sq.sum()

    def extra_repr(self) -> str:
        name = getattr(self.residual_fn, "__name__", repr(self.residual_fn))
        return f"residual_fn={name}, reduction={self.reduction!r}"
