"""
Supervised data losses for neural operator training.

Classes
-------
DataLoss
    Flexible supervised loss supporting MSE, relative L2, and MAE reductions.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from .base import PhysicsLoss


class DataLoss(PhysicsLoss):
    """Supervised data-fit loss between predicted and target fields.

    Supports three common reduction modes used in the neural-operator
    literature:

    - ``"mse"``         — Mean Squared Error: ``mean((u_pred - u_true)²)``
    - ``"rel_l2"``      — Relative L2 norm:
                          ``‖u_pred - u_true‖₂ / ‖u_true‖₂``
                          (averaged over the batch)
    - ``"mae"``         — Mean Absolute Error: ``mean(|u_pred - u_true|)``

    Parameters
    ----------
    reduction : {"mse", "rel_l2", "mae"}
        Loss reduction type.  Default is ``"rel_l2"`` (standard in FNO/DeepONet
        papers).
    eps : float
        Small constant added to the denominator in ``"rel_l2"`` mode to avoid
        division by zero.

    Examples
    --------
    >>> loss_fn = DataLoss(reduction="rel_l2")
    >>> loss = loss_fn(u_pred, u_true)   # scalar tensor

    >>> # With a model callable:
    >>> loss = loss_fn(model(x), u_true)
    """

    def __init__(
        self,
        reduction: Literal["mse", "rel_l2", "mae"] = "rel_l2",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if reduction not in {"mse", "rel_l2", "mae"}:
            raise ValueError(
                f"reduction must be 'mse', 'rel_l2', or 'mae', got '{reduction}'"
            )
        self.reduction = reduction
        self.eps = eps

    def forward(self, u_pred: Tensor, u_true: Tensor) -> Tensor:  # type: ignore[override]
        """Compute the data loss.

        Parameters
        ----------
        u_pred : Tensor
            Predicted field, shape ``(N, ...)``.
        u_true : Tensor
            Ground-truth field, shape ``(N, ...)``.

        Returns
        -------
        Tensor
            Scalar loss value.
        """
        if u_pred.shape != u_true.shape:
            raise ValueError(
                f"Shape mismatch: u_pred {u_pred.shape} vs u_true {u_true.shape}"
            )

        diff = u_pred - u_true

        if self.reduction == "mse":
            return diff.pow(2).mean()

        if self.reduction == "rel_l2":
            # Flatten all but the batch dimension
            diff_flat = diff.reshape(diff.shape[0], -1)
            true_flat = u_true.reshape(u_true.shape[0], -1)
            num = diff_flat.norm(p=2, dim=-1)        # (N,)
            den = true_flat.norm(p=2, dim=-1) + self.eps
            return (num / den).mean()

        # mae
        return diff.abs().mean()

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}, eps={self.eps}"
