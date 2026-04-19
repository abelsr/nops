"""
Base classes for physics-informed losses.

Classes
-------
PhysicsLoss
    Abstract base class that all loss functions in this library inherit from.
CombinedLoss
    Weighted combination of multiple ``PhysicsLoss`` instances. Returns both
    a scalar total loss and a dictionary of per-term scalar values for logging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class PhysicsLoss(nn.Module, ABC):
    """Abstract base class for physics-informed loss functions.

    All concrete losses must implement :meth:`forward`.  The signature of
    ``forward`` is intentionally flexible (``*args, **kwargs``) so that each
    PDE loss can declare exactly the inputs it needs without forcing a rigid
    shared interface.

    Subclasses should call ``super().__init__()`` in their ``__init__``.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return a scalar loss value.

        Returns
        -------
        Tensor
            Scalar loss (0-dimensional tensor).
        """
        ...

    # ------------------------------------------------------------------
    # Convenience: allow instances to be used as plain functions
    # ------------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:  # type: ignore[override]
        return super().__call__(*args, **kwargs)


class CombinedLoss(nn.Module):
    """Weighted sum of named :class:`PhysicsLoss` (or any callable) instances.

    Parameters
    ----------
    losses : dict[str, tuple[float, PhysicsLoss | Callable]]
        Mapping from a human-readable name to a ``(weight, loss_fn)`` pair.
        ``weight`` may be a Python float or a 0-d ``torch.Tensor``
        (e.g. a learnable parameter).

    Examples
    --------
    >>> from nops.losses import CombinedLoss, DataLoss
    >>> from nops.losses.pde import BurgersResidual, DirichletLoss
    >>>
    >>> loss_fn = CombinedLoss({
    ...     "data":     (1.0,  DataLoss()),
    ...     "pde":      (1e-3, BurgersResidual(nu=1e-3, dim=1)),
    ...     "boundary": (1e-2, DirichletLoss()),
    ... })
    >>>
    >>> total, terms = loss_fn(
    ...     data=(x_data, u_pred, u_true),
    ...     pde=(x_t,),
    ...     boundary=(x_bc, u_bc),
    ... )
    >>> # terms == {"data": tensor(...), "pde": tensor(...), "boundary": tensor(...)}

    Calling convention
    ------------------
    ``CombinedLoss.forward`` accepts keyword arguments whose keys match the
    names registered in ``losses``.  The value for each key is either:

    - A single tensor passed directly to that loss function, or
    - A tuple/list of positional arguments, or
    - A dict of keyword arguments (passed as ``**kwargs`` to the loss).

    Mixed positional/keyword can be passed as a 2-tuple
    ``(args_tuple, kwargs_dict)``.
    """

    def __init__(
        self,
        losses: Dict[str, Tuple[float | Tensor, PhysicsLoss | Callable]],
    ) -> None:
        super().__init__()
        self._weights: Dict[str, float | Tensor] = {}
        self._loss_fns: nn.ModuleDict = nn.ModuleDict()
        self._callable_fns: Dict[str, Callable] = {}

        for name, (weight, loss_fn) in losses.items():
            self._weights[name] = weight
            if isinstance(loss_fn, nn.Module):
                self._loss_fns[name] = loss_fn
            else:
                self._callable_fns[name] = loss_fn

    # ------------------------------------------------------------------
    # Make weights accessible as learnable parameters when they are tensors
    # ------------------------------------------------------------------
    def get_weight(self, name: str) -> float | Tensor:
        return self._weights[name]

    def set_weight(self, name: str, value: float | Tensor) -> None:
        self._weights[name] = value

    # ------------------------------------------------------------------
    def _get_fn(self, name: str) -> Callable:
        if name in self._loss_fns:
            return self._loss_fns[name]
        return self._callable_fns[name]

    @staticmethod
    def _call_fn(fn: Callable, inputs: Any) -> Tensor:
        """Dispatch inputs to ``fn`` using a flexible calling convention."""
        if inputs is None:
            return fn()
        if isinstance(inputs, Tensor):
            return fn(inputs)
        if isinstance(inputs, dict):
            return fn(**inputs)
        if (
            isinstance(inputs, (tuple, list))
            and len(inputs) == 2
            and isinstance(inputs[0], (tuple, list))
            and isinstance(inputs[1], dict)
        ):
            args, kwargs = inputs
            return fn(*args, **kwargs)
        if isinstance(inputs, (tuple, list)):
            return fn(*inputs)
        return fn(inputs)

    def forward(self, **named_inputs: Any) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute the weighted total loss.

        Parameters
        ----------
        **named_inputs
            One keyword argument per registered loss name.  See class docstring
            for the supported input formats.

        Returns
        -------
        total : Tensor
            Scalar weighted sum of all active losses.
        terms : dict[str, Tensor]
            Per-term scalar loss values (before weighting), for logging.
        """
        total = torch.tensor(0.0)
        terms: Dict[str, Tensor] = {}

        for name in self._weights:
            fn = self._get_fn(name)
            inputs = named_inputs.get(name, None)
            value = self._call_fn(fn, inputs)
            terms[name] = value.detach()
            weight = self._weights[name]
            total = total + weight * value

        return total, terms

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for name, weight in self._weights.items():
            fn = self._get_fn(name)
            lines.append(f"  {name}: weight={weight}, fn={fn}")
        lines.append(")")
        return "\n".join(lines)
