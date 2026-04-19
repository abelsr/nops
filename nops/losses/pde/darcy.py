"""
Darcy flow residual loss — 2D (and generalized to N-D).

PDE (steady-state Darcy / elliptic diffusion)
----------------------------------------------
  -∇·(a(x) ∇u(x)) = f(x)   in Ω ⊂ ℝᵈ

where:
  u   — pressure / head field (scalar)
  a   — permeability / diffusivity coefficient (scalar field, > 0)
  f   — source term (scalar field)

This is the standard benchmark PDE used in FNO papers.  The neural
operator maps (a, x) → u.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from nops.losses.base import PhysicsLoss
from nops.losses.derivatives import divergence, gradient


class DarcyResidual(PhysicsLoss):
    """Physics-informed residual loss for the steady-state Darcy flow equation.

    Parameters
    ----------
    dim : int
        Spatial dimensionality (typically 2).

    Calling convention
    ------------------
    ``forward(model, x, a, f=None)``

    - ``model`` : callable, ``(a, x) -> u``
      - ``a`` : permeability field at collocation points, shape ``(N, 1)`` or ``(N,)``
      - ``x`` : spatial coords, shape ``(N, dim)``, ``requires_grad=True``
      - output ``u`` : shape ``(N, 1)`` or ``(N,)``

    - ``a`` : Tensor, shape ``(N, 1)`` or ``(N,)``
      Permeability / diffusion coefficient at each collocation point.

    - ``f`` : Tensor or callable, optional
      Source term. If a Tensor of shape ``(N,)``; if callable, ``f(x) -> (N,)``.
      Defaults to 0.

    Returns
    -------
    Tensor
        Scalar mean-squared PDE residual.

    Notes
    -----
    The strong form -∇·(a ∇u) = f is expanded as:
      -(∇a · ∇u + a ∇²u) = f
    which is computed using autograd on both ``a`` (treated as a field
    provided at points) and ``u``.

    Examples
    --------
    >>> loss_fn = DarcyResidual(dim=2)
    >>> loss = loss_fn(model, x, a, f=rhs)
    """

    def __init__(self, dim: int = 2) -> None:
        super().__init__()
        self.dim = dim

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x: Tensor,
        a: Tensor,
        f: Optional[Tensor | Callable] = None,
    ) -> Tensor:
        """Compute the Darcy residual loss.

        Parameters
        ----------
        model : callable
            ``(a, x) -> u``.
        x : Tensor
            Spatial collocation points ``(N, dim)``, ``requires_grad=True``.
        a : Tensor
            Permeability at collocation points ``(N,)`` or ``(N, 1)``.
        f : Tensor or callable, optional
            Source term.

        Returns
        -------
        Tensor
            Scalar residual loss.
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)

        u = model(a, x)
        if u.dim() == 2 and u.shape[-1] == 1:
            u = u.squeeze(-1)   # (N,)

        if a.dim() == 2 and a.shape[-1] == 1:
            a = a.squeeze(-1)   # (N,)

        # ∇u  shape (N, dim)
        grad_u = gradient(u, x)

        # a ∇u  shape (N, dim)
        a_grad_u = a.unsqueeze(-1) * grad_u

        # ∇·(a ∇u) = Σᵢ ∂(a ∂u/∂xᵢ)/∂xᵢ
        div_flux = divergence(a_grad_u, x)   # (N,)

        # Source term
        if f is None:
            rhs = torch.zeros_like(div_flux)
        elif callable(f):
            rhs = f(x)
            if rhs.dim() == 2:
                rhs = rhs.squeeze(-1)
        else:
            rhs = f
            if rhs.dim() == 2:
                rhs = rhs.squeeze(-1)

        residual = -div_flux - rhs    # should be ≈ 0
        return residual.pow(2).mean()

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
