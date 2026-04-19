"""
Variational / energy-based physics-informed losses.

Instead of penalizing the strong-form PDE residual, these losses minimize
an energy functional whose Euler-Lagrange equation is the PDE of interest.

Classes
-------
DirichletEnergyLoss
    Minimizes ``½ ∫|∇u|² dx`` — equivalent to solving ∇²u = 0 (Laplace).
PoissonEnergyLoss
    Minimizes ``½ ∫|∇u|² dx - ∫ f u dx`` — Poisson weak form.
DarcyEnergyLoss
    Minimizes ``½ ∫ a|∇u|² dx - ∫ f u dx`` — Darcy weak form.
ElasticEnergyLoss
    Minimizes the linear elasticity strain energy
    ``μ ∫ ε:ε dx + λ/2 ∫ (∇·u)² dx``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .base import PhysicsLoss
from .derivatives import divergence, gradient


class DirichletEnergyLoss(PhysicsLoss):
    """Dirichlet energy loss ``½ ∫ |∇u|² dx``.

    Minimizing this functional (subject to boundary conditions) is
    equivalent to solving the Laplace equation ∇²u = 0.

    Parameters
    ----------
    reduction : {"mean", "sum"}
        Whether to average or sum over the collocation points
        (acts as a Monte-Carlo quadrature approximation of the integral).

    Calling convention
    ------------------
    ``forward(model, x)``

    - ``model`` : callable, ``x -> u``
    - ``x``     : collocation points ``(N, d)``, ``requires_grad=True``

    Examples
    --------
    >>> loss_fn = DirichletEnergyLoss()
    >>> loss = loss_fn(model, x)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, model: Callable, x: Tensor) -> Tensor:  # type: ignore[override]
        if not x.requires_grad:
            x = x.requires_grad_(True)

        u = model(x)
        if u.dim() == 2 and u.shape[-1] == 1:
            u = u.squeeze(-1)

        grad_u = gradient(u, x)                  # (N, d)
        energy = 0.5 * grad_u.pow(2).sum(-1)     # (N,)
        return energy.mean() if self.reduction == "mean" else energy.sum()

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class PoissonEnergyLoss(PhysicsLoss):
    """Variational energy for the Poisson equation.

    Minimizes ``½ ∫ |∇u|² dx - ∫ f u dx``.

    The Euler-Lagrange equation gives ``-∇²u = f``.

    Parameters
    ----------
    reduction : {"mean", "sum"}

    Calling convention
    ------------------
    ``forward(model, x, f=None)``

    - ``model`` : callable, ``x -> u``
    - ``x``     : collocation points ``(N, d)``, ``requires_grad=True``
    - ``f``     : source term, Tensor ``(N,)`` or callable ``f(x) -> (N,)``.
                  Defaults to 0 (→ Laplace energy).

    Examples
    --------
    >>> loss_fn = PoissonEnergyLoss()
    >>> loss = loss_fn(model, x, f=source_values)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x: Tensor,
        f: Optional[Tensor | Callable] = None,
    ) -> Tensor:
        if not x.requires_grad:
            x = x.requires_grad_(True)

        u = model(x)
        if u.dim() == 2 and u.shape[-1] == 1:
            u = u.squeeze(-1)

        grad_u = gradient(u, x)
        kinetic = 0.5 * grad_u.pow(2).sum(-1)   # (N,)

        if f is None:
            load = torch.zeros_like(u)
        elif callable(f):
            load = f(x)
            if load.dim() == 2:
                load = load.squeeze(-1)
        else:
            load = f
            if load.dim() == 2:
                load = load.squeeze(-1)

        energy = kinetic - load * u              # (N,)
        return energy.mean() if self.reduction == "mean" else energy.sum()

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class DarcyEnergyLoss(PhysicsLoss):
    """Variational energy for the Darcy flow equation.

    Minimizes ``½ ∫ a |∇u|² dx - ∫ f u dx``.

    The Euler-Lagrange equation gives ``-∇·(a ∇u) = f``.

    Parameters
    ----------
    reduction : {"mean", "sum"}

    Calling convention
    ------------------
    ``forward(model, x, a, f=None)``

    - ``model`` : callable, ``(a, x) -> u``
    - ``x``     : collocation points ``(N, d)``, ``requires_grad=True``
    - ``a``     : permeability, Tensor ``(N,)`` or ``(N, 1)``
    - ``f``     : source term, Tensor or callable (optional)

    Examples
    --------
    >>> loss_fn = DarcyEnergyLoss()
    >>> loss = loss_fn(model, x, a=permeability)
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(  # type: ignore[override]
        self,
        model: Callable,
        x: Tensor,
        a: Tensor,
        f: Optional[Tensor | Callable] = None,
    ) -> Tensor:
        if not x.requires_grad:
            x = x.requires_grad_(True)

        u = model(a, x)
        if u.dim() == 2 and u.shape[-1] == 1:
            u = u.squeeze(-1)

        if a.dim() == 2 and a.shape[-1] == 1:
            a = a.squeeze(-1)

        grad_u = gradient(u, x)                              # (N, d)
        kinetic = 0.5 * a * grad_u.pow(2).sum(-1)           # (N,)

        if f is None:
            load = torch.zeros_like(u)
        elif callable(f):
            load = f(x)
            if load.dim() == 2:
                load = load.squeeze(-1)
        else:
            load = f
            if load.dim() == 2:
                load = load.squeeze(-1)

        energy = kinetic - load * u                          # (N,)
        return energy.mean() if self.reduction == "mean" else energy.sum()

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}"


class ElasticEnergyLoss(PhysicsLoss):
    """Linear elasticity strain energy loss.

    Minimizes the stored elastic energy:

        W = μ ∫ ε:ε dx + λ/2 ∫ (∇·u)² dx

    where:
    - ``u``      — displacement vector field, shape ``(N, d)``
    - ``ε = ½(∇u + ∇uᵀ)`` — symmetric strain tensor
    - ``μ``, ``λ`` — Lamé parameters

    The Euler-Lagrange equation gives the Lamé equations of linear elasticity.

    Parameters
    ----------
    mu : float
        First Lamé parameter (shear modulus) μ > 0.
    lam : float
        Second Lamé parameter λ ≥ 0.
    dim : {2, 3}
        Spatial dimensionality.
    reduction : {"mean", "sum"}

    Calling convention
    ------------------
    ``forward(model, x)``

    - ``model`` : callable, ``x -> u`` with output shape ``(N, dim)``
    - ``x``     : collocation points ``(N, dim)``, ``requires_grad=True``

    Examples
    --------
    >>> loss_fn = ElasticEnergyLoss(mu=1.0, lam=0.5, dim=2)
    >>> loss = loss_fn(model, x)
    """

    def __init__(
        self,
        mu: float = 1.0,
        lam: float = 1.0,
        dim: int = 2,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        self.mu = mu
        self.lam = lam
        self.dim = dim
        self.reduction = reduction

    def forward(self, model: Callable, x: Tensor) -> Tensor:  # type: ignore[override]
        if not x.requires_grad:
            x = x.requires_grad_(True)

        u = model(x)   # (N, dim)

        from .derivatives import jacobian
        J = jacobian(u, x)                      # (N, dim, dim)  J[i,j] = ∂uᵢ/∂xⱼ

        # Symmetric strain tensor: ε = ½(J + Jᵀ)
        eps = 0.5 * (J + J.transpose(1, 2))     # (N, dim, dim)

        # Frobenius inner product ε:ε = Σᵢⱼ εᵢⱼ²
        eps_eps = eps.pow(2).sum(dim=(1, 2))     # (N,)

        # Divergence of displacement
        div_u = J.diagonal(dim1=1, dim2=2).sum(-1)   # (N,)  tr(J) = ∇·u

        energy = self.mu * eps_eps + 0.5 * self.lam * div_u.pow(2)  # (N,)
        return energy.mean() if self.reduction == "mean" else energy.sum()

    def extra_repr(self) -> str:
        return f"mu={self.mu}, lam={self.lam}, dim={self.dim}, reduction={self.reduction!r}"
