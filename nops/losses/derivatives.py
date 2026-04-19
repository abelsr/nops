"""
Autograd-based differential operators for physics-informed losses.

All operators use ``torch.autograd.grad`` with ``create_graph=True`` so that
loss gradients can be back-propagated through them during training.

Conventions
-----------
- ``u``  : output tensor of shape ``(N, ...)`` where N is the batch / point count.
- ``x``  : input tensor of shape ``(N, d)`` with ``requires_grad=True``,
           where ``d`` is the spatial (or spatio-temporal) dimension.
- ``t``  : time tensor of shape ``(N, 1)`` with ``requires_grad=True``.

All operators return tensors of the same leading batch size as ``u``.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _scalar_grad(u: Tensor, x: Tensor, *, create_graph: bool = True) -> Tensor:
    """Gradient of a scalar field u w.r.t. x.

    Parameters
    ----------
    u : Tensor
        Scalar field, shape ``(N,)`` or ``(N, 1)``.
    x : Tensor
        Coordinates, shape ``(N, d)``, ``requires_grad=True``.

    Returns
    -------
    Tensor
        Gradient ``∇u``, shape ``(N, d)``.
    """
    if u.dim() > 1 and u.shape[-1] == 1:
        u = u.squeeze(-1)
    grad_outputs = torch.ones_like(u)
    (grad_u,) = torch.autograd.grad(
        u,
        x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=True,
    )
    return grad_u  # (N, d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gradient(u: Tensor, x: Tensor, *, create_graph: bool = True) -> Tensor:
    """Gradient ``∇u`` of a scalar field.

    Parameters
    ----------
    u : Tensor
        Scalar field, shape ``(N,)`` or ``(N, 1)``.
    x : Tensor
        Coordinates with ``requires_grad=True``, shape ``(N, d)``.
    create_graph : bool
        Keep the computation graph for higher-order derivatives.

    Returns
    -------
    Tensor
        ``∇u``, shape ``(N, d)``.

    Examples
    --------
    >>> x = torch.rand(32, 2, requires_grad=True)
    >>> u = (x ** 2).sum(-1)          # u = x₁² + x₂²
    >>> grad_u = gradient(u, x)       # should be ≈ 2x
    """
    return _scalar_grad(u, x, create_graph=create_graph)


def jacobian(u: Tensor, x: Tensor, *, create_graph: bool = True) -> Tensor:
    """Full Jacobian ``J[i,j] = ∂uᵢ/∂xⱼ`` of a vector field.

    Parameters
    ----------
    u : Tensor
        Vector field, shape ``(N, m)``.
    x : Tensor
        Coordinates with ``requires_grad=True``, shape ``(N, d)``.

    Returns
    -------
    Tensor
        Jacobian, shape ``(N, m, d)``.
    """
    N, m = u.shape[0], u.shape[-1]
    rows = []
    for i in range(m):
        rows.append(_scalar_grad(u[..., i], x, create_graph=create_graph))
    return torch.stack(rows, dim=1)  # (N, m, d)


def divergence(u: Tensor, x: Tensor, *, create_graph: bool = True) -> Tensor:
    """Divergence ``∇·u`` of a vector field.

    Parameters
    ----------
    u : Tensor
        Vector field, shape ``(N, d)``.  Must have ``u.shape[-1] == x.shape[-1]``.
    x : Tensor
        Coordinates with ``requires_grad=True``, shape ``(N, d)``.

    Returns
    -------
    Tensor
        Divergence ``∇·u``, shape ``(N,)``.

    Examples
    --------
    >>> x = torch.rand(32, 3, requires_grad=True)
    >>> u = x                          # u = (x, y, z)  →  ∇·u = 3
    >>> div_u = divergence(u, x)       # should be ≈ 3
    """
    d = x.shape[-1]
    div = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
    for i in range(d):
        grad_outputs = torch.ones(u.shape[0], device=u.device, dtype=u.dtype)
        (g,) = torch.autograd.grad(
            u[..., i],
            x,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
        )
        div = div + g[..., i]
    return div  # (N,)


def laplacian(u: Tensor, x: Tensor, *, create_graph: bool = True) -> Tensor:
    """Scalar Laplacian ``∇²u = Σᵢ ∂²u/∂xᵢ²``.

    Parameters
    ----------
    u : Tensor
        Scalar field, shape ``(N,)`` or ``(N, 1)``.
    x : Tensor
        Coordinates with ``requires_grad=True``, shape ``(N, d)``.

    Returns
    -------
    Tensor
        ``∇²u``, shape ``(N,)``.

    Examples
    --------
    >>> x = torch.rand(32, 2, requires_grad=True)
    >>> u = (x ** 2).sum(-1)          # u = x₁² + x₂²  →  ∇²u = 4
    >>> lap_u = laplacian(u, x)       # should be ≈ 4
    """
    grad_u = _scalar_grad(u, x, create_graph=True)  # (N, d)
    d = x.shape[-1]
    lap = torch.zeros(grad_u.shape[0], device=u.device, dtype=u.dtype)
    for i in range(d):
        grad_outputs = torch.ones(grad_u.shape[0], device=grad_u.device, dtype=grad_u.dtype)
        (g2,) = torch.autograd.grad(
            grad_u[..., i],
            x,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
        )
        lap = lap + g2[..., i]
    return lap  # (N,)


def vector_laplacian(u: Tensor, x: Tensor, *, create_graph: bool = True) -> Tensor:
    """Component-wise Laplacian of a vector field ``∇²uᵢ`` for each component i.

    Parameters
    ----------
    u : Tensor
        Vector field, shape ``(N, m)``.
    x : Tensor
        Coordinates with ``requires_grad=True``, shape ``(N, d)``.

    Returns
    -------
    Tensor
        ``∇²u``, shape ``(N, m)``.
    """
    m = u.shape[-1]
    cols = []
    for i in range(m):
        cols.append(laplacian(u[..., i], x, create_graph=create_graph))
    return torch.stack(cols, dim=-1)  # (N, m)


def curl(u: Tensor, x: Tensor, *, create_graph: bool = True) -> Tensor:
    """Curl ``∇×u`` of a 2D or 3D vector field.

    - **2D**: ``u`` has shape ``(N, 2)``, returns scalar vorticity
      ``∂u₂/∂x₁ - ∂u₁/∂x₂``, shape ``(N,)``.
    - **3D**: ``u`` has shape ``(N, 3)``, returns curl vector, shape ``(N, 3)``.

    Parameters
    ----------
    u : Tensor
        Vector field.
    x : Tensor
        Coordinates with ``requires_grad=True``, shape ``(N, d)``.

    Returns
    -------
    Tensor
        Curl (scalar in 2D, vector in 3D).
    """
    d = x.shape[-1]
    J = jacobian(u, x, create_graph=create_graph)  # (N, m, d)
    if d == 2:
        # ω = ∂u₂/∂x - ∂u₁/∂y
        return J[:, 1, 0] - J[:, 0, 1]  # (N,)
    elif d == 3:
        cx = J[:, 2, 1] - J[:, 1, 2]
        cy = J[:, 0, 2] - J[:, 2, 0]
        cz = J[:, 1, 0] - J[:, 0, 1]
        return torch.stack([cx, cy, cz], dim=-1)  # (N, 3)
    else:
        raise ValueError(f"curl is only defined for d=2 or d=3, got d={d}")


def time_derivative(u: Tensor, t: Tensor, *, create_graph: bool = True) -> Tensor:
    """First-order time derivative ``∂u/∂t``.

    Parameters
    ----------
    u : Tensor
        Scalar or vector field, shape ``(N,)`` / ``(N, 1)`` / ``(N, m)``.
    t : Tensor
        Time coordinates with ``requires_grad=True``, shape ``(N, 1)`` or ``(N,)``.

    Returns
    -------
    Tensor
        ``∂u/∂t``, same shape as ``u``.
    """
    if u.dim() == 1 or (u.dim() == 2 and u.shape[-1] == 1):
        # Scalar case
        if u.dim() == 2:
            u = u.squeeze(-1)
        grad_outputs = torch.ones_like(u)
        (du_dt,) = torch.autograd.grad(
            u, t, grad_outputs=grad_outputs, create_graph=create_graph, retain_graph=True
        )
        return du_dt.squeeze(-1) if du_dt.dim() == 2 else du_dt
    else:
        # Vector case: differentiate each component
        m = u.shape[-1]
        cols = []
        for i in range(m):
            grad_outputs = torch.ones(u.shape[0], device=u.device, dtype=u.dtype)
            (g,) = torch.autograd.grad(
                u[..., i], t, grad_outputs=grad_outputs, create_graph=create_graph, retain_graph=True
            )
            cols.append(g.squeeze(-1) if g.dim() == 2 else g)
        return torch.stack(cols, dim=-1)


def second_time_derivative(u: Tensor, t: Tensor, *, create_graph: bool = True) -> Tensor:
    """Second-order time derivative ``∂²u/∂t²``.

    Parameters
    ----------
    u : Tensor
        Scalar field, shape ``(N,)`` or ``(N, 1)``.
    t : Tensor
        Time coordinates with ``requires_grad=True``, shape ``(N, 1)`` or ``(N,)``.

    Returns
    -------
    Tensor
        ``∂²u/∂t²``, shape ``(N,)``.
    """
    du_dt = time_derivative(u, t, create_graph=True)
    return time_derivative(du_dt, t, create_graph=create_graph)


def advection(u: Tensor, v: Tensor, x: Tensor, *, create_graph: bool = True) -> Tensor:
    """Advection term ``(v·∇)u`` for a scalar field u convected by velocity v.

    Parameters
    ----------
    u : Tensor
        Scalar field, shape ``(N,)`` or ``(N, 1)``.
    v : Tensor
        Velocity field, shape ``(N, d)``.
    x : Tensor
        Spatial coordinates with ``requires_grad=True``, shape ``(N, d)``.

    Returns
    -------
    Tensor
        ``(v·∇)u``, shape ``(N,)``.
    """
    grad_u = gradient(u, x, create_graph=create_graph)  # (N, d)
    return (v * grad_u).sum(-1)  # (N,)
