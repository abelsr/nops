"""
Tests for nops.losses — physics-informed loss functions.

Coverage
--------
- derivatives: gradient, laplacian, divergence, curl, time_derivative, etc.
- DataLoss: mse, rel_l2, mae
- PDE residuals: Burgers 1D/2D, Navier-Stokes 2D, Darcy 2D, Heat 1D/2D/3D,
                 Wave 1D/2D, Poisson 1D/2D/3D, GenericPDELoss
- Boundary losses: Dirichlet, Neumann, Periodic, Robin
- Initial condition losses: InitialConditionLoss, InitialVelocityLoss
- Energy losses: Dirichlet, Poisson, Darcy, Elastic
- CombinedLoss: weighting, term breakdown, gradient flow
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ── imports from the module under test ──────────────────────────────────────
from nops.losses import (
    BurgersResidual,
    CombinedLoss,
    DarcyEnergyLoss,
    DarcyResidual,
    DataLoss,
    DirichletEnergyLoss,
    DirichletLoss,
    ElasticEnergyLoss,
    GenericPDELoss,
    HeatResidual,
    InitialConditionLoss,
    InitialVelocityLoss,
    NavierStokesResidual,
    NeumannLoss,
    PeriodicLoss,
    PhysicsLoss,
    PoissonEnergyLoss,
    PoissonResidual,
    RobinLoss,
    WaveResidual,
    derivatives,
)

# ── helpers ──────────────────────────────────────────────────────────────────
N = 64   # number of collocation points


def rand_xt(dim: int, requires_grad_x: bool = True, requires_grad_t: bool = True):
    """Random (x, t) pair."""
    x = torch.rand(N, dim, requires_grad=requires_grad_x)
    t = torch.rand(N, 1,   requires_grad=requires_grad_t)
    return x, t


def rand_x(dim: int, requires_grad: bool = True):
    return torch.rand(N, dim, requires_grad=requires_grad)


def simple_mlp(in_dim: int, out_dim: int = 1) -> nn.Sequential:
    """Tiny MLP for shape / gradient-flow tests."""
    return nn.Sequential(
        nn.Linear(in_dim, 32), nn.Tanh(),
        nn.Linear(32, out_dim),
    )


def has_grads(model: nn.Module) -> bool:
    """Return True if at least one parameter has a non-None gradient.

    Note: for losses that differentiate u w.r.t. x (laplacian, gradient, etc.)
    the *output bias* of the last linear layer always has grad=None because
    ∂²(u + b)/∂x² = ∂²u/∂x² — the constant drops out. We therefore only
    require that *at least one* parameter received a gradient, not all.
    """
    return any(p.grad is not None for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Derivatives
# ─────────────────────────────────────────────────────────────────────────────

class TestDerivatives:

    def test_gradient_quadratic(self):
        """∇(x₁² + x₂²) should equal 2x."""
        x = torch.rand(N, 2, requires_grad=True)
        u = x.pow(2).sum(-1)
        g = derivatives.gradient(u, x)
        assert g.shape == (N, 2)
        assert torch.allclose(g, 2 * x, atol=1e-5)

    def test_gradient_returns_correct_shape_3d(self):
        x = rand_x(3)
        u = x.pow(2).sum(-1)
        g = derivatives.gradient(u, x)
        assert g.shape == (N, 3)

    def test_laplacian_quadratic(self):
        """∇²(x₁² + x₂²) should equal 4 (= 2*dim for dim=2)."""
        x = torch.rand(N, 2, requires_grad=True)
        u = x.pow(2).sum(-1)
        lap = derivatives.laplacian(u, x)
        assert lap.shape == (N,)
        assert torch.allclose(lap, torch.full((N,), 4.0), atol=1e-4)

    def test_laplacian_3d(self):
        x = torch.rand(N, 3, requires_grad=True)
        u = x.pow(2).sum(-1)
        lap = derivatives.laplacian(u, x)
        assert torch.allclose(lap, torch.full((N,), 6.0), atol=1e-4)

    def test_divergence_constant_field(self):
        """∇·x = d (identity vector field)."""
        x = torch.rand(N, 3, requires_grad=True)
        u = x                           # u_i = x_i  →  ∇·u = 3
        div = derivatives.divergence(u, x)
        assert div.shape == (N,)
        assert torch.allclose(div, torch.full((N,), 3.0), atol=1e-4)

    def test_curl_2d(self):
        """∇×(x₂, -x₁) = -2 (constant)."""
        x = torch.rand(N, 2, requires_grad=True)
        # u = (x₂, -x₁)  → curl = ∂(-x₁)/∂x - ∂(x₂)/∂y = -1 - 1 = -2
        u = torch.stack([x[:, 1], -x[:, 0]], dim=-1)
        c = derivatives.curl(u, x)
        assert c.shape == (N,)
        assert torch.allclose(c, torch.full((N,), -2.0), atol=1e-4)

    def test_curl_3d_shape(self):
        x = rand_x(3)
        u = torch.rand_like(x).requires_grad_(False)
        # Use a model-like function
        net = simple_mlp(3, 3)
        u_pred = net(x)
        c = derivatives.curl(u_pred, x)
        assert c.shape == (N, 3)

    def test_time_derivative_linear(self):
        """∂(t·sin(x))/∂t = sin(x)."""
        x = torch.rand(N, 1)
        t = torch.rand(N, 1, requires_grad=True)
        u = t.squeeze(-1) * torch.sin(x.squeeze(-1))
        du_dt = derivatives.time_derivative(u, t)
        expected = torch.sin(x.squeeze(-1))
        assert torch.allclose(du_dt, expected, atol=1e-5)

    def test_second_time_derivative(self):
        """∂²(t²)/∂t² = 2."""
        t = torch.rand(N, 1, requires_grad=True)
        u = t.squeeze(-1).pow(2)
        d2u = derivatives.second_time_derivative(u, t)
        assert torch.allclose(d2u, torch.full((N,), 2.0), atol=1e-4)

    def test_vector_laplacian_shape(self):
        x = rand_x(2)
        net = simple_mlp(2, 2)
        u = net(x)
        vl = derivatives.vector_laplacian(u, x)
        assert vl.shape == (N, 2)

    def test_advection(self):
        x = rand_x(2)
        net = simple_mlp(2, 1)
        u = net(x).squeeze(-1)
        v = torch.rand(N, 2)
        adv = derivatives.advection(u, v, x)
        assert adv.shape == (N,)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoss
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoss:

    @pytest.mark.parametrize("reduction", ["mse", "rel_l2", "mae"])
    def test_zero_when_perfect(self, reduction):
        loss_fn = DataLoss(reduction=reduction)
        u = torch.rand(16, 10)
        assert loss_fn(u, u).item() == pytest.approx(0.0, abs=1e-6)

    def test_mse_value(self):
        loss_fn = DataLoss(reduction="mse")
        pred = torch.ones(10)
        true = torch.zeros(10)
        assert loss_fn(pred, true).item() == pytest.approx(1.0, abs=1e-6)

    def test_rel_l2_shape(self):
        loss_fn = DataLoss(reduction="rel_l2")
        u = torch.rand(8, 5)
        v = torch.rand(8, 5)
        out = loss_fn(u, v)
        assert out.shape == ()

    def test_gradient_flows(self):
        net = simple_mlp(2, 1)
        x = torch.rand(16, 2)
        u_true = torch.rand(16, 1)
        loss = DataLoss()(net(x), u_true)
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            DataLoss()(torch.rand(4, 3), torch.rand(4, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Burgers Residual
# ─────────────────────────────────────────────────────────────────────────────

class TestBurgersResidual:

    def test_1d_shape_and_grad(self):
        loss_fn = BurgersResidual(nu=1e-3, dim=1)
        model = simple_mlp(2, 1)   # input: (x, t) cat → 2
        x, t = rand_xt(1)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        loss = loss_fn(wrapped, x, t)
        assert loss.shape == ()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_2d_shape_and_grad(self):
        loss_fn = BurgersResidual(nu=1e-3, dim=2)
        model = simple_mlp(3, 2)   # (x1, x2, t) → (u, v)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(2)
        loss = loss_fn(wrapped, x, t)
        assert loss.shape == ()
        loss.backward()

    def test_exact_1d_solution_near_zero(self):
        """For the exact solution u=sin(x-t), the residual is not zero (it has
        advection), but let's just check it's finite and non-negative."""
        loss_fn = BurgersResidual(nu=0.0, dim=1)
        x = torch.rand(N, 1, requires_grad=True)
        t = torch.rand(N, 1, requires_grad=True)

        def exact(x, t):
            return torch.sin(x - t)   # travelling wave, not exact Burgers solution

        loss = loss_fn(exact, x, t)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            BurgersResidual(dim=3)


# ─────────────────────────────────────────────────────────────────────────────
# Navier-Stokes Residual
# ─────────────────────────────────────────────────────────────────────────────

class TestNavierStokesResidual:

    def test_2d_forward_and_grad(self):
        loss_fn = NavierStokesResidual(nu=1e-3, dim=2)
        model = simple_mlp(3, 3)   # (x, y, t) → (u, v, p)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(2)
        loss = loss_fn(wrapped, x, t)
        assert loss.shape == ()
        loss.backward()

    def test_3d_forward(self):
        loss_fn = NavierStokesResidual(nu=1e-3, dim=3)
        model = simple_mlp(4, 4)   # (x, y, z, t) → (u, v, w, p)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(3)
        loss = loss_fn(wrapped, x, t)
        assert loss.shape == ()

    def test_with_forcing(self):
        loss_fn = NavierStokesResidual(nu=1e-3, dim=2)
        model = simple_mlp(3, 3)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        def forcing(x, t):
            return torch.zeros(x.shape[0], 2)

        x, t = rand_xt(2)
        loss = loss_fn(wrapped, x, t, forcing=forcing)
        assert torch.isfinite(loss)

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            NavierStokesResidual(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Darcy Residual
# ─────────────────────────────────────────────────────────────────────────────

class TestDarcyResidual:

    def test_2d_forward_and_grad(self):
        loss_fn = DarcyResidual(dim=2)
        model = simple_mlp(3, 1)   # (a, x, y) → u

        def wrapped(a, x):
            inp = torch.cat([a.unsqueeze(-1) if a.dim() == 1 else a, x], dim=-1)
            return model(inp)

        x = rand_x(2)
        a = torch.rand(N) + 0.1
        f = torch.rand(N)
        loss = loss_fn(wrapped, x, a, f=f)
        assert loss.shape == ()
        loss.backward()

    def test_constant_permeability_no_source(self):
        """With a=1, -∇²u = 0 → PoissonResidual should be same."""
        loss_fn = DarcyResidual(dim=1)
        model = simple_mlp(2, 1)   # (a, x) → u

        def wrapped(a, x):
            return model(torch.cat([a.unsqueeze(-1) if a.dim() == 1 else a, x], dim=-1))

        x = rand_x(1)
        a = torch.ones(N)
        loss = loss_fn(wrapped, x, a)
        assert torch.isfinite(loss)


# ─────────────────────────────────────────────────────────────────────────────
# Heat Residual
# ─────────────────────────────────────────────────────────────────────────────

class TestHeatResidual:

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_forward_dim(self, dim):
        loss_fn = HeatResidual(alpha=0.01, dim=dim)
        model = simple_mlp(dim + 1, 1)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(dim)
        loss = loss_fn(wrapped, x, t)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        loss_fn = HeatResidual(alpha=0.01, dim=2)
        model = simple_mlp(3, 1)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(2)
        loss = loss_fn(wrapped, x, t)
        loss.backward()
        assert has_grads(model)

    def test_with_forcing(self):
        loss_fn = HeatResidual(alpha=1.0, dim=1)
        model = simple_mlp(2, 1)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(1)
        forcing = torch.ones(N)
        loss = loss_fn(wrapped, x, t, forcing=forcing)
        assert torch.isfinite(loss)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            HeatResidual(alpha=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Wave Residual
# ─────────────────────────────────────────────────────────────────────────────

class TestWaveResidual:

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_forward_dim(self, dim):
        loss_fn = WaveResidual(c=1.0, dim=dim)
        model = simple_mlp(dim + 1, 1)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(dim)
        loss = loss_fn(wrapped, x, t)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_damped_wave(self):
        loss_fn = WaveResidual(c=1.0, dim=1, gamma=0.5)
        model = simple_mlp(2, 1)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(1)
        loss = loss_fn(wrapped, x, t)
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        loss_fn = WaveResidual(c=1.0, dim=1)
        model = simple_mlp(2, 1)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(1)
        loss = loss_fn(wrapped, x, t)
        loss.backward()
        assert has_grads(model)

    def test_invalid_c(self):
        with pytest.raises(ValueError):
            WaveResidual(c=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Poisson Residual
# ─────────────────────────────────────────────────────────────────────────────

class TestPoissonResidual:

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_forward_dim(self, dim):
        loss_fn = PoissonResidual(dim=dim)
        model = simple_mlp(dim, 1)
        x = rand_x(dim)
        loss = loss_fn(model, x)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_exact_quadratic_zero_residual(self):
        """For -∇²u = f, if u = ½x² and f = -1 (1D), the residual is zero."""
        loss_fn = PoissonResidual(dim=1, sign=-1)
        x = torch.rand(N, 1, requires_grad=True)

        def exact_model(x):
            return 0.5 * x.pow(2)

        # f = -∇²u = -1 (constant), sign=-1 → residual = -∇²u - f = -1 - (-1) = 0
        f = torch.full((N,), -1.0)
        loss = loss_fn(exact_model, x, f=f)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flow(self):
        loss_fn = PoissonResidual(dim=2)
        model = simple_mlp(2, 1)
        x = rand_x(2)
        loss = loss_fn(model, x)
        loss.backward()
        assert has_grads(model)

    def test_invalid_sign(self):
        with pytest.raises(ValueError):
            PoissonResidual(sign=0)


# ─────────────────────────────────────────────────────────────────────────────
# Generic PDE Loss
# ─────────────────────────────────────────────────────────────────────────────

class TestGenericPDELoss:

    def test_user_defined_residual(self):
        """Wrap the heat equation in GenericPDELoss."""

        def heat_residual(u, x, t, derivatives):
            u_s = u.squeeze(-1)
            du_dt = derivatives.time_derivative(u_s, t)
            lap_u = derivatives.laplacian(u_s, x)
            return du_dt - 0.01 * lap_u

        loss_fn = GenericPDELoss(heat_residual)
        model = simple_mlp(3, 1)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(2)
        loss = loss_fn(wrapped, x, t)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        def dummy_res(u, x, t, derivatives):
            return derivatives.laplacian(u.squeeze(-1), x)

        loss_fn = GenericPDELoss(dummy_res)
        model = simple_mlp(3, 1)

        def wrapped(x, t):
            return model(torch.cat([x, t], dim=-1))

        x, t = rand_xt(2)
        loss = loss_fn(wrapped, x, t)
        loss.backward()
        assert has_grads(model)

    def test_reduction_sum(self):
        def trivial(u, x, derivatives):
            return u.squeeze(-1)

        loss_fn = GenericPDELoss(trivial, reduction="sum")
        model = simple_mlp(2, 1)
        x = rand_x(2)
        loss = loss_fn(model, x)
        assert loss.shape == ()


# ─────────────────────────────────────────────────────────────────────────────
# Boundary Losses
# ─────────────────────────────────────────────────────────────────────────────

class TestDirichletLoss:

    def test_zero_when_exact(self):
        loss_fn = DirichletLoss()
        model = simple_mlp(2, 1)
        x_bc = torch.rand(16, 2)
        g = model(x_bc).detach()
        loss = loss_fn(model, x_bc, g)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_callable_g(self):
        loss_fn = DirichletLoss()
        model = simple_mlp(2, 1)
        x_bc = torch.rand(16, 2)
        loss = loss_fn(model, x_bc, g=lambda x: torch.zeros(x.shape[0], 1))
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        loss_fn = DirichletLoss()
        model = simple_mlp(2, 1)
        x_bc = torch.rand(16, 2)
        g = torch.zeros(16, 1)
        loss = loss_fn(model, x_bc, g)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestNeumannLoss:

    def test_shape(self):
        loss_fn = NeumannLoss()
        model = simple_mlp(2, 1)
        x_bc = torch.rand(16, 2, requires_grad=True)
        normals = torch.randn(16, 2)
        normals = normals / normals.norm(dim=-1, keepdim=True)
        loss = loss_fn(model, x_bc, normals, h=0.0)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        loss_fn = NeumannLoss()
        model = simple_mlp(2, 1)
        x_bc = torch.rand(16, 2, requires_grad=True)
        normals = torch.ones(16, 2) / 2**0.5
        loss = loss_fn(model, x_bc, normals)
        loss.backward()
        assert has_grads(model)


class TestPeriodicLoss:

    def test_zero_for_constant(self):
        """A constant model should have zero periodic loss."""
        loss_fn = PeriodicLoss(match_gradient=False)
        model = lambda x: torch.ones(x.shape[0], 1)  # noqa: E731
        x_left  = torch.rand(16, 2)
        x_right = torch.rand(16, 2)
        loss = loss_fn(model, x_left, x_right)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_with_gradient_match(self):
        loss_fn = PeriodicLoss(match_gradient=True)
        model = simple_mlp(2, 1)
        x_left  = torch.rand(16, 2, requires_grad=True)
        x_right = torch.rand(16, 2, requires_grad=True)
        loss = loss_fn(model, x_left, x_right)
        assert torch.isfinite(loss)
        loss.backward()


class TestRobinLoss:

    def test_shape_and_finite(self):
        loss_fn = RobinLoss(alpha=1.0, beta=0.5)
        model = simple_mlp(2, 1)
        x_bc = torch.rand(16, 2, requires_grad=True)
        normals = torch.ones(16, 2) / 2**0.5
        g = torch.zeros(16)
        loss = loss_fn(model, x_bc, normals, g)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        loss_fn = RobinLoss(alpha=1.0, beta=0.5)
        model = simple_mlp(2, 1)
        x_bc = torch.rand(16, 2, requires_grad=True)
        normals = torch.ones(16, 2) / 2**0.5
        g = torch.zeros(16)
        loss = loss_fn(model, x_bc, normals, g)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# Initial Condition Losses
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialConditionLoss:

    def test_zero_when_exact(self):
        model = simple_mlp(3, 1)
        x = torch.rand(16, 2)
        t0 = torch.zeros(16, 1)
        u0 = model(torch.cat([x, t0], dim=-1)).detach()

        loss_fn = InitialConditionLoss()
        loss = loss_fn(
            lambda x, t: model(torch.cat([x, t], dim=-1)),
            x, t0, u0,
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_callable_u0(self):
        model = simple_mlp(3, 1)
        x = torch.rand(16, 2)
        t0 = 0.0
        u0 = lambda x: torch.zeros(x.shape[0], 1)  # noqa: E731

        loss_fn = InitialConditionLoss()
        loss = loss_fn(
            lambda x, t: model(torch.cat([x, t], dim=-1)),
            x, t0, u0,
        )
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        model = simple_mlp(3, 1)
        x = torch.rand(16, 2)
        t0 = torch.zeros(16, 1)
        u0 = torch.zeros(16, 1)
        loss_fn = InitialConditionLoss()
        loss = loss_fn(
            lambda x, t: model(torch.cat([x, t], dim=-1)),
            x, t0, u0,
        )
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestInitialVelocityLoss:

    def test_shape_and_finite(self):
        model = simple_mlp(3, 1)
        x = torch.rand(16, 2)
        t0 = torch.zeros(16, 1, requires_grad=True)
        v0 = torch.zeros(16)

        loss_fn = InitialVelocityLoss()
        loss = loss_fn(
            lambda x, t: model(torch.cat([x, t], dim=-1)),
            x, t0, v0,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        model = simple_mlp(3, 1)
        x = torch.rand(16, 2)
        t0 = torch.zeros(16, 1, requires_grad=True)
        v0 = torch.zeros(16)
        loss_fn = InitialVelocityLoss()
        loss = loss_fn(
            lambda x, t: model(torch.cat([x, t], dim=-1)),
            x, t0, v0,
        )
        loss.backward()
        assert has_grads(model)


# ─────────────────────────────────────────────────────────────────────────────
# Energy Losses
# ─────────────────────────────────────────────────────────────────────────────

class TestDirichletEnergyLoss:

    def test_shape_and_nonneg(self):
        loss_fn = DirichletEnergyLoss()
        model = simple_mlp(2, 1)
        x = rand_x(2)
        loss = loss_fn(model, x)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_zero_for_constant(self):
        """A model with all-zero weights outputs a constant → Dirichlet energy ≈ 0."""
        loss_fn = DirichletEnergyLoss()
        model = simple_mlp(2, 1)
        # Zero out all weights so output = bias (constant w.r.t. x)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        x = rand_x(2)
        loss = loss_fn(model, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_flow(self):
        loss_fn = DirichletEnergyLoss()
        model = simple_mlp(2, 1)
        x = rand_x(2)
        loss = loss_fn(model, x)
        loss.backward()
        assert has_grads(model)


class TestPoissonEnergyLoss:

    def test_shape(self):
        loss_fn = PoissonEnergyLoss()
        model = simple_mlp(2, 1)
        x = rand_x(2)
        f = torch.rand(N)
        loss = loss_fn(model, x, f=f)
        assert loss.shape == ()

    def test_gradient_flow(self):
        loss_fn = PoissonEnergyLoss()
        model = simple_mlp(2, 1)
        x = rand_x(2)
        loss = loss_fn(model, x, f=torch.rand(N))
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestDarcyEnergyLoss:

    def test_shape(self):
        loss_fn = DarcyEnergyLoss()
        model = simple_mlp(3, 1)   # (a, x, y) → u

        def wrapped(a, x):
            a_exp = a.unsqueeze(-1) if a.dim() == 1 else a
            return model(torch.cat([a_exp, x], dim=-1))

        x = rand_x(2)
        a = torch.rand(N) + 0.1
        loss = loss_fn(wrapped, x, a)
        assert loss.shape == ()

    def test_gradient_flow(self):
        loss_fn = DarcyEnergyLoss()
        model = simple_mlp(3, 1)

        def wrapped(a, x):
            a_exp = a.unsqueeze(-1) if a.dim() == 1 else a
            return model(torch.cat([a_exp, x], dim=-1))

        x = rand_x(2)
        a = torch.rand(N) + 0.1
        loss = loss_fn(wrapped, x, a)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestElasticEnergyLoss:

    @pytest.mark.parametrize("dim", [2, 3])
    def test_shape_and_nonneg(self, dim):
        loss_fn = ElasticEnergyLoss(mu=1.0, lam=0.5, dim=dim)
        model = simple_mlp(dim, dim)
        x = rand_x(dim)
        loss = loss_fn(model, x)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_gradient_flow(self):
        loss_fn = ElasticEnergyLoss(mu=1.0, lam=0.5, dim=2)
        model = simple_mlp(2, 2)
        x = rand_x(2)
        loss = loss_fn(model, x)
        loss.backward()
        assert has_grads(model)

    def test_invalid_dim(self):
        with pytest.raises(ValueError):
            ElasticEnergyLoss(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# CombinedLoss
# ─────────────────────────────────────────────────────────────────────────────

class TestCombinedLoss:

    def _make_model_and_data(self):
        model = simple_mlp(2, 1)
        x = rand_x(2)
        u_true = torch.rand(N, 1)
        return model, x, u_true

    def test_returns_scalar_and_dict(self):
        model, x, u_true = self._make_model_and_data()

        loss_fn = CombinedLoss({
            "data": (1.0, DataLoss("mse")),
        })
        total, terms = loss_fn(data=(model(x), u_true))
        assert total.shape == ()
        assert "data" in terms
        assert terms["data"].shape == ()

    def test_weighted_sum_correctness(self):
        """total should equal w1*L1 + w2*L2."""
        u_pred = torch.rand(16, 1)
        u_true = torch.rand(16, 1)

        l1 = DataLoss("mse")
        l2 = DataLoss("mae")
        w1, w2 = 2.0, 0.5

        loss_fn = CombinedLoss({"l1": (w1, l1), "l2": (w2, l2)})
        total, terms = loss_fn(l1=(u_pred, u_true), l2=(u_pred, u_true))

        expected = w1 * l1(u_pred, u_true) + w2 * l2(u_pred, u_true)
        assert total.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_gradient_flow_through_combined(self):
        model = simple_mlp(2, 1)
        x = rand_x(2)
        u_true = torch.rand(N, 1)

        loss_fn = CombinedLoss({
            "data": (1.0, DataLoss("mse")),
            "energy": (0.1, DirichletEnergyLoss()),
        })

        total, _ = loss_fn(
            data=(model(x), u_true),
            energy=(model, x),
        )
        total.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_multiple_pde_terms(self):
        """Smoke test combining PDE + BC + IC losses."""
        pde_model = simple_mlp(3, 1)
        bc_model  = simple_mlp(2, 1)   # boundary model only needs spatial coords
        x, t = rand_xt(2)
        x_bc = torch.rand(16, 2)

        loss_fn = CombinedLoss({
            "pde": (1e-3, HeatResidual(alpha=0.01, dim=2)),
            "bc":  (1e-2, DirichletLoss()),
            "ic":  (1e-2, InitialConditionLoss()),
        })

        def heat_model(x, t):
            return pde_model(torch.cat([x, t], dim=-1))

        total, terms = loss_fn(
            pde=(heat_model, x, t),
            bc=(bc_model, x_bc, torch.zeros(16, 1)),
            ic=(heat_model, x, torch.zeros(N, 1), torch.zeros(N, 1)),
        )
        assert set(terms.keys()) == {"pde", "bc", "ic"}
        assert torch.isfinite(total)
        total.backward()

    def test_set_weight(self):
        loss_fn = CombinedLoss({"data": (1.0, DataLoss())})
        loss_fn.set_weight("data", 5.0)
        assert loss_fn.get_weight("data") == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# PhysicsLoss ABC
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysicsLossABC:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            PhysicsLoss()

    def test_concrete_subclass_works(self):
        class MyLoss(PhysicsLoss):
            def forward(self, x):
                return x.mean()

        loss_fn = MyLoss()
        assert loss_fn(torch.rand(10)).shape == ()
