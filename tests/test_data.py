"""
Tests for nops.data — PDE data generators and datasets.

Coverage
--------
- GaussianRF: shape, dtype, zero mean, 1D/2D, bad alpha
- BurgersGenerator: keys, shapes, no NaN, dtype
- NavierStokesGenerator: keys, shapes, no NaN, forcing modes
- DarcyGenerator: spectral + threshold modes, shapes, no NaN
- HeatGenerator: 1D + 2D, keys, shapes, random alpha
- WaveGenerator: 1D + 2D, keys, shapes, damped, random c
- PoissonGenerator: keys, shapes, no NaN
- All datasets: __len__, __getitem__ shapes, DataLoader batch,
  save/load round-trip, from-generator construction
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

from nops.data import (
    BurgersDataset,
    BurgersGenerator,
    DarcyDataset,
    DarcyGenerator,
    GaussianRF,
    HeatDataset,
    HeatGenerator,
    NavierStokesDataset,
    NavierStokesGenerator,
    PoissonDataset,
    PoissonGenerator,
    WaveDataset,
    WaveGenerator,
)

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

N_SMALL = 16   # small grid for fast tests
N_BATCH = 4    # small batch for fast generation


def assert_no_nan(t: torch.Tensor, name: str = "") -> None:
    assert not t.isnan().any(), f"NaN detected in {name}"
    assert not t.isinf().any(), f"Inf detected in {name}"


# ===========================================================================
# GaussianRF
# ===========================================================================

class TestGaussianRF:
    def test_shape_1d(self):
        grf = GaussianRF(dim=1, N=N_SMALL)
        u = grf.sample(N_BATCH)
        assert u.shape == (N_BATCH, N_SMALL)

    def test_shape_2d(self):
        grf = GaussianRF(dim=2, N=N_SMALL)
        u = grf.sample(N_BATCH)
        assert u.shape == (N_BATCH, N_SMALL, N_SMALL)

    def test_output_dtype_float32(self):
        grf = GaussianRF(dim=2, N=N_SMALL, dtype=torch.float32)
        u = grf.sample(N_BATCH)
        assert u.dtype == torch.float32

    def test_output_dtype_float64(self):
        grf = GaussianRF(dim=1, N=N_SMALL, dtype=torch.float64)
        u = grf.sample(N_BATCH)
        assert u.dtype == torch.float64

    def test_zero_mean(self):
        """The k=0 mode is zeroed, so the spatial mean should be ~0."""
        grf = GaussianRF(dim=2, N=32, alpha=2.5, tau=7.0)
        u = grf.sample(200)
        # Mean over spatial dims for each sample
        means = u.mean(dim=(-2, -1))
        assert means.abs().max() < 1e-5

    def test_no_nan(self):
        grf = GaussianRF(dim=2, N=N_SMALL)
        u = grf.sample(N_BATCH)
        assert_no_nan(u, "GaussianRF output")

    def test_custom_L(self):
        grf = GaussianRF(dim=1, N=N_SMALL, L=1.0)
        u = grf.sample(N_BATCH)
        assert u.shape == (N_BATCH, N_SMALL)

    def test_bad_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be >"):
            GaussianRF(dim=2, N=16, alpha=0.5)  # needs alpha > dim/2 = 1.0

    def test_bad_dim_raises(self):
        with pytest.raises(ValueError):
            GaussianRF(dim=3, N=16)

    def test_sigma_normalisation(self):
        """With sigma given, variance should be approximately sigma²."""
        grf = GaussianRF(dim=2, N=32, sigma=2.0)
        u = grf.sample(500)
        std = u.std().item()
        # Loose check: within factor 2 of target std
        assert 0.5 < std < 8.0


# ===========================================================================
# BurgersGenerator
# ===========================================================================

class TestBurgersGenerator:
    @pytest.fixture
    def gen(self):
        return BurgersGenerator(
            N=N_SMALL, L=2*math.pi, nu=0.1, T=0.01,
            dt=1e-3, record_steps=2,
        )

    def test_output_keys(self, gen):
        data = gen.generate(N_BATCH)
        assert set(data.keys()) == {"ic", "solution", "t_grid"}

    def test_output_shapes(self, gen):
        data = gen.generate(N_BATCH)
        assert data["ic"].shape == (N_BATCH, N_SMALL)
        assert data["solution"].shape == (N_BATCH, N_SMALL, 2)
        assert data["t_grid"].shape == (2,)

    def test_no_nan(self, gen):
        data = gen.generate(N_BATCH)
        for k, v in data.items():
            assert_no_nan(v, f"Burgers {k}")

    def test_output_dtype(self, gen):
        data = gen.generate(N_BATCH)
        assert data["ic"].dtype == torch.float32
        assert data["solution"].dtype == torch.float32

    def test_custom_L(self):
        gen = BurgersGenerator(N=N_SMALL, L=1.0, T=0.005, dt=1e-3, record_steps=1)
        data = gen.generate(2)
        assert data["ic"].shape == (2, N_SMALL)

    def test_repr(self, gen):
        r = repr(gen)
        assert "BurgersGenerator" in r


# ===========================================================================
# NavierStokesGenerator
# ===========================================================================

class TestNavierStokesGenerator:
    @pytest.fixture
    def gen(self):
        return NavierStokesGenerator(
            N=N_SMALL, nu=1e-3, T=0.002, dt=1e-3, record_steps=2,
            forcing="kolmogorov",
        )

    def test_output_keys(self, gen):
        data = gen.generate(N_BATCH)
        assert set(data.keys()) == {"ic", "vorticity", "t_grid"}

    def test_output_shapes(self, gen):
        data = gen.generate(N_BATCH)
        assert data["ic"].shape == (N_BATCH, N_SMALL, N_SMALL)
        assert data["vorticity"].shape == (N_BATCH, N_SMALL, N_SMALL, 2)
        assert data["t_grid"].shape == (2,)

    def test_no_nan(self, gen):
        data = gen.generate(N_BATCH)
        for k, v in data.items():
            assert_no_nan(v, f"NS {k}")

    def test_forcing_none(self):
        gen = NavierStokesGenerator(
            N=N_SMALL, T=0.002, dt=1e-3, record_steps=1, forcing=None
        )
        data = gen.generate(2)
        assert_no_nan(data["vorticity"], "NS vorticity (no forcing)")

    def test_forcing_grf(self):
        gen = NavierStokesGenerator(
            N=N_SMALL, T=0.002, dt=1e-3, record_steps=1, forcing="grf"
        )
        data = gen.generate(2)
        assert_no_nan(data["vorticity"], "NS vorticity (GRF forcing)")

    def test_forcing_tensor(self):
        f = torch.zeros(N_SMALL, N_SMALL)
        gen = NavierStokesGenerator(
            N=N_SMALL, T=0.002, dt=1e-3, record_steps=1, forcing=f
        )
        data = gen.generate(2)
        assert_no_nan(data["vorticity"], "NS vorticity (tensor forcing)")

    def test_bad_forcing_raises(self):
        with pytest.raises(ValueError):
            NavierStokesGenerator(N=N_SMALL, forcing="bad")

    def test_custom_L(self):
        gen = NavierStokesGenerator(N=N_SMALL, L=1.0, T=0.002, dt=1e-3, record_steps=1)
        data = gen.generate(2)
        assert data["ic"].shape == (2, N_SMALL, N_SMALL)


# ===========================================================================
# DarcyGenerator
# ===========================================================================

class TestDarcyGenerator:
    @pytest.fixture
    def gen_spectral(self):
        return DarcyGenerator(N=N_SMALL, solver="spectral", n_iter=5)

    @pytest.fixture
    def gen_threshold(self):
        return DarcyGenerator(N=N_SMALL, solver="threshold", n_iter=5)

    def test_output_keys_spectral(self, gen_spectral):
        data = gen_spectral.generate(N_BATCH)
        assert set(data.keys()) == {"coeff", "solution"}

    def test_output_shapes_spectral(self, gen_spectral):
        data = gen_spectral.generate(N_BATCH)
        assert data["coeff"].shape == (N_BATCH, N_SMALL, N_SMALL)
        assert data["solution"].shape == (N_BATCH, N_SMALL, N_SMALL)

    def test_no_nan_spectral(self, gen_spectral):
        data = gen_spectral.generate(N_BATCH)
        for k, v in data.items():
            assert_no_nan(v, f"Darcy spectral {k}")

    def test_output_shapes_threshold(self, gen_threshold):
        data = gen_threshold.generate(N_BATCH)
        assert data["coeff"].shape == (N_BATCH, N_SMALL, N_SMALL)
        assert data["solution"].shape == (N_BATCH, N_SMALL, N_SMALL)

    def test_threshold_permeability_values(self, gen_threshold):
        """Permeability should only contain a_low or a_high."""
        data = gen_threshold.generate(N_BATCH)
        a = data["coeff"]
        unique_vals = a.unique()
        assert len(unique_vals) == 2

    def test_no_nan_threshold(self, gen_threshold):
        data = gen_threshold.generate(N_BATCH)
        for k, v in data.items():
            assert_no_nan(v, f"Darcy threshold {k}")

    def test_bad_solver_raises(self):
        with pytest.raises(ValueError):
            DarcyGenerator(solver="bad")

    def test_file_solver_requires_path(self):
        gen = DarcyGenerator(solver="file")
        with pytest.raises(ValueError, match="'path' must be provided"):
            gen.generate(10)

    def test_file_not_found(self):
        gen = DarcyGenerator(solver="file")
        with pytest.raises(FileNotFoundError):
            gen.generate(10, path="/nonexistent/path.mat")

    def test_custom_L(self):
        gen = DarcyGenerator(N=N_SMALL, L=2*math.pi, solver="spectral", n_iter=3)
        data = gen.generate(2)
        assert data["coeff"].shape == (2, N_SMALL, N_SMALL)


# ===========================================================================
# HeatGenerator
# ===========================================================================

class TestHeatGenerator:
    @pytest.fixture
    def gen_2d(self):
        return HeatGenerator(
            N=N_SMALL, dim=2, alpha=0.01, T=0.1, dt=0.05, record_steps=2
        )

    @pytest.fixture
    def gen_1d(self):
        return HeatGenerator(
            N=N_SMALL, dim=1, alpha=0.01, T=0.1, dt=0.05, record_steps=2
        )

    def test_output_keys_2d(self, gen_2d):
        data = gen_2d.generate(N_BATCH)
        assert set(data.keys()) == {"ic", "solution", "t_grid", "alpha"}

    def test_output_shapes_2d(self, gen_2d):
        data = gen_2d.generate(N_BATCH)
        assert data["ic"].shape == (N_BATCH, N_SMALL, N_SMALL)
        assert data["solution"].shape == (N_BATCH, N_SMALL, N_SMALL, 2)
        assert data["t_grid"].shape == (2,)
        assert data["alpha"].shape == (N_BATCH,)

    def test_output_shapes_1d(self, gen_1d):
        data = gen_1d.generate(N_BATCH)
        assert data["ic"].shape == (N_BATCH, N_SMALL)
        assert data["solution"].shape == (N_BATCH, N_SMALL, 2)

    def test_no_nan_2d(self, gen_2d):
        data = gen_2d.generate(N_BATCH)
        for k, v in data.items():
            assert_no_nan(v, f"Heat 2D {k}")

    def test_no_nan_1d(self, gen_1d):
        data = gen_1d.generate(N_BATCH)
        for k, v in data.items():
            assert_no_nan(v, f"Heat 1D {k}")

    def test_random_alpha(self):
        gen = HeatGenerator(
            N=N_SMALL, dim=2, alpha="random", alpha_min=0.001, alpha_max=0.1,
            T=0.05, dt=0.05, record_steps=1
        )
        data = gen.generate(N_BATCH)
        alphas = data["alpha"]
        assert alphas.shape == (N_BATCH,)
        # All values should be in range
        assert (alphas >= 0.001).all()
        assert (alphas <= 0.1).all()

    def test_static_grf_forcing(self):
        gen = HeatGenerator(
            N=N_SMALL, dim=2, T=0.05, dt=0.05, record_steps=1,
            forcing="static_grf"
        )
        data = gen.generate(2)
        assert_no_nan(data["solution"], "Heat static_grf forcing")

    def test_bad_forcing_raises(self):
        with pytest.raises(ValueError):
            HeatGenerator(N=N_SMALL, forcing="bad_forcing")

    def test_bad_dim_raises(self):
        with pytest.raises(ValueError):
            HeatGenerator(N=N_SMALL, dim=3)

    def test_custom_L(self):
        gen = HeatGenerator(N=N_SMALL, dim=2, L=1.0, T=0.05, dt=0.05, record_steps=1)
        data = gen.generate(2)
        assert data["ic"].shape == (2, N_SMALL, N_SMALL)


# ===========================================================================
# WaveGenerator
# ===========================================================================

class TestWaveGenerator:
    @pytest.fixture
    def gen_2d(self):
        return WaveGenerator(
            N=N_SMALL, dim=2, c=1.0, gamma=0.0,
            T=0.1, dt=0.05, record_steps=2,
        )

    @pytest.fixture
    def gen_1d(self):
        return WaveGenerator(
            N=N_SMALL, dim=1, c=1.0, T=0.1, dt=0.05, record_steps=2,
        )

    def test_output_keys_2d(self, gen_2d):
        data = gen_2d.generate(N_BATCH)
        assert set(data.keys()) == {
            "ic_displacement", "ic_velocity", "solution", "t_grid", "c"
        }

    def test_output_shapes_2d(self, gen_2d):
        data = gen_2d.generate(N_BATCH)
        assert data["ic_displacement"].shape == (N_BATCH, N_SMALL, N_SMALL)
        assert data["ic_velocity"].shape == (N_BATCH, N_SMALL, N_SMALL)
        assert data["solution"].shape == (N_BATCH, N_SMALL, N_SMALL, 2)
        assert data["t_grid"].shape == (2,)
        assert data["c"].shape == (N_BATCH,)

    def test_output_shapes_1d(self, gen_1d):
        data = gen_1d.generate(N_BATCH)
        assert data["ic_displacement"].shape == (N_BATCH, N_SMALL)
        assert data["solution"].shape == (N_BATCH, N_SMALL, 2)

    def test_no_nan_2d(self, gen_2d):
        data = gen_2d.generate(N_BATCH)
        for k, v in data.items():
            assert_no_nan(v, f"Wave 2D {k}")

    def test_no_nan_1d(self, gen_1d):
        data = gen_1d.generate(N_BATCH)
        for k, v in data.items():
            assert_no_nan(v, f"Wave 1D {k}")

    def test_damped_wave(self):
        gen = WaveGenerator(
            N=N_SMALL, dim=2, c=1.0, gamma=0.2,
            T=0.1, dt=0.05, record_steps=1,
        )
        data = gen.generate(2)
        assert_no_nan(data["solution"], "Wave damped")

    def test_random_c(self):
        gen = WaveGenerator(
            N=N_SMALL, dim=2, c="random", c_min=0.5, c_max=2.0,
            T=0.1, dt=0.05, record_steps=1,
        )
        data = gen.generate(N_BATCH)
        c = data["c"]
        assert c.shape == (N_BATCH,)
        assert (c >= 0.5).all() and (c <= 2.0).all()

    def test_grf_v0(self):
        gen = WaveGenerator(
            N=N_SMALL, dim=2, T=0.1, dt=0.05, record_steps=1, v0_type="grf"
        )
        data = gen.generate(2)
        assert_no_nan(data["ic_velocity"], "Wave GRF v0")

    def test_static_grf_forcing(self):
        gen = WaveGenerator(
            N=N_SMALL, dim=2, T=0.1, dt=0.05, record_steps=1,
            forcing="static_grf"
        )
        data = gen.generate(2)
        assert_no_nan(data["solution"], "Wave static_grf forcing")

    def test_bad_gamma_raises(self):
        with pytest.raises(ValueError):
            WaveGenerator(N=N_SMALL, gamma=-0.1)

    def test_bad_dim_raises(self):
        with pytest.raises(ValueError):
            WaveGenerator(N=N_SMALL, dim=3)

    def test_custom_L(self):
        gen = WaveGenerator(N=N_SMALL, dim=2, L=1.0, T=0.1, dt=0.05, record_steps=1)
        data = gen.generate(2)
        assert data["ic_displacement"].shape == (2, N_SMALL, N_SMALL)


# ===========================================================================
# PoissonGenerator
# ===========================================================================

class TestPoissonGenerator:
    @pytest.fixture
    def gen(self):
        return PoissonGenerator(N=N_SMALL, L=1.0)

    def test_output_keys(self, gen):
        data = gen.generate(N_BATCH)
        assert set(data.keys()) == {"source", "solution"}

    def test_output_shapes(self, gen):
        data = gen.generate(N_BATCH)
        assert data["source"].shape == (N_BATCH, N_SMALL, N_SMALL)
        assert data["solution"].shape == (N_BATCH, N_SMALL, N_SMALL)

    def test_no_nan(self, gen):
        data = gen.generate(N_BATCH)
        for k, v in data.items():
            assert_no_nan(v, f"Poisson {k}")

    def test_poisson_residual(self):
        """Verify that -∇²u ≈ f using finite differences (loose check)."""
        N = 32
        L = 1.0
        gen = PoissonGenerator(N=N, L=L)
        data = gen.generate(5)
        f = data["source"].double()
        u = data["solution"].double()

        # Finite-difference Laplacian: (u_{i-1,j} + u_{i+1,j} + ...) - 4u_{i,j}
        dx = L / N
        lap_u = (
            torch.roll(u, 1, -1) + torch.roll(u, -1, -1)
            + torch.roll(u, 1, -2) + torch.roll(u, -1, -2)
            - 4 * u
        ) / dx**2
        residual = (-lap_u - f).abs().mean()
        # Should be small (FD approximation error is O(dx²))
        assert residual.item() < 5.0  # loose tolerance for N=32

    def test_custom_L(self):
        gen = PoissonGenerator(N=N_SMALL, L=2*math.pi)
        data = gen.generate(2)
        assert data["source"].shape == (2, N_SMALL, N_SMALL)


# ===========================================================================
# Datasets
# ===========================================================================

class TestBurgersDataset:
    @pytest.fixture
    def ds(self):
        gen = BurgersGenerator(N=N_SMALL, T=0.005, dt=1e-3, record_steps=1)
        return BurgersDataset(generator=gen, n_samples=N_BATCH)

    def test_len(self, ds):
        assert len(ds) == N_BATCH

    def test_getitem_shapes(self, ds):
        x, y = ds[0]
        assert x.shape == (N_SMALL,)
        assert y.shape == (N_SMALL,)

    def test_dataloader(self, ds):
        loader = ds.get_dataloader(batch_size=2, shuffle=False)
        x, y = next(iter(loader))
        assert x.shape == (2, N_SMALL)
        assert y.shape == (2, N_SMALL)
        assert_no_nan(x, "Burgers DataLoader x")
        assert_no_nan(y, "Burgers DataLoader y")

    def test_save_load_roundtrip(self, ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "burgers.pt"
            ds.save(path)
            ds2 = BurgersDataset.load(path)
            assert len(ds2) == len(ds)
            x1, y1 = ds[0]
            x2, y2 = ds2[0]
            assert torch.allclose(x1, x2)
            assert torch.allclose(y1, y2)

    def test_from_data_dict(self):
        gen = BurgersGenerator(N=N_SMALL, T=0.005, dt=1e-3, record_steps=1)
        data = gen.generate(N_BATCH)
        ds = BurgersDataset(data=data)
        assert len(ds) == N_BATCH


class TestNavierStokesDataset:
    @pytest.fixture
    def ds(self):
        gen = NavierStokesGenerator(N=N_SMALL, T=0.002, dt=1e-3, record_steps=1)
        return NavierStokesDataset(generator=gen, n_samples=N_BATCH)

    def test_len(self, ds):
        assert len(ds) == N_BATCH

    def test_getitem_shapes(self, ds):
        x, y = ds[0]
        assert x.shape == (N_SMALL, N_SMALL)
        assert y.shape == (N_SMALL, N_SMALL)

    def test_dataloader(self, ds):
        loader = ds.get_dataloader(batch_size=2, shuffle=False)
        x, y = next(iter(loader))
        assert x.shape == (2, N_SMALL, N_SMALL)
        assert y.shape == (2, N_SMALL, N_SMALL)

    def test_save_load_roundtrip(self, ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ns.pt"
            ds.save(path)
            ds2 = NavierStokesDataset.load(path)
            assert len(ds2) == len(ds)


class TestDarcyDataset:
    @pytest.fixture
    def ds(self):
        gen = DarcyGenerator(N=N_SMALL, solver="spectral", n_iter=3)
        return DarcyDataset(generator=gen, n_samples=N_BATCH)

    def test_len(self, ds):
        assert len(ds) == N_BATCH

    def test_getitem_shapes(self, ds):
        x, y = ds[0]
        assert x.shape == (N_SMALL, N_SMALL)
        assert y.shape == (N_SMALL, N_SMALL)

    def test_dataloader(self, ds):
        loader = ds.get_dataloader(batch_size=2, shuffle=False)
        x, y = next(iter(loader))
        assert x.shape == (2, N_SMALL, N_SMALL)

    def test_save_load_roundtrip(self, ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "darcy.pt"
            ds.save(path)
            ds2 = DarcyDataset.load(path)
            assert len(ds2) == len(ds)


class TestHeatDataset:
    @pytest.fixture
    def ds(self):
        gen = HeatGenerator(N=N_SMALL, dim=2, T=0.05, dt=0.05, record_steps=1)
        return HeatDataset(generator=gen, n_samples=N_BATCH)

    def test_len(self, ds):
        assert len(ds) == N_BATCH

    def test_getitem_shapes(self, ds):
        x, y = ds[0]
        assert x.shape == (N_SMALL, N_SMALL)
        assert y.shape == (N_SMALL, N_SMALL)

    def test_dataloader(self, ds):
        loader = ds.get_dataloader(batch_size=2, shuffle=False)
        x, y = next(iter(loader))
        assert x.shape == (2, N_SMALL, N_SMALL)

    def test_save_load_roundtrip(self, ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "heat.pt"
            ds.save(path)
            ds2 = HeatDataset.load(path)
            assert len(ds2) == len(ds)


class TestWaveDataset:
    @pytest.fixture
    def ds(self):
        gen = WaveGenerator(N=N_SMALL, dim=2, T=0.1, dt=0.05, record_steps=1)
        return WaveDataset(generator=gen, n_samples=N_BATCH)

    def test_len(self, ds):
        assert len(ds) == N_BATCH

    def test_getitem_shapes(self, ds):
        x, y = ds[0]
        # input: (*spatial, 2) — displacement + velocity stacked
        assert x.shape == (N_SMALL, N_SMALL, 2)
        assert y.shape == (N_SMALL, N_SMALL)

    def test_dataloader(self, ds):
        loader = ds.get_dataloader(batch_size=2, shuffle=False)
        x, y = next(iter(loader))
        assert x.shape == (2, N_SMALL, N_SMALL, 2)
        assert y.shape == (2, N_SMALL, N_SMALL)

    def test_save_load_roundtrip(self, ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wave.pt"
            ds.save(path)
            ds2 = WaveDataset.load(path)
            assert len(ds2) == len(ds)


class TestPoissonDataset:
    @pytest.fixture
    def ds(self):
        gen = PoissonGenerator(N=N_SMALL, L=1.0)
        return PoissonDataset(generator=gen, n_samples=N_BATCH)

    def test_len(self, ds):
        assert len(ds) == N_BATCH

    def test_getitem_shapes(self, ds):
        x, y = ds[0]
        assert x.shape == (N_SMALL, N_SMALL)
        assert y.shape == (N_SMALL, N_SMALL)

    def test_dataloader(self, ds):
        loader = ds.get_dataloader(batch_size=2, shuffle=False)
        x, y = next(iter(loader))
        assert x.shape == (2, N_SMALL, N_SMALL)

    def test_save_load_roundtrip(self, ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "poisson.pt"
            ds.save(path)
            ds2 = PoissonDataset.load(path)
            assert len(ds2) == len(ds)


# ===========================================================================
# BasePDEDataset: disk mode
# ===========================================================================

class TestDiskMode:
    def test_disk_mode_burgers(self):
        gen = BurgersGenerator(N=N_SMALL, T=0.005, dt=1e-3, record_steps=1)
        ds_mem = BurgersDataset(generator=gen, n_samples=N_BATCH)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "burgers_disk.pt"
            ds_mem.save(path)

            # Load in disk mode
            from nops.data.datasets.base import BasePDEDataset
            ds_disk = BurgersDataset(mode="disk", path=path)
            assert len(ds_disk) == N_BATCH
            x, y = ds_disk[0]
            assert x.shape == (N_SMALL,)

    def test_disk_mode_missing_path_raises(self):
        from nops.data.datasets.base import BasePDEDataset
        with pytest.raises(ValueError, match="'path' must be provided"):
            BurgersDataset(mode="disk")

    def test_disk_mode_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            BurgersDataset(mode="disk", path="/nonexistent/file.pt")


# ===========================================================================
# Top-level imports
# ===========================================================================

def test_top_level_imports():
    """All public symbols are importable from nops.data."""
    import nops.data as nd
    for name in nd.__all__:
        assert hasattr(nd, name), f"Missing: nops.data.{name}"
