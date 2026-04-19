"""
Data generator for the 2D steady-state Darcy flow equation.

PDE
---
    -∇·(a(x) ∇u(x)) = f(x)    in Ω = [0, L]²

with periodic boundary conditions (for the spectral solver) or zero Dirichlet
BC (for FNO benchmark-compatible data).

Expanding the divergence:
    -a(x)∇²u(x) - ∇a(x)·∇u(x) = f(x)

This is an elliptic PDE with variable coefficient ``a(x)``.  The solution
operator ``a ↦ u`` is the canonical FNO Darcy benchmark.

Solver Modes
------------
``solver="spectral"``
    Generates a **smooth** permeability field ``a(x)`` directly from a GRF
    (no thresholding) and solves the resulting variable-coefficient Poisson
    problem using a preconditioned Picard (Richardson) iteration in Fourier
    space.  The constant-coefficient Laplacian serves as the preconditioner.
    Converges reliably for slowly varying ``a``.  Pure PyTorch — GPU native.

``solver="threshold"``
    Generates a **piecewise-constant** permeability by thresholding a GRF:
    ``a = a_low * 1_{g ≤ 0} + a_high * 1_{g > 0}``.  Uses the same spectral
    Picard iteration.  Reproduces the FNO benchmark permeability distribution
    (``a ∈ {1, 12}``).  Requires more iterations than smooth ``a``.

``solver="file"``
    Loads pre-generated data from a ``.mat`` or ``.pt`` file.  Compatible
    with the original FNO benchmark files
    (``piececonst_r421_N1024_smooth1.mat``).

References
----------
- Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
  https://arxiv.org/abs/2010.08895
- Darcy benchmark data: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from nops.data.generators.base import BaseGenerator
from nops.data.utils.gaussian_rf import GaussianRF


_SolverType = Literal["spectral", "threshold", "file"]


class DarcyGenerator(BaseGenerator):
    """Pseudo-spectral + file-loader generator for 2D steady-state Darcy flow.

    Parameters
    ----------
    N : int
        Spatial resolution.  For FNO benchmark compatibility use ``421``.
        Default ``128``.
    L : float
        Domain length ``[0, L]²``.  Default ``1.0`` (FNO benchmark convention).
    solver : {"spectral", "threshold", "file"}
        How to produce the data.  See module docstring.  Default ``"spectral"``.
    n_iter : int
        Picard iteration count for ``solver ∈ {"spectral", "threshold"}``.
        Default ``500``.
    alpha_grf : float
        GRF smoothness for generating ``a(x)``.  Default ``2.0``.
    tau_grf : float
        GRF length-scale for generating ``a(x)``.  Default ``3.0``.
    a_low : float
        Low permeability value for threshold mode.  Default ``1.0``.
    a_high : float
        High permeability value for threshold mode.  Default ``12.0``.
    forcing : float or Tensor
        Source term ``f``.  If a scalar, a constant field is used (FNO standard
        ``f=1``).  If a Tensor of shape ``(N, N)``, used directly.
        Default ``1.0``.
    device : str or torch.device
        Computation device.  Default ``"cpu"``.
    dtype : torch.dtype
        Output dtype.  Default ``torch.float32``.

    Examples
    --------
    >>> # Smooth permeability (fast, GPU-native)
    >>> gen = DarcyGenerator(N=128, solver="spectral")
    >>> data = gen.generate(100)
    >>> data["coeff"].shape     # (100, 128, 128)
    >>> data["solution"].shape  # (100, 128, 128)

    >>> # Piecewise-constant permeability (FNO benchmark style)
    >>> gen = DarcyGenerator(N=421, solver="threshold", n_iter=800)
    >>> data = gen.generate(1024)

    >>> # Load from pre-generated .mat file
    >>> gen = DarcyGenerator(solver="file")
    >>> data = gen.generate(1024, path="piececonst_r421_N1024_smooth1.mat")
    """

    def __init__(
        self,
        N: int = 128,
        L: float = 1.0,
        solver: _SolverType = "spectral",
        n_iter: int = 500,
        alpha_grf: float = 2.0,
        tau_grf: float = 3.0,
        a_low: float = 1.0,
        a_high: float = 12.0,
        forcing: float | Tensor = 1.0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(N=N, L=L, device=device, dtype=dtype)

        if solver not in ("spectral", "threshold", "file"):
            raise ValueError(
                f"solver must be 'spectral', 'threshold', or 'file'; got {solver!r}"
            )

        self.solver = solver
        self.n_iter = n_iter
        self.a_low = a_low
        self.a_high = a_high

        # Source term
        if isinstance(forcing, Tensor):
            self._f = forcing.to(device=device, dtype=torch.float64)
        else:
            self._f = torch.full(
                (N, N), float(forcing), device=device, dtype=torch.float64
            )

        if solver in ("spectral", "threshold"):
            self._grf = GaussianRF(
                dim=2, N=N, alpha=alpha_grf, tau=tau_grf, L=L,
                device=device, dtype=dtype,
            )
            self._build_spectral_ops()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, n_samples: int, path: str | Path | None = None, **kwargs
    ) -> dict[str, Tensor]:
        """Generate or load ``n_samples`` Darcy flow samples.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        path : str or Path, optional
            Required when ``solver="file"``.  Path to a ``.mat`` or ``.pt``
            file.

        Returns
        -------
        dict with keys:
            ``"coeff"``    — permeability ``a(x)``, shape ``(n, N, N)``
            ``"solution"`` — pressure ``u(x)``,     shape ``(n, N, N)``
        """
        if self.solver == "file":
            if path is None:
                raise ValueError("'path' must be provided when solver='file'")
            return self._load_from_file(path, n_samples)

        # Generate permeability field a(x)
        a = self._generate_permeability(n_samples)  # (n, N, N), float64

        # Solve -∇·(a ∇u) = f for each sample
        u = self._picard_solve(a)  # (n, N, N), float64

        return {
            "coeff": a.to(self.dtype),
            "solution": u.to(self.dtype),
        }

    # ------------------------------------------------------------------
    # Permeability generation
    # ------------------------------------------------------------------

    def _generate_permeability(self, n: int) -> Tensor:
        """Generate permeability ``a(x)`` of shape ``(n, N, N)`` in float64."""
        g = self._grf.sample(n).to(torch.float64)  # (n, N, N)

        if self.solver == "threshold":
            # Threshold to piecewise-constant: a ∈ {a_low, a_high}
            a = torch.where(g > 0, self.a_high * torch.ones_like(g), self.a_low * torch.ones_like(g))
        else:
            # Smooth: shift and scale so a > 0 everywhere
            # a = exp(g) ensures positivity; alternatively a = 1 + exp(g)
            a = torch.exp(g)

        return a  # always positive

    # ------------------------------------------------------------------
    # Spectral operators
    # ------------------------------------------------------------------

    def _build_spectral_ops(self) -> None:
        """Precompute wavenumber grids and inverse Laplacian."""
        N = self.N
        scale = 2.0 * math.pi / self.L

        k0_int = torch.fft.fftfreq(N, d=1.0 / N, device=self.device).double()
        k1_int = torch.fft.rfftfreq(N, d=1.0 / N, device=self.device).double()
        K0, K1 = torch.meshgrid(k0_int, k1_int, indexing="ij")

        self._k0 = scale * K0  # (N, N//2+1)
        self._k1 = scale * K1  # (N, N//2+1)

        k2 = self._k0.pow(2) + self._k1.pow(2)

        # Inverse Laplacian preconditioner: P = (-∇²)^{-1}
        inv_lap = k2.clone()
        inv_lap[0, 0] = 1.0
        self._inv_lap = 1.0 / inv_lap
        self._inv_lap[0, 0] = 0.0  # enforce zero-mean solution

    def _divergence_flux(self, a: Tensor, u_hat: Tensor) -> Tensor:
        """Compute ``-∇·(a ∇u)`` in Fourier space.

        Parameters
        ----------
        a     : Tensor  shape (n, N, N), float64
        u_hat : Tensor  shape (n, N, N//2+1), complex128

        Returns
        -------
        Tensor  shape (n, N, N//2+1), complex128
            Fourier transform of ``-∇·(a ∇u)``.
        """
        N = self.N
        # Gradient components of u in physical space
        ux = torch.fft.irfft2(1j * self._k1 * u_hat, s=(N, N), dim=(-2, -1))
        uy = torch.fft.irfft2(1j * self._k0 * u_hat, s=(N, N), dim=(-2, -1))

        # Flux: q = a ∇u (pointwise in physical space)
        qx = a * ux
        qy = a * uy

        # Divergence of flux via FFT
        div_q = (
            1j * self._k1 * torch.fft.rfft2(qx, dim=(-2, -1))
            + 1j * self._k0 * torch.fft.rfft2(qy, dim=(-2, -1))
        )
        return -div_q  # = -∇·(a ∇u)

    @torch.no_grad()
    def _picard_solve(self, a: Tensor) -> Tensor:
        """Solve ``-∇·(a ∇u) = f`` via preconditioned Picard iteration.

        Uses the constant-coefficient Laplacian as a preconditioner:
            u ← u - (-∇²)^{-1} * (-∇·(a ∇u) - f)

        Parameters
        ----------
        a : Tensor  shape ``(n, N, N)``, float64

        Returns
        -------
        u : Tensor  shape ``(n, N, N)``, float64
        """
        N = self.N
        n = a.shape[0]

        # Initial guess: plain Poisson solve (a ≡ 1)
        f_hat = torch.fft.rfft2(self._f.unsqueeze(0).expand(n, -1, -1), dim=(-2, -1))
        u_hat = self._inv_lap * f_hat  # (n, N, N//2+1)

        for _ in range(self.n_iter):
            # Residual: r = -∇·(a ∇u) - f
            lhs_hat = self._divergence_flux(a, u_hat)
            res_hat = lhs_hat - f_hat  # should approach 0

            # Preconditioned correction
            u_hat = u_hat - self._inv_lap * res_hat
            # Enforce zero mean
            u_hat[..., 0, 0] = 0.0

        u = torch.fft.irfft2(u_hat, s=(N, N), dim=(-2, -1))
        return u.real

    # ------------------------------------------------------------------
    # File loader
    # ------------------------------------------------------------------

    def _load_from_file(
        self, path: str | Path, n_samples: int
    ) -> dict[str, Tensor]:
        """Load data from a ``.mat`` or ``.pt`` file.

        Supported formats:
          - MATLAB ``.mat`` (v5 / v7.3):  keys ``"coeff"`` and ``"sol"``
            (FNO benchmark convention).
          - PyTorch ``.pt``: dict with keys ``"coeff"`` and ``"solution"``.

        Parameters
        ----------
        path : str or Path
            Path to the data file.
        n_samples : int
            How many samples to return (first ``n_samples`` from the file).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".pt":
            raw = torch.load(path, map_location=self.device, weights_only=True)
            coeff = raw["coeff"] if "coeff" in raw else raw["a"]
            sol = raw["solution"] if "solution" in raw else raw["sol"]

        elif suffix == ".mat":
            coeff, sol = self._load_mat(path)

        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'.  Use '.mat' or '.pt'."
            )

        coeff = coeff[:n_samples].to(device=self.device, dtype=self.dtype)
        sol = sol[:n_samples].to(device=self.device, dtype=self.dtype)

        # Ensure 3-D: (n, H, W)
        if coeff.dim() == 2:
            coeff = coeff.unsqueeze(0)
        if sol.dim() == 2:
            sol = sol.unsqueeze(0)

        return {"coeff": coeff, "solution": sol}

    @staticmethod
    def _load_mat(path: Path) -> tuple[Tensor, Tensor]:
        """Load a ``.mat`` file, trying scipy then h5py."""
        try:
            import scipy.io

            data = scipy.io.loadmat(str(path))
            coeff_key = "coeff" if "coeff" in data else next(
                k for k in data if not k.startswith("_")
            )
            sol_key = "sol" if "sol" in data else "u"
            coeff = torch.from_numpy(data[coeff_key]).float()
            sol = torch.from_numpy(data[sol_key]).float()
            return coeff, sol

        except NotImplementedError:
            # v7.3 .mat (HDF5) — fall back to h5py
            import h5py

            with h5py.File(path, "r") as f:
                coeff = torch.from_numpy(f["coeff"][:]).float().T
                sol = torch.from_numpy(f["sol"][:]).float().T
            return coeff, sol
