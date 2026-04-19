"""
Gaussian Random Field (GRF) sampler for PDE initial conditions and forcings.

Generates samples from a Gaussian process whose covariance operator is
defined spectrally as:

    C(k) = (|k|² + τ²)^{-α}

where:
  k  — wavenumber vector
  α  — smoothness exponent  (larger → smoother fields)
  τ  — length-scale / shift parameter

This is a Matérn-like covariance on periodic domains.  The construction
mirrors the original FNO data generation (`GaussianRF` in Li et al. 2021).

Supports 1D and 2D domains with arbitrary length ``L`` (default ``2π``).
All FFT operations use ``torch.fft.rfft`` / ``torch.fft.rfft2`` so that the
spectral weights are computed using physical wavenumbers ``2π k / L``.

References
----------
- Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
  https://arxiv.org/abs/2010.08895
- Original GaussianRF: https://github.com/zongyi-li/fourier_neural_operator
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


class GaussianRF:
    """Spectral Gaussian Random Field sampler on a periodic domain.

    Samples fields from ``N(0, C)`` where the covariance is defined by its
    spectral density:

        S(k) = (|k|² + τ²)^{-α}

    with ``k`` in physical units ``2π·n/L`` (``n`` integer wavenumber).

    Parameters
    ----------
    dim : {1, 2}
        Spatial dimensionality.
    N : int
        Number of grid points along each dimension.
    alpha : float
        Smoothness exponent.  Larger values yield smoother fields.
        Must satisfy ``alpha > dim/2`` for L² samples.
        Default ``2.5`` (used in FNO Burgers and Navier-Stokes benchmarks).
    tau : float
        Length-scale shift.  Larger values yield shorter correlation lengths.
        Default ``7.0`` (FNO benchmark standard).
    sigma : float or None
        If given, the output is normalised so that its pointwise standard
        deviation is approximately ``sigma``.  If ``None`` (default), the
        raw spectral amplitude is used.
    L : float
        Domain length (same in all dimensions).  Default ``2π``.
    device : str or torch.device
        Device on which to precompute spectral weights.
    dtype : torch.dtype
        Output dtype of :meth:`sample`.  Internal computations always use
        ``float64`` for precision; the result is cast before returning.

    Examples
    --------
    >>> grf = GaussianRF(dim=2, N=64)
    >>> u0 = grf.sample(100)        # (100, 64, 64)

    >>> grf1d = GaussianRF(dim=1, N=1024, alpha=2.5, tau=7.0)
    >>> u0 = grf1d.sample(1000)     # (1000, 1024)
    """

    def __init__(
        self,
        dim: int,
        N: int,
        alpha: float = 2.5,
        tau: float = 7.0,
        sigma: float | None = None,
        L: float = 2.0 * math.pi,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim not in (1, 2):
            raise ValueError(f"dim must be 1 or 2, got {dim}")
        if alpha <= dim / 2:
            raise ValueError(
                f"alpha must be > dim/2 = {dim/2:.1f} for L² samples, got {alpha}"
            )

        self.dim = dim
        self.N = N
        self.L = L
        self.device = torch.device(device)
        self.dtype = dtype

        # Build spectral weights at float64 precision.
        sqrt_eig = self._build_sqrt_eig(dim, N, alpha, tau, L, self.device)

        if sigma is not None:
            # Normalise so that the pointwise std ≈ sigma.
            # Draw a small calibration batch, measure empirical std, rescale.
            # This avoids the rfft one-sided-spectrum bookkeeping.
            _tmp_grf = object.__new__(GaussianRF)
            _tmp_grf.dim = dim
            _tmp_grf.N = N
            _tmp_grf.L = L
            _tmp_grf.device = self.device
            _tmp_grf.dtype = torch.float64
            _tmp_grf.sqrt_eig = sqrt_eig
            cal = _tmp_grf.sample(64)   # (64, *spatial)
            empirical_std = cal.std().item()
            if empirical_std > 0:
                sqrt_eig = sqrt_eig * (sigma / empirical_std)

        # Store as complex64 weights for fast multiplication.
        self.sqrt_eig = sqrt_eig  # shape: (N,) 1D  or  (N, N//2+1) 2D

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, n_samples: int) -> Tensor:
        """Draw ``n_samples`` independent GRF realisations.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        Tensor
            Shape ``(n_samples, N)`` for 1D or ``(n_samples, N, N)`` for 2D.
            dtype matches ``self.dtype``.
        """
        if self.dim == 1:
            return self._sample_1d(n_samples)
        return self._sample_2d(n_samples)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sqrt_eig(
        dim: int,
        N: int,
        alpha: float,
        tau: float,
        L: float,
        device: torch.device,
    ) -> Tensor:
        """Compute ``sqrt(S(k))`` for all rfft wavenumbers."""
        scale = 2.0 * math.pi / L  # physical wavenumber factor

        if dim == 1:
            # rfft wavenumbers: 0, 1, ..., N//2
            k = torch.fft.rfftfreq(N, d=1.0 / N, device=device).double()
            k_phys = scale * k
            sqrt_eig = (k_phys.pow(2) + tau ** 2).pow(-alpha / 2.0)
            sqrt_eig[0] = 0.0  # zero mean
            return sqrt_eig  # (N//2+1,)

        else:  # dim == 2
            # rfft2 wavenumbers:
            #   dim-0 (full):  0, 1, ..., N//2, -N//2+1, ..., -1
            #   dim-1 (half):  0, 1, ..., N//2
            k0 = torch.fft.fftfreq(N, d=1.0 / N, device=device).double()
            k1 = torch.fft.rfftfreq(N, d=1.0 / N, device=device).double()
            K0, K1 = torch.meshgrid(k0, k1, indexing="ij")  # (N, N//2+1)
            K0_phys = scale * K0
            K1_phys = scale * K1
            k2 = K0_phys.pow(2) + K1_phys.pow(2)
            sqrt_eig = (k2 + tau ** 2).pow(-alpha / 2.0)
            sqrt_eig[0, 0] = 0.0  # zero mean
            return sqrt_eig  # (N, N//2+1)

    def _sample_1d(self, n: int) -> Tensor:
        M = self.N // 2 + 1  # rfft output length
        # White noise in Fourier space: Re + iIm ~ CN(0, I)
        xi = torch.randn(n, M, 2, dtype=torch.float64, device=self.device)
        xi = torch.view_as_complex(xi)  # (n, M)
        # Colour with spectral weights
        xi = self.sqrt_eig.unsqueeze(0) * xi  # (n, M)
        # Back to physical space
        u = torch.fft.irfft(xi, n=self.N, dim=-1)  # (n, N)
        return u.to(self.dtype)

    def _sample_2d(self, n: int) -> Tensor:
        N2 = self.N // 2 + 1  # rfft output size along last dim
        xi = torch.randn(n, self.N, N2, 2, dtype=torch.float64, device=self.device)
        xi = torch.view_as_complex(xi)  # (n, N, N//2+1)
        xi = self.sqrt_eig.unsqueeze(0) * xi  # (n, N, N//2+1)
        u = torch.fft.irfft2(xi, s=(self.N, self.N), dim=(-2, -1))  # (n, N, N)
        return u.to(self.dtype)
