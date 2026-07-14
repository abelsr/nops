"""
Navier-Stokes energy and enstrophy losses.

These losses operate on vorticity fields (scalars) and are particularly
useful for turbulence problems where preserving the energy cascade is critical.

Physics
-------
For 2D incompressible flow with vorticity omega and stream-function psi:

    - Kinetic energy:  E = 1/2 int |u|^2 dx = 1/2 int omega * psi dx
    - Enstrophy:       Z = 1/2 int omega^2 dx
    - Energy spectrum: E(k) = |hat{u}(k)|^2 = |i*k_perp * hat{omega}(k)|^2 / |k|^4

Reference: Chorin & Marsden, "A Mathematical Introduction to Fluid Dynamics"
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class KineticEnergyLoss(nn.Module):
    """Mean-square stream-function loss.

    Minimizes ``1/2 N^2 sum_i psi_i^2`` as a proxy for kinetic energy.

    For the vorticity-stream function formulation, the true kinetic energy
    is ``1/2 int omega * psi dx``, but this requires an inverse Laplacian.
    As a simpler regularizer, minimising the L2 norm of psi helps stabilise
    the pressure (stream-function) solution.

    Parameters
    ----------
    weight : float
        Scaling factor applied to the loss. Default ``0.1``.
    reduction : {"mean", "sum"}
        Aggregation method.
    """

    def __init__(self, weight: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, stream_fn: Tensor) -> Tensor:
        """Compute kinetic-energy proxy loss.

        Parameters
        ----------
        stream_fn : Tensor
            Stream function / pseudo-pressure field, shape ``(B, H, W)``
            or ``(H, W)``.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        loss = 0.5 * stream_fn.pow(2)
        if self.reduction == "mean":
            return self.weight * loss.mean()
        return self.weight * loss.sum()


class EnstrophyLoss(nn.Module):
    """Vortical enstrophy loss: ``1/2 int omega^2 dx``.

    Enstrophy is the squared-L2 norm of vorticity.  Minimising it
    acts as a higher-order regulariser that penalises sharp vortical
    structures.  Useful as a stabiliser against blow-up at high Reynolds
    numbers rather than a primary training objective.

    Parameters
    ----------
    weight : float
        Scaling factor. Default ``1e-4``.
    reduction : {"mean", "sum"}
    """

    def __init__(self, weight: float = 1e-4, reduction: str = "mean") -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, vorticity: Tensor) -> Tensor:
        """Compute enstrophy loss.

        Parameters
        ----------
        vorticity : Tensor
            Vorticity field, shape ``(B, H, W)`` or ``(H, W)``.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        loss = 0.5 * vorticity.pow(2)
        if self.reduction == "mean":
            return self.weight * loss.mean()
        return self.weight * loss.sum()


def compute_energy_spectrum(vorticity: Tensor, dx: float = 1.0) -> Tensor:
    """Compute the 1D energy spectrum E(k) from a vorticity field.

    The velocity field is obtained from vorticity via the stream-function
    (inverse 2D Laplacian).  The energy spectrum is:

        E(k) = |hat{u}(k)|^2 = |k_perp|^2 / |k|^4 * |hat{omega}(k)|^2

    Parameters
    ----------
    vorticity : Tensor
        Vorticity field, shape ``(N, H, W)``.  Must be on CPU.
    dx : float
        Grid spacing.  Default ``1.0`` (assumes domain size 1).

    Returns
    -------
    Tensor
        1D spectrum of shape ``(H//2 + 1,)`` for each sample.
        ``E(k) >= 0``.
    """
    if vorticity.device.type != "cpu":
        vorticity = vorticity.cpu()

    omega_hat = torch.fft.fft2(vorticity, norm="ortho")  # (N, H, W//2+1)
    N, H = vorticity.shape[:2]
    k0 = torch.fft.fftfreq(H, d=dx)
    k1 = torch.fft.rfftfreq(H, d=dx)
    K0, K1 = torch.meshgrid(k0, k1, indexing="ij")  # (H, W//2+1)
    k_sq = K0.pow(2) + K1.pow(2)  # (H, W//2+1)
    k_sq[0, 0] = 1.0  # guard zero mode

    # Perpendicular wave number in 2D: |k_perp|^2 = |k|^2
    # Because velocity = curl(psi) = (-d_psi/dy, d_psi/dx) with Laplacian(psi)=-omega
    # So k_perp_sq / |k|^4 = 1/|k|^2
    energy_per_mode = k_sq / k_sq.pow(2) * omega_hat.abs().pow(2)  # (N, H, W//2+1)

    # Radial binning
    radii = torch.sqrt(k_sq).round().long()  # (H, W//2+1)
    max_k = radii.max().item()
    spectrum = torch.zeros(N, max_k + 1, device=vorticity.device)
    batch_indices = torch.arange(N).unsqueeze(1).expand(N, max_k + 1).to(vorticity.device)
    for i in range(H):
        for j in range(energy_per_mode.shape[-1]):
            k_idx = radii[i, j].item()
            if k_idx <= max_k:
                spectrum[batch_indices[:, k_idx]].add_(energy_per_mode[:, i * max_k + k_idx % (H // 2 + 1)])

    # Simplified radial binning using torchscatter alternative
    spectrum = torch.zeros(N, max_k + 1, device=vorticity.device)
    for i in range(H):
        for j in range(energy_per_mode.shape[-1]):
            k_idx = radii[i, j].item()
            if k_idx <= max_k:
                for n in range(N):
                    spectrum[n, k_idx] += energy_per_mode[n, i, j]

    return spectrum


def energy_spectrum_loss(
    vort_pred: Tensor,
    vort_true: Tensor,
    dx: float = 1.0,
    weight: float = 1.0,
) -> Tensor:
    """Spectral mismatch loss between predicted and true vorticity.

    Computes the L2 difference between energy spectra of prediction and
    ground-truth.  Encourages correct turbulence cascade (e.g. Kolmogorov
    -5/3 law in 2D).

    Parameters
    ----------
    vort_pred : Tensor
        Predicted vorticity, shape ``(N, H, W)``.
    vort_true : Tensor
        Ground-truth vorticity, shape ``(N, H, W)``.
    dx : float
        Grid spacing.
    weight : float
        Loss scaling.

    Returns
    -------
    Tensor
        Scalar spectral loss.
    """
    spec_pred = compute_energy_spectrum(vort_pred, dx)
    spec_true = compute_energy_spectrum(vort_true, dx)
    return weight * (spec_pred.log1p() - spec_true.log1p()).pow(2).mean()
