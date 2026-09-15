"""Navier-Stokes step-pair dataset with L2-normalization and trajectory split.

Split strategy: all timesteps × sampled trajectories, L2-normalized.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

_GLOBAL_DATA = None


def _get_raw() -> np.ndarray:
    global _GLOBAL_DATA
    if _GLOBAL_DATA is None:
        p = Path.home() / ".cache" / "nops" / "navier_stokes_v1e-3_N1200_T20.pt"
        assert p.exists(), f"Data not found at {p}"
        _GLOBAL_DATA = torch.load(p, weights_only=False).float().numpy()  # [1200, 64, 64, 20]
    return _GLOBAL_DATA


class L2NormPairDataset(Dataset):

    def __init__(self, traj_indices: list[int]):
        raw = _get_raw()
        n_steps = 19  # t=0..18 → t=1..19

        # Pre-allocate: n_traj * n_steps samples
        nc = len(traj_indices) * n_steps
        self._ic = np.empty((nc, 64, 64), dtype=np.float32)
        self._tgt = np.empty((nc, 64, 64), dtype=np.float32)
        self._n_ic = np.empty(nc, dtype=np.float32)
        self._n_tgt = np.empty(nc, dtype=np.float32)

        k = 0
        for traj_i in traj_indices:
            traj = raw[traj_i]  # [64, 64, 20]
            for t in range(n_steps):
                ic = traj[:, :, t]
                tgt = traj[:, :, t + 1]
                n_ic = float(np.sqrt(np.mean(ic ** 2)))
                n_tgt = float(np.sqrt(np.mean(tgt ** 2)))
                eps = 1e-12
                self._ic[k] = ic / (n_ic + eps)
                self._tgt[k] = tgt / (n_tgt + eps)
                self._n_ic[k] = n_ic
                self._n_tgt[k] = n_tgt
                k += 1

    def __len__(self) -> int:
        return len(self._n_ic)

    def __getitem__(self, idx: int) -> dict:
        return {
            "vorticity_ic": torch.from_numpy(self._ic[idx]),
            "vorticity": torch.from_numpy(self._tgt[idx]),
            "norm_ic": float(self._n_ic[idx]),
            "norm_target": float(self._n_tgt[idx]),
        }


class SpaceTimeDataset(Dataset):
    """Space-time (FNO-3D) dataset: first ``n_in`` frames -> remaining frames.

    Each sample is a *volume* ``[64, 64, n_in]`` mapped to ``[64, 64, T-n_in]``.
    ``n_in`` defaults to ``T // 2`` (= 10 for T=20), so input and output
    volumes have equal temporal extent — what a 3-D FNO requires.
    """

    def __init__(self, traj_indices: list[int], n_in: int = 10):
        raw = _get_raw()                      # [N, 64, 64, T]
        T = raw.shape[-1]
        n_out = T - n_in
        n = len(traj_indices)
        self._ic = np.empty((n, 64, 64, n_in), dtype=np.float32)
        self._tgt = np.empty((n, 64, 64, n_out), dtype=np.float32)
        self._n_ic = np.empty(n, dtype=np.float32)
        self._n_tgt = np.empty(n, dtype=np.float32)

        eps = 1e-12
        for k, ti in enumerate(traj_indices):
            traj = raw[ti]                    # [64, 64, T]
            inp = traj[:, :, :n_in]
            tgt = traj[:, :, n_in:]
            n_ic = float(np.sqrt(np.mean(inp ** 2)))
            n_tgt = float(np.sqrt(np.mean(tgt ** 2)))
            self._ic[k] = inp / (n_ic + eps)
            self._tgt[k] = tgt / (n_tgt + eps)
            self._n_ic[k] = n_ic
            self._n_tgt[k] = n_tgt

    def __len__(self) -> int:
        return len(self._n_ic)

    def __getitem__(self, idx: int) -> dict:
        return {
            "vorticity_ic": torch.from_numpy(self._ic[idx]),
            "vorticity": torch.from_numpy(self._tgt[idx]),
            "norm_ic": float(self._n_ic[idx]),
            "norm_target": float(self._n_tgt[idx]),
        }


def make_dataloaders_3d(
    n_train: int = 800,
    n_val: int = 200,
    n_test: int = 200,
    n_in: int = 10,
    batch_size: int = 8,
    val_batch_size: int | None = None,
    device: str = "cpu",
) -> dict[str, DataLoader]:
    """Space-time variant of :func:`make_dataloaders` (same trajectory split)."""
    raw = _get_raw()
    n_all = raw.shape[0]
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_all)

    idx_tr = sorted(perm[:n_train].tolist())
    idx_va = sorted(perm[n_train:n_train + n_val].tolist())
    idx_te = sorted(perm[n_train + n_val:n_train + n_val + n_test].tolist())

    if val_batch_size is None:
        val_batch_size = len(idx_va)

    return {
        "train": DataLoader(SpaceTimeDataset(idx_tr, n_in), batch_size=batch_size, shuffle=True),
        "val": DataLoader(SpaceTimeDataset(idx_va, n_in), batch_size=val_batch_size, shuffle=False),
        "test": DataLoader(SpaceTimeDataset(idx_te, n_in), batch_size=val_batch_size, shuffle=False),
    }


class MultiFrameDataset(Dataset):
    """``K`` past frames (as channels) -> next frame  — FNO-2D + RNN context.

    This matches the setup of the FNO paper's ``FNO-2D`` benchmark on
    Navier-Stokes, which maps the previous 10 time steps to the next one.
    Each sample is ``[K, 64, 64]`` (input) -> ``[64, 64]`` (target).
    """

    def __init__(self, traj_indices: list[int], n_ctx: int = 10):
        raw = _get_raw()                      # [N, 64, 64, T]
        T = raw.shape[-1]
        starts = list(range(n_ctx - 1, T - 1))    # start index of last input frame
        n_pairs = len(starts)
        n = len(traj_indices) * n_pairs

        self._ic = np.empty((n, n_ctx, 64, 64), dtype=np.float32)
        self._tgt = np.empty((n, 64, 64), dtype=np.float32)
        self._n_ic = np.empty(n, dtype=np.float32)
        self._n_tgt = np.empty(n, dtype=np.float32)

        eps = 1e-12
        k = 0
        for ti in traj_indices:
            traj = raw[ti]                    # [64, 64, T]
            for t in starts:
                win = traj[:, :, t - n_ctx + 1:t + 1]   # [64, 64, K]
                tgt = traj[:, :, t + 1]                 # [64, 64]
                n_ic = float(np.sqrt(np.mean(win ** 2)))
                n_tgt = float(np.sqrt(np.mean(tgt ** 2)))
                self._ic[k] = np.transpose(win, (2, 0, 1)) / (n_ic + eps)  # [K,64,64]
                self._tgt[k] = tgt / (n_tgt + eps)
                self._n_ic[k] = n_ic
                self._n_tgt[k] = n_tgt
                k += 1

    def __len__(self) -> int:
        return len(self._n_ic)

    def __getitem__(self, idx: int) -> dict:
        return {
            "vorticity_ic": torch.from_numpy(self._ic[idx]),
            "vorticity": torch.from_numpy(self._tgt[idx]),
            "norm_ic": float(self._n_ic[idx]),
            "norm_target": float(self._n_tgt[idx]),
        }


class MultiFrameScaledDataset(Dataset):
    """``K`` past frames -> next frame, target in the SAME units as the input.

    Unlike :class:`MultiFrameDataset`, the target is **not** normalised by its
    own L2 norm — both the input window and the target are divided by the
    *window's* RMS.  This makes the model predict physically-scaled output:

    * no oracle leak — de-normalisation needs only the input window, which is
      available at inference time;
    * the (deterministic) amplitude growth the flow exhibits is something the
      model must actually learn, rather than being cancelled by the metric;
    * true autoregressive rollouts become possible.

    Sample: ``[K, 64, 64]`` -> ``[64, 64]`` (both in window-RMS units).
    """

    def __init__(self, traj_indices: list[int], n_ctx: int = 10):
        raw = _get_raw()                      # [N, 64, 64, T]
        T = raw.shape[-1]
        starts = list(range(n_ctx - 1, T - 1))
        n_pairs = len(starts)
        n = len(traj_indices) * n_pairs

        self._ic = np.empty((n, n_ctx, 64, 64), dtype=np.float32)
        self._tgt = np.empty((n, 64, 64), dtype=np.float32)
        self._n_win = np.empty(n, dtype=np.float32)

        eps = 1e-12
        k = 0
        for ti in traj_indices:
            traj = raw[ti]
            for t in starts:
                win = traj[:, :, t - n_ctx + 1:t + 1]     # [64, 64, K]
                tgt = traj[:, :, t + 1]                   # [64, 64]
                n_win = float(np.sqrt(np.mean(win ** 2)))
                self._ic[k] = np.transpose(win, (2, 0, 1)) / (n_win + eps)
                self._tgt[k] = tgt / (n_win + eps)        # SAME scale as input
                self._n_win[k] = n_win
                k += 1

    def __len__(self) -> int:
        return len(self._n_win)

    def __getitem__(self, idx: int) -> dict:
        return {
            "vorticity_ic": torch.from_numpy(self._ic[idx]),
            "vorticity": torch.from_numpy(self._tgt[idx]),
            "norm_ic": float(self._n_win[idx]),
            "norm_target": float(self._n_win[idx]),   # same scale by design
        }


def make_dataloaders_scaled(
    n_train: int = 1000,
    n_val: int = 100,
    n_test: int = 100,
    n_ctx: int = 10,
    batch_size: int = 16,
    val_batch_size: int | None = None,
    device: str = "cpu",
) -> dict[str, DataLoader]:
    """Window-scaled multi-frame variant (no oracle target norm)."""
    raw = _get_raw()
    n_all = raw.shape[0]
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_all)

    idx_tr = sorted(perm[:n_train].tolist())
    idx_va = sorted(perm[n_train:n_train + n_val].tolist())
    idx_te = sorted(perm[n_train + n_val:n_train + n_val + n_test].tolist())

    if val_batch_size is None:
        val_batch_size = len(idx_va)

    return {
        "train": DataLoader(MultiFrameScaledDataset(idx_tr, n_ctx), batch_size=batch_size, shuffle=True),
        "val": DataLoader(MultiFrameScaledDataset(idx_va, n_ctx), batch_size=val_batch_size, shuffle=False),
        "test": DataLoader(MultiFrameScaledDataset(idx_te, n_ctx), batch_size=val_batch_size, shuffle=False),
    }


def make_dataloaders_ctx(
    n_train: int = 800,
    n_val: int = 200,
    n_test: int = 200,
    n_ctx: int = 10,
    batch_size: int = 32,
    val_batch_size: int | None = None,
    device: str = "cpu",
) -> dict[str, DataLoader]:
    """Multi-frame-context variant of :func:`make_dataloaders` (same split)."""
    raw = _get_raw()
    n_all = raw.shape[0]
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_all)

    idx_tr = sorted(perm[:n_train].tolist())
    idx_va = sorted(perm[n_train:n_train + n_val].tolist())
    idx_te = sorted(perm[n_train + n_val:n_train + n_val + n_test].tolist())

    if val_batch_size is None:
        val_batch_size = len(idx_va)

    return {
        "train": DataLoader(MultiFrameDataset(idx_tr, n_ctx), batch_size=batch_size, shuffle=True),
        "val": DataLoader(MultiFrameDataset(idx_va, n_ctx), batch_size=val_batch_size, shuffle=False),
        "test": DataLoader(MultiFrameDataset(idx_te, n_ctx), batch_size=val_batch_size, shuffle=False),
    }


def make_dataloaders(
    n_train: int = 800,
    n_val: int = 200,
    n_test: int = 200,
    batch_size: int = 32,
    val_batch_size: int | None = None,
    device: str = "cpu",
) -> dict[str, DataLoader]:
    raw = _get_raw()
    n_all = raw.shape[0]
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(n_all)

    idx_tr = sorted(perm[:n_train].tolist())
    idx_va = sorted(perm[n_train:n_train + n_val].tolist())
    idx_te = sorted(perm[n_train + n_val:n_train + n_val + n_test].tolist())

    if val_batch_size is None:
        val_batch_size = len(idx_va)

    return {
        "train": DataLoader(L2NormPairDataset(idx_tr), batch_size=batch_size, shuffle=True),
        "val": DataLoader(L2NormPairDataset(idx_va), batch_size=val_batch_size, shuffle=False),
        "test": DataLoader(L2NormPairDataset(idx_te), batch_size=val_batch_size, shuffle=False),
    }
