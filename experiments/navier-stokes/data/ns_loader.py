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
