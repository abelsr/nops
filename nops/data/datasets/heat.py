"""
PyTorch Dataset for 1D/2D heat equation data.

Input / target convention
--------------------------
    input  : u(x, 0)   — initial temperature, shape ``(N,)`` or ``(N, N)``
    target : u(x, T)   — temperature at final snapshot, same shape

The per-sample diffusivity ``α`` and the full trajectory are available via
``dataset._get_data()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from nops.data.datasets.base import BasePDEDataset
from nops.data.generators.heat import HeatGenerator


class HeatDataset(BasePDEDataset):
    """Dataset of 1D/2D heat equation trajectories.

    Parameters
    ----------
    data : dict[str, Tensor] or None
        Pre-built data dict with keys ``"ic"``, ``"solution"``, ``"alpha"``.
    generator : HeatGenerator or None
        If provided (and ``data`` is ``None``), :meth:`generate` is called
        immediately with ``n_samples``.
    n_samples : int
        Number of samples to generate.  Default ``1000``.
    mode : {"memory", "disk"}
        Storage mode.  Default ``"memory"``.
    path : str or Path or None
        File path for disk mode or saved dataset.
    snapshot_idx : int
        Which snapshot index to use as target.  Default ``-1`` (last).

    Examples
    --------
    >>> gen = HeatGenerator(N=64, dim=2, alpha=0.01, T=1.0, record_steps=1)
    >>> ds = HeatDataset(generator=gen, n_samples=100)
    >>> x, y = ds[0]
    >>> x.shape, y.shape   # (64, 64), (64, 64)
    """

    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        generator: HeatGenerator | None = None,
        n_samples: int = 1000,
        mode: Literal["memory", "disk"] = "memory",
        path: str | Path | None = None,
        snapshot_idx: int = -1,
    ) -> None:
        self.snapshot_idx = snapshot_idx

        if mode == "disk" and data is None and generator is None:
            super().__init__(mode="disk", path=path)
            return

        if data is None:
            if generator is None:
                generator = HeatGenerator()
            data = generator.generate(n_samples)

        super().__init__(data=data, mode=mode)

    def _make_pair(self, data: dict[str, Tensor], idx: int) -> tuple[Tensor, Tensor]:
        inp = data["ic"][idx]                                # (*spatial)
        tgt = data["solution"][idx, ..., self.snapshot_idx]  # (*spatial)
        return inp, tgt

    @classmethod
    def load(cls, path: str | Path) -> "HeatDataset":
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data=raw["data"], mode="memory")
