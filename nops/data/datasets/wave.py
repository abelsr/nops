"""
PyTorch Dataset for 1D/2D wave equation data.

Input / target convention
--------------------------
    input  : stack(u(x,0), ∂u/∂t(x,0), dim=-1)  — shape ``(*spatial, 2)``
    target : u(x, T)                               — shape ``(*spatial,)``

Providing both initial displacement and velocity is essential because the
wave equation is 2nd-order in time; the solution is not determined by
displacement alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from nops.data.datasets.base import BasePDEDataset
from nops.data.generators.wave import WaveGenerator


class WaveDataset(BasePDEDataset):
    """Dataset of 1D/2D wave equation trajectories.

    Parameters
    ----------
    data : dict[str, Tensor] or None
        Pre-built data dict with keys ``"ic_displacement"``,
        ``"ic_velocity"``, ``"solution"``, ``"c"``.
    generator : WaveGenerator or None
        If provided (and ``data`` is ``None``), :meth:`generate` is called
        immediately with ``n_samples``.
    n_samples : int
        Number of samples.  Default ``1000``.
    mode : {"memory", "disk"}
        Storage mode.  Default ``"memory"``.
    path : str or Path or None
        File path for disk mode or saved dataset.
    snapshot_idx : int
        Which snapshot index to use as target.  Default ``-1`` (last).

    Examples
    --------
    >>> gen = WaveGenerator(N=64, dim=2, c=1.0, T=5.0, record_steps=1)
    >>> ds = WaveDataset(generator=gen, n_samples=50)
    >>> x, y = ds[0]
    >>> x.shape   # (64, 64, 2)  — displacement + velocity stacked
    >>> y.shape   # (64, 64)
    """

    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        generator: WaveGenerator | None = None,
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
                generator = WaveGenerator()
            data = generator.generate(n_samples)

        super().__init__(data=data, mode=mode)

    def _make_pair(self, data: dict[str, Tensor], idx: int) -> tuple[Tensor, Tensor]:
        u0 = data["ic_displacement"][idx]                       # (*spatial)
        v0 = data["ic_velocity"][idx]                           # (*spatial)
        # Stack along a new last dimension: (*spatial, 2)
        inp = torch.stack([u0, v0], dim=-1)
        tgt = data["solution"][idx, ..., self.snapshot_idx]     # (*spatial)
        return inp, tgt

    @classmethod
    def load(cls, path: str | Path) -> "WaveDataset":
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data=raw["data"], mode="memory")
