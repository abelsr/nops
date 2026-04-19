"""
PyTorch Dataset for 2D Navier-Stokes (vorticity) data.

Input / target convention
--------------------------
    input  : ω(x, 0)     — initial vorticity, shape ``(N, N)``
    target : ω(x, T)     — vorticity at final snapshot, shape ``(N, N)``

The full vorticity trajectory is available via
``dataset._get_data()["vorticity"]`` (shape ``(n, N, N, T_steps)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from nops.data.datasets.base import BasePDEDataset
from nops.data.generators.navier_stokes import NavierStokesGenerator


class NavierStokesDataset(BasePDEDataset):
    """Dataset of 2D Navier-Stokes vorticity trajectories.

    Parameters
    ----------
    data : dict[str, Tensor] or None
        Pre-built data dict with keys ``"ic"`` and ``"vorticity"``.
    generator : NavierStokesGenerator or None
        If provided (and ``data`` is ``None``), :meth:`generate` is called
        immediately with ``n_samples``.
    n_samples : int
        Number of samples to generate.  Default ``1000``.
    mode : {"memory", "disk"}
        Storage mode.  Default ``"memory"``.
    path : str or Path or None
        File path for disk mode or saved dataset.
    snapshot_idx : int
        Which vorticity snapshot to use as target.  Default ``-1`` (last).

    Examples
    --------
    >>> gen = NavierStokesGenerator(N=64, nu=1e-3, T=1.0, record_steps=1)
    >>> ds = NavierStokesDataset(generator=gen, n_samples=50)
    >>> x, y = ds[0]
    >>> x.shape, y.shape   # (64, 64), (64, 64)
    """

    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        generator: NavierStokesGenerator | None = None,
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
                generator = NavierStokesGenerator()
            data = generator.generate(n_samples)

        super().__init__(data=data, mode=mode)

    def _make_pair(self, data: dict[str, Tensor], idx: int) -> tuple[Tensor, Tensor]:
        inp = data["ic"][idx]                               # (N, N)
        tgt = data["vorticity"][idx, :, :, self.snapshot_idx]  # (N, N)
        return inp, tgt

    @classmethod
    def load(cls, path: str | Path) -> "NavierStokesDataset":
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data=raw["data"], mode="memory")
