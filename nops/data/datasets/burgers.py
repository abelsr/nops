"""
PyTorch Dataset for 1D Burgers equation data.

Input / target convention
--------------------------
    input  : u(x, 0)          — initial condition,  shape ``(N,)``
    target : u(x, T)          — solution at final snapshot, shape ``(N,)``

For multi-step training, the full trajectory is available via
``dataset._get_data()["solution"]`` (shape ``(n, N, T_steps)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from nops.data.datasets.base import BasePDEDataset
from nops.data.generators.burgers import BurgersGenerator


class BurgersDataset(BasePDEDataset):
    """Dataset of 1D Burgers equation trajectories.

    Can be constructed from:

    - A pre-generated data dict (``data=...``).
    - A :class:`~nops.data.generators.burgers.BurgersGenerator` that is
      called immediately on construction (``generator=...``).
    - A ``.pt`` file produced by :meth:`save` (``path=...``,
      ``mode="disk"``).

    Parameters
    ----------
    data : dict[str, Tensor] or None
        Pre-built data dict with keys ``"ic"`` and ``"solution"``.
    generator : BurgersGenerator or None
        If provided (and ``data`` is ``None``), :meth:`generate` is called
        with ``n_samples`` immediately.
    n_samples : int
        Number of samples to generate when ``generator`` is given.
        Default ``1000``.
    mode : {"memory", "disk"}
        Storage mode.  Default ``"memory"``.
    path : str or Path or None
        File path for disk mode or for loading a saved dataset.
    snapshot_idx : int or None
        Which snapshot to use as target.  Defaults to the **last** snapshot
        (``-1``).

    Examples
    --------
    >>> gen = BurgersGenerator(N=1024, nu=0.1, T=1.0, record_steps=1)
    >>> ds = BurgersDataset(generator=gen, n_samples=200)
    >>> x, y = ds[0]
    >>> x.shape, y.shape   # (1024,), (1024,)

    >>> loader = ds.get_dataloader(batch_size=32)
    """

    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        generator: BurgersGenerator | None = None,
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
                generator = BurgersGenerator()
            data = generator.generate(n_samples)

        super().__init__(data=data, mode=mode)

    def _make_pair(self, data: dict[str, Tensor], idx: int) -> tuple[Tensor, Tensor]:
        inp = data["ic"][idx]                           # (N,)
        tgt = data["solution"][idx, :, self.snapshot_idx]  # (N,)
        return inp, tgt

    @classmethod
    def load(cls, path: str | Path) -> "BurgersDataset":
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data=raw["data"], mode="memory")
