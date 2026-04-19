"""
PyTorch Dataset for 2D Poisson equation data.

Input / target convention
--------------------------
    input  : f(x)   — source term,  shape ``(N, N)``
    target : u(x)   — potential,    shape ``(N, N)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from nops.data.datasets.base import BasePDEDataset
from nops.data.generators.poisson import PoissonGenerator


class PoissonDataset(BasePDEDataset):
    """Dataset of 2D Poisson source-solution pairs.

    Parameters
    ----------
    data : dict[str, Tensor] or None
        Pre-built data dict with keys ``"source"`` and ``"solution"``.
    generator : PoissonGenerator or None
        If provided (and ``data`` is ``None``), :meth:`generate` is called
        immediately with ``n_samples``.
    n_samples : int
        Number of samples.  Default ``1000``.
    mode : {"memory", "disk"}
        Storage mode.  Default ``"memory"``.
    path : str or Path or None
        File path for disk mode or saved dataset.

    Examples
    --------
    >>> gen = PoissonGenerator(N=128, L=1.0)
    >>> ds = PoissonDataset(generator=gen, n_samples=200)
    >>> x, y = ds[0]
    >>> x.shape, y.shape   # (128, 128), (128, 128)
    """

    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        generator: PoissonGenerator | None = None,
        n_samples: int = 1000,
        mode: Literal["memory", "disk"] = "memory",
        path: str | Path | None = None,
    ) -> None:
        if mode == "disk" and data is None and generator is None:
            super().__init__(mode="disk", path=path)
            return

        if data is None:
            if generator is None:
                generator = PoissonGenerator()
            data = generator.generate(n_samples)

        super().__init__(data=data, mode=mode)

    def _make_pair(self, data: dict[str, Tensor], idx: int) -> tuple[Tensor, Tensor]:
        inp = data["source"][idx]    # (N, N)
        tgt = data["solution"][idx]  # (N, N)
        return inp, tgt

    @classmethod
    def load(cls, path: str | Path) -> "PoissonDataset":
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data=raw["data"], mode="memory")
