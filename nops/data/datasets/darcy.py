"""
PyTorch Dataset for 2D Darcy flow data.

Input / target convention
--------------------------
    input  : a(x)   — permeability coefficient, shape ``(N, N)``
    target : u(x)   — pressure solution,         shape ``(N, N)``
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from nops.data.datasets.base import BasePDEDataset
from nops.data.generators.darcy import DarcyGenerator


class DarcyDataset(BasePDEDataset):
    """Dataset of 2D Darcy flow permeability-solution pairs.

    Parameters
    ----------
    data : dict[str, Tensor] or None
        Pre-built data dict with keys ``"coeff"`` and ``"solution"``.
    generator : DarcyGenerator or None
        If provided (and ``data`` is ``None``), :meth:`generate` is called
        immediately with ``n_samples``.
    n_samples : int
        Number of samples to generate.  Default ``1000``.
    mode : {"memory", "disk"}
        Storage mode.  Default ``"memory"``.
    path : str or Path or None
        File path for disk mode or saved dataset.
        When ``generator`` uses ``solver="file"``, pass ``gen_path`` instead.
    gen_path : str or Path or None
        Path forwarded to :meth:`DarcyGenerator.generate` when
        ``generator.solver == "file"``.

    Examples
    --------
    >>> gen = DarcyGenerator(N=128, solver="spectral")
    >>> ds = DarcyDataset(generator=gen, n_samples=100)
    >>> x, y = ds[0]
    >>> x.shape, y.shape   # (128, 128), (128, 128)

    >>> # Load FNO benchmark .mat file
    >>> gen = DarcyGenerator(N=421, solver="file")
    >>> ds = DarcyDataset(generator=gen, n_samples=1024,
    ...                   gen_path="piececonst_r421_N1024_smooth1.mat")
    """

    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        generator: DarcyGenerator | None = None,
        n_samples: int = 1000,
        mode: Literal["memory", "disk"] = "memory",
        path: str | Path | None = None,
        gen_path: str | Path | None = None,
    ) -> None:
        if mode == "disk" and data is None and generator is None:
            super().__init__(mode="disk", path=path)
            return

        if data is None:
            if generator is None:
                generator = DarcyGenerator()
            data = generator.generate(n_samples, path=gen_path)

        super().__init__(data=data, mode=mode)

    def _make_pair(self, data: dict[str, Tensor], idx: int) -> tuple[Tensor, Tensor]:
        inp = data["coeff"][idx]     # (N, N)
        tgt = data["solution"][idx]  # (N, N)
        return inp, tgt

    @classmethod
    def load(cls, path: str | Path) -> "DarcyDataset":
        path = Path(path)
        raw = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data=raw["data"], mode="memory")
