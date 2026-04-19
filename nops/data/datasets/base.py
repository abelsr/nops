"""
Abstract base class for PDE datasets.

All datasets expose a standard :class:`torch.utils.data.Dataset` interface
returning ``(input, target)`` pairs, plus two extra capabilities:

1. **Two storage modes**

   - ``mode="memory"`` — tensors are held in RAM after generation or loading.
   - ``mode="disk"``   — a single ``.pt`` file on disk; samples are loaded
     lazily in :meth:`__getitem__` (useful when the full dataset does not fit
     in RAM).

2. **Save / load round-trip**

   :meth:`save` serialises the full raw ``data`` dict plus metadata to a
   ``.pt`` file.  :meth:`load` restores an identical dataset from that file.

Subclasses only need to implement :meth:`_make_pair`, which maps the raw
``dict[str, Tensor]`` entry at a given index to an ``(input, target)`` pair.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class BasePDEDataset(Dataset, ABC):
    """Base class for PDE datasets with memory and disk storage modes.

    Parameters
    ----------
    data : dict[str, Tensor] or None
        Pre-built raw data dict (e.g. as returned by a generator).
        Required when ``mode="memory"`` and no path/generator is given.
    mode : {"memory", "disk"}
        Storage mode.  In ``"disk"`` mode, ``data`` is ignored and the
        dataset is loaded lazily from ``path``.
    path : str or Path or None
        Path to a ``.pt`` file.  Required when ``mode="disk"``.

    Notes
    -----
    Subclasses must implement :meth:`_make_pair`.
    """

    def __init__(
        self,
        data: dict[str, Tensor] | None = None,
        mode: Literal["memory", "disk"] = "memory",
        path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode

        if mode == "memory":
            if data is None:
                raise ValueError("'data' must be provided when mode='memory'")
            self._data = data
            self._n = self._infer_length(data)

        elif mode == "disk":
            if path is None:
                raise ValueError("'path' must be provided when mode='disk'")
            self._path = Path(path)
            if not self._path.exists():
                raise FileNotFoundError(f"Dataset file not found: {self._path}")
            # Load just to know the length; keep the full file on disk
            meta = torch.load(self._path, map_location="cpu", weights_only=True)
            self._n = self._infer_length(meta["data"])
            self._disk_data: dict[str, Tensor] | None = None  # lazy

        else:
            raise ValueError(f"mode must be 'memory' or 'disk', got {mode!r}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _make_pair(self, data: dict[str, Tensor], idx: int) -> tuple[Tensor, Tensor]:
        """Extract ``(input, target)`` tensors for sample ``idx``.

        Parameters
        ----------
        data : dict[str, Tensor]
            The full raw data dict (all samples).
        idx : int
            Sample index.

        Returns
        -------
        input : Tensor
        target : Tensor
        """
        ...

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        data = self._get_data()
        return self._make_pair(data, idx)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the dataset to a ``.pt`` file.

        Parameters
        ----------
        path : str or Path
            Destination file.  The ``.pt`` extension is recommended.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._get_data()
        torch.save(
            {
                "data": data,
                "class": self.__class__.__name__,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BasePDEDataset":
        """Load a dataset saved with :meth:`save`.

        Parameters
        ----------
        path : str or Path
            Path to the ``.pt`` file produced by :meth:`save`.

        Returns
        -------
        BasePDEDataset
            A new instance in ``"memory"`` mode with the loaded data.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        return cls(data=raw["data"], mode="memory")

    # ------------------------------------------------------------------
    # DataLoader helper
    # ------------------------------------------------------------------

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """Wrap this dataset in a :class:`~torch.utils.data.DataLoader`.

        Parameters
        ----------
        batch_size : int
            Samples per batch.  Default ``32``.
        shuffle : bool
            Whether to shuffle at every epoch.  Default ``True``.
        num_workers : int
            Subprocesses for data loading.  Default ``0`` (main process).
        **kwargs
            Forwarded to :class:`~torch.utils.data.DataLoader`.

        Returns
        -------
        DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_data(self) -> dict[str, Tensor]:
        if self.mode == "memory":
            return self._data
        # Disk mode: load on first access and cache
        if self._disk_data is None:
            raw = torch.load(self._path, map_location="cpu", weights_only=True)
            self._disk_data = raw["data"]
        return self._disk_data

    @staticmethod
    def _infer_length(data: dict[str, Tensor]) -> int:
        """Return the batch dimension (first dim of any tensor in ``data``)."""
        for v in data.values():
            if isinstance(v, Tensor) and v.dim() >= 1:
                return v.shape[0]
        raise ValueError("Cannot infer dataset length: no tensors found in data dict")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n={self._n}, mode={self.mode!r})"
        )
