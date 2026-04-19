from nops.data.datasets.base import BasePDEDataset
from nops.data.datasets.burgers import BurgersDataset
from nops.data.datasets.darcy import DarcyDataset
from nops.data.datasets.heat import HeatDataset
from nops.data.datasets.navier_stokes import NavierStokesDataset
from nops.data.datasets.poisson import PoissonDataset
from nops.data.datasets.wave import WaveDataset

__all__ = [
    "BasePDEDataset",
    "BurgersDataset",
    "DarcyDataset",
    "HeatDataset",
    "NavierStokesDataset",
    "PoissonDataset",
    "WaveDataset",
]
