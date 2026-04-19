"""
PDE-specific residual losses.

Each class implements the strong-form residual of a classical PDE and is
usable as a stand-alone ``nn.Module`` or as a component of
:class:`~nops.losses.CombinedLoss`.
"""

from .burgers import BurgersResidual
from .darcy import DarcyResidual
from .generic import GenericPDELoss
from .heat import HeatResidual
from .navier_stokes import NavierStokesResidual
from .poisson import PoissonResidual
from .wave import WaveResidual

__all__ = [
    "BurgersResidual",
    "DarcyResidual",
    "GenericPDELoss",
    "HeatResidual",
    "NavierStokesResidual",
    "PoissonResidual",
    "WaveResidual",
]
