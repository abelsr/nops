"""
Graph Neural Operator (GNO) module for nops.
"""

from nops.gno.models.original import GNO, GNO2D, GNO3D
from nops.gno.models.mGNO import mGNO

__all__ = ["GNO", "GNO2D", "GNO3D", "mGNO"]
