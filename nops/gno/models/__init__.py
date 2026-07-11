"""
Models for Graph Neural Operator.
"""

from .original import GNO, GNO2D, GNO3D
from .mGNO import mGNO, mGNOBlock

__all__ = ["GNO", "GNO2D", "GNO3D", "mGNO", "mGNOBlock"]
