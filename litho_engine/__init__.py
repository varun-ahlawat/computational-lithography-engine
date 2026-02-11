"""
Computational Lithography Engine

A differentiable physics engine for computational lithography,
simulating Fraunhofer diffraction and implementing inverse mask optimization.
"""

__version__ = "0.1.0"

from .diffraction import FraunhoferDiffraction
from .optimizer import MaskOptimizer
from .thermal import ThermalExpansionModel

__all__ = ["FraunhoferDiffraction", "MaskOptimizer", "ThermalExpansionModel"]
