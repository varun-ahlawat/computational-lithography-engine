"""
Thermal Expansion Model for Silicon Wafer

Models the thermal expansion and contraction of silicon wafers during
the lithography process. The printed pattern on a hot wafer will shrink
when cooled to operating temperature, so the mask must be pre-compensated.

Silicon thermal expansion coefficient: ~2.6e-6 /°C (at room temperature)
This varies with temperature; we use a polynomial fit for accuracy.
"""

import torch
import torch.nn.functional as F
import numpy as np


class ThermalExpansionModel:
    """
    Models thermal expansion/contraction of a silicon wafer.

    During lithography, the wafer is at an elevated process temperature.
    When cooled to operating temperature, the pattern contracts. This model
    computes the scaling factor and applies geometric transformation to
    account for this effect.

    Args:
        process_temp (float): Temperature during lithography exposure (°C)
        operating_temp (float): Final operating temperature after cooling (°C)
        reference_temp (float): Reference temperature for CTE data (°C)
        cte_coefficients (tuple): Polynomial coefficients for CTE(T) in /°C.
            Default uses silicon: CTE(T) = a0 + a1*T + a2*T^2
            where T is in °C. Default values approximate silicon from
            literature (Okada & Tokumaru, 1984).
    """

    def __init__(self, process_temp=200.0, operating_temp=80.0,
                 reference_temp=25.0, cte_coefficients=None):
        self.process_temp = process_temp
        self.operating_temp = operating_temp
        self.reference_temp = reference_temp

        if cte_coefficients is None:
            # Silicon CTE polynomial coefficients (per °C)
            # CTE(T) ≈ 2.568e-6 + 3.477e-9 * T - 1.985e-12 * T^2
            # Valid roughly 20-500°C range
            self.cte_coefficients = (2.568e-6, 3.477e-9, -1.985e-12)
        else:
            self.cte_coefficients = cte_coefficients

    def cte_at_temperature(self, temp):
        """
        Compute coefficient of thermal expansion at a given temperature.

        Args:
            temp (float): Temperature in °C

        Returns:
            float: CTE in /°C
        """
        cte = 0.0
        for i, coeff in enumerate(self.cte_coefficients):
            cte += coeff * (temp ** i)
        return cte

    def compute_strain(self):
        """
        Compute the linear thermal strain from process to operating temperature.

        Uses numerical integration of CTE(T) over the temperature range.

        Returns:
            float: Linear thermal strain (dimensionless, negative = contraction)
        """
        # Numerical integration of CTE(T) dT from operating_temp to process_temp
        n_steps = 1000
        temps = np.linspace(self.operating_temp, self.process_temp, n_steps)
        dt = (self.process_temp - self.operating_temp) / (n_steps - 1)

        strain = 0.0
        for t in temps:
            strain += self.cte_at_temperature(t) * dt

        # Negative because cooling causes contraction
        return -strain

    def compute_scale_factor(self):
        """
        Compute the geometric scale factor for pattern transformation.

        When the wafer cools from process_temp to operating_temp,
        the pattern shrinks by this factor.

        Returns:
            float: Scale factor (< 1 means contraction)
        """
        strain = self.compute_strain()
        return 1.0 + strain

    def apply_thermal_contraction(self, pattern):
        """
        Apply thermal contraction to a pattern (simulates what happens
        when wafer cools from process temperature to operating temperature).

        Args:
            pattern (torch.Tensor): 2D pattern (H, W) or (B, C, H, W)

        Returns:
            torch.Tensor: Contracted pattern (same shape, zero-padded)
        """
        scale = self.compute_scale_factor()
        return self._rescale_pattern(pattern, scale)

    def apply_thermal_precompensation(self, pattern):
        """
        Pre-compensate a pattern for thermal contraction. The pattern is
        expanded so that after cooling, it matches the intended geometry.

        Args:
            pattern (torch.Tensor): 2D target pattern (H, W) or (B, C, H, W)

        Returns:
            torch.Tensor: Pre-compensated (expanded) pattern
        """
        scale = self.compute_scale_factor()
        # Inverse scale: expand so that after contraction we get original
        inverse_scale = 1.0 / scale
        return self._rescale_pattern(pattern, inverse_scale)

    def _rescale_pattern(self, pattern, scale):
        """
        Rescale a pattern by a given factor using bilinear interpolation.

        Args:
            pattern (torch.Tensor): Input pattern
            scale (float): Scale factor

        Returns:
            torch.Tensor: Rescaled pattern (same dimensions)
        """
        original_dim = pattern.dim()
        if pattern.dim() == 2:
            pattern = pattern.unsqueeze(0).unsqueeze(0)
        elif pattern.dim() == 3:
            pattern = pattern.unsqueeze(1)

        B, C, H, W = pattern.shape

        # Create affine transformation matrix for scaling
        theta = torch.tensor([
            [scale, 0, 0],
            [0, scale, 0]
        ], dtype=pattern.dtype, device=pattern.device).unsqueeze(0)
        theta = theta.expand(B, -1, -1)

        grid = F.affine_grid(theta, pattern.size(), align_corners=False)
        rescaled = F.grid_sample(pattern, grid, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)

        if original_dim == 2:
            return rescaled.squeeze(0).squeeze(0)
        elif original_dim == 3:
            return rescaled.squeeze(1)
        return rescaled

    def get_info(self):
        """
        Get a summary of thermal model parameters.

        Returns:
            dict: Model parameters and computed values
        """
        strain = self.compute_strain()
        scale = self.compute_scale_factor()
        return {
            'process_temp_C': self.process_temp,
            'operating_temp_C': self.operating_temp,
            'cte_at_process': self.cte_at_temperature(self.process_temp),
            'cte_at_operating': self.cte_at_temperature(self.operating_temp),
            'linear_strain': strain,
            'scale_factor': scale,
            'contraction_ppm': strain * 1e6,
        }
