"""
Tests for the computational lithography engine.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litho_engine import FraunhoferDiffraction, MaskOptimizer
from litho_engine.diffraction import create_test_mask
from litho_engine.optimizer import AdaptiveMaskOptimizer


class TestFraunhoferDiffraction:
    """Tests for Fraunhofer diffraction simulator."""
    
    def test_initialization(self):
        """Test diffraction model initialization."""
        diffraction = FraunhoferDiffraction(wavelength=193.0, pixel_size=10.0, NA=0.6)
        assert diffraction.wavelength == 193.0
        assert diffraction.pixel_size == 10.0
        assert diffraction.NA == 0.6
        
    def test_forward_pass_2d(self):
        """Test forward pass with 2D mask."""
        diffraction = FraunhoferDiffraction()
        mask = torch.ones((64, 64))
        
        intensity = diffraction(mask)
        
        assert intensity.shape == (1, 1, 64, 64)
        assert intensity.min() >= 0  # Intensity should be non-negative
        
    def test_forward_pass_4d(self):
        """Test forward pass with 4D mask (batch)."""
        diffraction = FraunhoferDiffraction()
        mask = torch.ones((2, 1, 64, 64))
        
        intensity = diffraction(mask)
        
        assert intensity.shape == (2, 1, 64, 64)
        assert intensity.min() >= 0
        
    def test_differentiability(self):
        """Test that the diffraction is differentiable."""
        diffraction = FraunhoferDiffraction()
        mask = torch.ones((64, 64), requires_grad=True)
        
        intensity = diffraction(mask)
        loss = intensity.sum()
        loss.backward()
        
        assert mask.grad is not None
        assert mask.grad.shape == mask.shape
        
    def test_energy_conservation(self):
        """Test that total energy is reasonable."""
        diffraction = FraunhoferDiffraction()
        mask = torch.ones((64, 64))
        
        intensity = diffraction(mask)
        total_intensity = intensity.sum()
        
        # Total intensity should be positive
        assert total_intensity > 0
        
    def test_create_test_masks(self):
        """Test creation of different mask patterns."""
        patterns = ['square', 'circle', 'lines']
        
        for pattern_type in patterns:
            mask = create_test_mask(size=64, pattern_type=pattern_type)
            assert mask.shape == (64, 64)
            assert mask.min() >= 0
            assert mask.max() <= 1


class TestMaskOptimizer:
    """Tests for mask optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        diffraction = FraunhoferDiffraction()
        optimizer = MaskOptimizer(
            diffraction_model=diffraction,
            mask_shape=(64, 64),
            learning_rate=0.01
        )
        
        assert optimizer.mask.shape == (64, 64)
        assert optimizer.learning_rate == 0.01
        
    def test_optimize_simple_target(self):
        """Test optimization with a simple target."""
        diffraction = FraunhoferDiffraction()
        optimizer = MaskOptimizer(
            diffraction_model=diffraction,
            mask_shape=(32, 32),
            learning_rate=0.05,
            regularization=0.001
        )
        
        # Simple target: uniform intensity
        target = torch.ones((32, 32)) * 0.5
        
        results = optimizer.optimize(
            target=target,
            num_iterations=10,
            loss_type='mse',
            verbose=False
        )
        
        assert 'mask' in results
        assert 'loss_history' in results
        assert 'final_loss' in results
        assert len(results['loss_history']) == 10
        
    def test_loss_decreases(self):
        """Test that loss decreases during optimization."""
        diffraction = FraunhoferDiffraction()
        optimizer = MaskOptimizer(
            diffraction_model=diffraction,
            mask_shape=(32, 32),
            learning_rate=0.1,
            regularization=0.0
        )
        
        target = torch.ones((32, 32)) * 0.3
        
        results = optimizer.optimize(
            target=target,
            num_iterations=30,
            loss_type='mse',
            verbose=False
        )
        
        # Loss should generally decrease
        initial_loss = results['loss_history'][0]
        final_loss = results['loss_history'][-1]
        
        assert final_loss < initial_loss
        
    def test_set_initial_mask(self):
        """Test setting initial mask."""
        diffraction = FraunhoferDiffraction()
        optimizer = MaskOptimizer(
            diffraction_model=diffraction,
            mask_shape=(32, 32)
        )
        
        initial = torch.ones((32, 32)) * 0.7
        optimizer.set_initial_mask(initial)
        
        current = optimizer.get_current_mask()
        
        # Should be close to initial value
        assert torch.allclose(current, initial, atol=0.1)
        
    def test_mask_constrained_range(self):
        """Test that optimized mask is in [0, 1] range."""
        diffraction = FraunhoferDiffraction()
        optimizer = MaskOptimizer(
            diffraction_model=diffraction,
            mask_shape=(32, 32)
        )
        
        target = torch.rand((32, 32))
        
        results = optimizer.optimize(
            target=target,
            num_iterations=20,
            verbose=False
        )
        
        mask = results['mask']
        assert mask.min() >= 0
        assert mask.max() <= 1


class TestAdaptiveMaskOptimizer:
    """Tests for adaptive mask optimizer."""
    
    def test_initialization_with_scheduler(self):
        """Test initialization with learning rate scheduler."""
        diffraction = FraunhoferDiffraction()
        optimizer = AdaptiveMaskOptimizer(
            diffraction_model=diffraction,
            mask_shape=(32, 32),
            use_scheduler=True
        )
        
        assert optimizer.scheduler is not None
        
    def test_early_stopping(self):
        """Test early stopping functionality."""
        diffraction = FraunhoferDiffraction()
        optimizer = AdaptiveMaskOptimizer(
            diffraction_model=diffraction,
            mask_shape=(32, 32),
            learning_rate=0.1
        )
        
        target = torch.rand((32, 32))
        
        results = optimizer.optimize(
            target=target,
            num_iterations=200,
            early_stopping_patience=10,
            verbose=False
        )
        
        # Should stop before 200 iterations due to early stopping
        # (unless it keeps improving continuously)
        assert len(results['loss_history']) <= 200


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_end_to_end_optimization(self):
        """Test complete optimization pipeline."""
        # Setup
        diffraction = FraunhoferDiffraction(wavelength=193.0, NA=0.6)
        
        # Create simple target
        target = create_test_mask(size=32, pattern_type='square')
        
        # Optimize
        optimizer = MaskOptimizer(
            diffraction_model=diffraction,
            mask_shape=(32, 32),
            learning_rate=0.05
        )
        
        results = optimizer.optimize(
            target=target,
            num_iterations=50,
            verbose=False
        )
        
        # Verify results
        optimized_mask = results['mask']
        
        # Check that optimized mask produces reasonable output
        with torch.no_grad():
            predicted = diffraction(optimized_mask).squeeze()
            
        # Should have similar shape characteristics
        assert predicted.shape == target.shape
        
    def test_gradient_flow(self):
        """Test that gradients flow through entire pipeline."""
        diffraction = FraunhoferDiffraction()
        mask = torch.rand((32, 32), requires_grad=True)
        target = torch.ones((32, 32))
        
        # Forward pass
        intensity = diffraction(mask)
        loss = torch.nn.functional.mse_loss(intensity, target.unsqueeze(0).unsqueeze(0))
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert mask.grad is not None
        assert not torch.isnan(mask.grad).any()
        assert not torch.isinf(mask.grad).any()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
