# Computational Lithography Engine

A differentiable physics engine for computational lithography, simulating Fraunhofer diffraction and implementing inverse mask optimization (in PyTorch).

## Overview

This engine provides a complete framework for computational lithography simulations, including:

- **Fraunhofer Diffraction Simulation**: Physics-based modeling of far-field diffraction patterns using Fast Fourier Transform (FFT)
- **Inverse Lithography Technology (ILT)**: Gradient-based optimization to find optimal mask patterns
- **Differentiable Architecture**: Built on PyTorch for automatic differentiation and GPU acceleration
- **Numerical Aperture Modeling**: Realistic optical system constraints

## Installation

### From Source

```bash
git clone https://github.com/yup-VARUN/computational-lithography-engine.git
cd computational-lithography-engine
pip install -r requirements.txt
pip install -e .
```

### Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0

## Quick Start

### Forward Diffraction Simulation

```python
import torch
from litho_engine import FraunhoferDiffraction
from litho_engine.diffraction import create_test_mask

# Create diffraction model (DUV lithography parameters)
diffraction = FraunhoferDiffraction(
    wavelength=193.0,  # nm
    pixel_size=10.0,   # nm
    NA=0.6             # Numerical aperture
)

# Create a test mask
mask = create_test_mask(size=128, pattern_type='square')

# Simulate diffraction
intensity = diffraction(mask)
```

### Inverse Mask Optimization

```python
from litho_engine import MaskOptimizer

# Define target pattern
target = create_test_mask(size=128, pattern_type='circle')

# Initialize optimizer
optimizer = MaskOptimizer(
    diffraction_model=diffraction,
    mask_shape=(128, 128),
    learning_rate=0.05,
    regularization=0.01
)

# Optimize mask to match target
results = optimizer.optimize(
    target=target,
    num_iterations=100,
    verbose=True
)

optimized_mask = results['mask']
```

## Examples

Run the provided examples to see the engine in action:

```bash
# Forward diffraction simulation
python examples/forward_diffraction.py

# Inverse mask optimization
python examples/inverse_optimization.py
```

## Architecture

### Core Components

#### 1. FraunhoferDiffraction

Simulates far-field diffraction using the Fraunhofer approximation:

- **Input**: Binary or grayscale mask pattern
- **Process**: FFT-based diffraction with pupil function filtering
- **Output**: Intensity distribution at the image plane

Key features:
- Fully differentiable for gradient-based optimization
- Supports both coherent and partially coherent illumination
- Configurable optical parameters (wavelength, NA, pixel size)

#### 2. MaskOptimizer

Implements inverse lithography technology (ILT):

- **Objective**: Find optimal mask that produces desired target pattern
- **Method**: Gradient descent using PyTorch's autograd
- **Regularization**: Total variation (TV) for smooth masks
- **Loss Functions**: MSE, L1, Binary Cross-Entropy

#### 3. AdaptiveMaskOptimizer

Enhanced optimizer with:
- Adaptive learning rate scheduling
- Early stopping to prevent overfitting
- Better convergence for complex patterns

## Physics Background

### Fraunhofer Diffraction

In the far-field regime, the diffraction pattern is the Fourier transform of the aperture function:

```
U(x,y) = F{A(ξ,η)}
I(x,y) = |U(x,y)|²
```

Where:
- `A(ξ,η)` is the aperture/mask function
- `U(x,y)` is the complex field at the image plane
- `I(x,y)` is the intensity distribution
- `F{}` denotes Fourier transform

### Numerical Aperture

The pupil function limits the spatial frequencies that can pass through:

```
P(fx,fy) = 1 if √(fx² + fy²) ≤ NA/λ
         = 0 otherwise
```

This creates the fundamental resolution limit in optical lithography.

### Inverse Optimization

The optimization problem:

```
minimize: L(I_predicted, I_target) + λ·R(mask)
```

Where:
- `L` is the loss function (e.g., MSE)
- `R` is the regularization term (e.g., total variation)
- `λ` is the regularization weight

## Testing

Run the test suite:

```bash
pytest tests/test_litho_engine.py -v
```

Tests cover:
- Forward diffraction correctness
- Gradient flow verification
- Optimization convergence
- Edge cases and constraints

## Applications

This engine can be used for:

1. **Semiconductor Manufacturing**: Optimize photomask designs for chip fabrication
2. **Optical Engineering**: Analyze diffraction patterns in optical systems
3. **Research**: Study inverse problems in computational imaging
4. **Education**: Demonstrate wave optics and optimization principles

## Performance Tips

- **GPU Acceleration**: Move models to GPU for faster computation:
  ```python
  diffraction = diffraction.cuda()
  mask = mask.cuda()
  ```

- **Batch Processing**: Process multiple masks simultaneously:
  ```python
  masks = torch.stack([mask1, mask2, mask3])
  intensities = diffraction(masks)
  ```

- **Memory Management**: Use smaller mask sizes or gradient checkpointing for large optimizations

## Limitations

- Assumes Fraunhofer (far-field) approximation
- Coherent or simple partially coherent illumination models
- Does not include resist effects or 3D mask topography
- Simplified pupil function model

## Future Enhancements

- [ ] Hopkins partially coherent imaging model
- [ ] Resist model integration (Mack model, etc.)
- [ ] 3D mask effects (thick mask modeling)
- [ ] Advanced source optimization
- [ ] Multi-objective optimization

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License

## References

- Wong, A. K. K. (2001). *Resolution Enhancement Techniques in Optical Lithography*
- Goodman, J. W. (2005). *Introduction to Fourier Optics*
- Ma, X., & Arce, G. R. (2010). "Computational Lithography"

## Citation

If you use this engine in your research, please cite:

```bibtex
@software{computational_lithography_engine,
  author = {VARUN},
  title = {Computational Lithography Engine},
  year = {2024},
  url = {https://github.com/yup-VARUN/computational-lithography-engine}
}
```

A good read [Nature Article](https://www.nature.com/articles/s41377-025-01923-w)
