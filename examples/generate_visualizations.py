#!/usr/bin/env python
"""
Generate Visualizations and Animations for Computational Lithography Engine

Produces:
  1. Animated GIF of inverse mask optimization converging (multiple patterns)
  2. Static comparison images showing forward diffraction for various configurations
  3. Thermal compensation visualization showing hot vs cooled patterns
  4. Multi-wavelength / multi-NA comparison
  5. Arbitrary input shape demonstration

All outputs saved to docs/images/ for embedding in README.md.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio.v2 as imageio
import os
import sys
import io

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litho_engine import FraunhoferDiffraction, MaskOptimizer, ThermalExpansionModel
from litho_engine.optimizer import AdaptiveMaskOptimizer, ThermalAwareMaskOptimizer
from litho_engine.diffraction import create_test_mask

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images')
FINAL_FRAME_HOLD_COUNT = 15


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Arbitrary shape helpers
# ---------------------------------------------------------------------------

def create_arbitrary_pattern(size, pattern_name):
    """Create various arbitrary target patterns beyond the built-in ones."""
    mask = torch.zeros((size, size))
    center = size // 2

    if pattern_name == 'L_shape':
        w = size // 8
        h = size // 2
        # Vertical bar
        mask[center - h:center + h, center - h:center - h + w] = 1.0
        # Horizontal bar (bottom)
        mask[center + h - w:center + h, center - h:center + h // 2] = 1.0

    elif pattern_name == 'T_shape':
        w = size // 8
        h = size // 3
        # Vertical stem
        mask[center - h // 2:center + h, center - w:center + w] = 1.0
        # Horizontal top bar
        mask[center - h // 2 - w:center - h // 2 + w,
             center - h:center + h] = 1.0

    elif pattern_name == 'ring':
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        dist = torch.sqrt((x.float() - center) ** 2 + (y.float() - center) ** 2)
        r_outer = size // 4
        r_inner = size // 6
        mask[(dist <= r_outer) & (dist >= r_inner)] = 1.0

    elif pattern_name == 'cross':
        w = size // 10
        h = size // 3
        mask[center - w:center + w, center - h:center + h] = 1.0
        mask[center - h:center + h, center - w:center + w] = 1.0

    elif pattern_name == 'diamond':
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        dist = (torch.abs(x.float() - center) + torch.abs(y.float() - center))
        side = size // 4
        mask[dist <= side] = 1.0

    elif pattern_name == 'zigzag':
        w = size // 12
        seg = size // 6
        for i in range(4):
            y_start = center - size // 3 + i * seg
            y_end = y_start + seg
            if i % 2 == 0:
                x_start = center - size // 6
            else:
                x_start = center + size // 12
            mask[y_start:y_end, x_start:x_start + w] = 1.0
            if i < 3:
                x_lo = min(center - size // 6, center + size // 12)
                x_hi = max(center - size // 6 + w, center + size // 12 + w)
                mask[y_end - w:y_end, x_lo:x_hi] = 1.0

    else:
        # Fallback: use built-in
        mask = create_test_mask(size, pattern_type=pattern_name)

    return mask


# ---------------------------------------------------------------------------
# 1. Inverse mask optimization animation (GIF)
# ---------------------------------------------------------------------------

def generate_optimization_animation(pattern_name='cross', size=128,
                                    n_iterations=200, gif_name=None,
                                    wavelength=193.0, NA=0.6):
    """
    Generate an animated GIF showing the inverse mask optimization converging.
    Frames show: target | current mask | predicted aerial image | loss curve.
    """
    ensure_output_dir()
    if gif_name is None:
        gif_name = f'optimization_{pattern_name}.gif'

    print(f"  Generating optimization animation: {pattern_name} "
          f"(λ={wavelength}nm, NA={NA}) ...")

    diffraction = FraunhoferDiffraction(wavelength=wavelength, NA=NA)
    target = create_arbitrary_pattern(size, pattern_name)

    import torch.nn as nn
    import torch.optim as optim

    mask_param = nn.Parameter(torch.rand(size, size))
    optimizer = optim.Adam([mask_param], lr=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20)

    frames = []
    loss_history = []
    capture_every = max(1, n_iterations // 60)  # ~60 frames max

    for it in range(n_iterations):
        optimizer.zero_grad()
        mask_c = torch.sigmoid(mask_param)
        predicted = diffraction(mask_c)
        target_4d = target.unsqueeze(0).unsqueeze(0)
        loss = torch.nn.functional.mse_loss(predicted, target_4d)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        loss_history.append(loss.item())

        if it % capture_every == 0 or it == n_iterations - 1:
            with torch.no_grad():
                pred_np = predicted.squeeze().cpu().numpy()
                mask_np = mask_c.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(target.numpy(), cmap='inferno', vmin=0, vmax=1)
            axes[0].set_title('Target Pattern', fontsize=11, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Mask (iter {it})', fontsize=11, fontweight='bold')
            axes[1].axis('off')

            axes[2].imshow(pred_np, cmap='inferno', vmin=0, vmax=1)
            axes[2].set_title('Aerial Image', fontsize=11, fontweight='bold')
            axes[2].axis('off')

            axes[3].plot(loss_history, color='#2196F3', linewidth=2)
            axes[3].set_xlabel('Iteration')
            axes[3].set_ylabel('Loss')
            axes[3].set_title('Convergence', fontsize=11, fontweight='bold')
            axes[3].set_yscale('log')
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xlim(0, n_iterations)

            fig.suptitle(f'Inverse Lithography Optimization — {pattern_name}  '
                         f'(λ={wavelength}nm, NA={NA})',
                         fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.93])

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frame = imageio.imread(buf)
            frames.append(frame)
            plt.close(fig)
            buf.close()

    # Hold final frame longer
    for _ in range(FINAL_FRAME_HOLD_COUNT):
        frames.append(frames[-1])

    gif_path = os.path.join(OUTPUT_DIR, gif_name)
    imageio.mimsave(gif_path, frames, duration=0.12, loop=0)
    print(f"    ✓ Saved: {gif_path}")
    return gif_path


# ---------------------------------------------------------------------------
# 2. Forward diffraction comparison (static)
# ---------------------------------------------------------------------------

def generate_forward_diffraction_image():
    """
    Generate a grid showing forward diffraction for multiple patterns,
    wavelengths, and NA values.
    """
    ensure_output_dir()
    print("  Generating forward diffraction comparison ...")

    patterns = ['square', 'circle', 'lines']
    configs = [
        {'wavelength': 193.0, 'NA': 0.6, 'label': 'DUV 193nm, NA=0.6'},
        {'wavelength': 248.0, 'NA': 0.5, 'label': 'KrF 248nm, NA=0.5'},
        {'wavelength': 13.5, 'NA': 0.33, 'label': 'EUV 13.5nm, NA=0.33'},
    ]
    size = 128

    fig, axes = plt.subplots(len(configs), len(patterns) * 2, figsize=(20, 10))

    for row, cfg in enumerate(configs):
        diffraction = FraunhoferDiffraction(
            wavelength=cfg['wavelength'], NA=cfg['NA'])
        for col, pat in enumerate(patterns):
            mask = create_test_mask(size, pat)
            with torch.no_grad():
                intensity = diffraction(mask).squeeze().cpu().numpy()

            ax_mask = axes[row, col * 2]
            ax_int = axes[row, col * 2 + 1]

            ax_mask.imshow(mask.numpy(), cmap='gray', vmin=0, vmax=1)
            ax_mask.set_title(f'{pat} mask', fontsize=9)
            ax_mask.axis('off')

            ax_int.imshow(intensity, cmap='inferno')
            ax_int.set_title(f'Aerial image', fontsize=9)
            ax_int.axis('off')

        # Row label
        axes[row, 0].set_ylabel(cfg['label'], fontsize=10, fontweight='bold',
                                rotation=90, labelpad=10)

    fig.suptitle('Forward Diffraction — Multiple Wavelengths & NA Settings',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, 'forward_diffraction_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 3. Thermal compensation visualization
# ---------------------------------------------------------------------------

def generate_thermal_compensation_image():
    """
    Show the effect of thermal expansion compensation:
      - target at 80 °C (desired)
      - pattern at process temp (expanded)
      - pattern after cooling WITHOUT compensation (mismatched)
      - pattern after cooling WITH compensation (matching target)
    """
    ensure_output_dir()
    print("  Generating thermal compensation visualization ...")

    size = 128
    target = create_arbitrary_pattern(size, 'cross')
    thermal = ThermalExpansionModel(process_temp=200.0, operating_temp=80.0)
    info = thermal.get_info()

    diffraction = FraunhoferDiffraction(wavelength=193.0, NA=0.6)

    # WITHOUT thermal compensation: optimize directly for target
    opt_no_comp = AdaptiveMaskOptimizer(
        diffraction, (size, size), learning_rate=0.05,
        regularization=0.005, use_scheduler=True)
    res_no_comp = opt_no_comp.optimize(
        target, num_iterations=200, verbose=False, early_stopping_patience=40)
    with torch.no_grad():
        aerial_no_comp = diffraction(res_no_comp['mask']).squeeze()
        cooled_no_comp = thermal.apply_thermal_contraction(aerial_no_comp)

    # WITH thermal compensation
    opt_comp = ThermalAwareMaskOptimizer(
        diffraction, (size, size), thermal,
        learning_rate=0.05, regularization=0.005, use_scheduler=True)
    res_comp = opt_comp.optimize(
        target, num_iterations=200, verbose=False, early_stopping_patience=40)
    with torch.no_grad():
        aerial_comp = diffraction(res_comp['mask']).squeeze()
        cooled_comp = thermal.apply_thermal_contraction(aerial_comp)

    # Difference maps
    diff_no_comp = torch.abs(cooled_no_comp - target)
    diff_comp = torch.abs(cooled_comp - target)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    titles_top = [
        f'Target (at {info["operating_temp_C"]}°C)',
        f'Aerial image at {info["process_temp_C"]}°C\n(no compensation)',
        f'After cooling to {info["operating_temp_C"]}°C\n(no compensation)',
        'Error map (no comp.)'
    ]
    imgs_top = [target.numpy(), aerial_no_comp.numpy(),
                cooled_no_comp.numpy(), diff_no_comp.numpy()]

    titles_bot = [
        f'Target (at {info["operating_temp_C"]}°C)',
        f'Aerial image at {info["process_temp_C"]}°C\n(WITH compensation)',
        f'After cooling to {info["operating_temp_C"]}°C\n(WITH compensation)',
        'Error map (WITH comp.)'
    ]
    imgs_bot = [target.numpy(), aerial_comp.numpy(),
                cooled_comp.numpy(), diff_comp.numpy()]

    for j in range(4):
        cmap = 'RdYlGn_r' if j == 3 else 'inferno'
        axes[0, j].imshow(imgs_top[j], cmap=cmap, vmin=0,
                          vmax=1 if j < 3 else None)
        axes[0, j].set_title(titles_top[j], fontsize=10, fontweight='bold')
        axes[0, j].axis('off')

        axes[1, j].imshow(imgs_bot[j], cmap=cmap, vmin=0,
                          vmax=1 if j < 3 else None)
        axes[1, j].set_title(titles_bot[j], fontsize=10, fontweight='bold')
        axes[1, j].axis('off')

    mse_no = torch.mean(diff_no_comp ** 2).item()
    mse_comp = torch.mean(diff_comp ** 2).item()
    fig.suptitle(
        f'Thermal Compensation  —  Si wafer {info["process_temp_C"]}°C → '
        f'{info["operating_temp_C"]}°C  '
        f'(contraction {info["contraction_ppm"]:.1f} ppm)\n'
        f'MSE without comp: {mse_no:.6f}  |  MSE with comp: {mse_comp:.6f}',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(OUTPUT_DIR, 'thermal_compensation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Arbitrary shapes gallery
# ---------------------------------------------------------------------------

def generate_arbitrary_shapes_gallery():
    """
    Demonstrate that the engine handles arbitrary input shapes and dimensions.
    Shows inverse optimization for multiple custom shapes.
    """
    ensure_output_dir()
    print("  Generating arbitrary shapes gallery ...")

    shapes = ['L_shape', 'T_shape', 'ring', 'cross', 'diamond', 'zigzag']
    size = 128
    diffraction = FraunhoferDiffraction(wavelength=193.0, NA=0.6)

    fig, axes = plt.subplots(len(shapes), 3, figsize=(12, len(shapes) * 3.2))

    for i, name in enumerate(shapes):
        target = create_arbitrary_pattern(size, name)
        opt = AdaptiveMaskOptimizer(
            diffraction, (size, size), learning_rate=0.05,
            regularization=0.005, use_scheduler=True)
        res = opt.optimize(target, num_iterations=200, verbose=False,
                           early_stopping_patience=40)
        with torch.no_grad():
            pred = diffraction(res['mask']).squeeze().cpu().numpy()

        axes[i, 0].imshow(target.numpy(), cmap='inferno', vmin=0, vmax=1)
        axes[i, 0].set_title('Target', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(res['mask'].squeeze().cpu().numpy(),
                          cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Optimized Mask', fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='inferno', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Aerial Image (loss={res["final_loss"]:.4f})',
                             fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')

        axes[i, 0].set_ylabel(name, fontsize=11, fontweight='bold',
                              rotation=0, labelpad=60, va='center')

    fig.suptitle('Inverse Lithography — Arbitrary Input Shapes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'arbitrary_shapes_gallery.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 5. Multi-size demonstration
# ---------------------------------------------------------------------------

def generate_multi_size_image():
    """Show that the engine works with arbitrary dimensions."""
    ensure_output_dir()
    print("  Generating multi-size demonstration ...")

    sizes = [64, 128, 256]
    diffraction = FraunhoferDiffraction(wavelength=193.0, NA=0.6)

    fig, axes = plt.subplots(len(sizes), 3, figsize=(12, len(sizes) * 3.5))

    for i, sz in enumerate(sizes):
        target = create_arbitrary_pattern(sz, 'cross')
        opt = AdaptiveMaskOptimizer(
            diffraction, (sz, sz), learning_rate=0.05,
            regularization=0.005, use_scheduler=True)
        res = opt.optimize(target, num_iterations=200, verbose=False,
                           early_stopping_patience=40)
        with torch.no_grad():
            pred = diffraction(res['mask']).squeeze().cpu().numpy()

        axes[i, 0].imshow(target.numpy(), cmap='inferno', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Target ({sz}×{sz})', fontsize=10,
                             fontweight='bold')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(res['mask'].squeeze().cpu().numpy(),
                          cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Optimized Mask', fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='inferno', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Result (loss={res["final_loss"]:.4f})',
                             fontsize=10, fontweight='bold')
        axes[i, 2].axis('off')

    fig.suptitle('Arbitrary Dimensions — 64×64, 128×128, 256×256',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, 'multi_size_demo.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    ✓ Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 6. Thermal-aware optimization animation
# ---------------------------------------------------------------------------

def generate_thermal_optimization_animation(size=128, n_iterations=200):
    """
    GIF showing thermal-aware optimization: the mask converges to produce
    a pattern that, after cooling, matches the target.
    """
    ensure_output_dir()
    print("  Generating thermal-aware optimization animation ...")

    target = create_arbitrary_pattern(size, 'diamond')
    thermal = ThermalExpansionModel(process_temp=200.0, operating_temp=80.0)
    diffraction = FraunhoferDiffraction(wavelength=193.0, NA=0.6)
    info = thermal.get_info()

    compensated_target = thermal.apply_thermal_precompensation(target)

    import torch.nn as nn
    import torch.optim as optim

    mask_param = nn.Parameter(torch.rand(size, size))
    optimizer = optim.Adam([mask_param], lr=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20)

    frames = []
    loss_history = []
    capture_every = max(1, n_iterations // 60)

    for it in range(n_iterations):
        optimizer.zero_grad()
        mask_c = torch.sigmoid(mask_param)
        predicted = diffraction(mask_c)
        target_4d = compensated_target.unsqueeze(0).unsqueeze(0)
        loss = torch.nn.functional.mse_loss(predicted, target_4d)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())
        loss_history.append(loss.item())

        if it % capture_every == 0 or it == n_iterations - 1:
            with torch.no_grad():
                pred_np = predicted.squeeze().cpu().numpy()
                mask_np = mask_c.squeeze().cpu().numpy()
                cooled = thermal.apply_thermal_contraction(
                    predicted.squeeze()).cpu().numpy()

            fig, axes = plt.subplots(1, 5, figsize=(22, 4))

            axes[0].imshow(target.numpy(), cmap='inferno', vmin=0, vmax=1)
            axes[0].set_title(f'Target ({info["operating_temp_C"]}°C)',
                              fontsize=10, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Mask (iter {it})', fontsize=10,
                              fontweight='bold')
            axes[1].axis('off')

            axes[2].imshow(pred_np, cmap='inferno', vmin=0, vmax=1)
            axes[2].set_title(f'Aerial @ {info["process_temp_C"]}°C',
                              fontsize=10, fontweight='bold')
            axes[2].axis('off')

            axes[3].imshow(cooled, cmap='inferno', vmin=0, vmax=1)
            axes[3].set_title(f'Cooled to {info["operating_temp_C"]}°C',
                              fontsize=10, fontweight='bold')
            axes[3].axis('off')

            axes[4].plot(loss_history, color='#FF5722', linewidth=2)
            axes[4].set_xlabel('Iteration')
            axes[4].set_ylabel('Loss')
            axes[4].set_title('Convergence', fontsize=10, fontweight='bold')
            axes[4].set_yscale('log')
            axes[4].grid(True, alpha=0.3)
            axes[4].set_xlim(0, n_iterations)

            fig.suptitle(
                f'Thermal-Aware ILT — '
                f'{info["process_temp_C"]}°C → {info["operating_temp_C"]}°C  '
                f'(contraction {info["contraction_ppm"]:.1f} ppm)',
                fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.92])

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close(fig)
            buf.close()

    for _ in range(FINAL_FRAME_HOLD_COUNT):
        frames.append(frames[-1])

    gif_path = os.path.join(OUTPUT_DIR, 'thermal_optimization.gif')
    imageio.mimsave(gif_path, frames, duration=0.12, loop=0)
    print(f"    ✓ Saved: {gif_path}")
    return gif_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Generating Visualizations & Animations")
    print("=" * 60)

    # 1. Optimization animations for various patterns
    for pat in ['cross', 'ring', 'diamond']:
        generate_optimization_animation(pat, size=128, n_iterations=200)

    # 2. Forward diffraction comparison (multiple wavelengths / NA)
    generate_forward_diffraction_image()

    # 3. Thermal compensation
    generate_thermal_compensation_image()

    # 4. Arbitrary shapes gallery
    generate_arbitrary_shapes_gallery()

    # 5. Multi-size demo
    generate_multi_size_image()

    # 6. Thermal-aware optimization animation
    generate_thermal_optimization_animation()

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
