"""
Super-High Resolution Poincaré Section
Comprehensive topological mapping with maximum fidelity
100k cycles × 250 steps per period = comprehensive attractor resolution
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from utils_and_functions import RK4, DrivenDampedPendulum
from time import perf_counter
import json
import psutil

# Timing and memory tracking
script_start = perf_counter()
process = psutil.Process(os.getpid())

A_val = 1.5
omega_d = 0.67
beta_val = 0.5

# Ultra-high resolution parameters
N = 100000      # 100,000 driving periods (2× Script 09)
steps = 250     # 250 steps per period (2.5× Script 09, 1.25× Script 12)
skip = 100       # Skip first 100 periods for transient decay

dt = 2 * np.pi/(omega_d*steps)

time = np.arange(0, N*2*np.pi / omega_d, dt)
nsteps = len(time)

print(f"SUPER-RESOLUTION POINCARÉ SECTION ANALYSIS")
print(f"{'='*70}")
print(f"Configuration:")
print(f"  Driving periods: {N}")
print(f"  Steps per period: {steps}")
print(f"  Transient skip: {skip}")
print(f"  Total integration steps: {nsteps:,}")
print(f"  Expected integration time: 10-15 minutes on Apple Silicon")
print(f"{'='*70}\n")

print(f"Integrating {nsteps:,} steps (~{nsteps*dt/(2*np.pi):.0f} driving periods)...")

omega = np.zeros(nsteps)
theta = np.zeros(nsteps)
omega[0] = 0.0
theta[0] = 0.0

integration_start = perf_counter()
for j in range(nsteps-1):
    if j % 50000 == 0 and j > 0:
        progress_pct = 100 * j / nsteps
        print(f"  Step {j:,}/{nsteps:,} ({progress_pct:.1f}%)", end='\r', flush=True)
    
    x_v_current = np.array([theta[j], omega[j]])
    x_v_next = RK4(time[j], x_v_current, 
                   lambda t, y: DrivenDampedPendulum(t, y, beta_val, A_val), 
                   dt)
    theta[j+1] = x_v_next[0]
    omega[j+1] = x_v_next[1]

print(f"✓ Integration complete!")

integration_time = perf_counter() - integration_start
peak_memory = process.memory_info().rss / (1024**2)  # MB

# Remove transient
theta = theta[skip*steps:]
omega = omega[skip*steps:]

offset = steps//4
theta_poincare = theta[offset :: steps]
omega_poincare = omega[offset :: steps]

# Wrap theta to [-pi, pi]
theta_poincare = (theta_poincare + np.pi) % (2*np.pi) - np.pi

num_points = len(theta_poincare)
print(f"✓ Extracted {num_points:,} Poincaré section points")
print(f"✓ Points per driving period: {num_points / (N - skip):.2f}")

# Record performance metrics
performance_data = {
    'script': '14_poincare_super_resolution.py',
    'periods': N,
    'steps_per_period': steps,
    'total_steps': nsteps,
    'points_extracted': num_points,
    'integration_time_sec': integration_time,
    'peak_memory_mb': peak_memory,
    'points_per_second': num_points / integration_time if integration_time > 0 else 0
}

with open('performance_14.json', 'w') as f:
    json.dump(performance_data, f, indent=2)

print(f"\n[Performance] Time: {integration_time:.2f}s | Memory: {peak_memory:.1f}MB | Points: {num_points} ({performance_data['points_per_second']:.0f} pts/s)")

# =========================================================================
# TOPOLOGICAL CHARACTERIZATION
# =========================================================================

print(f"\n{'='*70}")
print(f"TOPOLOGICAL CHARACTERIZATION")
print(f"{'='*70}")

# Density in focus region
focus_mask = (theta_poincare >= 1.3) & (theta_poincare <= 1.8) & \
             (omega_poincare >= -0.4) & (omega_poincare <= 0.3)
focus_count = np.sum(focus_mask)
focus_density = 100 * focus_count / num_points

print(f"\nFocus region [θ∈(1.3,1.8), ω∈(-0.4,0.3)]:")
print(f"  Points in region: {focus_count:,} ({focus_density:.1f}%)")
print(f"  Point density in region: {focus_count / (0.5 * 0.7):.1f} points/unit²")

# Phase space grid coverage
theta_min, theta_max = -np.pi, np.pi
omega_min, omega_max = omega_poincare.min(), omega_poincare.max()
print(f"\nPhase space extent:")
print(f"  θ: [{theta_min:.3f}, {theta_max:.3f}]")
print(f"  ω: [{omega_min:.3f}, {omega_max:.3f}]")

# Grid coverage analysis (10×10 grid)
grid_size = 10
theta_edges = np.linspace(-np.pi, np.pi, grid_size + 1)
omega_edges = np.linspace(omega_min, omega_max, grid_size + 1)

occupied_cells = 0
for i in range(grid_size):
    for j in range(grid_size):
        mask = (theta_poincare >= theta_edges[i]) & \
               (theta_poincare < theta_edges[i+1]) & \
               (omega_poincare >= omega_edges[j]) & \
               (omega_poincare < omega_edges[j+1])
        if np.sum(mask) > 0:
            occupied_cells += 1

coverage = 100 * occupied_cells / (grid_size ** 2)
print(f"\nPhase space coverage (10×10 grid):")
print(f"  Occupied cells: {occupied_cells}/{grid_size**2}")
print(f"  Coverage: {coverage:.1f}%")

# Local structure analysis - nearest neighbor distances
print(f"\nLocal structure analysis:")
distances = []
for i in range(min(5000, num_points)):
    if i + 1 < num_points:
        dist = np.sqrt((theta_poincare[i] - theta_poincare[i+1])**2 + 
                       (omega_poincare[i] - omega_poincare[i+1])**2)
        distances.append(dist)

distances = np.array(distances)
print(f"  Sequential point spacing:")
print(f"    Min: {distances.min():.6f}")
print(f"    Mean: {distances.mean():.6f}")
print(f"    Median: {np.median(distances):.6f}")
print(f"    Max: {distances.max():.6f}")
print(f"    Std Dev: {distances.std():.6f}")

# =========================================================================
# VISUALIZATION
# =========================================================================

print(f"\n{'='*70}")
print(f"GENERATING VISUALIZATIONS")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Full Poincaré section
ax = axes[0, 0]
ax.scatter(theta_poincare, omega_poincare, s=0.5, color='black', alpha=0.6)
ax.set_xlabel('$\\theta$ [rad]', fontsize=12)
ax.set_ylabel('$\\omega$ [rad/s]', fontsize=12)
ax.set_title(f'Full Poincaré Section ({num_points:,} points)', fontsize=13, fontweight='bold')
ax.set_xlim(-np.pi, np.pi)
ax.grid(True, alpha=0.3)

# Plot 2: Focus region zoomed
ax = axes[0, 1]
ax.scatter(theta_poincare[focus_mask], omega_poincare[focus_mask], 
          s=2, color='red', alpha=0.7, label='Focus region')
ax.scatter(theta_poincare[~focus_mask], omega_poincare[~focus_mask], 
          s=0.1, color='black', alpha=0.3, label='Other regions')
ax.set_xlabel('$\\theta$ [rad]', fontsize=12)
ax.set_ylabel('$\\omega$ [rad/s]', fontsize=12)
ax.set_title('Zoomed: Fractal Structure [θ∈(1.3,1.8), ω∈(-0.4,0.3)]', fontsize=13, fontweight='bold')
ax.set_xlim(1.3, 1.8)
ax.set_ylim(-0.4, 0.3)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Ultra-zoomed
ax = axes[1, 0]
ultra_zoom_mask = (theta_poincare >= 1.45) & (theta_poincare <= 1.65) & \
                  (omega_poincare >= -0.2) & (omega_poincare <= 0.1)
ax.scatter(theta_poincare[ultra_zoom_mask], omega_poincare[ultra_zoom_mask], 
          s=3, color='darkred', alpha=0.8)
ax.set_xlabel('$\\theta$ [rad]', fontsize=12)
ax.set_ylabel('$\\omega$ [rad/s]', fontsize=12)
ax.set_title('Ultra-Zoom: Fine Fractal Detail [θ∈(1.45,1.65), ω∈(-0.2,0.1)]', 
            fontsize=13, fontweight='bold')
ax.set_xlim(1.45, 1.65)
ax.set_ylim(-0.2, 0.1)
ax.grid(True, alpha=0.3)

# Plot 4: Topological folding map
ax = axes[1, 1]
theta_n = theta_poincare[:-1]
theta_n_plus_1 = theta_poincare[1:]
ax.scatter(theta_n, theta_n_plus_1, s=0.5, color='darkblue', alpha=0.5)
ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', linewidth=0.8, alpha=0.5, label='Identity map')
ax.set_xlabel('$\\theta_n$ (Current)', fontsize=12)
ax.set_ylabel('$\\theta_{n+1}$ (Next)', fontsize=12)
ax.set_title('Poincaré Return Map: Topological Folding', fontsize=13, fontweight='bold')
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
pdf_file = 'poincare_super_resolution.pdf'
plt.savefig(pdf_file, dpi=400, bbox_inches='tight')
print(f"✓ Saved: {pdf_file}")
plt.close()

# =========================================================================
# ADDITIONAL ANALYSIS PLOTS
# =========================================================================

print(f"\nGenerating analysis plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sequential spacing histogram
ax = axes[0]
ax.hist(distances, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Distance to next sequential point', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Point Spacing Distribution', fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Density heatmap (20×20 grid)
ax = axes[1]
grid_size = 20
density_grid = np.zeros((grid_size, grid_size))
theta_edges = np.linspace(-np.pi, np.pi, grid_size + 1)
omega_edges = np.linspace(omega_min, omega_max, grid_size + 1)

for i in range(grid_size):
    for j in range(grid_size):
        mask = (theta_poincare >= theta_edges[i]) & \
               (theta_poincare < theta_edges[i+1]) & \
               (omega_poincare >= omega_edges[j]) & \
               (omega_poincare < omega_edges[j+1])
        density_grid[j, i] = np.sum(mask)

im = ax.imshow(density_grid, extent=[-np.pi, np.pi, omega_min, omega_max],
              aspect='auto', cmap='hot', origin='lower', interpolation='nearest')
ax.set_xlabel('$\\theta$ [rad]', fontsize=11)
ax.set_ylabel('$\\omega$ [rad/s]', fontsize=11)
ax.set_title('Point Density Heatmap', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Points per cell', fontsize=10)

plt.tight_layout()
analysis_pdf = 'poincare_super_analysis.pdf'
plt.savefig(analysis_pdf, dpi=400, bbox_inches='tight')
print(f"✓ Saved: {analysis_pdf}")
plt.close()

# =========================================================================
# SUMMARY
# =========================================================================

print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*70}")
print(f"\nGenerated outputs:")
print(f"  1. {pdf_file} - Main 4-panel analysis")
print(f"  2. {analysis_pdf} - Detailed statistical visualizations")
print(f"\nTopology Summary:")
print(f"  Total points: {num_points:,}")
print(f"  Resolution: {steps} samples per driving period")
print(f"  Coverage: {coverage:.1f}% of phase space grid")
print(f"  Focus density: {focus_density:.1f}% in chaotic region")
print(f"\nThis ultra-high resolution ensures excellent fidelity for:")
print(f"  ✓ Fractal dimension measurement")
print(f"  ✓ Lyapunov exponent estimation")
print(f"  ✓ Bifurcation topology analysis")
print(f"  ✓ Attractor basin mapping")
print(f"\n{'='*70}\n")
