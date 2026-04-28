"""
Exercise 7: Damped Pendulum Phase Plots (Fine Range)
Zoomed-in phase space analysis with damping coefficients (1.5, 1.75, 2.25, 2.5)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from utils_and_functions import damped_pendulum, DampedODE_vector_field

betas = [1.5, 1.75, 2.25, 2.5]
theta_start = 90

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for idx, beta in enumerate(betas):
    t_vals, omega_vals, theta_vals = damped_pendulum(theta_start, beta)

    axs[idx].scatter(theta_vals, omega_vals, s=0.5, color='k')

    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    x_grid = np.linspace(x_min, x_max, 20)
    v_grid = np.linspace(y_min, y_max, 20)
    X_grid, V_grid = np.meshgrid(x_grid, v_grid)

    U_grid, W_grid = DampedODE_vector_field(X_grid, V_grid, beta)

    axs[idx].quiver(X_grid, V_grid, U_grid, W_grid, color='r', alpha=0.3, scale=20, width=0.002)

    axs[idx].set_title(f'Phase Plot ($\\beta = {beta}$)')
    axs[idx].set_xlabel('$\\theta$ [rad]')
    axs[idx].set_ylabel('$\\omega$ [rad/s]')
    axs[idx].set_xlim(x_min, x_max)
    axs[idx].set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('Phaseplot_damped_fine.pdf', dpi=700)
plt.show()
