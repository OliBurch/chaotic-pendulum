"""
Exercise 8: Driven Damped Pendulum - Phase Portraits and Animation
Visualizes phase space behavior and time-dependent vector fields
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from utils_and_functions import driven_damped_pendulum, DrivenDampedODE_vector_field

As = [0.9, 1.07, 1.47, 1.5]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for idx, A in enumerate(As):
    t_vals, omega_vals, theta_vals, omega0 = driven_damped_pendulum(1, 0, A)
    t_vals = np.array(t_vals).flatten()
    omega_vals = np.array(omega_vals).flatten()
    theta_vals = np.array(theta_vals).flatten()
    mask = t_vals > 100
    if np.any(mask):
        axs[idx].scatter(theta_vals[mask], omega_vals[mask], s=0.5, color='k')

    axs[idx].set_title(f'$A = {A}$')
    axs[idx].set_xlabel('$\\theta$ [rad]')
    axs[idx].set_ylabel('$\\omega$ [m/s]')

plt.xlabel('$\\theta$ $^0$')
plt.ylabel('$\\omega$ [m/s]')
plt.ylim(-3, 3)

plt.savefig('Phaseplot_driven_damped.png', dpi=700)
plt.tight_layout()
plt.show()

# Create animated vector field visualization
x_min, x_max = -3, 3
y_min, y_max = -3, 3
x_grid = np.linspace(x_min, x_max, 20)
v_grid = np.linspace(y_min, y_max, 20)
X_grid, V_grid = np.meshgrid(x_grid, v_grid)

parameters = [(0.25, 0.9), (0.25, 1.5), (1.0, 0.9), (1.0, 1.5)]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

quivers = []

for idx, (beta, A) in enumerate(parameters):
    ax = axs[idx]

    U, W = DrivenDampedODE_vector_field(X_grid, V_grid, beta, A, 0)

    Q = ax.quiver(X_grid, V_grid, U, W, color='r', alpha=0.3, scale=20, width=0.002)
    quivers.append(Q)

    ax.set_title(f"$\\beta={beta}$, $A={A}$ | Time: 0.00 s")
    ax.set_xlabel('$\\theta$ [rad]')
    ax.set_ylabel('$\\omega$ [m/s]')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

plt.tight_layout()

def update_quiver(t):
    for idx, (beta, A) in enumerate(parameters):
        ax = axs[idx]
        Q = quivers[idx]

        U_new, W_new = DrivenDampedODE_vector_field(X_grid, V_grid, beta, A, t)

        Q.set_UVC(U_new, W_new)

        ax.set_title(f"$\\beta={beta}$, $A={A}$ | Time: {t:.2f} s")

    return quivers

t_frames = np.linspace(0, 20, 50)
anim = animation.FuncAnimation(fig, update_quiver, frames=t_frames, interval=100, blit=False)

plt.rcParams['animation.embed_limit'] = 500.0

plt.close(fig)
anim.save('phase_portraits.gif', writer='pillow', fps=10)
