"""
Exercise 5: Pendulum Phase Space and Vector Field
Visualizes phase space trajectories and vector field for nonlinear pendulum
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from utils_and_functions import omega, Pendulum_vector_field

# Calculate trajectories for various initial angular velocities
t_vals, omega_vals, theta_vals, omega0 = omega(20, -10)

# Plot phase space trajectories
for i in range(len(omega_vals)):
    plt.scatter(theta_vals[i], omega_vals[i], s=0.5, color='k')

plt.xlabel('$\\theta$ $^0$')
plt.ylabel('$\\omega$ [m/s]')
plt.ylim(-3, 3)

# Add vector field
x_min, x_max = -3, 3
y_min, y_max = -3, 3
x_grid = np.linspace(x_min, x_max, 20)
v_grid = np.linspace(y_min, y_max, 20)
X_grid, V_grid = np.meshgrid(x_grid, v_grid)

U_grid, W_grid = Pendulum_vector_field(X_grid, V_grid)

plt.quiver(X_grid, V_grid, U_grid, W_grid, color='r', alpha=0.3, scale=20, width=0.002)

plt.savefig('Phaseplot (reg pend).png', dpi=700)

plt.show()
