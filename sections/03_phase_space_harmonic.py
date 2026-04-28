"""
Exercise 3: Phase Space of Harmonic Oscillator
Visualizes trajectories in phase space (position vs velocity)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from utils_and_functions import RK4, Mass_on_spring, Mass_on_spring_vector_field

N = np.arange(0, 10, 1)

x0_new = 0
v0_list = N*0.1

t0 = 0
tend = np.abs(10)

h_new = 0.01

n = int((tend - t0) / h_new) + 1

plt.figure(figsize=(8, 8))

# Create vector field
x_min, x_max = -1, 1
v_min, v_max = -1, 1
x_grid = np.linspace(x_min, x_max, 20)
v_grid = np.linspace(v_min, v_max, 20)
X_grid, V_grid = np.meshgrid(x_grid, v_grid)

U_grid, W_grid = Mass_on_spring_vector_field(X_grid, V_grid)

plt.quiver(X_grid, V_grid, U_grid, W_grid, color='r', alpha=0.5, scale=20, width=0.002)

# Plot trajectories for different initial velocities
for v0 in v0_list:
    x = np.zeros(n)
    v = np.zeros(n)
    t = np.zeros(n)

    x[0] = x0_new
    v[0] = v0
    t[0] = t0

    for i in range(n-1):
        x_v_current = np.array([x[i], v[i]])
        x_v_next = RK4(t[i], x_v_current, Mass_on_spring, h_new)
        x[i+1] = x_v_next[0]
        v[i+1] = x_v_next[1]
        t[i+1] = t[i] + h_new

    plt.plot(x, v, 'k')

plt.title('Phase Space of a Harmonic Oscillator')
plt.xlabel('$Position$ (m)')
plt.ylabel('$Velocity$ (ms$^-$$^1$)')

plt.tight_layout()
plt.show()
