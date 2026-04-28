"""
Bifurcation Diagram - Route to Chaos
Shows how system behavior changes with driving amplitude
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from utils_and_functions import RK4, DrivenDampedPendulum

omega_d = 0.67
beta_val = 0.5

# Sweep amplitude from 1 to 1.65 (balanced for speed and quality)
A_vals = np.linspace(1, 1.65, 1000)

steps = 300
skip = 100
N_cycles = 1000
dt = 2 * np.pi / (omega_d * steps)

nsteps = N_cycles * steps
time_arr = np.arange(0, nsteps * dt, dt)

# Arrays to hold plot data
A_plot = []
omega_plot = []

# Loop over amplitude values
for idx, A in enumerate(A_vals):
    print(f"Progress: {idx+1}/{len(A_vals)}", end='\r')
    theta = np.zeros(nsteps)
    omega = np.zeros(nsteps)

    theta[0] = 0.0
    omega[0] = 0.0

    # Integrate system
    for j in range(nsteps-1):
        x_v_current = np.array([theta[j], omega[j]])
        x_v_next = RK4(time_arr[j], x_v_current, lambda current_t, current_theta_w: DrivenDampedPendulum(current_t, current_theta_w, beta_val, A), dt)
        theta[j+1] = x_v_next[0]
        omega[j+1] = x_v_next[1]

    if theta[j+1] > np.pi:
        theta[j+1] -= 2*np.pi
    if theta[j+1] < -np.pi:
        theta[j+1] += 2*np.pi

    # Extract Poincaré section
    offset = steps // 4
    omega_p = omega[skip * steps + offset :: steps]

    # Store results
    for om_p in omega_p:
        A_plot.append(A)
        omega_plot.append(om_p)

# Plot bifurcation diagram
plt.figure(figsize=(12, 7))
plt.scatter(A_plot, omega_plot, s=0.05, color='black', alpha=0.3)

plt.xlabel('Amplitude (A)', fontsize=12)
plt.ylabel('$\\omega$ [rad/s]', fontsize=12)
plt.savefig('Bifurcation_diagram.png', dpi=700)
print("\n✓ Bifurcation diagram complete! Saved as 'Bifurcation_diagram.png'")
plt.show()
