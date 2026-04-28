"""
Exercise 2: Energy Conservation in Mass-on-Spring System
Tests if RK4 conserves energy over extended time periods
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from utils_and_functions import RK4, Mass_on_spring

# Initial conditions
x0 = 10
v0 = 0
t0 = 0

# Extended time for 1000 oscillations
t_end_new = np.abs(1000*(np.pi))
h = 0.1

num_points = int(round(t_end_new / h)) + 1

t_new = np.zeros(num_points)
x_new = np.zeros(num_points)
v_new = np.zeros(num_points)

# Create benchmark time array
t_bench_new = np.linspace(t0, t_end_new, num_points)
x_bench_new = x0*np.cos(t_bench_new)
v_bench_new = (-x0)*np.sin(t_bench_new)

t_new[0] = t0
x_new[0] = x0
v_new[0] = v0

# Perform RK4 integration
for i in range(num_points - 1):
    x_v_current = np.array([x_new[i], v_new[i]])
    x_v_next = RK4(t_new[i], x_v_current, Mass_on_spring, h)
    x_new[i+1] = x_v_next[0]
    v_new[i+1] = x_v_next[1]
    t_new[i+1] = t_new[i] + h

# Calculate energies
Ep_new = 0.5 * (x_new**2)
Ek_new = 0.5 * (v_new**2)

Ep_bench_new = 0.5 * (x_bench_new**2)
Ek_bench_new = 0.5 * (v_bench_new**2)

# Calculate errors
errorEp = np.abs(Ep_bench_new - Ep_new)
errorEk = np.abs(Ek_bench_new - Ek_new)

# Plot results
plt.rcParams['font.size'] = 16
fig, axes = plt.subplots(4, 1, figsize=(7, 14))

ax1 = axes[0]
ax1.plot(t_new, Ep_new, color='red', marker='.', markersize=3, linestyle='', label='Potential Energy (RK4)')
ax1.plot(t_bench_new, Ep_bench_new, color='pink', linestyle='-', linewidth=1.5, alpha=0.8, label='Potential Energy (Benchmark)')
ax1.set_xlabel('$t$ [s]')
ax1.set_ylabel('$E$ [J]')

ax2 = ax1.twinx()
ax2.plot(t_new, Ek_new, 'k', marker='.', markersize=3, linestyle='', label='Kinetic Energy (RK4)')
ax2.plot(t_bench_new, Ek_bench_new, color='gray', linestyle='-', linewidth=1.5, alpha=0.8, label='Kinetic Energy (Benchmark)')
ax2.set_xlabel('$t$ [s]', color='k')
ax2.tick_params(axis='y', labelcolor='k')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.set_title('Projectile Motion: Kinetic and Potential Energy')

plt.tight_layout()

ax_total_energy = axes[1]
ax_total_energy.plot(t_new, Ep_new + Ek_new, '-', label='Total Energy (RK4)')
ax_total_energy.set_ylabel('E(t) [J]')
ax_total_energy.set_xlabel('Time [s]')
ax_total_energy.ticklabel_format(useOffset=False, style='plain')
ax_total_energy.set_title('Total Energy over Time')

ax_err_y = axes[2]
ax_err_y.plot(t_new, errorEp, 'b', marker='.', linestyle='', markersize='0.5', label="Error in Ep")
ax_err_y.set_ylabel('Error [J]')
ax_err_y.set_xlabel('Time [s]')
ax_err_y.legend()
ax_err_y.set_title('Error in RK4 Approximation for Potential Energy')

ax_err_v = axes[3]
ax_err_v.plot(t_new, errorEk, 'r', marker='.', linestyle='', markersize='0.5', label="Error in Ek")
ax_err_v.set_ylabel('Error [J]')
ax_err_v.set_xlabel('Time [s]')
ax_err_v.legend()
ax_err_v.set_title('Error in RK4 Approximation for Kinetic energy')

plt.tight_layout()
plt.show()

print('Conservation of energy by E(t)/E(0) = ', (Ep_new + Ek_new)/(Ep_new[0] + Ek_new[0]))
