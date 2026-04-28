"""
Exercise 1: Benchmarking RK4 solver against mass-on-spring analytic solution
Compares RK4 numerical solution with known analytic solution
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure utils_and_functions can be imported
sys.path.insert(0, os.path.dirname(__file__))
from utils_and_functions import RK4, Mass_on_spring

# --- 1. Set Parameters ---
x0 = 10.0
v0 = 0.0
t0 = 0.0
tend = 20 * np.pi  # Cleaned up np.abs() since 20*pi is already positive
h = 0.01

n = int((tend - t0) / h)

# Initialize arrays
t = np.zeros(n)
x = np.zeros(n)
v = np.zeros(n)

# Initial conditions
t[0], x[0], v[0] = t0, x0, v0

# --- 2. Perform RK4 Integration ---
for i in range(n - 1):
    x_v_current = np.array([x[i], v[i]])
    x_v_next = RK4(t[i], x_v_current, Mass_on_spring, h)
    
    x[i+1] = x_v_next[0]
    v[i+1] = x_v_next[1]
    t[i+1] = t[i] + h

# --- 3. Benchmark Analytical Solution ---
# Using the exact `t` array from the loop guarantees matching dimensions for error calculation
x_bench = x0 * np.cos(t) 
v_bench = -x0 * np.sin(t) 

# --- 4. Calculate Errors ---
error_x = np.abs(x_bench - x)
error_v = np.abs(v_bench - v)

# --- 5. Plotting ---
plt.rcParams['font.size'] = 14

# PLOT 1: Position and Velocity (RK4 vs Benchmark)
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(t, x, color='red', marker='.', markersize=5, linestyle='', label='Position (RK4)')
ax1.plot(t, x_bench, color='pink', linestyle='-', linewidth=2, alpha=0.8, label='Position (Bench)')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Position [m]', color='k')
ax1.tick_params(axis='y', labelcolor='k')
ax1.set_ylim(-14, 14)

# Create a twin axis for velocity
ax2 = ax1.twinx()
ax2.plot(t, v, color='black', marker='.', markersize=5, linestyle='', label='Velocity (RK4)')
ax2.plot(t, v_bench, color='gray', linestyle='-', linewidth=2, alpha=0.8, label='Velocity (Bench)')
ax2.set_ylabel('Velocity [m/s]', color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(-14, 14)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=10, ncols = 4)

fig1.tight_layout()
fig1.savefig('Benchmarking_Rk4.png', dpi=1400)

# PLOT 2: Errors
fig2, ax3 = plt.subplots(figsize=(10, 5))

ax3.plot(t, error_x, 'b', label="Absolute Error in Position (x)")
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Error')

fig2.tight_layout()
fig2.savefig('Benchmarking_Rk4_Error.png', dpi=1400)

# Display both figures
plt.show()