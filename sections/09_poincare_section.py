"""
Poincaré Section Analysis - Chaotic Attractors
Shows stroboscopic sampling at driving frequency
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
theta0 = 0
omega0 = 0.0
t0 = 0.0

A_val = 1.5
omega_d = 0.67
beta_val = 0.5

N = 50000
steps = 100
skip = 100

dt = 2 * np.pi/(omega_d*steps)

# Renamed 'time' to 't_array' to prevent overriding the time module
t_array = np.arange(t0, N*2*np.pi / omega_d, dt)
nsteps = len(t_array)

omega = np.zeros(nsteps)
theta = np.zeros(nsteps)
omega[0] = omega0

theta[0] = theta0
omega_p = []
theta_p = []

integration_start = perf_counter()
for j in range(nsteps-1):
    x_v_current = np.array([theta[j], omega[j]])
    # Make sure to use t_array[j] here now!
    x_v_next = RK4(t_array[j], x_v_current, lambda current_t, current_theta_w: DrivenDampedPendulum(current_t, current_theta_w, beta_val, A_val), dt)
    theta[j+1] = x_v_next[0]
    omega[j+1] = x_v_next[1]

# Fixed timing calculation using perf_counter()
integration_time = perf_counter() - integration_start
peak_memory = process.memory_info().rss / (1024**2)  # MB

offset = steps//4
start_idx = skip * steps + offset

theta_p = theta[start_idx :: steps]
omega_p = omega[start_idx :: steps]

theta_p_wrapped = (theta_p + np.pi) % (2 * np.pi) - np.pi

# Record performance metrics
performance_data = {
    'script': '09_poincare_section.py',
    'periods': N,
    'steps_per_period': steps,
    'total_steps': nsteps,
    'points_extracted': len(theta_p),
    'integration_time_sec': integration_time,
    'peak_memory_mb': peak_memory,
    'points_per_second': len(theta_p) / integration_time if integration_time > 0 else 0
}

with open('performance_09.json', 'w') as f:
    json.dump(performance_data, f, indent=2)

print(f"\n[Performance] Time: {integration_time:.2f}s | Memory: {peak_memory:.1f}MB | Points: {len(theta_p)} ({performance_data['points_per_second']:.0f} pts/s)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(theta_p_wrapped, omega_p, s=0.1, color='black', alpha=0.5)
ax1.set_xlabel('$\\theta$ [rad]', fontsize=12)
ax1.set_ylabel('$\\omega$ [rad/s]', fontsize=12)
ax1.set_xlim(-np.pi, np.pi)

ax2.scatter(theta_p_wrapped, omega_p, s=0.5, color='k', alpha=0.6)
ax2.set_xlabel('$\\theta$ [rad]', fontsize=12)
ax2.set_ylabel('$\\omega$ [rad/s]', fontsize=12)

ax2.set_xlim(1.3, 1.8)
ax2.set_ylim(-0.4, 0.3)

plt.tight_layout()
plt.savefig('poincare_plot.png', dpi=700)

# Topological folding analysis
theta_n = theta_p_wrapped[:-1]
theta_n_plus_1 = theta_p_wrapped[1:]

plt.figure(figsize=(8, 8))
plt.scatter(theta_n, theta_n_plus_1, s=1.0, color='k', alpha=0.5)
plt.xlabel('$\\theta_n$ (Current State)', fontsize=12)
plt.ylabel('$\\theta_{n+1}$ (Next State)', fontsize=12)
plt.savefig('Topological_folding.png', dpi=700)
plt.show()