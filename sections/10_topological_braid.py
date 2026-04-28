"""
Topological Skeleton - Braiding of Unstable Periodic Orbits
3D visualization of chaotic braiding structures
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils_and_functions import RK4, DrivenDampedPendulum

# System Parameters
A_val = 1.5
omega_d = 0.67
beta_val = 0.5
T = 2 * np.pi / omega_d

# Exact Coordinates for unstable periodic orbits
p1_th_exact = -1.2291
p1_om_exact = -1.0664

p2_th_exact = 1.2587
p2_om_exact = 1.0452

def integrate_exact_path(th0, om0, num_periods, steps_per_period=200):
    """Integrate trajectory with proper angle wrapping"""
    dt = T / steps_per_period
    n_steps = num_periods * steps_per_period

    th_path = np.zeros(n_steps)
    om_path = np.zeros(n_steps)
    t_path = np.zeros(n_steps)

    th, om = th0, om0

    for i in range(n_steps):
        t = i * dt
        th_wrapped = (th + np.pi) % (2 * np.pi) - np.pi

        th_path[i] = th_wrapped
        om_path[i] = om
        t_path[i] = t

        # RK4 Steps
        k1_th = om * dt
        k1_om = (-np.sin(th) - beta_val * om + A_val * np.cos(omega_d * t)) * dt

        th2 = th + 0.5 * k1_th
        om2 = om + 0.5 * k1_om
        k2_th = om2 * dt
        k2_om = (-np.sin(th2) - beta_val * om2 + A_val * np.cos(omega_d * (t + 0.5*dt))) * dt

        th3 = th + 0.5 * k2_th
        om3 = om + 0.5 * k2_om
        k3_th = om3 * dt
        k3_om = (-np.sin(th3) - beta_val * om3 + A_val * np.cos(omega_d * (t + 0.5*dt))) * dt

        th4 = th + k3_th
        om4 = om + k3_om
        k4_th = om4 * dt
        k4_om = (-np.sin(th4) - beta_val * om4 + A_val * np.cos(omega_d * (t + dt))) * dt

        th += (k1_th + 2*k2_th + 2*k3_th + k4_th) / 6.0
        om += (k1_om + 2*k2_om + 2*k3_om + k4_om) / 6.0

    return th_path, om_path, t_path

num_periods_to_integrate = 5
th_a, om_a, t_a = integrate_exact_path(p1_th_exact, p1_om_exact, num_periods_to_integrate)
th_b, om_b, t_b = integrate_exact_path(p2_th_exact, p2_om_exact, num_periods_to_integrate)

def hide_wrapping_lines(th, om, t):
    """Insert NaN at wrapping discontinuities"""
    jumps = np.where(np.abs(np.diff(th)) > np.pi)[0] + 1
    th_clean = np.insert(th, jumps, np.nan)
    om_clean = np.insert(om, jumps, np.nan)
    t_clean = np.insert(t, jumps, np.nan)
    return th_clean, om_clean, t_clean

th_a_clean, om_a_clean, t_a_clean = hide_wrapping_lines(th_a, om_a, t_a)
th_b_clean, om_b_clean, t_b_clean = hide_wrapping_lines(th_b, om_b, t_b)

# Plot the 3D Braid
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(th_a_clean, om_a_clean, t_a_clean, color='blue', linewidth=1, label='Exact Orbit A')
ax.plot(th_b_clean, om_b_clean, t_b_clean, color='red', linewidth=1, label='Exact Orbit B')

ax.set_xlabel('$\\theta$ [rad]')
ax.set_ylabel('$\\omega$ [rad/s]')
ax.set_zlabel('Time [s]')

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-3, 3)

ax.view_init(elev=20, azim=45)

plt.legend(loc = (0.8,0.8))
plt.tight_layout()
plt.savefig('Topological_skeleton.png', dpi=700)
plt.show()
