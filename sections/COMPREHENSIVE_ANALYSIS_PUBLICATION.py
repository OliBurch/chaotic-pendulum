"""
Master Compilation: Pendulum Dynamics, Chaos, and Topology
Order of Execution:
1. Benchmarks
2. Phase space plots
3. Poincare plots
4. Bifurcation plots
5. Topology plots
6. Topology analysis
7. Computational analysis
"""

import sys
import os
import csv
import json
import psutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from time import perf_counter
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.special import comb
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# UTILITY AND CORE ODE FUNCTIONS (Filtered for active use)
# ============================================================================
def RK4(x, y, f, h):
    """Standard 4th-Order Runge-Kutta Numerical Integrator."""
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def Mass_on_spring(t, x_v):
    """ODE for a simple mass on a spring (Benchmark)."""
    x, v = x_v
    k = 1
    m = 1
    dxdt = x_v[1]
    dvdt = -(k / m) * x_v[0]
    return np.array([dxdt, dvdt])

def DampedPendulum(t, theta_w, B):
    """ODE for a Damped Pendulum."""
    theta, w = theta_w
    l = 9.81
    g = 9.81
    dthetadt = theta_w[1]
    dwdt = -g/l * np.sin(theta_w[0]) - B * theta_w[1]
    return np.array([dthetadt, dwdt])

def DampedODE_vector_field(theta_val, w_val, B):
    """Vector field generator for the Damped Pendulum."""
    l = 9.81
    g = 9.81
    dthetadt = w_val
    dwdt = -g/l * np.sin(theta_val) - B * w_val
    return dthetadt, dwdt

def damped_pendulum(initial_theta, beta_val):
    """Integration loop for Damped Pendulum."""
    t0 = 0
    tend = 500
    h = 0.1
    n = int((tend - t0) / h)
    theta_i = np.zeros(n)
    w_i = np.zeros(n)
    t = np.zeros(n)
    t[0] = t0
    theta_i[0] = initial_theta
    w_i[0] = 0

    for j in range(n-1):
        x_v_current = np.array([theta_i[j], w_i[j]])
        x_v_next = RK4(t[j], x_v_current, lambda current_t, current_theta_w: DampedPendulum(current_t, current_theta_w, beta_val), h)
        theta_i[j+1] = x_v_next[0]
        w_i[j+1] = x_v_next[1]
        t[j+1] = t[j] + h

        if theta_i[j+1] > np.pi:
            theta_i[j+1] -= 2*np.pi
        if theta_i[j+1] < -np.pi:
            theta_i[j+1] += 2*np.pi
    return t, w_i, theta_i

def DrivenDampedPendulum(t, theta_w, B, A):
    """ODE for a Driven Damped Pendulum."""
    theta, w = theta_w
    l = 9.81
    g = 9.81
    Wd = 0.67
    dthetadt = theta_w[1]
    dwdt = -(g/l) * np.sin(theta_w[0]) - B * theta_w[1] + A * np.cos(Wd * t)
    return np.array([dthetadt, dwdt])

def driven_damped_pendulum(N0_inputs, initial_omega, A):
    """Integration loop for Driven Damped Pendulum."""
    all_omega = []
    all_theta = []
    omega0 = np.zeros(N0_inputs)

    for i in range(N0_inputs):
        omega0[i] = (i**2 + initial_omega) * 0.3
        t0 = 0
        theta0 = 0
        tend = 500
        h = 0.1
        n = int((tend - t0) / h)
        theta_i = np.zeros(n)
        w_i = np.zeros(n)
        t = np.zeros(n)
        t[0] = t0
        theta_i[0] = theta0
        w_i[0] = omega0[i]

        for j in range(n-1):
            x_v_current = np.array([theta_i[j], w_i[j]])
            x_v_next = RK4(t[j], x_v_current, lambda current_t, current_theta_w: DrivenDampedPendulum(current_t, current_theta_w, 0.5, A), h)
            theta_i[j+1] = x_v_next[0]
            w_i[j+1] = x_v_next[1]
            t[j+1] = t[j] + h

        theta_wrapped = (theta_i + np.pi) % (2 * np.pi) - np.pi
        all_omega.append(w_i)
        all_theta.append(theta_wrapped)

    return t, all_omega, all_theta, omega0

def DrivenDampedODE_vector_field(theta_val, w_val, B, A, t_val):
    """Vector field generator for Driven Damped Pendulum."""
    l = 9.81
    g = 9.81
    Wd = 0.67
    dthetadt = w_val
    dwdt = -g/l * np.sin(theta_val) - B*w_val + A*np.cos(Wd*t_val)
    return dthetadt, dwdt


# ============================================================================
# 1. BENCHMARKS
# ============================================================================
def run_benchmarks():
    """
    Exercise 1: Benchmarking RK4 solver against mass-on-spring analytic solution
    Compares RK4 numerical solution with known analytic solution
    """
    print("\n--- Running Benchmarks ---")
    x0 = 10.0
    v0 = 0.0
    t0 = 0.0
    tend = 20 * np.pi
    h = 0.1
    n = int((tend - t0) / h)

    t = np.zeros(n)
    x = np.zeros(n)
    v = np.zeros(n)
    t[0], x[0], v[0] = t0, x0, v0

    for i in range(n - 1):
        x_v_current = np.array([x[i], v[i]])
        x_v_next = RK4(t[i], x_v_current, Mass_on_spring, h)
        x[i+1] = x_v_next[0]
        v[i+1] = x_v_next[1]
        t[i+1] = t[i] + h

    x_bench = x0 * np.cos(t) 
    v_bench = -x0 * np.sin(t) 
    error_x = np.abs(x_bench - x)

    plt.rcParams['font.size'] = 14
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t, x, color='red', marker='.', markersize=5, linestyle='', label='Position (RK4)')
    ax1.plot(t, x_bench, color='pink', linestyle='-', linewidth=2, alpha=0.8, label='Position (Bench)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [m]', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_ylim(-14, 14)

    ax2 = ax1.twinx()
    ax2.plot(t, v, color='black', marker='.', markersize=5, linestyle='', label='Velocity (RK4)')
    ax2.plot(t, v_bench, color='gray', linestyle='-', linewidth=2, alpha=0.8, label='Velocity (Bench)')
    ax2.set_ylabel('Velocity [m/s]', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(-14, 14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=6)

    fig1.tight_layout()
    fig1.savefig('Benchmarking_Rk4.png', dpi=1400)
    plt.close(fig1)

    fig2, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(t, error_x, 'b', label="Absolute Error in Position (x)")
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Error')
    fig2.tight_layout()
    fig2.savefig('Benchmarking_Rk4_Error.png', dpi=1400)
    plt.close(fig2)


# ============================================================================
# 2. PHASE SPACE PLOTS
# ============================================================================
def run_phase_space_plots():
    """
    Exercise 6 & 8: Damped & Driven Damped Pendulum Phase Plots
    Visualizes phase space behavior and time-dependent vector fields with wide damping range
    """
    print("\n--- Running Phase Space Plots ---")
    
    # 6: Damped Pendulum Wide
    betas = [0.25, 1, 2, 3]
    theta_start = 90
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for idx, beta in enumerate(betas):
        t_vals, omega_vals, theta_vals = damped_pendulum(theta_start, beta)
        axs[idx].scatter(theta_vals, omega_vals, s=0.5, color='k')
        x_min, x_max = -np.pi, np.pi
        y_min, y_max = -3, 3
        x_grid = np.linspace(x_min, x_max, 20)
        v_grid = np.linspace(y_min, y_max, 20)
        X_grid, V_grid = np.meshgrid(x_grid, v_grid)
        U_grid, W_grid = DampedODE_vector_field(X_grid, V_grid, beta)
        axs[idx].quiver(X_grid, V_grid, U_grid, W_grid, color='r', alpha=0.3, scale=20, width=0.002)
        axs[idx].set_title(f'($\\beta = {beta}$)')
        axs[idx].set_xlabel('$\\theta$ [rad]')
        axs[idx].set_ylabel('$\\omega$ [rad/s]')
        axs[idx].set_xlim(x_min, x_max)
        axs[idx].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig('Phaseplot_damped_wide.png', dpi=700)
    plt.close()

    # 8: Driven Damped Pendulum Phase
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

    plt.tight_layout()
    plt.savefig('Phaseplot_driven_damped.png', dpi=700)
    plt.close()

    # Animation block
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
    anim.save('phase_portraits.gif', writer='pillow', fps=10)
    plt.close(fig)


# ============================================================================
# 3. POINCARE PLOTS
# ============================================================================
def run_poincare_low_res():
    """
    Low Resolution Poincaré Section
    Fast computation with 25 steps per period for quick visualization
    """
    print("\n--- Running Low Resolution Poincare ---")
    script_start = perf_counter()
    process = psutil.Process(os.getpid())
    
    A_val = 1.5
    omega_d = 0.67
    beta_val = 0.5
    N = 25000
    steps = 25
    skip = 100
    dt = 2 * np.pi/(omega_d*steps)
    
    time_arr = np.arange(0.0, N*2*np.pi / omega_d, dt)
    nsteps = len(time_arr)
    omega = np.zeros(nsteps)
    theta = np.zeros(nsteps)

    integration_start = perf_counter()
    for j in range(nsteps-1):
        x_v_current = np.array([theta[j], omega[j]])
        x_v_next = RK4(time_arr[j], x_v_current, lambda t, yw: DrivenDampedPendulum(t, yw, beta_val, A_val), dt)
        theta[j+1] = x_v_next[0]
        omega[j+1] = x_v_next[1]

    integration_time = perf_counter() - integration_start
    peak_memory = process.memory_info().rss / (1024**2)

    offset = steps//4
    start_idx = skip * steps + offset
    theta_p = theta[start_idx :: steps]
    omega_p = omega[start_idx :: steps]
    theta_p_wrapped = (theta_p + np.pi) % (2 * np.pi) - np.pi

    performance_data = {
        'script': '13_poincare_low_resolution.py',
        'periods': N,
        'steps_per_period': steps,
        'total_steps': nsteps,
        'points_extracted': len(theta_p),
        'integration_time_sec': integration_time,
        'peak_memory_mb': peak_memory,
        'points_per_second': len(theta_p) / integration_time if integration_time > 0 else 0
    }
    with open('performance_13.json', 'w') as f:
        json.dump(performance_data, f, indent=2)

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
    plt.savefig('poincare_plot_low_def.pdf', dpi=700)
    plt.close()

def run_poincare_reg_res():
    """
    Regular Poincaré Section Analysis - Chaotic Attractors
    Shows stroboscopic sampling at driving frequency
    """
    print("\n--- Running Regular Poincare ---")
    script_start = perf_counter()
    process = psutil.Process(os.getpid())
    
    A_val = 1.5
    omega_d = 0.67
    beta_val = 0.5
    N = 50000
    steps = 100
    skip = 100
    dt = 2 * np.pi/(omega_d*steps)
    
    t_array = np.arange(0.0, N*2*np.pi / omega_d, dt)
    nsteps = len(t_array)
    omega = np.zeros(nsteps)
    theta = np.zeros(nsteps)

    integration_start = perf_counter()
    for j in range(nsteps-1):
        x_v_current = np.array([theta[j], omega[j]])
        x_v_next = RK4(t_array[j], x_v_current, lambda t, yw: DrivenDampedPendulum(t, yw, beta_val, A_val), dt)
        theta[j+1] = x_v_next[0]
        omega[j+1] = x_v_next[1]

    integration_time = perf_counter() - integration_start
    peak_memory = process.memory_info().rss / (1024**2)

    offset = steps//4
    start_idx = skip * steps + offset
    theta_p = theta[start_idx :: steps]
    omega_p = omega[start_idx :: steps]
    theta_p_wrapped = (theta_p + np.pi) % (2 * np.pi) - np.pi

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
    plt.close()

    theta_n = theta_p_wrapped[:-1]
    theta_n_plus_1 = theta_p_wrapped[1:]
    plt.figure(figsize=(8, 8))
    plt.scatter(theta_n, theta_n_plus_1, s=1.0, color='k', alpha=0.5)
    plt.xlabel('$\\theta_n$ (Current State)', fontsize=12)
    plt.ylabel('$\\theta_{n+1}$ (Next State)', fontsize=12)
    plt.savefig('Topological_folding.png', dpi=700)
    plt.close()

def run_poincare_super_res():
    """
    Super-High Resolution Poincaré Section
    Comprehensive topological mapping with maximum fidelity
    100k cycles × 250 steps per period = comprehensive attractor resolution
    """
    print("\n--- Running Super-High Resolution Poincare ---")
    script_start = perf_counter()
    process = psutil.Process(os.getpid())

    A_val = 1.5
    omega_d = 0.67
    beta_val = 0.5
    N = 100000 
    steps = 250   
    skip = 100    
    dt = 2 * np.pi/(omega_d*steps)
    time_arr = np.arange(0, N*2*np.pi / omega_d, dt)
    nsteps = len(time_arr)

    omega = np.zeros(nsteps)
    theta = np.zeros(nsteps)
    
    integration_start = perf_counter()
    for j in range(nsteps-1):
        x_v_current = np.array([theta[j], omega[j]])
        x_v_next = RK4(time_arr[j], x_v_current, 
                       lambda t, y: DrivenDampedPendulum(t, y, beta_val, A_val), dt)
        theta[j+1] = x_v_next[0]
        omega[j+1] = x_v_next[1]

    integration_time = perf_counter() - integration_start
    peak_memory = process.memory_info().rss / (1024**2)

    theta = theta[skip*steps:]
    omega = omega[skip*steps:]
    offset = steps//4
    theta_poincare = theta[offset :: steps]
    omega_poincare = omega[offset :: steps]
    theta_poincare = (theta_poincare + np.pi) % (2*np.pi) - np.pi
    num_points = len(theta_poincare)

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

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes[0, 0]
    ax.scatter(theta_poincare, omega_poincare, s=0.5, color='black', alpha=0.6)
    ax.set_xlim(-np.pi, np.pi)

    focus_mask = (theta_poincare >= 1.3) & (theta_poincare <= 1.8) & (omega_poincare >= -0.4) & (omega_poincare <= 0.3)
    ax = axes[0, 1]
    ax.scatter(theta_poincare[focus_mask], omega_poincare[focus_mask], s=2, color='red', alpha=0.7)
    ax.scatter(theta_poincare[~focus_mask], omega_poincare[~focus_mask], s=0.1, color='black', alpha=0.3)
    ax.set_xlim(1.3, 1.8)
    ax.set_ylim(-0.4, 0.3)

    ultra_zoom_mask = (theta_poincare >= 1.45) & (theta_poincare <= 1.65) & (omega_poincare >= -0.2) & (omega_poincare <= 0.1)
    ax = axes[1, 0]
    ax.scatter(theta_poincare[ultra_zoom_mask], omega_poincare[ultra_zoom_mask], s=3, color='darkred', alpha=0.8)
    ax.set_xlim(1.45, 1.65)
    ax.set_ylim(-0.2, 0.1)

    ax = axes[1, 1]
    theta_n = theta_poincare[:-1]
    theta_n_plus_1 = theta_poincare[1:]
    ax.scatter(theta_n, theta_n_plus_1, s=0.5, color='darkblue', alpha=0.5)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)

    plt.tight_layout()
    plt.savefig('poincare_super_resolution.pdf', dpi=400, bbox_inches='tight')
    plt.close()


# ============================================================================
# 4. BIFURCATION PLOTS
# ============================================================================
def run_bifurcation_diagram():
    """
    Bifurcation Diagram - Route to Chaos
    Shows how system behavior changes with driving amplitude
    """
    print("\n--- Running Bifurcation Diagram ---")
    omega_d = 0.67
    beta_val = 0.5
    A_vals = np.linspace(1, 1.65, 1000)
    steps = 300
    skip = 100
    N_cycles = 1000
    dt = 2 * np.pi / (omega_d * steps)
    nsteps = N_cycles * steps
    time_arr = np.arange(0, nsteps * dt, dt)

    A_plot = []
    omega_plot = []

    for idx, A in enumerate(A_vals):
        theta = np.zeros(nsteps)
        omega = np.zeros(nsteps)
        for j in range(nsteps-1):
            x_v_current = np.array([theta[j], omega[j]])
            x_v_next = RK4(time_arr[j], x_v_current, lambda t, yw: DrivenDampedPendulum(t, yw, beta_val, A), dt)
            theta[j+1] = x_v_next[0]
            omega[j+1] = x_v_next[1]

        offset = steps // 4
        omega_p = omega[skip * steps + offset :: steps]
        for om_p in omega_p:
            A_plot.append(A)
            omega_plot.append(om_p)

    plt.figure(figsize=(12, 7))
    plt.scatter(A_plot, omega_plot, s=0.05, color='black', alpha=0.3)
    plt.xlabel('Amplitude (A)', fontsize=12)
    plt.ylabel('$\\omega$ [rad/s]', fontsize=12)
    plt.savefig('Bifurcation_diagram.png', dpi=700)
    plt.close()


# ============================================================================
# 5. TOPOLOGY PLOTS
# ============================================================================
def run_topological_braid():
    """
    Topological Skeleton - Braiding of Unstable Periodic Orbits
    3D visualization of chaotic braiding structures
    """
    print("\n--- Running Topological Braid ---")
    A_val = 1.5
    omega_d = 0.67
    beta_val = 0.5
    T = 2 * np.pi / omega_d

    p1_th_exact = -1.2291
    p1_om_exact = -1.0664
    p2_th_exact = 1.2587
    p2_om_exact = 1.0452

    def integrate_exact_path(th0, om0, num_periods, steps_per_period=200):
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
        jumps = np.where(np.abs(np.diff(th)) > np.pi)[0] + 1
        th_clean = np.insert(th, jumps, np.nan)
        om_clean = np.insert(om, jumps, np.nan)
        t_clean = np.insert(t, jumps, np.nan)
        return th_clean, om_clean, t_clean

    th_a_clean, om_a_clean, t_a_clean = hide_wrapping_lines(th_a, om_a, t_a)
    th_b_clean, om_b_clean, t_b_clean = hide_wrapping_lines(th_b, om_b, t_b)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(th_b_clean, om_b_clean, t_b_clean, color='red', linewidth=1, label='Exact Orbit B')
    ax.set_xlabel('$\\theta$ [rad]')
    ax.set_ylabel('$\\omega$ [rad/s]')
    ax.set_zlabel('Time [s]')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-3, 3)
    ax.view_init(elev=20, azim=45)
    plt.savefig('Topological_skeleton.png', dpi=700)
    plt.close()


# ============================================================================
# 6. TOPOLOGY ANALYSIS
# ============================================================================
class DualLogger(object):
    """Custom logger for fidelity analysis output."""
    def __init__(self, filename="poincare_analysis_results.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def extract_poincare_points(N, steps, skip, A_val, omega_d, beta_val):
    dt = 2 * np.pi / (omega_d * steps)
    time_arr = np.arange(0, N * 2 * np.pi / omega_d, dt)
    nsteps = len(time_arr)
    omega = np.zeros(nsteps)
    theta = np.zeros(nsteps)
    
    for j in range(nsteps - 1):
        x_v_current = np.array([theta[j], omega[j]])
        x_v_next = RK4(time_arr[j], x_v_current,
                       lambda t, y: DrivenDampedPendulum(t, y, beta_val, A_val), dt)
        theta[j+1] = x_v_next[0]
        omega[j+1] = x_v_next[1]
    
    theta_clean = theta[skip*steps:]
    omega_clean = omega[skip*steps:]
    offset = steps // 4
    theta_p = theta_clean[offset :: steps]
    omega_p = omega_clean[offset :: steps]
    theta_p_wrapped = (theta_p + np.pi) % (2 * np.pi) - np.pi
    
    return theta_p_wrapped, omega_p, len(theta_p)

def count_points_in_region(theta, omega, theta_range, omega_range):
    mask = (theta >= theta_range[0]) & (theta <= theta_range[1]) & \
           (omega >= omega_range[0]) & (omega <= omega_range[1])
    return np.sum(mask)

def compute_point_density(theta, omega, grid_size=20):
    theta_edges = np.linspace(-np.pi, np.pi, grid_size + 1)
    omega_min, omega_max = omega.min(), omega.max()
    omega_edges = np.linspace(omega_min, omega_max, grid_size + 1)
    density = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            count = count_points_in_region(
                theta, omega,
                (theta_edges[i], theta_edges[i+1]),
                (omega_edges[j], omega_edges[j+1])
            )
            density[i, j] = count
    return density, theta_edges, omega_edges

def estimate_correlation_dimension(theta, omega, max_samples=5000):
    if len(theta) > max_samples:
        idx = np.random.choice(len(theta), max_samples, replace=False)
        theta_sample, omega_sample = theta[idx], omega[idx]
    else:
        theta_sample, omega_sample = theta, omega
        
    points = np.column_stack([theta_sample, omega_sample])
    distances = cdist(points, points)
    r_values = np.logspace(-2, 0, 20)
    C_r = []
    
    for r in r_values:
        count = np.sum(distances < r) - len(points)
        C_r.append(max(count, 1))
        
    C_r = np.array(C_r)
    valid_idx = C_r > 0
    log_r = np.log(r_values[valid_idx])
    log_C = np.log(C_r[valid_idx])
    coeffs = np.polyfit(log_r, log_C, 1)
    
    return coeffs[0], r_values, C_r

def detect_clusters(theta, omega, radius=0.1):
    points = np.column_stack([theta, omega])
    n = len(points)
    if n == 0: return 0, []
    
    visited = np.zeros(n, dtype=bool)
    clusters = []
    
    for i in range(n):
        if visited[i]: continue
        cluster = [i]
        visited[i] = True
        queue = [i]
        while queue:
            current = queue.pop(0)
            distances = np.linalg.norm(points - points[current], axis=1)
            neighbors = np.where((distances < radius) & ~visited)[0]
            for j in neighbors:
                visited[j] = True
                cluster.append(j)
                queue.append(j)
        clusters.append(cluster)
        
    return len(clusters), clusters

def compute_lyapunov_exponent_estimate(theta, omega, max_samples=2000):
    if len(theta) > max_samples:
        idx = np.random.choice(len(theta), max_samples, replace=False)
        theta_sample, omega_sample = theta[idx], omega[idx]
    else:
        theta_sample, omega_sample = theta, omega
        
    points = np.column_stack([theta_sample, omega_sample])
    divergences = []
    for i in range(min(100, len(points) - 10)):
        for j in range(i+1, min(i+20, len(points))):
            dist_now = np.linalg.norm(points[i] - points[j])
            if dist_now > 1e-6:
                divergences.append(np.log(dist_now))
                
    return np.mean(divergences) if divergences else 0.0

def analyze_coverage(theta, omega, grid_size=20):
    density, _, _ = compute_point_density(theta, omega, grid_size)
    non_empty_cells = np.sum(density > 0)
    total_cells = grid_size * grid_size
    return 100 * non_empty_cells / total_cells, density

def run_fidelity_analysis():
    print("\n--- Running Fidelity Analysis ---")
    original_stdout = sys.stdout
    sys.stdout = DualLogger("poincare_analysis_results.txt")

    print("\n" + "="*70)
    print("POINCARÉ SECTION TOPOLOGICAL FIDELITY ANALYSIS")
    print("="*70)
    
    A_val = 1.5
    omega_d = 0.67
    beta_val = 0.5
    
    # Extract points
    print("\n[SCRIPT 13] Low Resolution")
    theta_09, omega_09, periods_09 = extract_poincare_points(25000, 25, 100, A_val, omega_d, beta_val)
    print(f"✓ Extracted: {len(theta_09)} points")
    
    print("\n[SCRIPT 09] Regular Resolution")
    theta_12, omega_12, periods_12 = extract_poincare_points(50000, 100, 100, A_val, omega_d, beta_val)
    print(f"✓ Extracted: {len(theta_12)} points")
    
    print("\n[SCRIPT 14] Super-High Resolution")
    theta_14, omega_14, periods_14 = extract_poincare_points(100000, 250, 100, A_val, omega_d, beta_val)
    print(f"✓ Extracted: {len(theta_14)} points")

    # Spatial Analysis
    print("\n" + "="*70 + "\nSPATIAL ANALYSIS\n" + "="*70)
    focus_region = {"theta": (1.3, 1.8), "omega": (-0.4, 0.3)}
    
    c_09 = count_points_in_region(theta_09, omega_09, focus_region["theta"], focus_region["omega"])
    c_12 = count_points_in_region(theta_12, omega_12, focus_region["theta"], focus_region["omega"])
    c_14 = count_points_in_region(theta_14, omega_14, focus_region["theta"], focus_region["omega"])
    
    print(f"Focus Region Points (Script 13): {c_09} ({100*c_09/max(len(theta_09),1):.1f}%)")
    print(f"Focus Region Points (Script 09): {c_12} ({100*c_12/max(len(theta_12),1):.1f}%)")
    print(f"Focus Region Points (Script 14): {c_14} ({100*c_14/max(len(theta_14),1):.1f}%)")

    cov_09, _ = analyze_coverage(theta_09, omega_09)
    cov_12, _ = analyze_coverage(theta_12, omega_12)
    cov_14, _ = analyze_coverage(theta_14, omega_14)
    print(f"\nCoverage: Script 13 ({cov_09:.1f}%), Script 09 ({cov_12:.1f}%), Script 14 ({cov_14:.1f}%)")

    # Topological Properties
    print("\n" + "="*70 + "\nTOPOLOGICAL PROPERTIES\n" + "="*70)
    D2_09, _, _ = estimate_correlation_dimension(theta_09, omega_09)
    D2_12, _, _ = estimate_correlation_dimension(theta_12, omega_12)
    D2_14, _, _ = estimate_correlation_dimension(theta_14, omega_14)
    print(f"Correlation Dimension (D_2): Script 13 ({D2_09:.3f}), Script 09 ({D2_12:.3f}), Script 14 ({D2_14:.3f})")

    lyap_09 = compute_lyapunov_exponent_estimate(theta_09, omega_09)
    lyap_12 = compute_lyapunov_exponent_estimate(theta_12, omega_12)
    lyap_14 = compute_lyapunov_exponent_estimate(theta_14, omega_14)
    print(f"Lyapunov Est (λ): Script 13 ({lyap_09:.4f}), Script 09 ({lyap_12:.4f}), Script 14 ({lyap_14:.4f})")

    sys.stdout = original_stdout
    print("✓ Full Fidelity output written to 'poincare_analysis_results.txt'")
    
# ============================================================================
# 7. COMPUTATIONAL ANALYSIS
# ============================================================================
def run_performance_analysis():
    """
    PERFORMANCE ANALYSIS - Computational Cost Assessment
    Analyzes runtime, memory usage, and efficiency across all Poincaré simulations
    """
    print("\n--- Running Performance Analysis ---")
    performance_data = []
    
    for json_file in Path('.').rglob('performance_*.json'):
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        name = entry.get('script', entry.get('name', 'Unknown'))
                        if '12' not in name and 'Ultra' not in name:
                            performance_data.append(entry)
                else:
                    name = data.get('script', data.get('name', 'Unknown'))
                    if '12' not in name and 'Ultra' not in name:
                        performance_data.append(data)
            except Exception as e:
                pass

    if not performance_data:
        print("⚠ WARNING: No performance data files found! Ensure Poincare scripts successfully generated JSONs.")
        return

    extracted_dict = {}
    for entry in performance_data:
        name = entry.get('script', entry.get('name', 'Unknown'))
        if '13' in name or 'Low' in name:
            label, order, color = 'Low resolution', 0, '#2ca02c'
        elif '09' in name or 'Regular' in name:
            label, order, color = 'Middle ground', 1, '#1f77b4'
        elif '14' in name or 'Super' in name:
            label, order, color = 'High resolution', 2, '#d62728'
        else:
            continue
        
        time_val = entry.get('integration_time_sec', entry.get('time_sec', 0))
        mem_val = entry.get('peak_memory_mb', entry.get('memory_mb', 0))
        points = entry.get('points_extracted', entry.get('points', 0))
        extracted_dict[label] = (order, label, time_val, mem_val, points, color)

    extracted_data = list(extracted_dict.values())
    extracted_data.sort(key=lambda x: x[0])

    scripts = [item[1] for item in extracted_data]
    times = [item[2] for item in extracted_data]
    memories = [item[3] for item in extracted_data]
    points_list = [item[4] for item in extracted_data]
    colors_list = [item[5] for item in extracted_data]

    if scripts:
        fig, ax = plt.subplots(figsize=(10, 7))
        sizes = [mem * 3 for mem in memories]
        for time_val, points, color, label, size in zip(times, points_list, colors_list, scripts, sizes):
            ax.scatter(time_val, points, s=size, alpha=0.6, color=color, edgecolors='black', linewidth=2, label=label)
        
        ax.set_xlabel('Runtime (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Points Extracted from Poincaré Section', fontsize=11, fontweight='bold')
        ax.set_title('Computational Cost vs. Data Yield Tradeoff', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig('performance_tradeoff.png', dpi=700)
        plt.close()
        print("✓ Saved: performance_tradeoff.png")

# ============================================================================
# MASTER EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("Starting Main Execution Pipeline...")
    
    run_benchmarks()
    run_phase_space_plots()
    
    # Poincare sections (Generate JSON logs for final step)
    run_poincare_low_res()
    run_poincare_reg_res()
    run_poincare_super_res()
    
    run_bifurcation_diagram()
    run_topological_braid()
    run_fidelity_analysis()
    run_performance_analysis()
    
    print("\nPipeline execution complete! All plots, logs, and GIFs generated.")