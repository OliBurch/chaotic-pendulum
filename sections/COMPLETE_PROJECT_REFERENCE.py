"""
COMPLETE PENDULUM PROJECT - ALL EXERCISES 01-14 COMBINED
=========================================================
Complete working code from all 14 exercises + topological analysis
Full compilation for personal reference and execution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial.distance import cdist
import warnings
import time
import json
import psutil
import os

warnings.filterwarnings('ignore')

print("="*70)
print("COMPLETE PENDULUM PROJECT - All Exercises 01-14")
print("="*70)

# Timing tracking
performance_log = []
process = psutil.Process(os.getpid())

# ============================================================================
# CORE NUMERICAL SOLVER: RK4 (4th-order Runge-Kutta)
# ============================================================================

def RK4(x, y, f, h):
    """4th-order Runge-Kutta ODE solver"""
    k1 = h * f(x, y)
    k2 = h * f(x + 0.5*h, y + 0.5*k1)
    k3 = h * f(x + 0.5*h, y + 0.5*k2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# ============================================================================
# ODE SYSTEM DEFINITIONS
# ============================================================================

def Mass_on_spring(t, x_v):
    """Harmonic oscillator: d²x/dt² = -x"""
    x, v = x_v
    k = 1
    m = 1
    dxdt = x_v[1]
    dvdt = -(k/m) * x_v[0]
    return np.array([dxdt, dvdt])

def Mass_on_spring_vector_field(x_val, v_val):
    k = 1
    m = 1
    dxdt_val = v_val
    dvdt_val = -(k / m) * x_val
    return dxdt_val, dvdt_val

def Pendulum(t, theta_w):
    """Simple pendulum: d²θ/dt² = -(g/ℓ)sin(θ)"""
    theta, w = theta_w
    l = 9.81
    g = 9.81
    dthetadt = theta_w[1]
    dwdt = -g/l * np.sin(theta_w[0])
    return np.array([dthetadt, dwdt])

def Pendulum_vector_field(theta_val, w_val):
    l = 9.81
    g = 9.81
    dthetadt = w_val
    dwdt = -g/l * np.sin(theta_val)
    return dthetadt, dwdt

def DampedPendulum(t, theta_w, B):
    """Damped pendulum: d²θ/dt² = -(g/ℓ)sin(θ) - β·ω"""
    theta, w = theta_w
    l = 9.81
    g = 9.81
    dthetadt = theta_w[1]
    dwdt = -g/l * np.sin(theta_w[0]) - B*theta_w[1]
    return np.array([dthetadt, dwdt])

def DampedODE_vector_field(theta_val, w_val, B):
    l = 9.81
    g = 9.81
    dthetadt = w_val
    dwdt = -g/l * np.sin(theta_val) - B*w_val
    return dthetadt, dwdt

def DrivenDampedPendulum(t, theta_w, B, A):
    """Driven-damped pendulum: d²θ/dt² = -(g/ℓ)sin(θ) - β·ω + A·cos(ωd·t)"""
    theta, w = theta_w
    l = 9.81
    g = 9.81
    Wd = 0.67
    dthetadt = theta_w[1]
    dwdt = -(g/l) * np.sin(theta_w[0]) - B*theta_w[1] + A*np.cos(Wd*t)
    return np.array([dthetadt, dwdt])

def DrivenDampedODE_vector_field(theta_val, w_val, B, A, t_val):
    l = 9.81
    g = 9.81
    Wd = 0.67
    dthetadt = w_val
    dwdt = -g/l * np.sin(theta_val) - B*w_val + A*np.cos(Wd*t_val)
    return dthetadt, dwdt

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def theta_single(N0_inputs, initial_theta):
    """Solve simple pendulum for multiple initial angles"""
    all_thetas = []
    theta0 = np.zeros(N0_inputs)

    for i in range(N0_inputs):
        theta0[i] = initial_theta
        t0 = 0
        w0 = 0
        tend = 10
        h = 0.1
        n = int((tend - t0) / h)

        theta_i = np.zeros(n)
        w_i = np.zeros(n)
        t = np.zeros(n)

        t[0] = t0
        theta_i[0] = theta0[i]
        w_i[0] = w0

        for j in range(n-1):
            x_v_current = np.array([theta_i[j], w_i[j]])
            x_v_next = RK4(t[j], x_v_current, Pendulum, h)
            theta_i[j+1] = x_v_next[0]
            w_i[j+1] = x_v_next[1]
            t[j+1] = t[j] + h

        all_thetas.append(theta_i)

    return t, all_thetas, theta0

def Zero_cross_alg(N0_inputs, theta_init):
    """Calculate period via zero-crossing algorithm"""
    t_out, thetas_out, theta0_out = theta_single(N0_inputs, theta_init)

    zero_crossing_times = []
    for i in range(1, len(thetas_out[0])):
        if np.sign(thetas_out[0][i-1]) != np.sign(thetas_out[0][i]):
            zero_crossing_times.append(t_out[i])

    period_differences = []
    for j in range(len(zero_crossing_times) - 1):
        period_differences.append(zero_crossing_times[j+1] - zero_crossing_times[j])

    period = np.mean(period_differences) * 2
    return period

def damped_pendulum(initial_theta, beta_val):
    """Solve damped pendulum ODE"""
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
        x_v_next = RK4(t[j], x_v_current, lambda ct, ctw: DampedPendulum(ct, ctw, beta_val), h)
        theta_i[j+1] = x_v_next[0]
        w_i[j+1] = x_v_next[1]
        t[j+1] = t[j] + h

        if theta_i[j+1] > np.pi:
            theta_i[j+1] = theta_i[j+1] - 2*np.pi
        if theta_i[j+1] < -np.pi:
            theta_i[j+1] = theta_i[j+1] + 2*np.pi

    return t, w_i, theta_i

def driven_damped_pendulum(N0_inputs, initial_omega, A):
    """Solve driven-damped pendulum for multiple initial conditions"""
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
            x_v_next = RK4(t[j], x_v_current, lambda ct, ctw: DrivenDampedPendulum(ct, ctw, 0.5, A), h)
            theta_i[j+1] = x_v_next[0]
            w_i[j+1] = x_v_next[1]
            t[j+1] = t[j] + h

        theta_wrapped = (theta_i + np.pi) % (2 * np.pi) - np.pi
        all_omega.append(w_i)
        all_theta.append(theta_wrapped)

    return t, all_omega, all_theta, omega0

# ============================================================================
# EXERCISE 01: MASS ON SPRING - BENCHMARK
# ============================================================================
print("\n[EXERCISE 01] Mass on Spring Benchmark")
x0 = 1.0
v0 = 0.0
t0 = 0.0
tend = 4 * np.pi
h = 0.01
n = int((tend - t0) / h)

t_array = np.zeros(n)
x_array = np.zeros(n)
v_array = np.zeros(n)

t_array[0] = t0
x_array[0] = x0
v_array[0] = v0

for i in range(n-1):
    x_v_current = np.array([x_array[i], v_array[i]])
    x_v_next = RK4(t_array[i], x_v_current, Mass_on_spring, h)
    x_array[i+1] = x_v_next[0]
    v_array[i+1] = x_v_next[1]
    t_array[i+1] = t_array[i] + h

x_analytical = np.cos(t_array)
error = np.abs(x_array - x_analytical)
print(f"✓ Max position error: {np.max(error):.2e}")

# ============================================================================
# EXERCISE 02: ENERGY CONSERVATION
# ============================================================================
print("[EXERCISE 02] Energy Conservation")
energy_numerical = 0.5 * v_array**2 + 0.5 * x_array**2
energy_analytical = 0.5 * (x0**2 + v0**2)
energy_error = np.abs(energy_numerical - energy_analytical)
print(f"✓ Max energy error: {np.max(energy_error):.2e}")
print(f"✓ Total Energy: {energy_analytical:.4f}")

# ============================================================================
# EXERCISE 03: PHASE SPACE HARMONIC OSCILLATOR
# ============================================================================
print("[EXERCISE 03] Phase Space - Harmonic Oscillator")
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_array, v_array, 'k-', linewidth=1, label='Numerical (RK4)')
ax.plot(x_analytical, -np.sin(t_array), 'r--', linewidth=1, label='Analytical')
ax.set_xlabel('Position $x$', fontsize=12)
ax.set_ylabel('Velocity $v$', fontsize=12)
ax.set_title('Phase Portrait: Harmonic Oscillator', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ex03_phase_space.pdf', dpi=200)
plt.close()
print("✓ Saved: ex03_phase_space.pdf")

# ============================================================================
# EXERCISE 04: SIMPLE PENDULUM - NONLINEARITIES
# ============================================================================
print("[EXERCISE 04] Simple Pendulum - Nonlinearities")
initial_angles = [0.1, 0.5, 1.0, 1.5]
periods = []

fig, ax = plt.subplots(figsize=(10, 6))
for angle in initial_angles:
    period = Zero_cross_alg(1, angle)
    periods.append(period)
    t, thetas, _ = theta_single(1, angle)
    ax.plot(t[t <= 20], thetas[0][t <= 20], label=f'θ₀ = {angle:.1f} rad, T = {period:.3f}')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Angle θ (rad)', fontsize=12)
ax.set_title('Simple Pendulum: Period vs Amplitude', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ex04_pendulum_nonlinear.pdf', dpi=200)
plt.close()
print(f"✓ Periods: {periods}")
print("✓ Saved: ex04_pendulum_nonlinear.pdf")

# ============================================================================
# EXERCISE 05: PENDULUM PHASE SPACE
# ============================================================================
print("[EXERCISE 05] Pendulum Phase Space")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
for idx, angle in enumerate(initial_angles):
    t, thetas, _ = theta_single(1, angle)
    ax = axes[idx]
    ax.plot(thetas[0], np.gradient(thetas[0], t), 'k-', linewidth=1)
    ax.set_xlabel('θ (rad)', fontsize=11)
    ax.set_ylabel('dθ/dt (rad/s)', fontsize=11)
    ax.set_title(f'Phase Space: θ₀ = {angle:.1f} rad', fontsize=12)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ex05_pendulum_phase.pdf', dpi=200)
plt.close()
print("✓ Saved: ex05_pendulum_phase.pdf")

# ============================================================================
# EXERCISE 06: DAMPED PENDULUM - WIDE RANGE
# ============================================================================
print("[EXERCISE 06] Damped Pendulum - Wide Damping Range")
damping_vals = [0.25, 1.0, 1.5, 3.0]
fig, ax = plt.subplots(figsize=(10, 6))

for beta in damping_vals:
    t, w, theta = damped_pendulum(0.5, beta)
    mask = t <= 100
    ax.plot(t[mask], theta[mask], label=f'β = {beta}', linewidth=1)

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Angle θ (rad)', fontsize=12)
ax.set_title('Damped Pendulum: Multiple Damping Coefficients', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ex06_damped_wide.pdf', dpi=200)
plt.close()
print("✓ Saved: ex06_damped_wide.pdf")

# ============================================================================
# EXERCISE 07: DAMPED PENDULUM - FINE RESOLUTION
# ============================================================================
print("[EXERCISE 07] Damped Pendulum - Fine Resolution")
beta_fine = np.linspace(0.1, 5, 50)
settling_times = []

for beta in beta_fine[:10]:  # Sample for speed
    t, w, theta = damped_pendulum(1.5, beta)
    settling_idx = np.argmax(np.abs(theta) < 0.01)
    if settling_idx > 0:
        settling_times.append(t[settling_idx])
    else:
        settling_times.append(t[-1])

print(f"✓ Settling time range: {np.min(settling_times):.2f} - {np.max(settling_times):.2f} s")

# ============================================================================
# EXERCISE 08: DRIVEN-DAMPED PHASE PORTRAITS
# ============================================================================
print("[EXERCISE 08] Driven-Damped Pendulum - Phase Portraits")
A_vals = [0.9, 1.07, 1.47, 1.5]
fig, axs = plt.subplots(2, 2, figsize=(12, 11))
axs = axs.flatten()

for idx, A in enumerate(A_vals):
    t_vals, omega_vals, theta_vals, omega0 = driven_damped_pendulum(1, 0, A)
    t_vals = np.array(t_vals).flatten()
    omega_vals = np.array(omega_vals).flatten()
    theta_vals = np.array(theta_vals).flatten()
    mask = t_vals > 100
    axs[idx].scatter(theta_vals[mask], omega_vals[mask], s=0.5, color='k', alpha=0.6)
    axs[idx].set_title(f'Phase Portrait (A = {A})', fontsize=12)
    axs[idx].set_xlabel('θ [rad]', fontsize=11)
    axs[idx].set_ylabel('ω [rad/s]', fontsize=11)
    axs[idx].grid(True, alpha=0.3)
    axs[idx].set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('ex08_phase_portraits.pdf', dpi=200)
plt.close()
print("✓ Saved: ex08_phase_portraits.pdf")

# ============================================================================
# EXERCISE 09: POINCARÉ SECTION - REGULAR RESOLUTION
# ============================================================================
print("[EXERCISE 09] Poincaré Section - Regular Resolution (50k periods)")
theta0, omega0, t0 = 0, 0.0, 0.0
A_val, omega_d, beta_val = 1.5, 0.67, 0.5
N, steps, skip = 50000, 100, 100

dt = 2 * np.pi / (omega_d * steps)
time = np.arange(t0, N * 2 * np.pi / omega_d, dt)
nsteps = len(time)

omega_arr = np.zeros(nsteps)
theta_arr = np.zeros(nsteps)
omega_arr[0] = omega0
theta_arr[0] = theta0

ex09_start = time.perf_counter()
for j in range(nsteps-1):
    x_v_current = np.array([theta_arr[j], omega_arr[j]])
    x_v_next = RK4(time[j], x_v_current, lambda ct, ctw: DrivenDampedPendulum(ct, ctw, beta_val, A_val), dt)
    theta_arr[j+1] = x_v_next[0]
    omega_arr[j+1] = x_v_next[1]
ex09_time = time.perf_counter() - ex09_start
ex09_memory = process.memory_info().rss / (1024**2)

offset = steps // 4
start_idx = skip * steps + offset
theta_p = theta_arr[start_idx :: steps]
omega_p = omega_arr[start_idx :: steps]
theta_p_wrapped = (theta_p + np.pi) % (2 * np.pi) - np.pi

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(theta_p_wrapped, omega_p, s=0.1, color='black', alpha=0.5)
ax1.set_xlabel('θ [rad]', fontsize=12)
ax1.set_ylabel('ω [rad/s]', fontsize=12)
ax1.set_title('Full Poincaré Section', fontsize=13)
ax1.set_xlim(-np.pi, np.pi)
ax1.grid(True, alpha=0.3)

ax2.scatter(theta_p_wrapped, omega_p, s=0.5, color='k', alpha=0.6)
ax2.set_xlabel('θ [rad]', fontsize=12)
ax2.set_ylabel('ω [rad/s]', fontsize=12)
ax2.set_title('Zoomed Detail (Fractal)', fontsize=13)
ax2.set_xlim(1.3, 1.8)
ax2.set_ylim(-0.4, 0.3)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ex09_poincare_regular.pdf', dpi=200)
plt.close()
print(f"✓ Points extracted: {len(theta_p)}")
print(f"  [Performance] Time: {ex09_time:.2f}s | Memory: {ex09_memory:.1f}MB | {len(theta_p)/ex09_time:.0f} pts/s")
performance_log.append({
    'exercise': '09',
    'name': 'Poincaré Regular (50k)',
    'periods': N,
    'steps_per_period': steps,
    'points': len(theta_p),
    'time_sec': ex09_time,
    'memory_mb': ex09_memory,
    'pts_per_sec': len(theta_p) / ex09_time
})
print("✓ Saved: ex09_poincare_regular.pdf")

# ============================================================================
# EXERCISE 10: TOPOLOGICAL FOLDING (Return Map)
# ============================================================================
print("[EXERCISE 10] Topological Folding - Return Map")
theta_n = theta_p_wrapped[:-1]
theta_n_plus_1 = theta_p_wrapped[1:]

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(theta_n, theta_n_plus_1, s=1.0, color='k', alpha=0.5)
ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', alpha=0.5, label='y=x')
ax.set_xlabel('θₙ (Current)', fontsize=12)
ax.set_ylabel('θₙ₊₁ (Next)', fontsize=12)
ax.set_title('Topological Folding Map', fontsize=13)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('ex10_topological_folding.pdf', dpi=200)
plt.close()
print("✓ Saved: ex10_topological_folding.pdf")

# ============================================================================
# EXERCISE 11: BIFURCATION DIAGRAM
# ============================================================================
print("[EXERCISE 11] Bifurcation Diagram (Period-Doubling Route to Chaos)")
A_values = np.linspace(1.0, 1.65, 100)
bifurc_theta = []
bifurc_A = []

for A in A_values:
    t_vals, omega_vals, theta_vals, _ = driven_damped_pendulum(1, 0, A)
    t_vals = np.array(t_vals).flatten()
    theta_vals = np.array(theta_vals).flatten()
    mask = t_vals > 400  # Transient removal
    if np.sum(mask) > 0:
        bifurc_theta.extend(theta_vals[mask])
        bifurc_A.extend([A] * np.sum(mask))

fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(bifurc_A, bifurc_theta, s=0.1, color='black', alpha=0.4)
ax.set_xlabel('Driving Amplitude A', fontsize=12)
ax.set_ylabel('θ [rad]', fontsize=12)
ax.set_title('Bifurcation Diagram: Period-Doubling to Chaos', fontsize=13)
ax.axvline(1.1, color='r', linestyle='--', alpha=0.3, label='1st bifurcation')
ax.axvline(1.4, color='b', linestyle='--', alpha=0.3, label='Chaos onset')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('ex11_bifurcation.pdf', dpi=200)
plt.close()
print("✓ Saved: ex11_bifurcation.pdf")

# ============================================================================
# EXERCISE 12: POINCARÉ SECTION - ULTRA-HIGH RESOLUTION
# ============================================================================
print("[EXERCISE 12] Poincaré Section - Ultra-High Resolution (30k periods)")
N12, steps12 = 30000, 200
dt12 = 2 * np.pi / (omega_d * steps12)
time12 = np.arange(t0, N12 * 2 * np.pi / omega_d, dt12)
nsteps12 = len(time12)

omega_arr12 = np.zeros(nsteps12)
theta_arr12 = np.zeros(nsteps12)
omega_arr12[0] = omega0
theta_arr12[0] = theta0

ex12_start = time.perf_counter()
for j in range(nsteps12-1):
    x_v_current = np.array([theta_arr12[j], omega_arr12[j]])
    x_v_next = RK4(time12[j], x_v_current, lambda ct, ctw: DrivenDampedPendulum(ct, ctw, beta_val, A_val), dt12)
    theta_arr12[j+1] = x_v_next[0]
    omega_arr12[j+1] = x_v_next[1]
ex12_time = time.perf_counter() - ex12_start
ex12_memory = process.memory_info().rss / (1024**2)

offset12 = steps12 // 4
start_idx12 = skip * steps12 + offset12
theta_p12 = theta_arr12[start_idx12 :: steps12]
omega_p12 = omega_arr12[start_idx12 :: steps12]
theta_p12_wrapped = (theta_p12 + np.pi) % (2 * np.pi) - np.pi

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(theta_p12_wrapped, omega_p12, s=0.1, color='black', alpha=0.5)
ax1.set_xlabel('θ [rad]', fontsize=12)
ax1.set_ylabel('ω [rad/s]', fontsize=12)
ax1.set_title('Full Poincaré Section (30k periods)', fontsize=13)
ax1.set_xlim(-np.pi, np.pi)
ax1.grid(True, alpha=0.3)

ax2.scatter(theta_p12_wrapped, omega_p12, s=0.5, color='k', alpha=0.6)
ax2.set_xlabel('θ [rad]', fontsize=12)
ax2.set_ylabel('ω [rad/s]', fontsize=12)
ax2.set_title('Zoomed Detail (30k points)', fontsize=13)
ax2.set_xlim(1.3, 1.8)
ax2.set_ylim(-0.4, 0.3)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ex12_poincare_ultra.pdf', dpi=200)
plt.close()
print(f"✓ Points extracted: {len(theta_p12)}")
print(f"  [Performance] Time: {ex12_time:.2f}s | Memory: {ex12_memory:.1f}MB | {len(theta_p12)/ex12_time:.0f} pts/s")
performance_log.append({
    'exercise': '12',
    'name': 'Poincaré Ultra (30k)',
    'periods': N12,
    'steps_per_period': steps12,
    'points': len(theta_p12),
    'time_sec': ex12_time,
    'memory_mb': ex12_memory,
    'pts_per_sec': len(theta_p12) / ex12_time
})
print("✓ Saved: ex12_poincare_ultra.pdf")

# ============================================================================
# EXERCISE 13: POINCARÉ SECTION - LOW RESOLUTION (QUICK PREVIEW)
# ============================================================================
print("[EXERCISE 13] Poincaré Section - Low Resolution (Quick Preview)")
N13, steps13 = 50000, 25
dt13 = 2 * np.pi / (omega_d * steps13)
time13 = np.arange(t0, N13 * 2 * np.pi / omega_d, dt13)
nsteps13 = len(time13)

omega_arr13 = np.zeros(nsteps13)
theta_arr13 = np.zeros(nsteps13)
omega_arr13[0] = omega0
theta_arr13[0] = theta0

for j in range(nsteps13-1):
    x_v_current = np.array([theta_arr13[j], omega_arr13[j]])
    x_v_next = RK4(time13[j], x_v_current, lambda ct, ctw: DrivenDampedPendulum(ct, ctw, beta_val, A_val), dt13)
    theta_arr13[j+1] = x_v_next[0]
    omega_arr13[j+1] = x_v_next[1]

offset13 = steps13 // 4
start_idx13 = skip * steps13 + offset13
theta_p13 = theta_arr13[start_idx13 :: steps13]
omega_p13 = omega_arr13[start_idx13 :: steps13]

fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(theta_p13, omega_p13, s=1, color='k', alpha=0.6)
ax.set_xlabel('θ [rad]', fontsize=12)
ax.set_ylabel('ω [rad/s]', fontsize=12)
ax.set_title('Low Resolution Preview (25 steps/period)', fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ex13_poincare_low.pdf', dpi=200)
plt.close()
print(f"✓ Points extracted: {len(theta_p13)}")
print("✓ Saved: ex13_poincare_low.pdf")

# ============================================================================
# EXERCISE 14: POINCARÉ SECTION - SUPER-HIGH RESOLUTION
# ============================================================================
print("[EXERCISE 14] Poincaré Section - Super-High Resolution (100k periods)")
N14, steps14 = 100000, 250
dt14 = 2 * np.pi / (omega_d * steps14)
time14 = np.arange(t0, N14 * 2 * np.pi / omega_d, dt14)
nsteps14 = len(time14)

omega_arr14 = np.zeros(nsteps14)
theta_arr14 = np.zeros(nsteps14)
omega_arr14[0] = omega0
theta_arr14[0] = theta0

print(f"  Integrating {nsteps14} steps...")
ex14_start = time.perf_counter()
for j in range(nsteps14-1):
    if (j+1) % 5000000 == 0:
        print(f"  Progress: {j+1}/{nsteps14-1}")
    x_v_current = np.array([theta_arr14[j], omega_arr14[j]])
    x_v_next = RK4(time14[j], x_v_current, lambda ct, ctw: DrivenDampedPendulum(ct, ctw, beta_val, A_val), dt14)
    theta_arr14[j+1] = x_v_next[0]
    omega_arr14[j+1] = x_v_next[1]
ex14_time = time.perf_counter() - ex14_start
ex14_memory = process.memory_info().rss / (1024**2)

offset14 = steps14 // 4
start_idx14 = skip * steps14 + offset14
theta_p14 = theta_arr14[start_idx14 :: steps14]
omega_p14 = omega_arr14[start_idx14 :: steps14]
theta_p14_wrapped = (theta_p14 + np.pi) % (2 * np.pi) - np.pi

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(theta_p14_wrapped, omega_p14, s=0.05, color='black', alpha=0.6)
ax1.set_xlabel('θ [rad]', fontsize=12)
ax1.set_ylabel('ω [rad/s]', fontsize=12)
ax1.set_title('Full Poincaré Section (100k periods, 250 steps/period)', fontsize=14)
ax1.set_xlim(-np.pi, np.pi)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(theta_p14_wrapped, omega_p14, s=0.1, color='k', alpha=0.6)
ax2.set_xlabel('θ [rad]', fontsize=11)
ax2.set_ylabel('ω [rad/s]', fontsize=11)
ax2.set_title('Zoom 1: Main Structure', fontsize=12)
ax2.set_xlim(1.3, 1.8)
ax2.set_ylim(-0.4, 0.3)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(theta_p14_wrapped, omega_p14, s=0.2, color='k', alpha=0.7)
ax3.set_xlabel('θ [rad]', fontsize=11)
ax3.set_ylabel('ω [rad/s]', fontsize=11)
ax3.set_title('Zoom 2: Fine Detail (Fractal)', fontsize=12)
ax3.set_xlim(1.45, 1.65)
ax3.set_ylim(-0.2, 0.1)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ex14_poincare_super.pdf', dpi=200)
plt.close()
print(f"✓ Points extracted: {len(theta_p14)}")
print(f"  [Performance] Time: {ex14_time:.2f}s | Memory: {ex14_memory:.1f}MB | {len(theta_p14)/ex14_time:.0f} pts/s")
performance_log.append({
    'exercise': '14',
    'name': 'Poincaré Super (100k)',
    'periods': N14,
    'steps_per_period': steps14,
    'points': len(theta_p14),
    'time_sec': ex14_time,
    'memory_mb': ex14_memory,
    'pts_per_sec': len(theta_p14) / ex14_time
})
print("✓ Saved: ex14_poincare_super.pdf")

# ============================================================================
# TOPOLOGICAL ANALYSIS: CORRELATION DIMENSION & LYAPUNOV EXPONENTS
# ============================================================================
print("\n[TOPOLOGICAL ANALYSIS]")

def estimate_correlation_dimension(theta, omega, max_lag=50):
    """Estimate correlation dimension D2"""
    points = np.column_stack([theta, omega])
    if len(points) < 100:
        return np.nan
    distances = cdist(points, points, metric='euclidean')
    np.fill_diagonal(distances, np.inf)
    C_r = []
    r_vals = np.linspace(0.01, 2, 20)
    for r in r_vals:
        count = np.sum(distances < r)
        C_r.append(count)
    C_r = np.array(C_r)
    if np.any(C_r > 0):
        log_r = np.log(r_vals[C_r > 0])
        log_C = np.log(C_r[C_r > 0])
        if len(log_r) > 2:
            D2 = np.polyfit(log_r, log_C, 1)[0]
            return max(0, min(D2, 3.0))
    return np.nan

def compute_lyapunov_estimate(theta, omega):
    """Estimate largest Lyapunov exponent"""
    points = np.column_stack([theta, omega])
    N = len(points)
    if N < 2:
        return np.nan
    
    lyap_sum = 0
    count = 0
    for i in range(N-1):
        nearest_neighbor_dist = np.min(np.linalg.norm(points[i+1:] - points[i], axis=1))
        if nearest_neighbor_dist > 1e-8:
            lyap_sum += np.log(nearest_neighbor_dist)
            count += 1
    
    if count > 0:
        lambda_est = lyap_sum / (count * 0.1)  # Scale by dt approximation
        return lambda_est
    return np.nan

# Analysis for all three resolutions
resolutions = [
    ("Script 09 (50k periods, 100 steps/period)", theta_p_wrapped, omega_p),
    ("Script 12 (30k periods, 200 steps/period)", theta_p12_wrapped, omega_p12),
    ("Script 14 (100k periods, 250 steps/period)", theta_p14_wrapped, omega_p14)
]

print("\nTOPOLOGICAL PROPERTIES:")
print("-" * 70)
for name, theta, omega in resolutions:
    n_points = len(theta)
    D2 = estimate_correlation_dimension(theta, omega)
    lyap = compute_lyapunov_estimate(theta, omega)
    coverage = np.sum((theta > 1.3) & (theta < 1.8) & (omega > -0.4) & (omega < 0.3)) / n_points * 100
    
    print(f"{name}:")
    print(f"  Points: {n_points:,}")
    print(f"  D₂ (Correlation Dimension): {D2:.3f}")
    print(f"  λ (Lyapunov Exponent): {lyap:.4f}")
    print(f"  Focus region coverage: {coverage:.1f}%")

print("\nCOMPLETION STATUS:")
print("="*70)
print("✓ All 14 exercises completed successfully")
print("✓ All visualizations saved as PDF")
print("✓ Topological analysis complete")
print("="*70)

# ============================================================================
# SAVE PERFORMANCE METRICS
# ============================================================================
print("\n[PERFORMANCE SUMMARY]")
print("-" * 70)
print(f"{'Script':<30} {'Time (s)':<12} {'Memory (MB)':<15} {'Points':<12} {'Pts/Sec':<10}")
print("-" * 70)
for entry in performance_log:
    print(f"{entry['name']:<30} {entry['time_sec']:<12.2f} {entry['memory_mb']:<15.1f} {entry['points']:<12} {entry['pts_per_sec']:<10.0f}")
print("-" * 70)

# Save to both CSV and JSON
import csv
with open('performance_complete.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['exercise', 'name', 'periods', 'steps_per_period', 'points', 'time_sec', 'memory_mb', 'pts_per_sec'])
    writer.writeheader()
    writer.writerows(performance_log)

with open('performance_complete.json', 'w') as f:
    json.dump(performance_log, f, indent=2)

print("✓ Performance data saved to: performance_complete.csv")
print("✓ Performance data saved to: performance_complete.json")
