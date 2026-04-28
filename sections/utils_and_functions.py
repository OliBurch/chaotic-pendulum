"""
Utility functions and ODE definitions for nonlinear dynamical systems
RK4 numerical solver and all system ODEs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def RK4(x,y,f,h):
  k1 = h*f(x,y)
  k2 = h*f(x+0.5*h, y + 0.5*k1)
  k3 = h*f(x+0.5*h, y + 0.5*k2)
  k4 = h*f(x+h, y + k3)
  return y + (k1 + 2*k2 + 2*k3 + k4)/6

def Mass_on_spring(t,x_v):
  x,v = x_v
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

def Pendulum(t,theta_w): #Define ODE
  theta,w = theta_w
  l = 9.81
  g = 9.81
  dthetadt = theta_w[1]
  dwdt = -g/l * np.sin(theta_w[0])
  return np.array([dthetadt, dwdt])

def theta(N0_inputs, initial_theta):
    all_thetas = []
    all_w = []
    theta0 = np.zeros(N0_inputs)

    for i in range(N0_inputs):

        theta0[i] = np.deg2rad(((i**2) + initial_theta))

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

        for j in range(n-1): # Perform RK4 on ODE
            x_v_current = np.array([theta_i[j], w_i[j]])
            x_v_next = RK4(t[j], x_v_current, Pendulum, h)
            theta_i[j+1] = x_v_next[0]
            w_i[j+1] = x_v_next[1]
            t[j+1] = t[j] + h

        all_thetas.append(theta_i)
        all_w.append(w_i)

    return t, all_thetas, theta0, all_w

def theta_single(N0_inputs, initial_theta):
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

        for j in range(n-1): # Perform RK4
            x_v_current = np.array([theta_i[j], w_i[j]])
            x_v_next = RK4(t[j], x_v_current, Pendulum, h)
            theta_i[j+1] = x_v_next[0]
            w_i[j+1] = x_v_next[1]
            t[j+1] = t[j] + h

        all_thetas.append(theta_i)

    return t, all_thetas, theta0


def Zero_cross_alg(N0_inputs,theta_init):
  t_out, thetas_out, theta0_out = theta_single(N0_inputs,theta_init)

  zero_crossing_times = []

  for i in range(1, len(thetas_out[0])):
    if np.sign(thetas_out[0][i-1]) != np.sign(thetas_out[0][i]):
       zero_crossing_times.append(t_out[i])

  period_differences = []
  for j in range(len(zero_crossing_times) - 1):
    period_differences.append(zero_crossing_times[j+1] - zero_crossing_times[j])

  period = np.mean(period_differences) * 2

  return period

def omega(N0_inputs, initial_omega):
    all_omega = []
    all_theta = []
    omega0 = np.zeros(N0_inputs)

    for i in range(N0_inputs):

        omega0[i] = (i + initial_omega)*0.3

        t0 = 0

        theta0 = 0

        tend = 50

        h = 0.1

        n = int((tend - t0) / h)

        theta_i = np.zeros(n)

        w_i = np.zeros(n)

        t = np.zeros(n)

        t[0] = t0

        theta_i[0] = theta0

        w_i[0] = omega0[i]

        for j in range(n-1): # Perform RK4 on ODE
            x_v_current = np.array([theta_i[j], w_i[j]])
            x_v_next = RK4(t[j], x_v_current, Pendulum, h)
            theta_i[j+1] = x_v_next[0]
            w_i[j+1] = x_v_next[1]
            t[j+1] = t[j] + h

            if theta_i[j+1] > np.pi:
              theta_i[j+1] = theta_i[j+1] - 2*np.pi
            if theta_i[j+1] < -np.pi:
              theta_i[j+1] = theta_i[j+1] + 2*np.pi

        all_omega.append(w_i)
        all_theta.append(theta_i)

    return t, all_omega, all_theta, omega0

def Pendulum_vector_field(theta_val, w_val):
  l = 9.81
  g = 9.81
  dthetadt = w_val
  dwdt = -g/l * np.sin(theta_val)
  return dthetadt, dwdt

def DampedPendulum(t,theta_w,B):
  theta,w = theta_w
  l = 9.81
  g = 9.81
  B = B
  dthetadt = theta_w[1]
  dwdt = -g/l * np.sin(theta_w[0]) -B*theta_w[1]
  return np.array([dthetadt, dwdt])

def DampedODE_vector_field(theta_val, w_val, B):
    l = 9.81
    g = 9.81
    dthetadt = w_val
    dwdt = -g/l * np.sin(theta_val) - B*w_val
    return dthetadt, dwdt

def damped_pendulum(initial_theta, beta_val):
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
            theta_i[j+1] = theta_i[j+1] - 2*np.pi
        if theta_i[j+1] < -np.pi:
            theta_i[j+1] = theta_i[j+1] + 2*np.pi

    return t, w_i, theta_i

def DrivenDampedPendulum(t,theta_w,B,A): #Define ODE
  theta,w = theta_w
  l = 9.81
  g = 9.81
  B = B
  A = A
  Wd = 0.67
  dthetadt = theta_w[1]
  dwdt = -(g/l) * np.sin(theta_w[0]) -B*theta_w[1] + A*np.cos(Wd*t)
  return np.array([dthetadt, dwdt])

def driven_damped_pendulum(N0_inputs, initial_omega, A):
    all_omega = []
    all_theta = []
    omega0 = np.zeros(N0_inputs)

    for i in range(N0_inputs):

        omega0[i] = (i**2 + initial_omega)*0.3

        t0 = 0

        theta0 = 0

        tend = 500 # Changed from 50 to 200

        h = 0.1

        n = int((tend - t0) / h)

        theta_i = np.zeros(n)

        w_i = np.zeros(n)

        t = np.zeros(n)

        t[0] = t0

        theta_i[0] = theta0

        w_i[0] = omega0[i]

        for j in range(n-1): # Perform RK4 on ODE
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
  l = 9.81
  g = 9.81
  Wd = 0.67
  dthetadt = w_val
  dwdt = -g/l * np.sin(theta_val) - B*w_val + A*np.cos(Wd*t_val)
  return dthetadt, dwdt
