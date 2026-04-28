"""
Exercise 4: Pendulum Analysis - Period vs Initial Angle
Analyzes how pendulum period varies with amplitude
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from utils_and_functions import theta, Zero_cross_alg

# Analyze multiple initial angles
t_out, thetas_out, theta0_out, w_out = theta(10, -15)

# Plot theta vs time for different initial angles
for i in range(len(thetas_out)):
    plt.plot(t_out, np.rad2deg(thetas_out[i]), marker='.', markersize=0.5, linestyle='-', alpha=0.6, 
             label='theta0 = %.1f degrees' % np.rad2deg(theta0_out[i]))

plt.xlabel('t [s]')
plt.ylabel('$\\theta$ $^0$')
plt.legend(fontsize=5, loc=(1.05, 0.3))
plt.show()

# Plot phase space for different initial conditions
for j in range(len(w_out)):
    plt.plot(np.rad2deg(thetas_out[j]), w_out[j], color='k')

plt.xlabel('$\\theta$ $^0$')
plt.ylabel('w [rad/s]')
plt.show()

# Analyze period as function of initial angle
periods = []
theta_list = []

for i in np.arange(-(np.pi) + 0.01, np.pi - 0.01, 0.1):
    theta_list.append(i)
    periods.append(Zero_cross_alg(1, i))

plt.figure(figsize=(10, 6))
plt.plot(theta_list, periods, '.-')
plt.xlabel('$\\theta$ [rad]')
plt.ylabel('Mean Period (s)')
plt.show()
