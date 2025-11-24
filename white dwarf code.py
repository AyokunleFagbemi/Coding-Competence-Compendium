# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:49:24 2023

@author: David F
"""

import numpy as np
import matplotlib.pyplot as plt


# Define the system of differential equations
def model(r, z, R_O, M_O):
    x, y = z
    if y < 0:
        s = np.nan
    else:
        s = y ** (2 / 3) / (3 * np.sqrt(1 + y ** (2 / 3)))
    dx_dr = R_O * r ** 2 * y
    dy_dr = -M_O * x * y / (s * r ** 2)
    return np.array([dx_dr, dy_dr])


# Set up initial conditions and constants
r_start = 0.0001
r_end = 50
h = 0.001
M_O = 1
R_O = 4
z_start = np.array([0.001, 0.3])


# Define the RK4 method to solve the system of differential equations
def RK4_step(r, z, h, R_O, M_O, model):
    k1 = h * model(r, z, R_O, M_O)
    k2 = h * model(r + h / 2, z + k1 / 2, R_O, M_O)
    k3 = h * model(r + h / 2, z + k2 / 2, R_O, M_O)
    k4 = h * model(r + h, z + k3, R_O, M_O)
    return z + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# Perform the RK4 method to solve the system of differential equations
num_steps = int((r_end - r_start) / h)
r_vals = np.linspace(r_start, r_end, num_steps + 1)
z_vals = np.zeros((num_steps + 1, 2))
z_vals[0, :] = z_start

for i in range(num_steps):
    z_vals[i + 1, :] = RK4_step(r_vals[i], z_vals[i, :], h, R_O, M_O, model)

# Plot the results
plt.plot(r_vals, z_vals[:, 0], label='x')
plt.plot(r_vals, z_vals[:, 1], label='y')
plt.xlabel('r')
plt.legend()
plt.show()


