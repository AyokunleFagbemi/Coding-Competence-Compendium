# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:59:51 2024

@author: David F
"""

import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize

def hair_bvp_2d(theta_0_list, L, R, f_x, f_g=0.1):
    """
    Solving the BVP to model hair.
   
    Parameters
    ----------
   
    theta_0_list : vector, contains N_hairs elements
        Angles at which the N_hairs hairs meet the head.
    L : scalar
        Length of the hairs (all the same)
    R : scalar
        Radius of the head
    f_x : scalar
        Force due to wind (parameter)
    f_g : scalar
        Force due to gravity (parameter), default is 0.1
       
    Returns
    -------
   
    x : vector
        x coordinates of the hairs
    z : vector
        z coordinates of the hairs
    """
   
    # Algorithm choice and reasons:
    # The solve_bvp algorithm from scipy.integrate is chosen for its suitability in solving boundary value problems (BVPs).
    # This problem involves hair dynamics with specified angles and forces, making BVPs an appropriate approach.

    def hair_ode(s, y, L, f_x, f_g):
        # ODE system describing hair motion
        theta, dtheta_ds = y
        ds = s / (L - s)  # Arc length parameterization
        dy_ds = [dtheta_ds, s * f_g * np.cos(theta) + s * f_x * np.sin(theta)]
        return dy_ds

    def hair_bc(ya, yb, p):
        # Boundary conditions
        L, f_x = p
        return [ya[0] - theta_0, yb[1] - L * f_x]
 
    x = []
    z = []
 
    for theta_0 in theta_0_list:
        # Initial guess for the solution
        s_values = np.linspace(0, L, 100)
        initial_guess = np.zeros((2, len(s_values)))
        initial_guess[0] = theta_0
 
        # Solving the boundary value problem
        from scipy.integrate import solve_bvp
        solution = solve_bvp(fun=lambda s, y: hair_ode(s, y, L, f_x, f_g),
                             bc=lambda ya, yb: hair_bc(ya, yb, (L, f_x)),
                             x=np.linspace(0, L, 100),
                             y=initial_guess)
        # Extracting the solution
        s_values = np.linspace(0, L, 100)
        theta_values = solution.sol(s_values)[0]
 
        # Convert polar coordinates to Cartesian coordinates
        x_values = R * np.cos(theta_0) + np.cumsum(np.cos(theta_values) * (s_values[1] - s_values[0]))
        z_values = R * np.sin(theta_0) + np.cumsum(np.sin(theta_values) * (s_values[1] - s_values[0]))
 
        x.append(x_values)
        z.append(z_values)
 
    return x, z

#Advantages:
#Accuracy: solve_bvp provides accurate solutions to BVPs, ensuring reliable results for hair simulation. 
#Adaptability: It handles complex boundary conditions, crucial for modeling hair motion with specified angles and forces.
#Disadvantages:
#Computational Cost: Solving BVPs can be computationally expensive, especially for large-scale simulations. 
#Sensitivity to Initial Guess: Convergence may depend on the choice of the initial guess, impacting the reliability of the solution.
#Alternative Methods:
#Finite Difference Methods: Finite difference schemes can be used, but they may require fine grids and careful handling of boundary conditions. 
#Shooting Methods: Solving the problem as an initial value problem using shooting methods is an alternative, but it may require additional effort in handling boundary conditions.


if __name__ == "__main__":
   
    L = 4   # Hair length [cm]
    R = 10  # Sphere radius [cm]
   
    # Call hair_bvp_2d and plot the location of 20 different hairs
    # with and without the wind (Tasks 2 and 3).
   
    # Task 2: Plot hairs with no wind force (fx = 0)
    f_x = 0
    theta_0_list = np.linspace(0, np.pi, 20)
 
    x_coords, z_coords = hair_bvp_2d(theta_0_list, L, R, f_x, f_g=0.1)
 
    plt.figure(figsize=(8, 6))
    for x, z in zip(x_coords, z_coords):
        plt.plot(x, z)
 
    plt.title('Hair Simulation with No Wind Force (fx = 0)')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()
 
    # Task 3: Plot hairs with wind force (fx = 0.1)
    f_x = 0.1
   
    x_coords, z_coords = hair_bvp_2d(theta_0_list, L, R, f_x, f_g=0.1)
   
    plt.figure(figsize=(8, 6))
    for x, z in zip(x_coords, z_coords):
        plt.plot(x, z)
 
    plt.title('Hair Simulation with Wind Force (fx = 0.1)')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()


