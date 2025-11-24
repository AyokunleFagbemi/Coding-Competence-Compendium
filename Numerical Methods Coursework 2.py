# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:07:30 2021

@author: David F
"""


import numpy as np
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt

fx = 0
fg= 0.1


def rhs(s, q, fg, fx):
    dqds = np.zeros_like(q)
    theta = q[0]
    dthetads = q[1]
    dqds[0] = s*fg*np.cos(theta)
    dqds[1] = s*fg*np.cos(theta)+s*fx*np.sin(theta)
    dqds[2] = np.cos(theta) 
    dqds[3] = np.sin(theta)
    return dqds

s=0
q=np.array([1.0, 2.0])
rhs(s,q)

 
def shooting_error(theta_hat, z0, interval, bcs, fg, fx):
    q0 = [theta_0, theta_hat_0, x0, z0]
    args = (Fg, Fx)
    dense_output = True
    sol = scipy.integrate.solve_ivp(rhs, interval, R, fg, fx, q0, theta_0)
    z_end = sol.y[1, -1]
    err = z_end - bcs[1]
    return err

theta
     
def shooting(interval, bcs, N):
    x = np.linspace(interval[0], interval[1], N)
    z0 = scipy.optimize.newton(shooting_error, 1.0, args = (interval, bcs))
    # You can use brentq, fsolve, root
    q0 = np.array([bcs[0], z0])
    sol = scipy.integrate.solve_ivp(rhs, interval, q0, dense_output=True)
    return x, sol.sol(x)[0, :]
                           
                               
     
interval = np.array([0, 5.0])
bcs = np.array([10.0, 2.0])

# Solve the BVP
x, T = shooting(interval, bcs, 100)
 
# Plotting
plt.plot(x, T)
plt.xlabel(r'$x$')
plt.ylabel(r'$T$')
plt.title('T(x) - Shooting method')
plt.grid()

theta_0_all = np.linspace(0, np.pi, 20)
L = 4
R = 10
fg = 0.1
fx = 0

#2d head
head_theta = np.linspace(0, 2*np.pi, 100)
head_x = R * np.cos(head_theta)
head_z = R * np.sin(head_theta)
interval=[0,L] 

plt.plot(head_x, head_z, 'r-', lw=3)
for theta_0 in theta_0_all:
    # bcs = p[theta_0, 0]
    # s, theta = shooting(interval, bcs, 100)
    # plt.plot(s, theta)
    # plt.xlabel(r'$x$')
    # plt.ylabel(r'$theta$')
    #plt.title('theta(x) - Shooting method')
    # plt.grid()
    # plt.show()
    x, zc = hair_bvp_2d(L, R, fx, fg, theta_0)
    plt.plot(x, zc, 'k-', lw=1)
    # for x_hair, z_hair in zip(x, zc):
      # plt.plot(x_hair, z_hair, 'k-', lw=1)

plt.xlabel('x')
plt.ylabel('z')
plt.title('Task 2 - No wind')
plt.gca().set_aspect('equal')
plt.show()

theta_0_all = np.linspace(0, np.pi,20)
L = 4
R = 10
fg = 0.1
fx = 0.1

#2d head
head_theta = np.linspace(0, 2*np.pi, 100)
head_x = R * np.cos(head_theta)
head_z = R * np.sin(head_theta)
interval = [0,L]

plt.plot(head_x, head_z, 'r-', lw=3)
for theta_0 in theta_0_all:
    # bcs = [theta_0, 0]
    # s. theta = shooting(interval, bcs, 100)
    # plt.plot(s, theta)
    # plt.ylabel(r'$theta$')
    # plt.title('theta(x) - Shooting method')
    # plt.grid()
    # plt.show()
    x,zc = hair_bvp_2d(L, R, fx, fg, theta_0)
    plt.plot(x, zc, 'k-',lw=1)
    # fpr x_hair, z_hair in zip(x,zc):
    # plt.plot(x_hair, z_hair, 'k-', lw=1)
    
plt.xlabel('x')
plt.ylabel('z')
plt.title('Task 2 - wind')
plt.gca().set_aspect('equal')
plt.show

def hair_bvp_3d(L, R, fx, fg, theta_0,phi_0):
    
    bcs = [theta_0, 0,phi_0, 0]
    s, theta, phi,z0,p0 = shooting(interval, bcs, 100)
    x0 = R * np.cos(theta_0) * np.sin(phi_0)
    y0 = -R * np.cos(theta_0) * np.sin(phi_0)
    z1 = R * np.sin(theta_0)
    q1 = [theta_0, z0, x0, z1,phi_0, p0, y0]
    sol = scipy.integrate.solve_ivp(rhs_all, interval, q1, dense_output=True)
    return sol.sol(s)[2], sol.sol(s)[6], sol.sol(s) #x y z ycoordinates for each hair
