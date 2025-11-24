# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:14:57 2022

@author: David F
"""
import numpy as np
from math import e
import matplotlib.pyplot as plt
import scipy 
options = [-2,5,0.05]

def MyRK3_step(f, tn, qn, dt, options):
    """
    Function defining the RHS of the system of equations.
    
    Parameters:
    ----------
    t: float
       time t^n of input data
    q: array of floats
       Input data q^n
    Returns:
    -------
    rhs: array of floats
         The RHS dy/dt at time t^n
    """
    k1 = f(tn,qn,options)
    k2 = f(tn + 0.5 * dt, qn + 0.5 * dt * k1, options)
    k3 = f(tn + dt, qn + dt * (-k1 + 2 * k2), options )
    qn = qn + (1/6)* dt * (k1 + 4 * k2 + k3)
    return qn

def rhs(t, q, options):
    """
    Function defining the RHS of the system of equations.
    
    Parameters:
    ----------
    t: float
       time t^n of input data
    q: array of floats
       Input data q^n
    options: list
        constants of the problems
    Returns:
    -------
    rhs: array of floats
         The RHS dy/dt at time t^n
    """
    #assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and\
    #np.all(np.isreal(t)) and np.isscalar(t) and t >= 0 ), \
    #"t must be real, finite and scalar"
    #assert((not np.any(np.isnan(q))) and np.all(np.isfinite(q))\
    #and np.all(np.isreal(q)) and len(q) == 2), \
    #"q must be real, finite, length 2"
    
    x, y = q
    gamma = options[0]
    omega = options[1]
    epsilon = options[2]
    
    A = np.array([[gamma,epsilon],[epsilon,-1]])
    B = np.array([[(-1+x**2-np.cos(t))/(2*x)],[(-2+y**2-np.cos(omega*t))/(2*y)]])
    C = np.array([[np.sin(t)/(2*x)], [omega*np.sin(omega*t)/(2*y)]])
    print(A)
    #term1 = -1000 * x
    #term2 = 1000*x-y
    
    rhs = A*B-C
    print(rhs)

t_end = 0.1
t_start = 0.0
dt = (t_end-t_start)/400

Nt = int(t_end/dt) +1
print(Nt)

t = np.linspace(0,t_end,Nt)

q_sol = np.zeros((2,Nt))

q0 = np.array([1,0])
q_sol[:,0] = q0

for i in range(1,Nt):
    q_sol[:,i] = MyRK3_step(rhs,t[i-i], q_sol[:,i-1], dt, options)
    

xvalues = q_sol[0,:]
yvalues = q_sol[1,:]

fig, axes = plt.subplots(1,2)
axes[0].semilogy(t,xvalues,)
axes[0].set(xlabel = r'$t$', ylabel = r'$x(t)$' )
axes[1].semilogy(t,yvalues)
axes[1].set(xlabel = r'$t$', ylabel = r'$y(t)$')
plt.tight_layout()

def MyGRRK3_step(f, t, qn, dt, options):
    """
    Function defining the RHS of the system of equations.
    
    Parameters:
    ----------
    t: float
       time t^n of input data
    q: array of floats
       Input data q^n
    Returns:
    -------
    rhs: array of floats
         The RHS dy/dt at time t^n
    """
    K = np.array([k1,k2])
    def F(K):     
        return K - np.array([f(tn + 0.33 * dt, qn + 0.083 * dt * (-k2 + 5 * k1)),f(tn + dt, qn + 0.25 * dt * (3*k1 + k2))])
    from scipy.optimize import fsolve
    print(fsolve(F, 0))
    #k1 = f(tn + 0.33 * dt, qn + 0.083 * dt * (-k2 + 5 * k1))
    #k2 = f(tn + dt, qn + 0.25 * dt * (3*k1 + k2))
    #qn+1 = qn + 0.25 * dt * (3*k1 + k2)