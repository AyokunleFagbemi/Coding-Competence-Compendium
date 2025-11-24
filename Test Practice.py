# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 19:57:09 2021

@author: David F
"""

print('Hello, World!')
"""Check you know the difference between typing commands into the
console (where they run immediately) and into the editor (where
they can be saved to a file and then run, maybe many times).
Ensure you can save your script to a py-file. For the Class Test you
need to submit a file."""


a = 1
one_half = 0.5
a_very_small_number = 1.0e-10
some_condition = True
hello = 'Hello, world!'

"""Rules for PYTHON variables:
• A variable name must start with a letter or the underscore
character (_).
• A variable name cannot start with a number.
• A variable name can only contain alpha-numeric characters and
underscores (A-z, 0-9, and _).
• Variable names are case-sensitive (age, Age and AGE are three
different variables)."""


print(1/3)
print((1.2 + 3.4 - (5.6 * 7.8)**(9/11)) / 12)

import numpy as np

x = np.array([1, 2, 5])

print(x)
print(len(x))
print(np.sum(x))
print(np.sin(np.pi * x)**2)

print(np.arange(1, 11, 1))
print(np.zeros(5))
print(np.ones(5) * 6)
print(np.linspace(0, 1, 11))

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(A.size)
print(A.shape)
print(A)
print(np.transpose(A))
print(np.dot(A, x)) # or print(A @ x)
print(A * A) # Element-wise multiplication!

A = np.array([[2, 3], [6, 5]])
print(np.linalg.eig(A))

evals, evecs = np.linalg.eig(A)
first_evec = evecs[:, 0]
print(first_evec)

x, dx = np.linspace(0, 1, 11, retstep=True)
print('x =', x)
print('dx =', dx)

print('the last element: ', x[-1])
print('elements 1, 2, 3: ', x[1:4])
print('last 3 elements: ', x[-3:])
print('every 2nd element:', x[::2])

i = 1
a = np.array([1, 2])

print(f'Element {i} of array {a} is {a[i]}.')

x = 12
y = 15

x = 12
y = 15

if x == y:
    print(f'x={x} is equal to y={y}')
elif x > y:
    print(f'x={x} is bigger than y={y}')
elif x < y:
    print(f'x={x} is less than y={y}')
else:
    print('This should never happen.')
    
x = np.arange(0, 5)
i = 0

while i < 5:
    print(f'x[{i}] = {x[i]}')
    i += 1 # This is equivalent to i = i + 1
    
x = np.arange(0, 5)

for i in range(0, 5, 1):
    print(f'x[{i}] = {x[i]}')
    
def gaussian(x, mu=0, sigma=1):
    coef = 1 / (sigma * np.sqrt(2 * np.pi))
    return coef * np.exp(-0.5 * ((x - mu) / sigma)**2)

x, dx = np.linspace(-10, 10, 100, retstep=True)

print(sum(gaussian(x)) * dx) # A bit rough but ok
print(sum(gaussian(x, mu=1.0, sigma=0.5)) * dx)

def gaussian(x, mu=0, sigma=1):
    '''
    Gaussian distribution function.

    Parameters
    ----------
    x : float or array
    Argument(s) of the function.
    mu : float, optional
    The mean (or expectation) of the distribution.
    sigma : float, optional
    The standard deviation of the distribution.
    
    Returns
    -------
    float or array
    Gaussian distribution function value(s) at point(s) x.

    '''
    coef = 1 / (sigma * np.sqrt(2 * np.pi))
    return coef * np.exp(-0.5 * ((x - mu) / sigma)**2)

import matplotlib.pyplot as plt
def gaussian(x, mu=0, sigma=1):
    coef = 1 / (sigma * np.sqrt(2 * np.pi))
    return coef * np.exp(-0.5 * ((x - mu) / sigma)**2)

x = np.linspace(-5, 5, 100)
y = gaussian(x)

plt.plot(x, y) # NOTE: x and y should be the same size
plt.show()

plt.plot(x, y, '.-', color='blue') # Markers and colour
plt.grid() # Grid lines
plt.title('Gaussian function') # Title
plt.xlabel(r'$x$') # Axis label
plt.ylabel(r'$f~(x, \mu=0, \sigma=1)$') # TeX notation!
plt.xlim(-5, 5) # Axis limits
plt.ylim(0) # Bottom limit
plt.show()

x = np.linspace(-5, 5, 100)
y1 = gaussian(x)
y2 = gaussian(x, 2.0, 0.5)
plt.grid() # Grid lines
plt.plot(x, y1, '.-', label='Normal distribution')
plt.plot(x, y2, 'x-', label=r'$\mu=2, \sigma=0.5$')
plt.legend() # Shows the legend
plt.show()

plt.subplot(1, 2, 1) # 1x2 grid, 1st plot
plt.plot(x, y1, '.-', color='red')
plt.title('Normal distribution')
plt.xlabel(r'$x$')
plt.ylabel(r'$f~(x)$')
plt.xlim(-5, 5)
plt.ylim(0, 0.85)
plt.grid()

plt.subplot(1, 2, 2) # 1x2 grid, 2nd plot
plt.plot(x, y2, 'x-', color='blue')
plt.title(r'$\mu=2, \sigma=0.5$')
plt.xlabel(r'$x$')
plt.xlim(-5, 5)
plt.ylim(0, 0.85)
plt.grid()

plt.show()

fig, axs = plt.subplots(1, 2) # 1x2 grid

axs[0].plot(x, y1, ls='-', lw='1', color='red', marker='.')
axs[0].set_title('Normal distribution')
axs[0].set_xlabel(r'$x$')
axs[0].set_ylabel(r'$f~(x)$')
axs[0].set_xlim(-5, 5)
axs[0].set_ylim(0, 0.85)
axs[0].grid()

axs[1].plot(x, y2, ls='-', lw='1', color='blue', marker='x')
axs[1].set_title(r'$\mu=2, \sigma=0.5$')
axs[1].set_xlabel(r'$x$')
axs[1].set_xlim(-5, 5)
axs[1].set_ylim(0, 0.85)
axs[1].grid()

plt.show()

plt.semilogy(x, y1, '.-', color='red') # Log-scale for Y
plt.title('Normal distribution in log-scale')
plt.xlabel(r'$x$')
plt.ylabel(r'$f~(x)$')
plt.xlim(-5, 5)
plt.grid()
plt.show()

n = np.arange(1, 11)
N = 2.0**n # Number of mesh points
tol = 1 / N # Mimics tolerance
err1 = 1 / N # Mimics numerical error 1
err2 = 1 / N**2 # Numerical error 2

plt.loglog(N, err1, 'o-', label='I order method')
plt.loglog(N, err2, 'v-', label='II order method')
plt.title("Convergence plot example")
plt.xlabel(r'$N$')
plt.ylabel('Error')
plt.grid()
plt.legend(loc='upper right') # Position of the legend
plt.show()

x = np.linspace(0, 1, 100)

plt.figure(1)
plt.plot(x, x) # Figure 1
plt.figure(2)
plt.plot(x, x*x) # Figure 2
plt.figure(3)
plt.plot(x, np.sqrt(x)) # Figure 3
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize

r = np.random.rand(5) # 5 random numbers
A = np.random.rand(3, 3) # 3x3 matrix of random numbers
u = np.random.randn(10) # 10 normally distributed numbers

print(np.trace(A)) # Sum of the diagonal elements
print(np.linalg.det(A)) # Determinant of A
print(np.linalg.inv(A)) # Inverse of A
print(np.linalg.eig(A)) # Eigenvalues and eigenvectors

eigvals, eigvecs = np.linalg.eig(A) # Unpacking
print(eigvecs[:, 0]) # Slicing

def integrand(x):
    return x**2

I, err = scipy.integrate.quad(integrand, 0, 1)
print(f"Integral = {I}, error = {err}")

def f(x):
    return np.cos(x) - x

print(scipy.optimize.brentq(f, 0, 1))

sol = scipy.optimize.root(f, 0)
print(sol)
print(sol.x)

def f(x):
    return np.cos(x) - x

print(scipy.optimize.brentq(f, 0, 1))

sol = scipy.optimize.root(f, 0)
print(sol)
print(sol.x)

def rhs(t, y):
    return -y

sol = scipy.integrate.solve_ivp(rhs, [0, 10], [1],
dense_output=True)
t = np.linspace(0, 10, 500)
y = sol.sol(t)
plt.plot(t, y[0, :])
plt.show()

def integrand2(s):
    return np.sin(s)**2

def rhs2(t, y):
    quadrature = scipy.integrate.quad(integrand2, 0, t)
    C = 1 + quadrature[0]
    return -C * y

sol = scipy.integrate.solve_ivp(rhs2, [0, 10], [1],
dense_output=True)

def rhs3(t, q):
    x, y = q # Unpacking of q
    rhs = np.zeros_like(q) # Creates vector rhs
    rhs[0] = -y
    rhs[1] = x
    return rhs

sol = scipy.integrate.solve_ivp(rhs3, [0, 5], [1, 0],
                                
dense_output=True)
t = np.linspace(0, 5, 500)
q = sol.sol(t)
x = q[0, :] # Use plt.plot(t, x),
y = q[1, :] # and plt.plot(t, y) to plot both x and y

A = np.array([[1,1,-1],[2,2,4],[1,4,-10]])
b = np.array([2,2,-1])
sol = np.linalg.solve(A,b)
print(sol)
"""
def integrand (x:
    return (np.exp(-2*x**2))*(np.tan(x/1000)/(1+x**2))
integral, error = scipy.integrate.quad(integrand, 0,4)
return integral,error
"""

def f(x):
    return (x**2)*np.exp(-x)
grid = np.linspace(0,10,1000)
f_values = f(grid)
plt.plot(grid,f_values)
plt.xlabel(r"$x$")
plt.ylabel(r"$f(x) = x^2 \exp(-x)$")
plt.title("Plot of the function $f(x) = x^2 \exp(-x)$ between 0 and 10")
plt.show
