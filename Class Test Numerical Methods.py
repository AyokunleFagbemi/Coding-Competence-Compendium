# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print('Hello, World!')

a = 1
one_half = 0.5
a_very_small_number = 1.0e-10
some_condition = True
hello = 'Hello, world!'

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