# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from math import cos , fabs , pi
def g(x):
    n=1; total=term=cos(x) # First term
    while fabs (term)>(1.0e-7*fabs(total) + 1.0e-13):
        n+=2 # Advance to next term
        term = cos (n*x)/(n*n)
        total+=term # Add term to total
    return total

import matplotlib. pyplot as plt
import numpy as np
x=np.arange(0,4*pi,pi/10) #creates an array of values starting from 0 and ending in 4π in steps of π/10
a=[]
"""creates an array of g(x) values corresponding to the x values"""
for i in x:
    a.append(g(i)) #appends g(x) values to array 'a'


plt.plot(a)
plt.xlabel ('x')
plt.ylabel('g(x)')
plt.title ('Fourier series')
plt.savefig('Fourier series.png')

import os
curdir = os.getcwd()
print('Plot saved in:', curdir )




 