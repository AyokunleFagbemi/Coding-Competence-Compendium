# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 01:18:39 2022

@author: David F
"""

import matplotlib .pyplot as plt # for plotting
from math import cos
import numpy as np

def f(x):
    """Sums Fourier series"""
    sum=0.0
    term =1.0
    n=1.0
    while abs(term) > 1e-5*(abs(sum+1e-4)):
        term=cos(n * x)/(n * n)
        sum=sum+term
        n = n + 2
    return sum

#####
    
n=40
g=[f((4* np.pi * i)/n) for i in range(0,n)]
# plotting
fig = plt.figure ()
ax = fig.add_axes ([0 ,0 ,1 ,1])
line = ax.plot(g)
plt.setp(line)
plt.show ()
