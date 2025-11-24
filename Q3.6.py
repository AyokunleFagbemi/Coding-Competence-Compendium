# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 22:44:58 2020

@author: David F
"""

"""Q3.6"""

from numpy import exp #imports exponential
from minimise import gmin

a=0 #x=a
c=2 #x=c
def f(x):
    return exp(x)+1/x #returns x=b and f(b)


print(gmin(f, a, c))