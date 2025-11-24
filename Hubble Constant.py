# -*- coding: utf-8 -*-
"""
Created on Fri May 15 21:22:28 2020

@author: David F
"""
import numpy as np
import matplotlib.pyplot as plt 
import random

np.loadtxt('MW_Cepheids.dat')
"""
parallax, err(par), period, m, A, err(A) = np.loadtxt('MW_Cepheids(edit).dat',\
                                 unpack=True, \
                                 usecols=(1,2,3,4,5,6), \
                                 skiprows=2, \
                                 dtype='float')


print(parallax)
print(err(par))
print(period)
print(m)
"""