# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 05:23:31 2020

@author: David F
"""

"""Q8.1"""
from numpy import *
L=0.1 #for 8.1, L is arbitrary but I set it to 0.1m or 10cm because of Q8.3
a=0.017 #a is 1.7cm which is equal to 0.017m in standard SI units
b=0.21 #b is 21cm which is equal to 0.21m
R=(2*a*b)**(1/2)
count=0

for i in range(100): #for loop which simulates 100 initial fissions
    position=L*random.random()
    for i in range(2): #for loop which produces secondary neutrons
        if (random.random()>0.5):
            direction = position+R
        else:
            direction = position-R
        if 0<=direction<=L: #condition for secondary fission to take place
            count +=1
            print(count, direction)
