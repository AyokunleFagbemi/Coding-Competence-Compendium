# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:15:47 2020

@author: David F
"""

"""Q8.3"""
from numpy import *
from neutrons import neutrons
L=0.1
a=0.017
b=0.21
R=(2*a*b)**(1/2)
count=0
while count != 100:
    count=0
    L+=0.00001
    for i in range(100):
        position=L*random.random()
        for i in range(neutrons()):
            if (random.random()>0.5):
                direction = position+R
            else:
                direction = position-R
            if 0<=direction<=L:
                count +=1
print(L)
