# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 06:54:19 2020

@author: David F
"""
"""Q8.2"""
from numpy import *
from neutrons import neutrons
L=0.1
a=0.017
b=0.21
R=(2*a*b)**(1/2)
count=0

for i in range(100):
    position=L*random.random()
    for i in range(neutrons()): #range is neutrons rather than 2 as the number of secondary fissions varies in practice
        if (random.random()>0.5):
            direction = position+R
        else:
            direction = position-R
        if 0<=direction<=L:
            count +=1
            print(count, direction)
