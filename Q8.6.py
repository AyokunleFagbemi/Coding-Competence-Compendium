# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:49:19 2020

@author: David F
"""
"""Q8.4"""
from numpy import *
from neutrons import neutrons, diffusion
L=0.1
a=0.017
b=0.21
R=(2*a*b)**(1/2)
count=0
phi = 2.0* pi * random . random ()
theta = arccos(2.0* random . random () -1.0)
x=L*random.random()
y=L*random.random()
z=L*random.random()

for i in range(100):
    position=diffusion()*R
    xprime=x+diffusion()*R*sin(theta)*cos(phi)
    yprime=y+diffusion()*R*sin(theta)*cos(phi)
    zprime=z+diffusion()*R*cos(theta)
    for i in range(neutrons()): #range is neutrons rather than 2 as the number of secondary fissions varies in practice
        if (random.random()>0.5):
            direction = position+R
        else:
            direction = position-R
        if 0<=direction<=L:
            count +=1
            print(count, direction)
