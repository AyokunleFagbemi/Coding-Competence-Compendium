# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 13:23:18 2022

@author: David F
"""

n=100000
for k in range(1 ,6):
    c=0
    for i in range(0,n):
        x=uniform (1)
        phi=pi * uniform (1)
        sx =0.5* sin(phi)
        if ((x<sx) or (1-x<sx )):
            c+=1
    print( k, float(c)/n, 2/pi)