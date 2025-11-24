# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:34:10 2023

@author: David F
"""

import numpy as np 
from numpy import cos as cos
from numpy import sin as sin
from numpy import log as log
import matplotlib.pyplot as plt


x = np.arange(0, 4, 0.000001) #Sets the range and resolution of the global variable x

def func1(x):
    
    return cos((log(x-1))/(log(x))) #Function that performs the calculation using the variable x

def func2(x):
    
    return cos((sin(x))/(cos(x))) #Function that performs the calculation using variable x 



def plotting(x, func, L_x, Xlow, Xup, Ylow, Yup, pos):
    plt.xlim(Xlow,Xup)
    plt.ylim(Ylow,Yup)
    plt.subplot(2,1,pos)
    plt.plot(x, func, label = L_x)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    
 
    return 
'''Plotting takes in the variable x, the functions defined above, the legend label, X lower and upper limits, Y upper and lower limits then the position of the subplot'''
        

plt.figure(dpi = 300) #Sets the resolution of the plot
plotting(x, func1(x), 'Func1', -1, 1, 1, 4, 1)#Plots function 1 
plotting(x, func2(x), 'Func2', 0, 4, -1, 1, 2)#plots function 2 


plt.subplots_adjust(hspace = 0.4, wspace = 0.5, top = 5, bottom = 3.5)#Adjusts the subplots space width and the height then the top and bottom lengths
    