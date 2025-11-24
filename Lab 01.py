# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:15:18 2023

@author: David F
"""

import numpy as np
from matplotlib.pyplot import plot, show

x1 = np.arange(-10, 10, 1) #get values between -10 and 10 with 0.01 step and set to y
y1 = 0.5-x1 #get x values from y

x2 =  np.arange(-10, 10, 1) #get values between -10 and 10 with 0.01 step and set to y
y2 = 3*x2/2 #get x values from y


plot(x1, y1, x2, y2)
show()

x3 = np.arange(-10, 10, 1) #get values between -10 and 10 with 0.01 step and set to y
y3 = 1-x3 #get x values from y

x4 = np.arange(-10, 10, 1) #get values between -10 and 10 with 0.01 step and set to y
y4 = (x4**2)-5 #get x values from y

plot(x3, y3, x4, y4)
show()

x5 = np.arange(-4, 4, 0.5) #get values between -10 and 10 with 0.01 step and set to y
y5 = (9-(x5**2))**0.5 #get x values from y

x6 = np.arange(-4, 4, 0.5) #get values between -10 and 10 with 0.01 step and set to y
y6 = -(9-(x6**2))**0.5 #get x values from y

x7 = np.arange(-4, 2, 0.5) #get values between -4 and 10 with 0.01 step and set to y
y7 = (x7**2)+(3*x7)+1 #get x values from y

plot(x5, y5, x6, y6, x7, y7)
show()
