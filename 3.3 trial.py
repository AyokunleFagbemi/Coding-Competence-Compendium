# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 11:10:40 2020

@author: David F
"""

from numpy import array
import matplotlib.pyplot as plt
f=open('data1.txt','r') #reads in code for data1.txt file and assigns it to 'f'
ea=[]; na=[]; dn=[] #energy, no. scattered pions and error respectively
"""Loop which converts strings in data1.txt file into floats"""
for line in f:
    estr,nstr,errstr=line.split() #extracts the number strings
    ea.append(float(estr)) #converts energy strings into floats
    na.append( float(nstr))  #converts no. scattered pions strings into floats
    dn.append(float(errstr)) #converts error strings into floats
f.close() #closes file
ea=array(ea)
na=array(na)
dn=array(dn)
print(len(ea)) #22 elements in each array; the middle value would be the 11th
print()
w=(1.467*(4+(ea[11]-12.32)**2)/na[11])**0.5
print(w)


import numpy as np
th=np.zeros((22,1))
#Calculates the theoretical values of na
def theory(ea, w):
    for i in th:
        th[i]=1.467e7/(w**2/(4+(ea[i]-12.32)**2))
        th.append(th[i])
    print(th)
    

#print(theory(ea, w))
