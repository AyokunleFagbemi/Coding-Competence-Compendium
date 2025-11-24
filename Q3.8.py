# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 02:27:20 2020

@author: David F
"""

from numpy import array
from scipy.optimize import curve_fit

def theory(ea, w):
    na=1.467e7/(((w**2)/4)+(ea-1232)**2)
    return na

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

popt, pcov=curve_fit(theory, ea, na, None, dn)
# popt contains the optimal values for the parameters
print(popt)
# pcov contains the covariance matrix
print(pcov)

from numpy import sqrt , diag
print(sqrt(diag(pcov)))
