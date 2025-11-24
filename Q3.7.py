# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:01:35 2020

@author: David F
"""

from minimise import gmin
from discrepancy import discrepancy
print(gmin(discrepancy, 111, 112))

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
"""
ea=array(ea)
na=array(na)
dn=array(dn)


w=111.9999991303221
th=[]
#Calculates the theoretical values of na

def theory(ea, w):     
    x=1.467e7/(((w**2)/4)+(ea-1232)**2)
    th.append(x)
    return th[0]
print(theory(ea, w))   

plt.plot(ea, na)
plt.plot(ea, th[0])
#plt.errorbar(ea,na,dn)
plt.xlabel('Energy, E (MeV)')
plt.ylabel('No. scattered pions, n(E)')
plt.title('Best Fit Graph')
plt.savefig('Best Fit.png')

import os
curdir = os.getcwd()
print('Plot saved in:', curdir )
"""