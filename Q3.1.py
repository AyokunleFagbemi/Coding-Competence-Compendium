# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 07:52:33 2020

@author: David F
"""

from numpy import array
import matplotlib. pyplot as plt
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

plt. plot(ea, na)
plt.errorbar(ea,na,dn)
plt.xlabel('Energy, E (MeV)')
plt.ylabel('No. scattered pions, n(E)')
plt.title('Energy against No.scattered pions')
plt.savefig('section3graph.png')

import os
curdir = os.getcwd()
print('Plot saved in:', curdir )
