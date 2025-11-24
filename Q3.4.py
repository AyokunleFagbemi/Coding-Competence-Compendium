# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 21:25:24 2020

@author: David F
"""
from numpy import array
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
w=2*((1.467e7/na[0])-(ea[0]-1232)**2)**0.5
 #produces the minimum discrepancy
print(w)


"""function which calculates the discrepancy of the data and the theory"""
def discrepancy(w):
   for i in ea:
        while i<=1466: 
            ri=[]
            x=na-
            ri.append(x)
            d=(ri[0]/(dn))**2 #ri[0] is an array of the residuals
            return sum(d) #d is equal to the discrepancy

print(discrepancy(w))


