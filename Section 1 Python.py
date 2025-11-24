# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:11:19 2020

@author: David F
"""
#Q1.1

a=[3,54,26,90]
b=[3,54,'llama',a,3.5]
print(b[-3][3])
print()
print(a[-0])

print()

#Q1.2
fac=1;i=1
while i<10:
    fac=fac*i
    print(i, fac)
    i=i+1

    
#Q1.5
def square_root(x):   
    y=x
    while not y*y<= x+1e-15:
        y=0.5*(y+x/y)
        print(y, y*y)
    #return y
    print(y, "is the square root of", x)

square_root(4)

#Q1.6
from math import sqrt

def a(x):
    return x - sqrt (x*x -1)

def b(x):
    return 1/( x+ sqrt (x*x -1))

number=[150000]
for i in number:
    while i<150010:
        print([a(i), b(i)])
        i = i + 1
 
    





