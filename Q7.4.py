# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 17:13:10 2020

@author: David F
"""

"""Q7.4"""
from numpy import zeros, fabs, append

def sweep(v,p,q,r,s,α):
  for i in range(1,len(v)-1):
    for j in range(1,len(v)-1):
      c=0.0
      if i==p and j==q: c=1.0
      if i==r and j==s: c=-1.0
      v[i,j]=(v[i-1,j]+v[i+1,j]+v[i,j-1]+v[i,j+1]-α*v[i,j]+c)/(4-α) 

N=22
α=1.1
v=zeros((N,N),float)
p=q=int((len(v)-1)/2)
r=p+1
s=q

R=[]

"""The next set of lines are used to calculate the voltage across node A and node B"""
dv=1.0e10
lastdv=0
count=0
while (fabs(dv-lastdv)>1.0e-7*fabs(dv)):
  lastdv=dv
  sweep(v,p,q,r,s,α)
  dv=v[p,q]-v[r,s]
  count+=1
R.append(dv)


"""The next set of lines are used to calculate the voltage acrpss node A and node C"""
t=p+2 #I changed the coordinates to node C to (t,u) in order to distinguish it from node B 

dv=1.0e10
lastdv=0
count=0
while (fabs(dv-lastdv)>1.0e-7*fabs(dv)):
  lastdv=dv
  sweep(v,p,q,t,s,α)
  dv=v[p,q]-v[t,s]
  count+=1
R.append(dv)

u=p+3 #I changed the coordinates to node C to (t,u) in order to distinguish it from node B 

dv=1.0e10
lastdv=0
count=0
while (fabs(dv-lastdv)>1.0e-7*fabs(dv)):
  lastdv=dv
  sweep(v,p,q,u,s,α)
  dv=v[p,q]-v[u,s]
  count+=1
R.append(dv)

u=p+4 #I changed the coordinates to node C to (t,u) in order to distinguish it from node B 

dv=1.0e10
lastdv=0
count=0
while (fabs(dv-lastdv)>1.0e-7*fabs(dv)):
  lastdv=dv
  sweep(v,p,q,u,s,α)
  dv=v[p,q]-v[u,s]
  count+=1
R.append(dv)

x=[1,2,3,4]
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8)) 
plt.plot(x,R)
plt.xlabel('x')
plt.ylabel('Resistance (Ω)')
plt.title('Resistances between various nodes')
plt.savefig('Q7.4.png')
