
    
from numpy import *
N=array([7,12,17,22])
R_ab=[0.48296421806607115, 0.49540056465742344, 0.49782997170379595, 0.49875299630324765]
R_ac=[0.6032925128733665, 0.6275177788944413, 0.632302019211515, 0.6341351594545814]
x=1/N**2
import matplotlib.pyplot as plt
plt.figure(figsize=(8,8)) 
plt.plot(x,R_ab)
plt.plot(x,R_ac)
plt.xlabel('1/N^2')
plt.ylabel('Resistance (Ω)')
plt.title('Resistances between nodes A,B (blue line) and A,C (orange line)')
plt.savefig('Q7.2.png')

