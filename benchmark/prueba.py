# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:19:19 2019

@author: Eduardo
"""

import numpy as np
from numpy.linalg   import pinv
from scipy.optimize import root
import matplotlib.pyplot as plt

def f_action(u_actual,p,pd_vaca,pd_hunt):
     
    Q = 5*((p)/(np.abs(p-u_actual[0])**3))#+((p)/(np.abs(p-u_actual[1])**3))
    M = (-5)/(np.abs(p-u_actual[0])**3)#,(-1)/(np.abs(p-u_actual[1])**3)])
#    MMt = M*M
#    K1  = 1/(MMt)
#    sol = u_actual + (1/M)*p + (1/M)*(Q+pd-p)
    sol = M*u_actual+Q+pd_vaca-pd_hunt

#    """ Penalty """
#    for i in range(2*N_hunters):
#         if (u_actual[i] > 4 or u_actual[i] < -4):
#             for k in range(len(sol)):
#                 sol[k] = sol[k] + (k+1)*10000 - np.sqrt(k+1)*1000
#             break 
    
    return sol

p = 0
pd_vaca = 2
pd_hunt = 1
h = 3
T = 0.01
k = 0

P = [p]
H = [h]

plt.close('all')

while (k<1000):
    
    pdot = 5*((p-h)/(np.abs(p-h)**3)) + (pd_vaca-p)
    if k<10:
        print(pdot)
    p += T*pdot#+((p-h2)/(np.abs(p-h2)**3)))
    u_actual = h #[h1,h2]
    u = root(f_action, u_actual, args = (p,pd_vaca,pd_hunt), method='lm',tol = 10e-10)
    if np.abs(u.x[0]-h)<0.2:
        h = u.x[0]
    else:
        h -= 0.02*np.sign(u.x[0])
         
    P.append(p)
    H.append(h)
    k += 1
   
    
plt.plot(P)
plt.show()
plt.plot(H)
plt.show()