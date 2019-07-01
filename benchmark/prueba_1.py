# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:45:32 2019

@author: Eduardo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:19:19 2019

@author: Eduardo
"""

import numpy as np
from numpy.linalg   import pinv
from scipy.optimize import root
import matplotlib.pyplot as plt
import time
import random

plt.close('all')

def f_action(u_actual,p,pd):
     
    Q = ((p)/(np.abs(p-u_actual[0])**3))+((p)/(np.abs(p-u_actual[1])**3))
    Qd = ((pd)/(np.abs(p-u_actual[0])**3))+((pd)/(np.abs(p-u_actual[1])**3))
    M = np.matrix([(-1)/(np.abs(p-u_actual[0])**3),(-1)/(np.abs(p-u_actual[1])**3)])
    MMt =M*np.transpose(M)
    K1  = pinv(MMt,10e-4)[0,0]
    solPre = np.transpose(np.matrix([u_actual[0],u_actual[1]])) + np.transpose(M)*p + np.transpose(M)*K1*(Q+Qd)
    sol = [solPre[0,0],solPre[1,0]]
      
    return sol


for j in range(1):
    p = random.uniform(-2,2)
    pd = random.uniform(-1,1) + 0.5
    print(p)
    print(pd)
    h1 = 4
    h2 = -4
    T = 0.1
    k = 0
    
    P = []
    PD = []
    H1 = []
    H2 = []
    C=[]
    while (k<800):
        
        start = time.time()
        
        pdot = ((p-h1)/(np.abs(p-h1)**3)+((p-h2)/(np.abs(p-h2)**3)))
        p = p + T*pdot
        u_actual = [h1,h2]
        u = root(f_action, u_actual, args = (p,pd), method='lm',tol = 10e-15)
        
        if np.abs(u.x[0]-u_actual[0])>0.1:
            u.x[0] = u_actual[0]+0.01*np.sign(u.x[0]-u_actual[0])
        if np.abs(u.x[1]-u_actual[1])>0.1:
            u.x[1]=u_actual[1]+0.01*np.sign(u.x[1]-u_actual[1])
            
        if np.abs(p-pd)>0.1:
            h1 = u.x[0]
            h2 = u.x[1]
        else:
            
            h1 = pd + 1
            h2 = pd - 1


        P.append(p)
        PD.append(pd)
        H1.append(h1)
        H2.append(h2)
        end = time.time()
        C.append(end-start)
        k += 1
        
    
    
    fig = plt.figure()
    ax  = plt.axes()  
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Position evolution (m)')
    plt.plot(PD,'r')
    plt.plot(P,'g')
    plt.plot(H1,'b')
    plt.plot(H2,'b')
    plt.show()
    
    
    fig = plt.figure()
    ax  = plt.axes()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Computing time (s)')
    plt.plot(C)
    plt.show()