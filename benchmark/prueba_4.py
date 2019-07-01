# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:26:47 2019

@author: Eduardo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:45:34 2019

@author: Eduardo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:19:19 2019

@author: Eduardo
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import random
from numpy.linalg   import pinv

plt.close('all')

for j in range(1):
    N = 2
    M = 1
    p = [random.uniform(-3,3) for j in range(M)]
    h = [0 for i in range(N)]
    for i in range(N):
        if i%2==0:
            h[i] = random.uniform(3,5)
        else:
            h[i] = random.uniform(-5,-3)
            
    d = [0 for k in range(M*N)]
    for j in range(M):
        for i in range(N):
            d[N*j+i] = p[j]-h[i]
    dd = [random.uniform(0,1)*np.sign(d[j]) for j in range(N*M)]
    
    print(d)
    print(dd)
    T = 0.01
    k = 0
    
    D = [[] for k in range(N*M)]
    DD = [[] for k in range(N*M)]
    
    while (k<50):
        
        start = time.time()
        for j in range(M):
            for i in range(N):
                d[N*j+i] = p[j]-h[i]
                D[N*j+i].append(d[N*j+i])
                DD[N*j+i].append(dd[N*j+i])
        
        for j in range(M):
            for i in range(N):
                p[j]+= T*(d[N*j+i]/(np.abs(d[N*j+i])**3))
            
        P = np.zeros([N*M,N])
        for i in range(N):
            for j in range(M):
                P[N*j+i,i] = 1                
                
        PPt = np.matrix(P)*np.matrix(np.transpose(P))
        K1 = np.inv(PPt)
        
        Q = np.zeros([N*M,1])
        for j in range(M):
            for i in range(N):
                Q[N*j+i] += (d[N*j+i]/(np.abs(d[N*j+i])**3))
                
        ddd  = np.transpose( np.matrix(d))
        dddd = np.transpose(np.matrix(dd))
        u_actual = np.transpose(P)*K1*Q + 10*np.transpose(P)*(ddd-dddd) #np.transpose(P)*K1*Q + np.transpose(P)*(ddd-dddd)
            
        for i in range(N):
            h[i] += T*u_actual[i,0]
        end = time.time()
        
        k += 1
        
    fig = plt.figure()
    ax  = plt.axes()
    
       
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Distance evolution (m)')
    
    for k in range(N*M):
        plt.plot(D[k],'b')
        plt.plot(DD[k],'r')
    plt.show()
