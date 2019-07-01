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
from scipy.optimize import root
import time
import matplotlib.pyplot as plt
import random

plt.close('all')

for j in range(1):
    p = random.uniform(0,3)
    h1 = random.uniform(3,5)
    h2 = random.uniform(-3,0)
    d1 = p-h1
    d2 = p-h2
    print(d1)
    print(d2)
    dd1 = -random.uniform(0,1)
    dd2 = random.uniform(0,1)
    print(dd1)
    print(dd2)
    T = 0.1
    k = 0
    u = [0,0]
    
    D1 = []
    D2 = []
    DD1 = []
    DD2 = []
    C = []
    
    while (k<300):
        
        start = time.time()
        d1 = p-h1
        d2 = p-h2
        p += T*((d1/(np.abs(d1)**3))+(d2/(np.abs(d2)**3)))
        D1.append(d1)
        D2.append(d2)
        DD1.append(dd1)
        DD2.append(dd2)
        u = [(d1/(np.abs(d1)**3))+(d2/(np.abs(d2)**3))+(d1-dd1),(d1/(np.abs(d1)**3))+(d2/(np.abs(d2)**3))+(d2-dd2)]
        h1 += T*u[0]
        h2 += T*u[1]
        end = time.time()
        C.append(end-start)
        k += 1
        
    fig = plt.figure()
    ax  = plt.axes()
    
       
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Distance evolution (m)')
    plt.plot(D1,'b')
    plt.plot(D2,'b')
    plt.plot(DD1,'r')
    plt.plot(DD2,'r')
    plt.show()
