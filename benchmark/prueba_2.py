# -*- coding: utf-8 -*-
"""
Created on Mon May 13 19:45:33 2019

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

for j in range(10):
    p = random.uniform(0,3)
    h = random.uniform(3,5)
    d = p-h
    dd = -random.uniform(0,1)
    T = 0.1
    k = 0
    u = 0
    
    D = []
    DD = []
    C = []
    
    while (k<300):
        
        start = time.time()
        d = p-h
        p += T*((d/(np.abs(d)**3)))
        D.append(d)
        DD.append(dd)
        u_actual = (d/(np.abs(d)**3))+0.2*(d-dd)
        u = u_actual
        h += T*u
        end = time.time()
        C.append(end-start)
        k += 1
        
    fig = plt.figure()
    ax  = plt.axes()
    
       
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Distance evolution (m)')
    plt.plot(D,'b')
    plt.plot(DD,'r')
    plt.show()
    
