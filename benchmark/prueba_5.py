# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:50:29 2019

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
import matplotlib.pyplot as plt
import random

plt.close('all')

for j in range(10):
    p1 = random.uniform(-2,2)
    h1 = random.uniform(3,5)
    h2 = random.uniform(-5,-3)
    h3 = random.uniform(3,5)
    h4 = random.uniform(-5,-3)
    d11 = p1-h1
    d12 = p1-h2
    d13 = p1-h3
    d14 = p1-h4
    dd11 = np.sign(d11)*random.uniform(0,1)
    dd12 = np.sign(d12)*random.uniform(0,1)
    dd13 = np.sign(d13)*random.uniform(0,1)
    dd14 = np.sign(d14)*random.uniform(0,1)
    
    T = 0.1
    k = 0
    u = [0,0]
    
    D1 = []
    D2 = []
    D3 = []
    D4 = []
    DD1 = []
    DD2 = []
    DD3 = []
    DD4 = []
    
    while (k<300):
        
        d11 = p1-h1
        d12 = p1-h2
        d13 = p1-h3
        d14 = p1-h4
        p1 += T*((d11/(np.abs(d11)**3))+(d12/(np.abs(d12)**3))+(d13/(np.abs(d13)**3))+(d14/(np.abs(d14)**3)))
        
        D1.append(d11)
        D2.append(d12)
        D3.append(d13)
        D4.append(d14)
        DD1.append(dd11)
        DD2.append(dd12)
        DD3.append(dd13)
        DD4.append(dd14)
        
        h1 += T*((d11/(np.abs(d11)**3))+(d12/(np.abs(d12)**3))+(d13/(np.abs(d13)**3))+(d14/(np.abs(d14)**3))+(d11-dd11))
        h2 += T*((d11/(np.abs(d11)**3))+(d12/(np.abs(d12)**3))+(d13/(np.abs(d13)**3))+(d14/(np.abs(d14)**3))+(d12-dd12))
        h3 += T*((d11/(np.abs(d11)**3))+(d12/(np.abs(d12)**3))+(d13/(np.abs(d13)**3))+(d14/(np.abs(d14)**3))+(d13-dd13))
        h4 += T*((d11/(np.abs(d11)**3))+(d12/(np.abs(d12)**3))+(d13/(np.abs(d13)**3))+(d14/(np.abs(d14)**3))+(d14-dd14))
        
        k += 1
        
    fig = plt.figure()
    ax  = plt.axes()
    
       
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Distance evolution (m)')
    plt.plot(D1,'b')
    plt.plot(D2,'b')
    plt.plot(D3,'b')
    plt.plot(D4,'b')
    plt.plot(DD1,'r')
    plt.plot(DD2,'r')
    plt.plot(DD3,'r')
    plt.plot(DD4,'r')
    plt.show()
