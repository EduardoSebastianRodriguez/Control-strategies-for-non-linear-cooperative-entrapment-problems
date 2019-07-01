# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 08:23:18 2019

@author: Eduardo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 09:54:54 2019

@author: Eduardo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:00:40 2019

@author: Eduardo
"""


import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from numpy.linalg   import norm
from sklearn.preprocessing import normalize

def f_action(u_actual,p,pd_hunt):
     
    W = np.zeros([2,2])
    W[0,0] = 0.02
    W[0,1] = -0.002
    W[1,0] = -0.007
    W[1,1] = 0.06
    
    h1 =np.matrix([u_actual[0],u_actual[1]])
    h2 =np.matrix([u_actual[2],u_actual[3]])

    X1 = (p[0,0]-h1[0,0])**2+(p[0,1]-h1[0,1])**2
    X2 = (p[0,0]-h2[0,0])**2+(p[0,1]-h2[0,1])**2
    if (norm(p-h1)<1.0 or norm(p-h2)<1.0):
         Q = 1.5*(p)*np.exp(-(1/(1**2))*X1) + 1.5*(p)*np.exp(-(1/(1**2))*X2)
         Qd = 1.5*(pd_hunt)*np.exp(-(1/(1**2))*X1) + 1.5*(pd_hunt)*np.exp(-(1/(1**2))*X2)
         M = np.matrix([[-1.5*np.exp(-(1/(1**2))*X1),0,-1.5*np.exp(-(1/(1**2))*X2),0],[0,-1.5*np.exp(-(1/(1**2))*X1),0,-1.5*np.exp(-(1/(1**2))*X2)]])
    else:
         Q = 0.5*1.5*(p)*np.exp(-(1/(1**2))*X1) + 0.5*1.5*(p)*np.exp(-(1/(1**2))*X2)
         Qd = 0.5*1.5*(pd_hunt)*np.exp(-(1/(1**2))*X1) + 0.5*1.5*(pd_hunt)*np.exp(-(1/(1**2))*X2)
         M = np.matrix([[-0.5*1.5*np.exp(-(1/(1**2))*X1),0,-0.5*1.5*np.exp(-(1/(1**2))*X2),0],[0,-0.5*1.5*np.exp(-(1/(1**2))*X1),0,-0.5*1.5*np.exp(-(1/(1**2))*X2)]])
         
    solX = np.transpose(Q) + np.transpose(Qd) + M*np.transpose(np.matrix([u_actual[0],u_actual[1],u_actual[2],u_actual[3]])) + 2*(np.transpose(p) - np.transpose(pd_hunt)) +  W*(np.transpose(p)+np.transpose(pd_hunt))  
    sol = [solX[0,0],solX[1,0],0,0]
    
    return sol

p = np.matrix([0.0,0.0])
pd_hunt = np.matrix([1,1])
h1 = np.matrix([0,-3])
h2 = np.matrix([3,0])
T = 0.1
k = 0
W = np.zeros([2,2])
W[0,0] = 0.02
W[0,1] = -0.002
W[1,0] = -0.007
W[1,1] = 0.06

PX = [p[0,0]]
PY = [p[0,1]]
H1X = [h1[0,0]]
H1Y = [h1[0,1]]
H2X = [h2[0,0]]
H2Y = [h2[0,1]]
plt.close('all')

while (k<5000):
    
    X1 = (p[0,0]-h1[0,0])**2+(p[0,1]-h1[0,1])**2
    X2 = (p[0,0]-h2[0,0])**2+(p[0,1]-h2[0,1])**2
    if norm(p-h1)<1.0 or norm(p-h2)<1.0:
        pdot = 1.5*(p-h1)*np.exp(-(1/(1**2))*X1) + 1.5*(p-h2)*np.exp(-(1/(1**2))*X2) + np.matrix([W[0,0]*p[0,0]+W[0,1]*p[0,1],W[1,0]*p[0,0]+W[1,1]*p[0,1]])
    else:
        pdot = 0.5*1.5*(p-h1)*np.exp(-(1/(1**2))*X1) + 0.5*1.5*(p-h2)*np.exp(-(1/(1**2))*X2) + np.matrix([W[0,0]*p[0,0]+W[0,1]*p[0,1],W[1,0]*p[0,0]+W[1,1]*p[0,1]]) 
        
    if k%10==0:
        print(pdot)
    p = p + T*pdot
    u_actual = [h1[0,0],h1[0,1],h2[0,0],h2[0,1]]
    u = root(f_action, u_actual, args = (p,pd_hunt), method='lm',tol = 10e-10)
    
    if norm(np.matrix([u.x[0],u.x[1]])-h1)>0.1:
        u.x[0] = (h1 + 0.1*normalize(np.matrix([u.x[0],u.x[1]])-h1))[0,0]
        u.x[1] = (h1 + 0.1*normalize(np.matrix([u.x[0],u.x[1]])-h1))[0,1]
        
    if norm(np.matrix([u.x[2],u.x[3]])-h2)>0.1:
        u.x[2] = (h2 + 0.1*normalize(np.matrix([u.x[2],u.x[3]])-h2))[0,0]
        u.x[3] = (h2 + 0.1*normalize(np.matrix([u.x[2],u.x[3]])-h2))[0,1]
      
    h1 = np.matrix([u.x[0],u.x[1]])
    h2 = np.matrix([u.x[2],u.x[3]])
         
    PX.append(p[0,0])
    PY.append(p[0,1])
    H1X.append(h1[0,0])
    H1Y.append(h1[0,1])
    H2X.append(h2[0,0])
    H2Y.append(h2[0,1])
    
    k += 1
   
    
plt.plot(PX,PY,'g')
plt.show()
plt.plot(H1X,H1Y,'b')
plt.show()
plt.plot(H2X,H2Y,'b')
plt.show()
