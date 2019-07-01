# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:41:20 2019

@author: Eduardo
"""


import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from numpy.linalg   import norm
from sklearn.preprocessing import normalize

def f_action(u_actual,p1,p2,p1d_hunt,p2d_hunt,p1d_vaca,p2d_vaca):
     
    h1 =np.matrix([u_actual[0],u_actual[1]])

    Q1 = 5*((p1)/(norm(p1-h1)**3))
    Q2 = 5*((p2)/(norm(p2-h1)**3))

    Q = np.matrix([Q1[0,0],Q1[0,1],Q2[0,0],Q2[0,1]])
    PD_HUNT = np.matrix([p1d_hunt[0,0],p1d_hunt[0,1],p2d_hunt[0,0],p2d_hunt[0,1]])
    PD_VACA = np.matrix([p1d_vaca[0,0],p1d_vaca[0,1],p2d_vaca[0,0],p2d_vaca[0,1]])
    
    M = np.matrix([[(-5)/(norm(p1-h1)**3),0],[0,(-5)/(norm(p1-h1)**3)],[(-5)/(norm(p2-h1)**3),0],[0,(-5)/(norm(p2-h1)**3)]])
    
    solX = np.transpose(Q) + M*np.transpose(np.matrix([u_actual[0],u_actual[1]])) + +5*np.transpose(PD_VACA-PD_HUNT)
    sol = [solX[0,0],solX[1,0],solX[2,0],solX[3,0]]

    return sol

p1 = np.matrix([0,0])
pd1_hunt = np.matrix([1,1])
pd1_vaca = np.matrix([0.2,0.5])
p2 = np.matrix([0.5,0.5])
pd2_hunt = np.matrix([2,2])
pd2_vaca = np.matrix([0.4,1.3])
h1 = np.matrix([-3,-3])

T = 0.01
k = 0

P1X = [p1[0,0]]
P1Y = [p1[0,1]]

P2X = [p2[0,0]]
P2Y = [p2[0,1]]

H1X = [h1[0,0]]
H1Y = [h1[0,1]]

plt.close('all')

while (k<4000):
    
    pdot1 = 5*((p1-h1)/(norm(p1-h1)**3)) 
    pdot2 = 5*((p2-h1)/(norm(p2-h1)**3))
    
    if k%10==0:
        print(k)
    p1 = p1 + T*pdot1
    p2 = p2 + T*pdot2
    u_actual = [h1[0,0],h1[0,1],0,0]
    u = root(f_action, u_actual, args = (p1,p2,pd1_hunt,pd2_hunt,pd1_vaca,pd2_vaca), method='lm',tol = 10e-10)
    
    if norm(np.matrix([u.x[0],u.x[1]])-h1)>0.1:
        u.x[0] = (h1 + 0.1*normalize(np.matrix([u.x[0],u.x[1]])-h1))[0,0]
        u.x[1] = (h1 + 0.1*normalize(np.matrix([u.x[0],u.x[1]])-h1))[0,1]
        
        
    h1 = np.matrix([u.x[0],u.x[1]])
         
    P1X.append(p1[0,0])
    P1Y.append(p1[0,1])
    P2X.append(p2[0,0])
    P2Y.append(p2[0,1])

    H1X.append(h1[0,0])
    H1Y.append(h1[0,1])
    
    k += 1
   
fig = plt.figure()
ax = plt.axes()
ax.plot(pd1_hunt[0,0],pd1_hunt[0,1],'s', ms=12, c='g') 
ax.plot(pd2_hunt[0,0],pd2_hunt[0,1],'s', ms=12, c='r') 
ax.plot(pd1_vaca[0,0],pd1_vaca[0,1],'v', ms=12, c='g') 
ax.plot(pd2_vaca[0,0],pd2_vaca[0,1],'v', ms=12, c='r') 
ax.plot(P1X,P1Y,'g')
ax.plot(P2X,P2Y,'r')
ax.plot(H1X,H1Y,'b')
plt.show()
