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
from matplotlib.path import Path
import matplotlib.patches as patches

def f_action(u_actual,p,pd_hunt):
     

    h1 =np.matrix([u_actual[0],u_actual[1]])
    h2 =np.matrix([u_actual[2],u_actual[3]])

    Q = 5*((p)/(norm(p-h1)**3))+5*((p)/(norm(p-h2)**3))
    Qd = 5*((pd_hunt)/(norm(p-h1)**3))+5*((pd_hunt)/(norm(p-h2)**3))
    M = np.matrix([[(-5)/(norm(p-h1)**3),0,(-5)/(norm(p-h2)**3),0],[0,(-5)/(norm(p-h1)**3),0,(-5)/(norm(p-h2)**3)]])
    solX = np.transpose(Q) + np.transpose(Qd) + M*np.transpose(np.matrix([u_actual[0],u_actual[1],u_actual[2],u_actual[3]])) + 0.5*(np.transpose(p) - np.transpose(pd_hunt)) 
    sol = [solX[0,0],solX[1,0],0,0]
    
    return sol

p = np.matrix([0.0,0.0])
pd_hunt = np.matrix([1.3,1.5])
h1 = np.matrix([-3.0,0.0])
h2 = np.matrix([0.0,3.0])
T = 0.01
k = 0

PX = [p[0,0]]
PY = [p[0,1]]
H1X = [h1[0,0]]
H1Y = [h1[0,1]]
H2X = [h2[0,0]]
H2Y = [h2[0,1]]
plt.close('all')

while (k<5000):
    
    pdot = 5*((p-h1)/(norm(p-h1)**3)) + 5*((p-h2)/(norm(p-h2)**3))
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
    
#    if (h1[0,0]>5.0):
#        h1[0,0] = 5.0
#    if (h1[0,0]<-5.0):
#        h1[0,0] = -5.0
#    if (h2[0,0]>5.0):
#        h2[0,0] = 5.0
#    if (h2[0,0]<-5.0):
#        h2[0,0] = -5.0
#    if (h1[0,1]>5.0):
#        h1[0,1] = 5.0
#    if (h1[0,1]<-5.0):
#        h1[0,1] = -5.0
#    if (h2[0,1]>5.0):
#        h2[0,1] = 5.0
#    if (h2[0,1]<-5.0):
#        h2[0,1] = -5.0
         
    PX.append(p[0,0])
    PY.append(p[0,1])
    H1X.append(h1[0,0])
    H1Y.append(h1[0,1])
    H2X.append(h2[0,0])
    H2Y.append(h2[0,1])
    
    k += 1
   
    
D = 10.0


fig = plt.figure()
ax = plt.axes(xlim=(-D, D), ylim=(-D, D))
ax.set_title("Preserve")
ax.set_xlabel("X dimension")
ax.set_ylabel("Y dimension")
          
"""
Plot preserve
"""
vertsP = [
                                (-D, -D),  # left, bottom
                                (-D, D),  # left, top
                                (D, D),  # right, top
                                (D, -D),  # right, bottom
                                (-D, -D),  # ignored
                        ]
vertsPCodes = [
                                Path.MOVETO,
                                Path.LINETO,
                                Path.LINETO,
                                Path.LINETO,
                                Path.CLOSEPOLY,
               ]
                
vertsPPath = Path(vertsP, vertsPCodes)
vertsPPatch = patches.PathPatch(vertsPPath, facecolor='none', lw=2)
ax.add_patch(vertsPPatch)

ax.plot(PX[0], PY[0], 's', ms=12, c='y')
ax.plot(H1X[0], H1Y[0], 's', ms=12, c='y')
ax.plot(H2X[0], H2Y[0], 's', ms=12, c='y')
ax.plot(p[0,0], p[0,1], '^', ms=12, c='m')
ax.plot(h1[0,0], h1[0,1], '^', ms=12, c='m')
ax.plot(h2[0,0], h2[0,1] , '^', ms=12, c='m')
ax.plot(pd_hunt[0,0], pd_hunt[0,1], 'v', ms=12, c='r')
ax.plot(PX,PY,'g')
ax.plot(H1X,H1Y,'b')
ax.plot(H2X,H2Y,'b')
plt.show()

