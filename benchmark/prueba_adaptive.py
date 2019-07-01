# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:00:40 2019

@author: Eduardo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 25 08:17:55 2019

@author: Eduardo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:19:19 2019

@author: Eduardo
"""

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from numpy.linalg   import norm
from sklearn.preprocessing import normalize
from matplotlib.path import Path
import matplotlib.patches as patches


def f_action(u_actual,p,pd_hunt, coeff):
     

    h1 =np.matrix([u_actual[0],u_actual[1]])
    h2 =np.matrix([u_actual[2],u_actual[3]])

    Q = coeff*((p)/(norm(p-h1)**3))+coeff*((p)/(norm(p-h2)**3))
    Qd = coeff*((pd_hunt)/(norm(p-h1)**3))+coeff*((pd_hunt)/(norm(p-h2)**3))
    M = coeff*np.matrix([[(-1)/(norm(p-h1)**3),0,(-1)/(norm(p-h2)**3),0],[0,(-1)/(norm(p-h1)**3),0,(-1)/(norm(p-h2)**3)]])
    solX = np.transpose(Q) + np.transpose(Qd) + M*np.transpose(np.matrix([u_actual[0],u_actual[1],u_actual[2],u_actual[3]])) + 0.5*(np.transpose(p) - np.transpose(pd_hunt)) 
    sol = [solX[0,0],solX[1,0],0,0]
    
    return sol

p = np.matrix([0.0,0.0])
pd_hunt = np.matrix([1.3,1.5])
h1 = np.matrix([-3.0,0.0])
h2 = np.matrix([0.0,3.0])
T = 0.001
k = 0
coeff = 2.0

PX = [p[0,0]]
PY = [p[0,1]]
H1X = [h1[0,0]]
H1Y = [h1[0,1]]
H2X = [h2[0,0]]
H2Y = [h2[0,1]]
COEFF = [coeff]
plt.close('all')

D = 5.0

while (k<7800):
    
    pdot = 5*((p-h1)/(norm(p-h1)**3)) + 5*((p-h2)/(norm(p-h2)**3))
    p = p + T*pdot
    
    
    coeffdot = -(pd_hunt+p) * np.transpose(((p)/(norm(p-h1)**3)) + ((p)/(norm(p-h2)**3)) + 
                    ((pd_hunt)/(norm(p-h1)**3)) + ((pd_hunt)/(norm(p-h2)**3)) ) * 0.5
    coeff = coeff + T*coeffdot[0,0]
    
    if k%10==0:
        print(p)
        print(k)
        
    u_actual = [h1[0,0],h1[0,1],h2[0,0],h2[0,1]]
    u = root(f_action, u_actual, args = (p,pd_hunt,coeff), method='lm',tol = 10e-10)
    
    if norm(np.matrix([u.x[0],u.x[1]])-h1)>0.1:
        u.x[0] = (h1 + 0.1*normalize(np.matrix([u.x[0],u.x[1]])-h1))[0,0]
        u.x[1] = (h1 + 0.1*normalize(np.matrix([u.x[0],u.x[1]])-h1))[0,1]
        
    if norm(np.matrix([u.x[2],u.x[3]])-h2)>0.1:
        u.x[2] = (h2 + 0.1*normalize(np.matrix([u.x[2],u.x[3]])-h2))[0,0]
        u.x[3] = (h2 + 0.1*normalize(np.matrix([u.x[2],u.x[3]])-h2))[0,1]
      
    if (u.x[0]>D):
        u.x[0]=D
    if (u.x[0]<-D):
        u.x[0]=-D
    if (u.x[1]>D):
        u.x[1]=D
    if (u.x[1]<-D):
        u.x[1]=-D
    if (u.x[2]>D):
        u.x[2]=D
    if (u.x[2]<-D):
        u.x[2]=-D
    if (u.x[3]>D):
        u.x[3]=D
    if (u.x[3]<-D):
        u.x[3]=-D
        
    h1 = np.matrix([u.x[0],u.x[1]])
    h2 = np.matrix([u.x[2],u.x[3]])
         
    PX.append(p[0,0])
    PY.append(p[0,1])
    H1X.append(h1[0,0])
    H1Y.append(h1[0,1])
    H2X.append(h2[0,0])
    H2Y.append(h2[0,1])
    COEFF.append(coeff)
    
    k += 1
   


ax.set_title("Preserve")
ax.set_xlabel("X dimension")
ax.set_ylabel("Y dimension")
fig = plt.figure()
ax = plt.axes(xlim=(-D, D), ylim=(-D, D))

          
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

fig = plt.figure()   
ax = plt.axes() 
ax.set_xlabel("Tiempo (k)")
ax.set_ylabel("Valor del coeficiente")
ax.plot(COEFF)
plt.show()