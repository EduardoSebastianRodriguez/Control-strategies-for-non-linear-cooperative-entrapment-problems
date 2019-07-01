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
from matplotlib import animation
import time
import sys


def animate(i,xp,yp,xh,yh,line1,line2,line3,line4):      
    
    line1.set_data(xp[i], yp[i])
    line2.set_data(xh[i], yh[i])
    line3.set_data(xp[i+3], yp[i+3])
    line4.set_data(xh[i+3], yh[i+3])
    
    return line1,line2,line3,line4,

def f_action(u_actual,p,pd_vaca,pd_hunt):
     
    W = np.zeros([2,2])
    W[0,0] = 0.02
    W[0,1] = -0.002
    W[1,0] = -0.007
    W[1,1] = 0.06

    Q = 5*((p)/(norm(p-u_actual)**3))
    M = (-5)/(norm(p-u_actual)**3)
    solX = M*u_actual+Q+1*(pd_vaca-pd_hunt)+W*(np.transpose(p)+np.transpose(pd_hunt))
    sol = [solX[0,0],solX[0,1]]
    
    return sol

p = np.matrix([0,0])
pd_vaca = np.matrix([2,1.5])
pd_hunt = np.matrix([1,1])
h = np.matrix([-0.5,-0.5])
T = 0.01
k = 0
D = 5
W = np.zeros([2,2])
W[0,0] = 0.02
W[0,1] = -0.002
W[1,0] = -0.007
W[1,1] = 0.06

saveAnim = False
K = 6000

PX = [p[0,0]]
PY = [p[0,1]]
HX = [h[0,0]]
HY = [h[0,1]]

plt.close('all')

while (k<K):
    
    pdot = 5*((p-h)/(norm(p-h)**3)) + (pd_vaca-p) + np.matrix([W[0,0]*p[0,0]+W[0,1]*p[0,1],W[1,0]*p[0,0]+W[1,1]*p[0,1]])
    if k%100==0:
        print(k)
    p = p + T*pdot
    u_actual = h 
    u = root(f_action, u_actual, args = (p,pd_vaca,pd_hunt), method='lm',tol = 10e-10)
    if norm(np.matrix([u.x[0],u.x[1]])-h)<0.2:
        h =np.matrix([u.x[0],u.x[1]])
    else:
        h = h + T*0.2*normalize(np.matrix([u.x[0],u.x[1]])-h)
         
    PX.append(p[0,0])
    PY.append(p[0,1])
    HX.append(h[0,0])
    HY.append(h[0,1])
    k += 1
   
    
plt.plot(PX,PY,'g')
plt.show()
plt.plot(HX,HY,'b')
plt.show()
print(p)

#fig = plt.figure()
#ax = plt.axes(xlim=(-D, D), ylim=(-D, D))
#ax.set_title("Preserve")
#ax.set_xlabel("X dimension")
#ax.set_ylabel("Y dimension")
#          
#"""
#Plot preserve
#"""
#vertsP = [
#                                (-D, -D),  # left, bottom
#                                (-D, D),  # left, top
#                                (D, D),  # right, top
#                                (D, -D),  # right, bottom
#                                (-D, -D),  # ignored
#                        ]
#vertsPCodes = [
#                                Path.MOVETO,
#                                Path.LINETO,
#                                Path.LINETO,
#                                Path.LINETO,
#                                Path.CLOSEPOLY,
#               ]
#                
#vertsPPath = Path(vertsP, vertsPCodes)
#vertsPPatch = patches.PathPatch(vertsPPath, facecolor='none', lw=2)
#ax.add_patch(vertsPPatch)
#
#ax.plot(pd_vaca[0,0], pd_vaca[0,1], 's', ms=12, c='y')
#ax.plot(pd_hunt[0,0], pd_hunt[0,1], 'v', ms=12, c='r')
#ax.plot(HX[0], HY[0], 'x', ms=12, c='b')
#
#line1, = ax.plot([], [], 'o', ms=8, c='g')
#line2, = ax.plot([], [], 'o', ms=8, c='b')
#line3, = ax.plot([], [], 'o', ms=4, c='#3AAD74')
#line4, = ax.plot([], [], 'o', ms=4, c='#7984D0')
#
#xp = np.transpose(PX)
#yp = np.transpose(PY)
#xh = np.transpose(HY)
#yh = np.transpose(HX)
#
#try:
#    anim = animation.FuncAnimation(fig, 
#                                   animate,
#                                   fargs = (xp,yp,xh,yh,line1,line2,line3,line4), 
#                                   frames=K, 
#                                   interval=1,
#                                   blit=True,
#                                   repeat=False
#                                   )
#
#    if(saveAnim):   
#        anim.save('G:/Unidades de equipo/EduardoSebastian/TFG/05_Hemeroteca/AnimationNONLIN_'+str(int(time.time()))+'.mp4',writer='ffmpeg', fps=60)
#    
#except:
#    print("Unexpected error:", sys.exc_info()[0])
#    pass