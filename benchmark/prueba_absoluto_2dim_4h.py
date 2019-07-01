# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:41:20 2019

@author: Eduardo
"""

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
from numpy.linalg   import pinv

def f_action(u_actual,p1,p2,p1d_hunt,p2d_hunt):
     
    h1 =np.matrix([u_actual[0],u_actual[1]])
    h2 =np.matrix([u_actual[2],u_actual[3]])
    h3 =np.matrix([u_actual[4],u_actual[5]])
    h4 =np.matrix([u_actual[6],u_actual[7]])

    Q1 = 5*((p1)/(norm(p1-h1)**3))+5*((p1)/(norm(p1-h2)**3))+5*((p1)/(norm(p1-h3)**3))+5*((p1)/(norm(p1-h4)**3))
    Qd1 = 5*((pd1_hunt)/(norm(p1-h1)**3))+5*((pd1_hunt)/(norm(p1-h2)**3))+5*((pd1_hunt)/(norm(p1-h3)**3))+5*((pd1_hunt)/(norm(p1-h4)**3))
    Q2 = 5*((p2)/(norm(p2-h1)**3))+5*((p2)/(norm(p2-h2)**3))+5*((p2)/(norm(p2-h3)**3))+5*((p2)/(norm(p2-h4)**3))
    Qd2 = 5*((pd2_hunt)/(norm(p2-h1)**3))+5*((pd2_hunt)/(norm(p2-h2)**3))+5*((pd2_hunt)/(norm(p2-h3)**3))+5*((pd2_hunt)/(norm(p2-h4)**3))

    Q = np.matrix([Q1[0,0],Q1[0,1],Q2[0,0],Q2[0,1]])
    Qd = np.matrix([Qd1[0,0],Qd1[0,1],Qd2[0,0],Qd2[0,1]])
    P = np.matrix([p1[0,0],p1[0,1],p2[0,0],p2[0,1]])
    PD = np.matrix([p1d_hunt[0,0],p1d_hunt[0,1],p2d_hunt[0,0],p2d_hunt[0,1]])
    
    M = np.matrix([[(-5)/(norm(p1-h1)**3),0,(-5)/(norm(p1-h2)**3),0,(-5)/(norm(p1-h3)**3),0,(-5)/(norm(p1-h4)**3),0],[0,(-5)/(norm(p1-h1)**3),0,(-5)/(norm(p1-h2)**3),0,(-5)/(norm(p1-h3)**3),0,(-5)/(norm(p1-h4)**3)],[0,(-5)/(norm(p2-h1)**3),0,(-5)/(norm(p2-h2)**3),0,(-5)/(norm(p2-h3)**3),0,(-5)/(norm(p2-h4)**3)],[0,(-5)/(norm(p2-h1)**3),0,(-5)/(norm(p2-h2)**3),0,(-5)/(norm(p2-h3)**3),0,(-5)/(norm(p2-h4)**3)]])
    solX = np.transpose(Q) + np.transpose(Qd) + M*np.transpose(np.matrix([u_actual[0],u_actual[1],u_actual[2],u_actual[3],u_actual[4],u_actual[5],u_actual[6],u_actual[7]])) + 5*np.transpose(P) - 5*np.transpose(PD)
    sol = [solX[0,0],solX[1,0],solX[2,0],solX[3,0],0,0,0,0]

    return sol

p1 = np.matrix([0,0])
pd1_hunt = np.matrix([1,1])
p2 = np.matrix([0.5,0.5])
pd2_hunt = np.matrix([-2,-2])
h1 = np.matrix([-3,0])
h2 = np.matrix([-3,0])
h3 = np.matrix([0,3])
h4 = np.matrix([0,3])
T = 0.01
k = 0

P1X = [p1[0,0]]
P1Y = [p1[0,1]]

P2X = [p2[0,0]]
P2Y = [p2[0,1]]

H1X = [h1[0,0]]
H1Y = [h1[0,1]]
H2X = [h2[0,0]]
H2Y = [h2[0,1]]
H3X = [h3[0,0]]
H3Y = [h3[0,1]]
H4X = [h4[0,0]]
H4Y = [h4[0,1]]
plt.close('all')

while (k<4000):
    
    pdot1 = 5*((p1-h1)/(norm(p1-h1)**3)) + 5*((p1-h2)/(norm(p1-h2)**3)) + 5*((p1-h3)/(norm(p1-h3)**3)) + 5*((p1-h4)/(norm(p1-h4)**3))  
    pdot2 = 5*((p2-h1)/(norm(p2-h1)**3)) + 5*((p2-h2)/(norm(p2-h2)**3)) + 5*((p2-h3)/(norm(p2-h3)**3)) + 5*((p2-h4)/(norm(p2-h4)**3))  

    if k%1==0:
        print(k)
    p1 = p1 + T*pdot1
    p2 = p2 + T*pdot2
    u_actual = [h1[0,0],h1[0,1],h2[0,0],h2[0,1],h3[0,0],h3[0,1],h4[0,0],h4[0,1]]
    u = root(f_action, u_actual, args = (p1,p2,pd1_hunt,pd2_hunt), method='lm',tol = 10e-10)
    
    if norm(np.matrix([u.x[0],u.x[1]])-h1)>0.1:
        u.x[0] = (h1 + 0.1*normalize(np.matrix([u.x[0],u.x[1]])-h1))[0,0]
        u.x[1] = (h1 + 0.1*normalize(np.matrix([u.x[0],u.x[1]])-h1))[0,1]
        
    if norm(np.matrix([u.x[2],u.x[3]])-h2)>0.1:
        u.x[2] = (h2 + 0.1*normalize(np.matrix([u.x[2],u.x[3]])-h2))[0,0]
        u.x[3] = (h2 + 0.1*normalize(np.matrix([u.x[2],u.x[3]])-h2))[0,1]
        
    if norm(np.matrix([u.x[4],u.x[5]])-h3)>0.1:
        u.x[4] = (h3 + 0.1*normalize(np.matrix([u.x[4],u.x[5]])-h3))[0,0]
        u.x[5] = (h3 + 0.1*normalize(np.matrix([u.x[4],u.x[5]])-h3))[0,1]
        
    if norm(np.matrix([u.x[6],u.x[7]])-h4)>0.1:
        u.x[6] = (h4 + 0.1*normalize(np.matrix([u.x[6],u.x[7]])-h4))[0,0]
        u.x[7] = (h4 + 0.1*normalize(np.matrix([u.x[6],u.x[7]])-h4))[0,1]
        
    h1 = np.matrix([u.x[0],u.x[1]])
    h2 = np.matrix([u.x[2],u.x[3]])
    h3 = np.matrix([u.x[4],u.x[5]])
    h4 = np.matrix([u.x[6],u.x[7]])
         
    P1X.append(p1[0,0])
    P1Y.append(p1[0,1])
    P2X.append(p2[0,0])
    P2Y.append(p2[0,1])

    H1X.append(h1[0,0])
    H1Y.append(h1[0,1])
    H2X.append(h2[0,0])
    H2Y.append(h2[0,1])
    H3X.append(h3[0,0])
    H3Y.append(h3[0,1])
    H4X.append(h4[0,0])
    H4Y.append(h4[0,1])
    
    k += 1
   
    
plt.plot(P1X,P1Y,'g')
plt.show()
plt.plot(P2X,P2Y,'r')
plt.show()
plt.plot(H1X,H1Y,'b')
plt.show()
plt.plot(H2X,H2Y,'b')
plt.show()
plt.plot(H3X,H3Y,'b')
plt.show()
plt.plot(H4X,H4Y,'b')
plt.show()