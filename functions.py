# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:31:46 2018

@author: Eduardo

"""
from __future__ import division, print_function
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib import animation
import time
import sys
from numpy.linalg   import norm
from sklearn.preprocessing import normalize
import scipy.linalg
import matplotlib.patches as mpatches
from scipy.optimize import root

"""
Class defining the behavior of a hunter
"""
class hunter:
    
    def __init__(self):
        
        self.newPosition = np.matrix([0.,0.])    #Present postion of the hunter
        self.prePosition = np.matrix([0.,0.])    #Previous position of the hunter
        self.velocity = np.matrix([0.0,0.0])     #Velocity of the hunter
        self.saturation = 0.1                      #Saturation level
        return
    
    def setLocalization(self 
                        ,p    #new position of the hunter
                        ):
        
        self.prePosition = p
        self.newPosition = p
        return
    
        
    def interaction(self
                    ,p    #new position of the hunter
                    ,D    #limits of the preserve
                    ,T    #sample time of the system
                    ):
        
        self.prePosition = self.newPosition
        if (norm(p-self.prePosition) < self.saturation):
            
            self.newPosition = p
            v = norm(p-self.prePosition)/T 
            
        else:
            
            try:
                
                inc    = np.matrix(normalize((p-self.prePosition)))*self.saturation 
            
            except:
                
                inc = np.matrix([0,0])
                
            self.newPosition += inc
            v = self.saturation/T
            
        """
        Limits the movements of the object: it is not possible to cross the preserve
        """
        if (self.newPosition[0,0] < -D):
            self.newPosition[0,0] = -D
        
        if (self.newPosition[0,0] > D):
            self.newPosition[0,0] = D
        
        if (self.newPosition[0,1] < -D):
            self.newPosition[0,1] = -D
        
        if (self.newPosition[0,1] > D):
            self.newPosition[0,1] = D
            
        if (self.prePosition[0,0] < -D):
            self.prePosition[0,0] = -D
        
        if (self.prePosition[0,0] > D):
            self.prePosition[0,0] = D
        
        if (self.prePosition[0,1] < -D):
            self.prePosition[0,1] = -D
        
        if (self.prePosition[0,1] > D):
            self.prePosition[0,1] = D
        
        return v

"""
Class defining the behavior of a prey.
The main structure of the class defines the behavior of a standard prey: animal, 
fire focus, epidemic disease, ...
Thus, different dynamics can be selected by switching the flag MODE:
    
    *Mode 0: Pierson & Schwager
        
    *Mode 1: Licitra et al.
    
    *Mode 2: Pierson & Schwager + Browning & Wexler
    
    *Mode 3: Licitra et al. + Browning & Wexler
    
"""
class prey:
    
    def __init__(self):
        
        self.newPosition = np.matrix([0.,0.])     #Present postion of the prey
        self.prePosition = np.matrix([0.,0.])     #Previous position of the prey
        self.angryMode = False                    #Determines if the prey is angry or not
        self.repulsionVelocity =np.matrix([0.,0.])#Repulsion due to hunters and other preys
        self.label = 0                            #Tag labelling the prey
        self.mode = 0                             #Flag selecting the mode of the prey
        self.saturation = 100                     #Saturation level for velocity
        return
    
    def setLocalization(self 
                        ,p #new position of the prey
                        ):
        
        self.prePosition = p
        self.newPosition = p
        return
    
    def setRadius(self
                  ,r #new radius for the angry mode
                  ):
    
        self.radius = r
        return
    
    def setAngryMode(self
                     ,a #boolean that sets if the prey is angry or not
                     ):
        
        self.angryMode = a
        return
    
    def setLabel(self
                 ,l #int that names the object
                 ):
    
        self.label = l
        return
    
    def setMode(self
                ,m #int that selects the dynamics of the prey
                ):
        
        self.mode = m
        return
    
    """
    Main method of the class: describes the interaction of the object with the environment
    
        *Mode 0: Pierson & Schwager
        
        *Mode 1: Licitra et al.
        
        *Mode 2: fluid mechanics + Browning & Wexler
        
        *Mode 3: non-linear model 2
    
    """
    
    def interaction(self
                    ,H #vector of hunters 
                    ,P #vector of preys
                    ,T #sample time
                    ,D #dimension of the preserve
                    ,k #time
                    ):
        
        """
        Reset the variables
        """
        self.repulsionVelocity = np.matrix([0.,0.])
        self.prePosition = self.newPosition
        
        """ Load hunters positions """
        
        h = np.matrix(np.zeros([len(H),2]))
        for i in range(len(H)):
            h[i,0] = H[i].prePosition[0,0]
            h[i,1] = H[i].prePosition[0,1]

        """
        Mode 0: A standard artificial potential dynamic is implemented: a repulsion
                force decides the velocity (vector) of the prey.
        """
        if (self.mode == 1):           
            self.repulsionVelocity = dd1(self.prePosition,h,len(h))
                
        """
        Mode 1: A Gaussian standard artificial potential dynamic is implemented: 
                a repulsion force decides the velocity (vector) of the prey.
        """

        
        if (self.mode == 2):
            self.repulsionVelocity = dd2(self.prePosition,h,len(h))
            
                   
        """
        Mode 2: Browning & Wexler proposed a bi dimensional model for wind in 1968.
        Thus, and adapted by using a general velocity vector field, a fire focus 
        model is proposed: the focuses (preys) move following the wind field, except 
        when a fireman (hunter) is closed. In this case, the fire is repulsed with a
        force proporcional to the distance and with the opposite direction with respect
        to the fireman. This is added to the mode 0 as a perturbance.
        """

        if (self.mode == 3):
            self.repulsionVelocity = dd3(self.prePosition,h,len(h))

        """
        Mode 3: Browning & Wexler proposed a bi dimensional model for wind in 1968.
        Thus, and adapted by using a general velocity vector field, a fire focus 
        model is proposed: the focuses (preys) move following the wind field, except 
        when a fireman (hunter) is closed. In this case, the fire is repulsed with a
        force proporcional to the distance and with the opposite direction with respect
        to the fireman. This is added to the mode 1 as a perturbance.
        """  
            
        if (self.mode == 4):
            self.repulsionVelocity = dd4(self.prePosition,h,len(h)) 
          
        """
        Calculate the new position with respect to a saturation rule
        """
        if (norm(self.repulsionVelocity)>self.saturation):
            self.repulsionVelocity = np.matrix(normalize(self.repulsionVelocity))*self.saturation

        self.newPosition = self.newPosition + np.matrix([T*self.repulsionVelocity[0,0],T*self.repulsionVelocity[0,1]])
        
        """
        Limits the movements of the object: it is not possible to cross the preserve
        """
        
        if (self.newPosition[0,0] < -D):
            self.newPosition[0,0] = -D
        
        if (self.newPosition[0,0] > D):
            self.newPosition[0,0] = D
        
        if (self.newPosition[0,1] < -D):
            self.newPosition[0,1] = -D
        
        if (self.newPosition[0,1] > D):
            self.newPosition[0,1] = D
            
        if (self.prePosition[0,0] < -D):
            self.prePosition[0,0] = -D
        
        if (self.prePosition[0,0] > D):
            self.prePosition[0,0] = D
        
        if (self.prePosition[0,1] < -D):
            self.prePosition[0,1] = -D
        
        if (self.prePosition[0,1] > D):
            self.prePosition[0,1] = D
        
        v = norm(self.repulsionVelocity) 
        
        return v
    
    
def dd1(p,H,N):
    
    dp = np.matrix([0.0,0.0])
    gamma = 1
    for i in range (N):
        dp += gamma*(p-H[i])/(norm(p-H[i])**3)
    
    return dp

def dd2(p,H,N):
    
    var   = 1.0
    alpha = 100.5
    beta  = 0.5
    radius = 1.0
    angry = False
        
    dp = np.matrix([0.0,0.0])
    
    for i in range (N):
        if norm(p-H[i]) < radius:
            angry = True
            break
        else:
            angry = False
        
    for i in range (N):
        Xi = (p[0,0]-H[i,0])**2+(p[0,1]-H[i,1])**2
        if angry:
            dp += alpha*(p-H[i])*np.exp(-(1/(var**2))*Xi)
        else:
            dp += alpha*beta*(p-H[i])*np.exp(-(1/(var**2))*Xi)
    
    return dp

def dd3(p,H,N):
    
    dp = np.matrix([0.0,0.0])
    W = np.zeros([2,2])
    W[0,0] = 0.02
    W[0,1] = -0.002
    W[1,0] = -0.007
    W[1,1] = 0.06
    gamma = 1
#    W = 45*W
    for i in range (N):
        dp += gamma*(p-H[i])/(norm(p-H[i])**3)
    
    dp += np.array([p[0,0]*W[0,0] + p[0,1]*W[1,0],p[0,0]*W[0,1] + p[0,1]*W[1,1]])
    
    return dp

def dd4(p,H,N):
    
    var   = 1.0
    alpha = 100.5
    beta  = 0.5
    radius = 1.0
    angry = False
        
    dp = np.matrix([0.0,0.0])
    W = np.zeros([2,2])
    W[0,0] = 0.02
    W[0,1] = -0.002
    W[1,0] = -0.007
    W[1,1] = 0.06
    W = 45*W
    
    for i in range (N):
        if norm(p-H[i]) < radius:
            angry = True
            break
        else:
            angry = False
   
    for i in range (N):
        Xi = (p[0,0]-H[i,0])**2+(p[0,1]-H[i,1])**2
        if angry:
            dp += alpha*(p-H[i])*np.exp(-(1/(var**2))*Xi)
        else:
            dp += alpha*beta*(p-H[i])*np.exp(-(1/(var**2))*Xi)
    
    dp += np.array([p[0,0]*W[0,0] + p[0,1]*W[1,0],p[0,0]*W[0,1] + p[0,1]*W[1,1]])
   
    return dp

"""
This program constructs the setup function which, during the 
project, will execute and represent the interactions between
hunters and preys. As arguments, we have to establish:
    
    

    * The number of desired hunters N_hunters
    * The number of desired preys M_preys
    * The mode of behavior of the preys (0, 1 or 2)
    * A config structure which passes the following parameters:
        
        * The sample time T
        * The dimension of the preserve D
        * The number of desired samples K 
        * If we want to draw the history of positions of the hunters and preys, draw = True
        * If we want an animation of the system, anim = True
        * Sets the time interval between frames in the animation
        * The pole placement feedback matrix for the controller L
        * The initial position of the hunters H0
        * The initial position of the preys P0
        * The desired position of the hunters h_desired
        * The desired position of the preys p_desired
        

It is also included a function, called f, which implements a numeric solver:
given a desired set of prey positions, the function calculates the desired
final position of the hunters which makes the system stable and with no velocity
in the desired prey's position. 


In order to test the performance, the distance from the centroid of the vertices
of the polygon formed by the preys is calculated. Thus, a quick function is used
to implemented.     
"""

def setup(N_hunters  #Number of desired hunters
          ,M_preys   #Number of desired preys
          ,mode      #mode of behavior of the preys
          ,config    #parameters of configuration
          ):
    """
    1. Instantiate the main local variables
    """
    H = [hunter() for i in range(N_hunters)] #Vector of N_hunters x 1 hunters
    P = [prey() for j in range(M_preys)]       #Vector of M_preys x 1 preys
    XH = [[] for i in range(N_hunters)]             #Vector storing the X coordinate of the hunters in each sample
    YH = [[] for i in range(N_hunters)]             #Vector storing the Y coordinate of the hunters in each sample
    XP = [[] for j in range(M_preys)]               #Vector storing the X coordinate of the preys in each sample
    YP = [[] for j in range(M_preys)]               #Vector storing the Y coordinate of the preys in each sample
    
    if (len(config)<13):
        
        if 'SampleTime' in config:
            T    = config['SampleTime']
        else:
            T        = 1
        
        if 'D' in config:
            D    = config['D']
        else:
            D        = 5.0
            
        if 'K' in config:
            K    = config['K']
        else:
            K        = 100
            
        if 'draw' in config:
            draw    = config['draw']
        else:
            draw        = False
            
        if 'anim' in config:
            anim    = config['anim']
        else:
            anim    = False
            
        if 'saveAnim' in config:
            saveAnim  = config['saveAnim']
        else:
            saveAnim  = False
            
        if 'interval' in config:
            interval    = config['interval']
        else:
            interval        = 0.1
            
        if 'L' in config:
            L    = config['L']
        else:
            L    = np.zeros([2*N_hunters,1])
            
        if 'controller' in config:
            controller = config['controller']
        else:
            controller = "linear"
            
        if 'H0' in config:
            h_actual = config['H0']
        else:
            for i in range(N_hunters): 
                h_actual.append(0.0)
                h_actual.append(0.0)
                
        if 'P0' in config:
            p_actual = config['P0']
        else:
            for j in range(M_preys): 
                p_actual.append(0.0)
                p_actual.append(0.0)

        if 'H_desired' in config:
            h_desired = config['H_desired']
            for i in range(N_hunters):
                H[i].setLocalization(np.matrix([h_desired[2*i],h_desired[2*i+1]]))
        else:
            for i in range(N_hunters):
                H[i].setLocalization(np.matrix([random.uniform(-3*D/4,3*D/4),random.uniform(-3*D/4,3*D/4)]))
                
        if 'P_desired' in config:
            p_desired = config['P_desired']
            for j in range(M_preys):
                P[j].setLabel(j)
                P[j].setMode(mode)
                P[j].setLocalization(np.matrix([p_desired[2*j],p_desired[2*j+1]]))
        else:
            for j in range(M_preys):
                P[j].setLabel(j)
                P[j].setMode(mode)
                P[j].setLocalization(np.matrix([random.uniform(-3*D/4,3*D/4),random.uniform(-3*D/4,3*D/4)]))
    else:
        
        T        = config['SampleTime']
        D        = config['D']
        K        = config['K']
        draw     = config['draw']
        anim     = config['anim']
        interval = config['interval']
        h_actual = config['H0']
        p_actual = config['P0']
        h_desired = config['H_desired']
        p_desired = config['P_desired']
        saveAnim  = config['saveAnim']
        controller = config['controller']
        
        if controller == "linear":
            L        = np.matrix(config['L']) 
        if controller == "adaptive":
            coeff    = np.matrix(config['L'])
        
#        mode = 4
        for i in range(N_hunters):
            H[i].setLocalization(np.matrix([config['H0'][2*i],config['H0'][2*i+1]]))
        for j in range(M_preys):
            P[j].setLabel(j)
            P[j].setMode(mode)
            P[j].setLocalization(np.matrix([config['P0'][2*j],config['P0'][2*j+1]]))
    
    """
    Error metric: norm of the distance of each prey from its desired final position
    """
    error            = np.zeros([K,M_preys])
    v_preys          = np.zeros([K,M_preys])
    v_hunters        = np.zeros([K,N_hunters])
    times            = []
    converge = False
    
    """
    2. If we want to draw the historical of positions of both
       hunters and preys, first we need to set the preserve. If
       draw == FALSE, jump to point 3
    """
    if (draw):
        
            """ The main layout of the animation """
            ax = plt.axes(xlim=(-D,D), ylim=(-D,D))
            
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
            
            for j in range(M_preys):  
                ax.plot(p_desired[2*j], p_desired[2*j+1], '^', ms=10, c='m')
    
    
    """
    3. Control law
    """
    
    x_actual = np.transpose(np.matrix(p_actual))
    x_desired= np.transpose(np.matrix(p_desired))
    u_desired= np.transpose(np.matrix(h_desired))
    u_actual = np.transpose(np.matrix(h_actual))
            
    """
    4. Play the simulation
    """    
    k = 0
    while (k < K):  
        
    
#        if k%100==0:
#        print(k)
        """
        Run the interaction of the preys, this is applied with every prey
        """                              
        for j in range(M_preys):
            
            """ Interaction of the prey """
            v_p = P[j].interaction(H,P,T,D,k)
                
            """ Store the coordinates of the prey """
            XP[j].append(P[j].prePosition[0,0])
            YP[j].append(P[j].prePosition[0,1])
            x_actual[2*j] = P[j].prePosition[0,0]
            x_actual[2*j+1] = P[j].prePosition[0,1]
            
            """ Calc actual error metric of this prey """
            error[k,j] = np.sqrt((x_actual[2*j] - x_desired[2*j])**2 + (x_actual[2*j+1] - x_desired[2*j+1])**2)
            v_preys[k,j] = v_p
                        
            """ If draw == TRUE, it is time to plot the actual position of the prey """
            if (draw):
                if(k==0):
                    ax.plot(P[j].prePosition[0,0],P[j].prePosition[0,1], 's', ms=10, c='y')
                else:
                    ax.plot(P[j].prePosition[0,0],P[j].prePosition[0,1], 'o', ms=3, c='g')
            
        
        """ Calc control action (hunters' position) """
        if controller == "linear":
            
            inc_x = x_actual - x_desired
            inc_u = -L*inc_x
            u_actual = inc_u + u_desired
            
        elif controller == "non-linear":  
                    
            start = time.time()
            u_root  = root(f_action, u_actual, args = (x_actual,x_desired,mode,N_hunters,M_preys), method='lm',tol = 10e-10)                        
            end = time.time()
            times.append(end-start)
            
        elif controller == "adaptive":
            
            Q = np.matrix(np.zeros([2*M_preys,2*M_preys]))
            Qd = np.matrix(np.zeros([2*M_preys,2*M_preys]))
            W = np.eye(2*M_preys)*1
            
            var   = 1.0
            beta  = 0.5
            radius = 1.0
            angry = False
            
            for j in range (M_preys):
                vel = np.matrix([0,0])
                pj = np.matrix([x_actual[2*j,0],x_actual[2*j+1,0]])
                angry = False
                if mode==2 or mode==4:
                    for i in range (N_hunters):
                        hi = np.matrix([u_actual[2*i,0],u_actual[2*i+1,0]])
                        dij = pj-hi
                        if norm(dij)<radius:
                            angry=True
                            break
                        
                for i in range (N_hunters):
                    hi = np.matrix([u_actual[2*i,0],u_actual[2*i+1,0]])
                    dij = pj-hi
                    if mode == 1 or mode == 3:
                        vel = vel + dij/(norm(dij)**3)
                    else:
                        X = (pj[0,0]-hi[0,0])**2+(pj[0,1]-hi[0,1])**2
                        if angry:
                            vel = vel + dij*np.exp(-(1/(var**2))*X)
                        else:
                            vel = vel + beta*dij*np.exp(-(1/(var**2))*X)
                        
                Q[2*j,2*j] = vel[0,0]
                Q[2*j+1,2*j+1] = vel[0,1]
                
            for j in range (M_preys):
                vel = np.matrix([0,0])
                pj = np.matrix([x_desired[2*j,0],x_desired[2*j+1,0]])
                angry = False
                if mode==2 or mode==4:
                    for i in range (N_hunters):
                        hi = np.matrix([u_actual[2*i,0],u_actual[2*i+1,0]])
                        dij = pj-hi
                        if norm(dij)<radius:
                            angry=True
                            break
                        
                for i in range (N_hunters):
                    hi = np.matrix([u_actual[2*i,0],u_actual[2*i+1,0]])
                    dij = pj-hi
                    if mode == 1 or mode == 3:
                        vel = vel + dij/(norm(dij)**3)
                    else:
                        X = (pj[0,0]-hi[0,0])**2+(pj[0,1]-hi[0,1])**2
                        if angry:
                            vel = vel + dij*np.exp(-(1/(var**2))*X)
                        else:
                            vel = vel + beta*dij*np.exp(-(1/(var**2))*X)

                Qd[2*j,2*j] = vel[0,0]
                Qd[2*j+1,2*j+1] = vel[0,1]
        
            coeffdot = np.transpose(x_desired-x_actual)*(Q+Qd)*W
            
            coeff = coeff - T*coeffdot
            
            start = time.time()
            u_root  = root(f_action_adaptive, u_actual, args = (x_actual,x_desired,mode,N_hunters,M_preys,coeff), method='lm',tol = 10e-10)                        
            end = time.time()
            times.append(end-start)            
        else:
            start = time.time()
            u_root  = root(f_action, u_actual, args = (x_actual,x_desired,mode,N_hunters,M_preys), method='lm',tol = 10e-10)                        
            end = time.time()
            times.append(end-start) 
            
            
        """
        Run the interaction of the hunters, this is applied by every hunter
        """           
        
        for i in range(N_hunters):
            
            
            """ Interaction of the hunter """
            if controller == "linear":
#                HINI = [-0.8,1.2,-1.3,1.5,0.2,0.2,0.3,1.7]
#                v_h = H[i].interaction(np.matrix([HINI[2*i],HINI[2*i+1]]),D,T)
                v_h = H[i].interaction(np.transpose(np.matrix([np.array(u_actual)[2*i],np.array(u_actual)[2*i+1]])),D,T)
            elif controller == "non-linear":
                v_h = H[i].interaction(np.matrix([u_root.x[2*i],u_root.x[2*i+1]]),D,T)
            elif controller == "adaptive":
                v_h = H[i].interaction(np.matrix([u_root.x[2*i],u_root.x[2*i+1]]),D,T)
            else:
                v_h = H[i].interaction(np.matrix([u_root.x[2*i],u_root.x[2*i+1]]),D,T)

            """ Store the coordinates of the hunter """
            XH[i].append(H[i].prePosition[0,0])
            YH[i].append(H[i].prePosition[0,1])
            
            u_actual[2*i] = H[i].prePosition[0,0]
            u_actual[2*i+1] = H[i].prePosition[0,1]
                
            """ Save v of the hunters """
            v_hunters[k,i] = v_h
                        
            """ If draw == TRUE, it is time to plot the actual position of the hunter """
            if (draw):
                if(k==0):
                    ax.plot(H[i].prePosition[0,0],H[i].prePosition[0,1], 's', ms=10, c='y')
                else:
                    ax.plot(H[i].prePosition[0,0],H[i].prePosition[0,1], 'o', ms=3, c='b')

        if np.average(v_preys[k,:]) < 0.25:
            converge = True
            
        k = k + 1
        

        
    if(draw):
        for j in range(M_preys):  
            ax.plot(P[j].prePosition[0,0],P[j].prePosition[0,1], 'v', ms=10, c='r')
        for i in range(N_hunters):  
            ax.plot(H[i].prePosition[0,0],H[i].prePosition[0,1], 'v', ms=10, c='r')

        plt.show()
#        plt.savefig("G:/Unidades de equipo/EduardoSebastian/TFG/04_Imagenes/Trayectoria.png")
#        plt.close('all')
    
    if(anim):
        
        
        """ The frame is cleared to create the next frame """
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
        for j in range(M_preys):  
            ax.plot(p_desired[2*j], p_desired[2*j+1], 'x', ms=12, c='g')
        for i in range(N_hunters):  
            ax.plot(h_desired[2*i], h_desired[2*i+1], 'x', ms=12, c='b')
        
        line1, = ax.plot([], [], 'o', ms=8, c='g')
        line2, = ax.plot([], [], 'o', ms=8, c='b')
        line3, = ax.plot([], [], 'o', ms=4, c='#3AAD74')
        line4, = ax.plot([], [], 'o', ms=4, c='#7984D0')
        
        xp = np.transpose(XP)
        yp = np.transpose(YP)
        xh = np.transpose(XH)
        yh = np.transpose(YH)
        
        try:
            anim = animation.FuncAnimation(fig, 
                                           animate,
                                           fargs = (xp,yp,xh,yh,line1,line2,line3,line4), 
                                           frames=K-4, 
                                           interval=interval, 
                                           blit=True,
                                           repeat=False
                                           )
        
            if(saveAnim):
                anim
            
                anim.save('G:/Unidades de equipo/EduardoSebastian/TFG/05_Hemeroteca/Animation_'+str(int(time.time()))+'.mp4',writer='ffmpeg', fps=60)
            
            plt.pause(5)
        
        except:
            print("Unexpected error:", sys.exc_info()[0])
            pass
     
#    if controller != "linear":
#        print(np.max(times))
#        MEAN = 0
#        for i in range(len(times)):
#            MEAN+=times[i]
#        MEAN = MEAN/K
#        print(MEAN)
#        
#    if controller == "adaptive":
#        print(coeff)
    
    return XP,YP,XH,YH,error,v_preys,v_hunters,converge


"""
Solver for mode 0
"""
def f_1(h,p,N,M):
    
    """ Read the hunters initial positions """
    N_hunters = N
    h0 = list2mat(h)
        
    """ We need input and output must have same dim """
    M_preys = M
    p0 = list2mat(p) 
        
    """ Create output preys buffer,with the same issue """
    if N_hunters > M_preys:
        pf  = np.zeros([N_hunters,2])
    else:
        pf  = np.zeros([M_preys,2])
        
    sol = []

    """ Apply the function to all the preys """
    for j in range (M_preys):
        pf[j] = dd1(p0[j],h0,N_hunters)

    sol = mat2list(pf)
    
    """ Penalty """
    for i in range(N_hunters):
         if ((h[2*i] > 4.9 or h[2*i] < -4.9) or (h[2*i+1] > 4.9 or h[2*i+1] < -4.9)):
             for k in range(len(sol)):
                 sol[k] = sol[k] + (k+1)*10000 - np.sqrt(k+1)*1000
             break
    return sol

"""
Solver for mode 1
"""
 
def f_2(h,p,N,M):
    
    """ Read the hunters initial positions """
    N_hunters = N
    h0 = list2mat(h)
        
    """ We need input and output must have same dim """
    M_preys = M
    p0 = list2mat(p) 
        
    """ Create output preys buffer,with the same issue """
    if N_hunters > M_preys:
        pf  = np.zeros([N_hunters,2])
    else:
        pf  = np.zeros([M_preys,2])
    
    sol = []

    """ Apply the function to all the preys """
    for j in range (M_preys):
        pf[j] = dd2(p0[j],h0,N_hunters)

    sol = mat2list(pf)
        
    """ Penalty """
    for i in range(N_hunters):
         if ((h[2*i] > 4.9 or h[2*i] < -4.9) or (h[2*i+1] > 4.9 or h[2*i+1] < -4.9)):
             for k in range(len(sol)):
                 sol[k] = sol[k] + (k+1)*10000 - np.sqrt(k+1)*1000
             break
    
    return sol

"""
Solver for mode 2
"""

def f_3(h,p,N,M):
    
    """ Read the hunters initial positions """
    N_hunters = N
    h0 = list2mat(h)
        
    """ We need input and output must have same dim """
    M_preys = M
    p0 = list2mat(p)

    """ Create output preys buffer,with the same issue """
    if N_hunters > M_preys:
        pf  = np.zeros([N_hunters,2])
    else:
        pf  = np.zeros([M_preys,2])
    
    sol = []
        
    """ Apply the function to all the preys """
    for j in range (M_preys):
        pf[j] = dd3(p0[j],h0,N_hunters)
         
    sol = mat2list(pf)
    
    """ Penalty """
    for i in range(N_hunters):
         if ((h[2*i] > 4.9 or h[2*i] < -4.9) or (h[2*i+1] > 4.9 or h[2*i+1] < -4.9)):
             for k in range(len(sol)):
                 sol[k] = sol[k] + (k+1)*10000 - np.sqrt(k+1)*1000
             break
        
    return sol

"""
Solver for mode 3
"""
def f_4(h,p,N,M): 
    
    """ Read the hunters initial positions """
    N_hunters = N
    h0 = list2mat(h)
        
    """ We need input and output must have same dim """
    M_preys = M
    p0 = list2mat(p) 
        
    """ Create output preys buffer,with the same issue """
    if N_hunters > M_preys:
        pf  = np.zeros([N_hunters,2])
    else:
        pf  = np.zeros([M_preys,2])
    
    sol = []

    """ Apply the function to all the preys """
    for j in range (M_preys):
       pf[j] = dd4(p0[j],h0,N_hunters)
        
    sol = mat2list(pf)
    
    """ Penalty """
    
    for i in range(N_hunters):
         if ((h[2*i] > 4.9 or h[2*i] < -4.9) or (h[2*i+1] > 4.9 or h[2*i+1] < -4.9)):
             for k in range(len(sol)):
                 sol[k] = sol[k] + (k+1)*10000 - np.sqrt(k+1)*1000
             break  
         
    return sol

def f_action(u_actual,x_actual,x_desired,mode,N_hunters,M_preys):
     
    Q = np.matrix(np.zeros([2*M_preys,1]))
    Qd = np.matrix(np.zeros([2*M_preys,1]))
    M = np.matrix(np.zeros([2*M_preys,2*N_hunters]))
    W = np.matrix(np.zeros([2*M_preys,1]))
    
    var   = 1.0
    alpha = 100.5
    beta  = 0.5
    radius = 1.0
    angry = False
    gamma = 1          
                           
    for j in range (M_preys):
        vel = np.matrix([0,0])
        pj = np.matrix([x_actual[2*j,0],x_actual[2*j+1,0]])
        angry = False
        if mode==2 or mode==4:
            for i in range (N_hunters):
                hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
                dij = pj-hi
                if norm(dij)<radius:
                    angry=True
                    break
                
        for i in range (N_hunters):
            hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
            dij = pj-hi
            if mode == 1 or mode == 3:
                vel = vel + gamma*dij/(norm(dij)**3)
            else:
                X = (pj[0,0]-hi[0,0])**2+(pj[0,1]-hi[0,1])**2
                if angry:
                    vel = vel + alpha*dij*np.exp(-(1/(var**2))*X)
                else:
                    vel = vel + beta*alpha*dij*np.exp(-(1/(var**2))*X)
                
        Q[2*j] = vel[0,0]
        Q[2*j+1] = vel[0,1]
        
    for j in range (M_preys):
        vel = np.matrix([0,0])
        pj = np.matrix([x_desired[2*j,0],x_desired[2*j+1,0]])
        angry = False
        if mode==2 or mode==4:
            for i in range (N_hunters):
                hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
                dij = pj-hi
                if norm(dij)<radius:
                    angry=True
                    break
                
        for i in range (N_hunters):
            hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
            dij = pj-hi
            if mode == 1 or mode == 3:
                vel = vel + gamma*dij/(norm(dij)**3)
            else:
                X = (pj[0,0]-hi[0,0])**2+(pj[0,1]-hi[0,1])**2
                if angry:
                    vel = vel + alpha*dij*np.exp(-(1/(var**2))*X)
                else:
                    vel = vel + beta*alpha*dij*np.exp(-(1/(var**2))*X)
                
        Qd[2*j] = vel[0,0]
        Qd[2*j+1] = vel[0,1]
    
    
    for j in range (M_preys):
        pj = np.matrix([x_actual[2*j,0],x_actual[2*j+1,0]])
        angry = False
        if mode==2 or mode==4:
            for i in range (N_hunters):
                hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
                dij = pj-hi
                if norm(dij)<radius:
                    angry=True
                    break
                
        for i in range (N_hunters):
            hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
            dij = pj-hi
            if mode == 1 or mode == 3:
                vel = -gamma/(norm(dij)**3)
            else:
                X = (pj[0,0]-hi[0,0])**2+(pj[0,1]-hi[0,1])**2
                if angry:
                    vel = -alpha*np.exp(-(1/(var**2))*X)
                else:
                    vel = -beta*alpha*np.exp(-(1/(var**2))*X)
                
            M[2*j,2*i]     = vel
            M[2*j+1,2*i+1] = vel
   
    if mode == 1:
        solPre = Q + Qd + M*np.transpose(np.matrix(u_actual)) + 4*( x_actual - x_desired ) #2
    elif mode == 2:
        solPre = Q + Qd + M*np.transpose(np.matrix(u_actual)) + 3*( x_actual - x_desired )
    elif mode == 3:
        for j in range(M_preys):
            W[2*j]   = 0.02*(x_actual[2*j,0]+x_desired[2*j,0]) - 0.002*(x_actual[2*j+1,0]+x_desired[2*j+1,0]) 
            W[2*j+1] = -0.007*(x_actual[2*j,0]+x_desired[2*j,0]) + 0.06*(x_actual[2*j+1,0]+x_desired[2*j+1,0]) 
        
        solPre = Q + Qd + M*np.transpose(np.matrix(u_actual)) + 6*( x_actual - x_desired ) + W

    else:
        for j in range(M_preys):
            W[2*j]   = 0.02*(x_actual[2*j,0]+x_desired[2*j,0]) - 0.002*(x_actual[2*j+1,0]+x_desired[2*j+1,0]) 
            W[2*j+1] = -0.007*(x_actual[2*j,0]+x_desired[2*j,0]) + 0.06*(x_actual[2*j+1,0]+x_desired[2*j+1,0]) 
        
        solPre = Q + Qd + M*np.transpose(np.matrix(u_actual)) + 25*( x_actual - x_desired ) + W
        
    sol = []
    for i in range(2*M_preys):
        sol.append(solPre[i,0])
    for j in range(2*(N_hunters-M_preys)):
        sol.append(0)
    
        
    return sol

def f_action_adaptive(u_actual,x_actual,x_desired,mode,N_hunters,M_preys,coeff):
     
    Q = np.matrix(np.zeros([2*M_preys,2*M_preys]))
    Qd = np.matrix(np.zeros([2*M_preys,2*M_preys]))
    M = np.matrix(np.zeros([2*M_preys,2*N_hunters]))
    W = np.matrix(np.zeros([2*M_preys,1]))
    
    var   = 1.0
    beta  = 0.5
    radius = 1.0
    angry = False
        
    for j in range (M_preys):
        vel = np.matrix([0,0])
        pj = np.matrix([x_actual[2*j,0],x_actual[2*j+1,0]])
        angry = False
        if mode==2 or mode==4:
            for i in range (N_hunters):
                hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
                dij = pj-hi
                if norm(dij)<radius:
                    angry=True
                    break
                
        for i in range (N_hunters):
            hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
            dij = pj-hi
            if mode == 1 or mode == 3:
                vel = vel + dij/(norm(dij)**3)
            else:
                X = (pj[0,0]-hi[0,0])**2+(pj[0,1]-hi[0,1])**2
                if angry:
                    vel = vel + dij*np.exp(-(1/(var**2))*X)
                else:
                    vel = vel + beta*dij*np.exp(-(1/(var**2))*X)
                
        Q[2*j,2*j] = vel[0,0]
        Q[2*j+1,2*j+1] = vel[0,1]
        
    for j in range (M_preys):
        vel = np.matrix([0,0])
        pj = np.matrix([x_desired[2*j,0],x_desired[2*j+1,0]])
        angry = False
        if mode==2 or mode==4:
            for i in range (N_hunters):
                hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
                dij = pj-hi
                if norm(dij)<radius:
                    angry=True
                    break
                
        for i in range (N_hunters):
            hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
            dij = pj-hi
            if mode == 1 or mode == 3:
                vel = vel + dij/(norm(dij)**3)
            else:
                X = (pj[0,0]-hi[0,0])**2+(pj[0,1]-hi[0,1])**2
                if angry:
                    vel = vel + dij*np.exp(-(1/(var**2))*X)
                else:
                    vel = vel + beta*dij*np.exp(-(1/(var**2))*X)

        Qd[2*j,2*j] = vel[0,0]
        Qd[2*j+1,2*j+1] = vel[0,1]
                
    
    for j in range (M_preys):
        vel = np.matrix([0,0])
        pj = np.matrix([x_actual[2*j,0],x_actual[2*j+1,0]])
        angry = False
        if mode==2 or mode==4:
            for i in range (N_hunters):
                hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
                dij = pj-hi
                if norm(dij)<radius:
                    angry=True
                    break
                
        for i in range (N_hunters):
            hi = np.matrix([u_actual[2*i],u_actual[2*i+1]])
            dij = pj-hi
            if mode == 1 or mode == 3:
                vel = -1/(norm(dij)**3)
            else:
                X = (pj[0,0]-hi[0,0])**2+(pj[0,1]-hi[0,1])**2
                if angry:
                    vel = -np.exp(-(1/(var**2))*X)
                else:
                    vel = -beta*np.exp(-(1/(var**2))*X)

        M[2*j,2*j] = vel
        M[2*j+1,2*j+1] = vel
        
    if mode == 1:
        solPre = (Q + Qd)*np.transpose(coeff) + M*np.transpose(np.matrix(u_actual)) + 2*( x_actual - x_desired )
    elif mode == 2:
        solPre = (Q + Qd)*np.transpose(coeff) + M*np.transpose(np.matrix(u_actual)) + 0.1*( x_actual - x_desired )
    elif mode == 3:
        for j in range(M_preys):
            W[2*j]   = 0.02*(x_actual[2*j,0]+x_desired[2*j,0]) - 0.002*(x_actual[2*j+1,0]+x_desired[2*j+1,0]) 
            W[2*j+1] = -0.007*(x_actual[2*j,0]+x_desired[2*j,0]) + 0.06*(x_actual[2*j+1,0]+x_desired[2*j+1,0]) 
        
        (Q + Qd)*np.transpose(coeff) + M*np.transpose(np.matrix(u_actual)) + 2*( x_actual - x_desired ) + W

    else:
        for j in range(M_preys):
            W[2*j]   = 0.02*(x_actual[2*j,0]+x_desired[2*j,0]) - 0.002*(x_actual[2*j+1,0]+x_desired[2*j+1,0]) 
            W[2*j+1] = -0.007*(x_actual[2*j,0]+x_desired[2*j,0]) + 0.06*(x_actual[2*j+1,0]+x_desired[2*j+1,0]) 
        
        (Q + Qd)*np.transpose(coeff) + M*np.transpose(np.matrix(u_actual)) + 1*( x_actual - x_desired ) + W
   
    sol = []
    for i in range(2*M_preys):
        sol.append(solPre[i,0])
    for j in range(2*(N_hunters-M_preys)):
        sol.append(0)
    
        
    return sol

"""
Recursive calculator of the centroid of a N-dimensional polynom
"""   
def centroid(points):
    
    centroid   = np.matrix([0,0])
    A          = 0
    num_points = len(points)+1
    
    verts      = np.zeros([num_points,2])  
    for i in range(num_points-1):
        verts[i,0] = points[i][0]
        verts[i,1] = points[i][1]
    verts[num_points-1,:] = verts[0,:]    

    
    if (num_points > 3):
        """ Calc the polygon's signed area """
        for i in range(num_points-1):
            
            A += verts[i,0]*verts[i+1,1]-verts[i+1,0]*verts[i,1]
           
        A = 0.5*A
        
        """ Calc centroid coordinates """
        for i in range(num_points-1):
            centroid[0,0] += (verts[i,0]+verts[i+1,0])*(verts[i,0]*verts[i+1,1]-verts[i+1,0]*verts[i,1])
            centroid[0,1] += (verts[i,1]+verts[i+1,1])*(verts[i,0]*verts[i+1,1]-verts[i+1,0]*verts[i,1])
            
        centroid = centroid/(6*A) 
    else:
        
        centroid = np.matrix([verts[0,0]+(np.abs(verts[1,0]-verts[0,0])/2),verts[0,1]+(np.abs(verts[1,1]-verts[0,1])/2)])
        
    return centroid

"""
Iterator which creates the animation
"""

def animate(i,xp,yp,xh,yh,line1,line2,line3,line4):      
    
    line1.set_data(xp[i], yp[i])
    line2.set_data(xh[i], yh[i])
    line3.set_data(xp[i+3], yp[i+3])
    line4.set_data(xh[i+3], yh[i+3])
    
    return line1,line2,line3,line4,

"""
Converts a matrix of 2 columns in a one-dimensional list
"""

def mat2list(mat):
    
    list = []
    
    for i in range(len(mat)):
    
        list.append(mat[i,0])
        list.append(mat[i,1])
    
    return list

"""
Converts a one-dimensional list in a matrix of 2 columns
"""
def list2mat(list):
    
    mat = np.zeros([int(len(list)/2),2])
    
    for i in range(int(len(list)/2)):
        
        mat[i,0] = list[2*i]
        mat[i,1] = list[2*i+1]
        
    mat = np.matrix(mat)
    return mat

"""
Solve the continuous time lqr controller.
 
dx/dt = A x + B u
 
cost = integral x.T*Q*x + u.T*R*u
"""    
def lqr(A,B,Q,R):

    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
 
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
 
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
 
    return K, X, eigVals
 
    
"""
Solve the discrete time lqr controller.
 
x[k+1] = A x[k] + B u[k]
 
cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
"""
def dlqr(A,B,Q,R):

    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
     
    return K, X, eigVals


"""
This function draws an object in the location specified by loc,
with the colour specified by 'colour' and with the size specified
by 'size':
    
    * 'loc' must be a 3 dimension vector [x,y,th]
    
    * 'size' must be a scalar related to the dimension of the field (D)
    
"""

def drawbrobot(loc,colour,size):
    
    """ Setting the characteristic dimension of the object """
    side           = 0.5*size/50

    """ Setting the square which represents the object """
    bottom_right   = np.array([side,-side,1])
    bottom_left    = np.array([-side,-side,1])
    top_right      = np.array([side,side,1])
    top_left       = np.array([-side,side,1])
    
    """ Extract the x,y and theta variables from loc """
    x              = loc[0]
    y              = loc[1]
    th             = loc[2]
    
    """ Homogeneous transform matrix constructed from x,y aand theta """
    H              = np.array([[np.cos(th), -np.sin(th), x],
                              [np.sin(th), np.cos(th), y], 
                              [0,        0 ,        1]])
    
    """ The square is built up """
    square         = np.array([bottom_right, bottom_left, top_left, top_right, bottom_right])
    
    """ The square is transformed to represent the location of the object """
    figure         = np.dot(H,np.transpose(square))
    
    """ The square is plotted with the colour specified """
    plt.plot(figure[0,:], figure[1,:], colour)
  
  
def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None, legend = None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean, label = legend)
    
class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed
 
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)
 
        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)
 
        return patch