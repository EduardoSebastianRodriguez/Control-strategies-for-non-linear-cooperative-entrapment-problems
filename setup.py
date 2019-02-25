# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:31:46 2018

@author: Eduardo

"""

import numpy as np
import random
import prey
import hunter
import plotObject
import csv
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from numpy.linalg   import norm
from matplotlib import animation
import time

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
    H = [hunter.hunter() for i in range(N_hunters)] #Vector of N_hunters x 1 hunters
    P = [prey.prey() for j in range(M_preys)]       #Vector of M_preys x 1 preys
    XH = [[] for i in range(N_hunters)]             #Vector storing the X coordinate of the hunters in each sample
    YH = [[] for i in range(N_hunters)]             #Vector storing the Y coordinate of the hunters in each sample
    XP = [[] for j in range(M_preys)]               #Vector storing the X coordinate of the preys in each sample
    YP = [[] for j in range(M_preys)]               #Vector storing the Y coordinate of the preys in each sample
    
    if (len(config)<12):
        
        if 'SampleTime' in config:
            T    = config['SampleTime']
        else:
            T        = 1
        
        if 'D' in config:
            D    = config['D']
        else:
            D        = 10
            
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
            
        if 'H0' in config:
            h_actual = config['H0']
        else:
            h_actual = [[0,0] for i in range (N_hunters)]
                
        if 'P0' in config:
            p_actual = config['P0']
        else:
            p_actual = [[0,0] for j in range(M_preys)]
    
    
        if 'H_desired' in config:
            h_desired = config['H_desired']
            for i in range(N_hunters):
                H[i].setLabel(i) 
                H[i].setLocalization(np.matrix([config['H_desired'][i][0],config['H_desired'][i][1]]),0)
        else:
            for i in range(N_hunters):
                H[i].setLabel(i) 
                H[i].setLocalization(np.matrix([random.uniform(D/4,3*D/4),random.uniform(D/4,3*D/4)]),random.random()*2*np.pi)
                
        if 'P_desired' in config:
            p_desired = config['P_desired']
            for j in range(M_preys):
                P[j].setLabel(j)
                P[j].setMode(mode)
                P[j].setLocalization(np.matrix([config['H_desired'][i][0],config['H_desired'][i][1]]),0)
        else:
            for j in range(M_preys):
                P[j].setLabel(j)
                P[j].setMode(mode)
                P[j].setLocalization(np.matrix([random.uniform(D/4,3*D/4),random.uniform(D/4,3*D/4)]),random.random()*2*np.pi)
    else:
        
        T        = config['SampleTime']
        D        = config['D']
        K        = config['K']
        draw     = config['draw']
        anim     = config['anim']
        interval = config['interval']
        L        = np.matrix(config['L'])
        h_actual = config['H0']
        p_actual = config['P0']
        h_desired = config['H_desired']
        p_desired = config['P_desired']
        saveAnim  = config['saveAnim']
        
        for i in range(N_hunters):
            H[i].setLabel(i) 
            H[i].setLocalization(np.matrix(config['H0'][i]),0)
        for j in range(M_preys):
            P[j].setLabel(j)
            P[j].setMode(mode)
            P[j].setLocalization(np.matrix(config['P0'][j]),0)
    
    """
    Error metric: norm of the distance of each prey from its desired final position
    """
    error            = np.zeros([K,M_preys])
    v_preys          = np.zeros([K,M_preys])
    v_hunters        = np.zeros([K,N_hunters])
    converge = False
    
    """
    2. If we want to draw the historical of positions of both
       hunters and preys, first we need to set the preserve. If
       draw == FALSE, jump to point 3
    """
    if (draw):
        
            """ The main layout of the animation """
            ax = plt.axes(xlim=(0,D), ylim=(0,D))
            ax.set_title("Preserve")
            ax.set_xlabel("X dimension")
            ax.set_ylabel("Y dimension")
            
            """
            Plot preserve
            """
            vertsP = [
                    (0, 0),  # left, bottom
                    (0, D),  # left, top
                    (D, D),  # right, top
                    (D, 0),  # right, bottom
                    (0, 0),  # ignored
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
            ax.plot([row[0] for row in p_desired], [row[1] for row in p_desired], 'x', ms=8, c='g')
            ax.plot([row[0] for row in h_desired], [row[1] for row in h_desired], 'x', ms=8, c='b')
    
    """
    3. Open the LOG file
    """
    csvFile = open('LOG.csv', 'a')
    csvWriter = csv.writer(csvFile)
    
    """
    4. Control law
    """
  
    x_actual  = []
    x_desired = []
    u_desired = []
    u_actual  = []
    for j in range(M_preys):          
        x_actual.append (p_actual[j][0])
        x_actual.append (p_actual[j][1])
        x_desired.append(p_desired[j][0])
        x_desired.append(p_desired[j][1])
        
    for i in range(N_hunters):
        u_desired.append(h_desired[i][0])
        u_desired.append(h_desired[i][1])
        u_actual.append(h_actual[i][0])
        u_actual.append(h_actual[i][1])
        
    x_actual = np.transpose(np.matrix(x_actual))
    x_desired= np.transpose(np.matrix(x_desired))
    u_desired= np.transpose(np.matrix(u_desired))
    u_actual = np.transpose(np.matrix(u_actual))
    
    
    """
    5. Play the simulation
    """    

    for k in range(K): 
        
        
        """
        Run the interaction of the preys, this is applied with every prey
        """                              
        for j in range(M_preys):
            
            """ Interaction of the prey """
            v_p = P[j].interaction(H,P,T,D)
            
            """ Store the coordinates of the prey """
            XP[j].append(P[j].prePosition[0,0])
            YP[j].append(P[j].prePosition[0,1])
            x_actual[2*j] = P[j].prePosition[0,0]
            x_actual[2*j+1] = P[j].prePosition[0,1]
            p_actual[j][0] = P[j].prePosition[0,0]
            p_actual[j][1] = P[j].prePosition[0,1]
            
            """ Calc actual error metric of this prey """
            error[k,j] = norm(np.matrix(p_actual[j]) - np.matrix(p_desired[j]))
            v_preys[k,j] = v_p
            
            """ Save the coordinates in the LOG file """
            csvWriter.writerow([k,j,'P',P[j].prePosition[0,0],P[j].prePosition[0,1]])
            
            """ If draw == TRUE, it is time to plot the actual position of the prey """
            if (draw):
                plotObject.drawbrobot([P[j].prePosition[0,0],P[j].prePosition[0,1],P[j].orientation],'g',D)
            
        
        """ Calc control action (hunters' position) """
        inc_x = x_actual - x_desired
        inc_u = -L*inc_x
        u_actual = inc_u + u_desired

        """
        Run the interaction of the hunters, this is applied by every hunter
        """           
        
        for i in range(N_hunters):
            
            
            """ Interaction of the hunter """
            v_h = H[i].interaction(np.transpose(np.matrix([np.array(u_actual)[2*i],np.array(u_actual)[2*i+1]])),D,T)
            
            """ Store the coordinates of the hunter """
            XH[i].append(H[i].prePosition[0,0])
            YH[i].append(H[i].prePosition[0,1])
            u_actual[2*i] = H[i].prePosition[0,0]
            u_actual[2*i+1] = H[i].prePosition[0,1]
            
            """ Save v of the hunters """
            v_hunters[k,i] = v_h
            
            """ Save the coordinates in the LOG file """
            csvWriter.writerow([k,i,'H',H[i].prePosition[0,0],H[i].prePosition[0,1]])
            
            """ If draw == TRUE, it is time to plot the actual position of the hunter """
            if (draw):
                plotObject.drawbrobot([H[i].prePosition[0,0],H[i].prePosition[0,1],H[i].orientation],'b',D)    

        if np.average(v_preys[k,:]) < 0.25:
            converge = True
            K = k
            break
        
    csvFile.close() 
    
    if(anim):
        
        
        """ The frame is cleared to create the next frame """
        fig = plt.figure()
        ax = plt.axes(xlim=(0, D), ylim=(0, D))
        ax.set_title("Preserve")
        ax.set_xlabel("X dimension")
        ax.set_ylabel("Y dimension")
                  
        """
        Plot preserve
        """
        vertsP = [
                                        (0, 0),  # left, bottom
                                        (0, D),  # left, top
                                        (D, D),  # right, top
                                        (D, 0),  # right, bottom
                                        (0, 0),  # ignored
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

        ax.plot([row[0] for row in p_desired], [row[1] for row in p_desired], 'x', ms=8, c='g')
        ax.plot([row[0] for row in h_desired], [row[1] for row in h_desired], 'x', ms=8, c='b')
        
        line1, = ax.plot([], [], 'o', ms=8, c='g')
        line2, = ax.plot([], [], 'o', ms=8, c='b')
        line3, = ax.plot([], [], 'o', ms=4, c='#3AAD74')
        line4, = ax.plot([], [], 'o', ms=4, c='#7984D0')
        
        xp = np.transpose(XP)
        yp = np.transpose(YP)
        xh = np.transpose(XH)
        yh = np.transpose(YH)
        
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
        
    return XP,YP,XH,YH,error,v_preys,v_hunters,converge



def f_0(h,p,N,M):
    
    """ Read the hunters initial positions """
    N_hunters = N
    h0 = np.zeros([N_hunters,2])
    for i in range(N_hunters):
        h0[i,0] = h[2*i]  
        h0[i,1] = h[2*i+1]
        
    """ We need input and output must have same dim """
    M_preys = M
    p0 = np.zeros([M_preys,2])
    for j in range(M_preys):
        p0[j,0] = p[2*j] 
        p0[j,1] = p[2*j+1]  
        
    """ Create output preys buffer,with the same issue """
    if N_hunters > M_preys:
        pf  = np.zeros([N_hunters,2])
    else:
        pf  = np.zeros([M_preys,2])
        
    sol = []

    """ Apply the function to all the preys """
    for j in range (M_preys):
        for i in range (N_hunters):
            pf[j] -= (p0[j]-h0[i])/(norm(h0[i]-p0[j])**3) 

    if N_hunters > M_preys:
        for j in range(N_hunters):
            sol.append(pf[j,0])
            sol.append(pf[j,1])
    else:
        for i in range(M_preys):
            sol.append(pf[i,0])
            sol.append(pf[i,1])
    
    return sol
    
def f_1(h,p,N,M):
    
    """ Read the hunters initial positions """
    N_hunters = N
    h0 = np.zeros([N_hunters,2])
    for i in range(N_hunters):
        h0[i,0] = h[2*i]  
        h0[i,1] = h[2*i+1]
        
    """ We need input and output must have same dim """
    M_preys = M
    p0 = np.zeros([M_preys,2])
    for j in range(M_preys):
        p0[j,0] = p[2*j] 
        p0[j,1] = p[2*j+1]  
        
    """ Create output preys buffer,with the same issue """
    if N_hunters > M_preys:
        pf  = np.zeros([N_hunters,2])
    else:
        pf  = np.zeros([M_preys,2])
    
    
    sol = []

    """ Apply the function to all the preys """
    for j in range (M_preys):
        for i in range (N_hunters):
#            if norm(p0[j]-h0[i]) < 0.5:
            pf[j] -= 0.5*(p0[j]-h0[i])*np.exp(-(1/(50**2))*(p0[j]-h0[i])*np.transpose(p0[j]-h0[i]))
#            else:
#                pf[j] -= 0.05*(p0[j]-h0[i])*np.exp(-(1/(50**2))*(p0[j]-h0[i])*np.transpose(p0[j]-h0[i]))

    if N_hunters > M_preys:
        for j in range(N_hunters):
            sol.append(pf[j,0])
            sol.append(pf[j,1])
    else:
        for i in range(M_preys):
            sol.append(pf[i,0])
            sol.append(pf[i,1])
      
    return sol

def f_2(h,p,N,M):
    
    """ Read the hunters initial positions """
    N_hunters = N
    h0 = np.zeros([N_hunters,2])
    for i in range(N_hunters):
        h0[i,0] = h[2*i]  
        h0[i,1] = h[2*i+1]
        
    """ We need input and output must have same dim """
    M_preys = M
    p0 = np.zeros([M_preys,2])
    for j in range(M_preys):
        p0[j,0] = p[2*j] 
        p0[j,1] = p[2*j+1]  
        
    """ Create output preys buffer,with the same issue """
    if N_hunters > M_preys:
        pf  = np.zeros([N_hunters,2])
    else:
        pf  = np.zeros([M_preys,2])
    
    sol = []
    
    W = np.zeros([2,2])
    W[0,0] = 0.01
    W[0,1] = -0.001
    W[1,0] = -0.005
    W[1,1] = 0.03 
    
    """ Apply the function to all the preys """
    for j in range (M_preys):
        for i in range (N_hunters):
            pf[j] -= (p0[j]-h0[i])/(norm(h0[i]-p0[j])**3) 
                
        
        pf[j] += np.array([p0[j,0]*W[0,0] + p0[j,1]*W[1,0],p0[j,0]*W[0,1] + p0[j,1]*W[1,1]])
        
        
    if N_hunters > M_preys:
        for j in range(N_hunters):
            sol.append(pf[j,0])
            sol.append(pf[j,1])
    else:
        for i in range(M_preys):
            sol.append(pf[i,0])
            sol.append(pf[i,1])
        
    return sol


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


def animate(i,xp,yp,xh,yh,line1,line2,line3,line4):      
    
    line1.set_data(xp[i], yp[i])
    line2.set_data(xh[i], yh[i])
    line3.set_data(xp[i+3], yp[i+3])
    line4.set_data(xh[i+3], yh[i+3])
    
    return line1,line2,line3,line4,