# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:32:59 2018

@author: Eduardo
"""
from functions import dlqr,setup,f_1,f_2,f_3,f_4,list2mat#,drawbrobot #,dd1
import random
import numpy as np
from scipy.optimize import root
import control
from numpy.linalg   import norm
#import matplotlib.pyplot as plt
#from matplotlib.path import Path
#import matplotlib.patches as patches

"""

The idea is to implement, in a single function, the whole pack of perfomance of
the algoritm:
    
    1) We calculate, given a desired position of the preys, a desired hunters' 
    position which makes possible to linearalize the system, using the numeric 
    solver
    
    2) With both desired position sets, the control law is obteined
    
    3) The config dictionary is constructed and the setup is played

"""
def program(M0,N0,mode,h0,p0,noise_h,noise_p,controller):
    
    M_preys = M0
    N_hunters = N0
    Ts = 0.0005
#    plt.rcParams.update({'font.size': 15})
#    h_desired = [3,3,3,-3,-3,-3,-3,3]
#    p_desired = [0,0]
#    M_preys = 1
#    N_hunters = 4
#    D = 5
#    """ The main layout of the animation """
#    ax = plt.axes(xlim=(-D,D), ylim=(-D,D))
#    
#    """
#    Plot preserve
#    """
#    vertsP = [
#            (-D, -D),  # left, bottom
#            (-D, D),  # left, top
#            (D, D),  # right, top
#            (D, -D),  # right, bottom
#            (-D, -D),  # ignored
#                                    ]
#    vertsPCodes = [
#            Path.MOVETO,
#            Path.LINETO,
#            Path.LINETO,
#            Path.LINETO,
#            Path.CLOSEPOLY,
#            ]
#                            
#    vertsPPath = Path(vertsP, vertsPCodes)
#    vertsPPatch = patches.PathPatch(vertsPPath, facecolor='none', lw=2)
#    ax.add_patch(vertsPPatch)
#    for j in range(M_preys):  
#        ax.plot(p_desired[2*j], p_desired[2*j+1], 'o', ms=15, c='g')
#    for i in range(N_hunters):  
#        ax.plot(h_desired[2*i], h_desired[2*i+1], 'D', ms=15, c='b')
#        
#    plt.show()
#    plt.savefig("G:/Unidades de equipo/EduardoSebastian/TFG/04_Imagenes/Figura3.png")
#    
#    plt.close('all')
#    
#    h_desired = [2,0,0,2,-2,0,0,-2]
#    p_desired = [0,0]
#    M_preys = 1
#    N_hunters = 4
#    D = 5
#    """ The main layout of the animation """
#    ax = plt.axes(xlim=(-D,D), ylim=(-D,D))
#    
#    """
#    Plot preserve
#    """
#    vertsP = [
#            (-D, -D),  # left, bottom
#            (-D, D),  # left, top
#            (D, D),  # right, top
#            (D, -D),  # right, bottom
#            (-D, -D),  # ignored
#                                    ]
#    vertsPCodes = [
#            Path.MOVETO,
#            Path.LINETO,
#            Path.LINETO,
#            Path.LINETO,
#            Path.CLOSEPOLY,
#            ]
#                            
#    vertsPPath = Path(vertsP, vertsPCodes)
#    vertsPPatch = patches.PathPatch(vertsPPath, facecolor='none', lw=2)
#    ax.add_patch(vertsPPatch)
#    for j in range(M_preys):  
#        ax.plot(p_desired[2*j], p_desired[2*j+1], 'o', ms=15, c='g')
#    for i in range(N_hunters):  
#        ax.plot(h_desired[2*i], h_desired[2*i+1], 'D', ms=15, c='b')
#        
#    plt.show()
#    plt.savefig("G:/Unidades de equipo/EduardoSebastian/TFG/04_Imagenes/Figura4.png")
#
#    plt.close('all')
#
#    h_desired = h0
#    p_desired = p0
#    M_preys = 3
#    N_hunters = 6
#    D = 5
#    """ The main layout of the animation """
#    ax = plt.axes(xlim=(-D,D), ylim=(-D,D))
#    
#    """
#    Plot preserve
#    """
#    vertsP = [
#            (-D, -D),  # left, bottom
#            (-D, D),  # left, top
#            (D, D),  # right, top
#            (D, -D),  # right, bottom
#            (-D, -D),  # ignored
#                                    ]
#    vertsPCodes = [
#            Path.MOVETO,
#            Path.LINETO,
#            Path.LINETO,
#            Path.LINETO,
#            Path.CLOSEPOLY,
#            ]
#                            
#    vertsPPath = Path(vertsP, vertsPCodes)
#    vertsPPatch = patches.PathPatch(vertsPPath, facecolor='none', lw=2)
#    ax.add_patch(vertsPPatch)
#    for j in range(M_preys):  
#        ax.plot(p_desired[2*j], p_desired[2*j+1], 'o', ms=12, c='g')
#    for i in range(N_hunters):  
#        ax.plot(h_desired[2*i], h_desired[2*i+1], 'D', ms=12, c='b')
#        
#    plt.show()
#    plt.savefig("G:/Unidades de equipo/EduardoSebastian/TFG/04_Imagenes/Figura1.png")
#
#    plt.close('all')

    if controller == "linear":
        if mode == 1:
            h  = root(f_1, h0, args = (p0,N0,M0), method = 'hybr', tol = 10e-10)
        elif mode == 2:
            h  = root(f_2, h0, args = (p0,N0,M0), method = 'lm', tol = 10e-10)
        elif mode == 3:
            h  = root(f_3, h0, args = (p0,N0,M0), method = 'hybr', tol = 10e-10)
        elif mode == 4:
            h  = root(f_4, h0, args = (p0,N0,M0), method = 'lm', tol = 10e-10)
        else:
            h  = root(f_1, h0, args = (p0,N0,M0), method = 'hybr', tol = 10e-10)
            
            
        p  = p0  
        
        """ REMOVE NEGLIBLE HUNTERS """
        hunt = []
        N_hunters = N0
        for i in range(N_hunters):
            if ((h.x[2*i] < 10 and h.x[2*i] > -10) and (h.x[2*i+1] < 10 and h.x[2*i+1] > -10)):
                hunt.append(h.x[2*i])
                hunt.append(h.x[2*i+1])
            else:
                N0 -= 1
                print("Hunter neglected is: " + str(i))  

    
        """ CHECK THE SOLVER """
        
        hi0 = list2mat(hunt)
        pj0 = list2mat(p)  
        
        if not h.success:
            print("not success")
            XP = -1
            YP = -1
            XH = -1
            YH = -1
            error = -1
            found = False
            v_preys = None
            v_hunters = None
            converge = False
            return XP,YP,XH,YH,error,v_preys,v_hunters,found,converge
        else:
            """
            Now is time to simulate the behavior from the positions calculated previously
            """    
            print("success")
            N_hunters = N0
            M_preys   = M0
            mode      = mode
            h         = hunt
            
    else:
        p  = p0 
        h = h0
        hi0 = list2mat(h)
        pj0 = list2mat(p)
        
#    h_desired = h.x
#    p_desired = p0
#    M_preys = 3
#    N_hunters = 6
#    D = 5
#    """ The main layout of the animation """
#    ax = plt.axes(xlim=(-D,D), ylim=(-D,D))
#    
#    """
#    Plot preserve
#    """
#    vertsP = [
#            (-D, -D),  # left, bottom
#            (-D, D),  # left, top
#            (D, D),  # right, top
#            (D, -D),  # right, bottom
#            (-D, -D),  # ignored
#                                    ]
#    vertsPCodes = [
#            Path.MOVETO,
#            Path.LINETO,
#            Path.LINETO,
#            Path.LINETO,
#            Path.CLOSEPOLY,
#            ]
#                            
#    vertsPPath = Path(vertsP, vertsPCodes)
#    vertsPPatch = patches.PathPatch(vertsPPath, facecolor='none', lw=2)
#    ax.add_patch(vertsPPatch)
#    for j in range(M_preys):  
#        ax.plot(p_desired[2*j], p_desired[2*j+1], 'o', ms=12, c='g')
#    for i in range(N_hunters):  
#        ax.plot(h_desired[2*i], h_desired[2*i+1], 'D', ms=12, c='b')
#        
#    plt.show()
#    plt.savefig("G:/Unidades de equipo/EduardoSebastian/TFG/04_Imagenes/Figura2.png")
#
#    plt.close('all')
#    time.sleep(2)
    
    
        

###############################################################################
######  LINEAR CONTROLLER #####################################################
###############################################################################    
    
    if (controller == "linear"):
        
        """ Calculate the state-space system """
        A = np.zeros([2*M_preys, 2*M_preys])
        B = np.zeros([2*M_preys, 2*N_hunters]) 
        C = np.eye(2*M_preys)
        D = np.zeros([2*M_preys, 2*N_hunters])
    ###############################################################################
    ######  MODE 1 CONTROL SYSTEM #################################################
    ###############################################################################
     
        if mode == 1:        
            for j in range(M_preys):
                
                a = np.zeros([2,2])
                b = np.zeros([2,2])
                I = np.matrix([[1,0],[0,1]])
                for i in range(N_hunters):                 
                    term = (I*norm(hi0[i]-pj0[j])**3)
                    a += (term+3*(norm(hi0[i]-pj0[j])*np.transpose(hi0[i]-pj0[j])*(hi0[i]-pj0[j])))/(norm(hi0[i]-pj0[j])**5)
                    b = (-term+3*(norm(hi0[i]-pj0[j])*np.transpose(hi0[i]-pj0[j])*(hi0[i]-pj0[j])))/(norm(hi0[i]-pj0[j])**5)
                    
                    B[2*j,2*i]    = b[0,0]
                    B[2*j,2*i+1]  = b[0,1]
                    B[2*j+1,2*i]  = b[1,0]
                    B[2*j+1,2*i+1]= b[1,1]
                    
                A[2*j,2*j]    = a[0,0]
                A[2*j,2*j+1]  = a[0,1]
                A[2*j+1,2*j]  = a[1,0]
                A[2*j+1,2*j+1]= a[1,1]
 
    ###############################################################################
    ######  MODE 2 CONTROL SYSTEM #################################################
    ###############################################################################
               
        elif mode == 2:
            
            var   = 1.0
            alpha = 100.5
            beta  = 0.5
            radius = 1.0
            angry = False
            
            for j in range(M_preys):
            
                a = np.zeros([2,2])
                b = np.zeros([2,2])
                I = np.matrix([[1,0],[0,1]])
                
                for i in range(N_hunters):
                    if norm(pj0[j]-hi0[i]) < radius:
                        angry = True
                        break
                    else:
                        angry = False
                
                for i in range(N_hunters):
                    Xi = (pj0[j,0]-hi0[i,0])**2 + (pj0[j,1]-hi0[i,1])**2

                    term1 = np.exp(-(1/(var**2))*Xi)
                    term2 = np.matrix([[0,0],[0,0]])
                    term2[0,0] = (pj0[0,0]-hi0[0,0])*(pj0[0,0]-hi0[0,0])
                    term2[0,1] = (pj0[0,0]-hi0[0,0])*(pj0[0,1]-hi0[0,1])
                    term2[1,0] = (pj0[0,0]-hi0[0,0])*(pj0[0,1]-hi0[0,1])
                    term2[1,1] = (pj0[0,1]-hi0[0,1])*(pj0[0,1]-hi0[0,1])
                    
                    if angry:  

                        a +=  I*alpha*term1 - 2*alpha*(1/(var**2))*term1*term2 
                        b =   -I*alpha*term1 + 2*alpha*(1/(var**2))*term1*term2
                    else:
                        a +=  I*alpha*beta*term1 - 2*alpha*beta*(1/(var**2))*term1*term2  #
                        b =   -I*alpha*beta*term1 + 2*alpha*beta*(1/(var**2))*term1*term2 #
                    
                    B[2*j,2*i]    = b[0,0]
                    B[2*j,2*i+1]  = b[0,1]
                    B[2*j+1,2*i]  = b[1,0]
                    B[2*j+1,2*i+1]= b[1,1]
                    
                A[2*j,2*j]    = a[0,0]
                A[2*j,2*j+1]  = a[0,1]
                A[2*j+1,2*j]  = a[1,0]
                A[2*j+1,2*j+1]= a[1,1]
                    
    
    ###############################################################################
    ######  MODE 3 CONTROL SYSTEM #################################################
    ###############################################################################
                   
        elif mode == 3:
            
            for j in range(M_preys):
                
                a = np.zeros([2,2])
                b = np.zeros([2,2])
                I = np.matrix([[1,0],[0,1]])
                
                for i in range(N_hunters):
                    term = (I*norm(hi0[i]-pj0[j])**3)
                    a += (term+3*(norm(hi0[i]-pj0[j])*np.transpose(hi0[i]-pj0[j])*(hi0[i]-pj0[j])))/(norm(hi0[i]-pj0[j])**5)
                    b = (-term+3*(norm(hi0[i]-pj0[j])*np.transpose(hi0[i]-pj0[j])*(hi0[i]-pj0[j])))/(norm(hi0[i]-pj0[j])**5)
                    
                    B[2*j,2*i]    = b[0,0]
                    B[2*j,2*i+1]  = b[0,1]
                    B[2*j+1,2*i]  = b[1,0]
                    B[2*j+1,2*i+1]= b[1,1]
                
                a += np.matrix([[0.02,-0.002],[-0.007,0.06]])
                A[2*j,2*j]    = a[0,0]
                A[2*j,2*j+1]  = a[0,1]
                A[2*j+1,2*j]  = a[1,0]
                A[2*j+1,2*j+1]= a[1,1]
    
    
    ###############################################################################
    ######  MODE 4 CONTROL SYSTEM #################################################
    ###############################################################################
                   
        elif mode == 4:
            
            var   = 1.0
            alpha = 100.5
            beta  = 0.5
            radius = 1.0
            angry = False
            
            for j in range(M_preys):
            
                a = np.zeros([2,2])
                b = np.zeros([2,2])
                I = np.matrix([[1,0],[0,1]])
                
                for i in range(N_hunters):
                    if norm(pj0[j]-hi0[i])  < radius:
                        angry = True
                        break
                    else:
                        angry = False
                        
                for i in range(N_hunters):
                    Xi = (pj0[j,0]-hi0[i,0])**2 + (pj0[j,1]-hi0[i,1])**2

                    term1 = np.exp(-(1/(var**2))*Xi)
                    term2 = np.matrix([[0,0],[0,0]])
                    term2[0,0] = (pj0[0,0]-hi0[0,0])*(pj0[0,0]-hi0[0,0])
                    term2[0,1] = (pj0[0,0]-hi0[0,0])*(pj0[0,1]-hi0[0,1])
                    term2[1,0] = (pj0[0,0]-hi0[0,0])*(pj0[0,1]-hi0[0,1])
                    term2[1,1] = (pj0[0,1]-hi0[0,1])*(pj0[0,1]-hi0[0,1])
                    
                    if angry:  

                        a +=  I*alpha*term1 - 2*alpha*(1/(var**2))*term1*term2 
                        b =   -I*alpha*term1 + 2*alpha*(1/(var**2))*term1*term2
                    else:
                        a +=  I*alpha*beta*term1 - 2*alpha*beta*(1/(var**2))*term1*term2  #
                        b =   -I*alpha*beta*term1 + 2*alpha*beta*(1/(var**2))*term1*term2 #
                        
                    B[2*j,2*i]    = b[0,0]
                    B[2*j,2*i+1]  = b[0,1]
                    B[2*j+1,2*i]  = b[1,0]
                    B[2*j+1,2*i+1]= b[1,1]
                
                a += np.matrix([[0.02,-0.002],[-0.007,0.06]])
                A[2*j,2*j]    = a[0,0]
                A[2*j,2*j+1]  = a[0,1]
                A[2*j+1,2*j]  = a[1,0]
                A[2*j+1,2*j+1]= a[1,1]
                
    ###############################################################################
    ######  DEFAULT MODE CONTROL SYSTEM ###########################################
    ###############################################################################
                    
        else: 
            
            for j in range(M_preys):
                
                a = np.zeros([2,2])
                b = np.zeros([2,2])
                I = np.matrix([[1,0],[0,1]])
                for i in range(N_hunters):                 
                    term = (I*norm(hi0[i]-pj0[j])**3)
                    a += (term+3*(norm(hi0[i]-pj0[j])*np.transpose(hi0[i]-pj0[j])*(hi0[i]-pj0[j])))/(norm(hi0[i]-pj0[j])**5)
                    b = (-term+3*(norm(hi0[i]-pj0[j])*np.transpose(hi0[i]-pj0[j])*(hi0[i]-pj0[j])))/(norm(hi0[i]-pj0[j])**5)
                    
                    B[2*j,2*i]    = b[0,0]
                    B[2*j,2*i+1]  = b[0,1]
                    B[2*j+1,2*i]  = b[1,0]
                    B[2*j+1,2*i+1]= b[1,1]
                    
                A[2*j,2*j]    = a[0,0]
                A[2*j,2*j+1]  = a[0,1]
                A[2*j+1,2*j]  = a[1,0]
                A[2*j+1,2*j+1]= a[1,1]
                    
                
    
        """ Ensures control functions do not delete rows"""
        for i in range(2*M_preys):
            if sum(A[i])==0.0:
                A[i] = A[i] + 1e-10
            if sum(B[i])==0.0:
                B[i] = B[i] + 1e-10
        
        
        cont_system = control.StateSpace(A,B,C,D)
        disc_system = control.c2d(cont_system,Ts)
        
        
        """ LQR METHOD """
        try:
            if mode == 1 or mode == 3:
                [L,S,e] = dlqr(disc_system.A,disc_system.B,1*np.eye(2*M_preys),2*np.eye(2*N_hunters))
            else:
                [L,S,e] = dlqr(disc_system.A,disc_system.B,1*np.eye(2*M_preys),0.1*np.eye(2*N_hunters))
        except:
            print("not success")
            XP = -1
            YP = -1
            XH = -1
            YH = -1
            error = -1
            found = False
            v_preys = None
            v_hunters = None
            converge = False
            return XP,YP,XH,YH,error,v_preys,v_hunters,found,converge
                
        """ POLE ASSIGNMENT METHOD """
        #poles = np.exp(-1 * np.ones([N_hunters])*Ts)
        #L = control.place(disc_system.A,disc_system.B,poles) 

###############################################################################
######  NON - LINEAR CONTROLLER ###############################################
############################################################################### 
            
    elif(controller == "non-linear"):
        Ts = 0.01
        L = 0

###############################################################################
######  NON - LINEAR CONTROLLER WITH ADAPTIVE BEHAVIOR ########################
############################################################################### 
        
    elif(controller == "adaptive"):
        Ts = 0.1
        L = np.ones([1,2*M_preys])*1
        

###############################################################################
######  NON - LINEAR CONTROLLER WITH ADAPTIVE BEHAVIOR AND DISTRIBUTED FORM ###
############################################################################### 
                           
    else:
        return -1

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
    """ Place zero position of hunters and preys """
    
    h_desired = []
    h_ini     = []
    for i in range(len(h)):
        h_desired.append(h[i])
        h_ini.append(h[i] + random.uniform(-noise_h,noise_h))  
    
    
    p_desired = []
    p_ini     = []       
    for j in range(len(p)):
        if (p[j]!=0.0):
            p_desired.append(p[j])
            p_ini.append(p[j] + random.uniform(-noise_p,noise_p))
               

#    p_ini = [1.0,1.0] 
#    p_desired = [0.0,0.0]
#    h_ini = [-3.0,0.0,0.0,3.0]
#    p_desired = [0.3,-0.1,-1.0,1.0,-0.8,1.5]
#    p_ini = [1.0,-0.7,-0.5,1.5,-0.4,2.0]
#    h_ini = [-0.8,1.2,-1.3,1.5,0.2,0.2,0.3,1.7,3.0,0.0]

    config = {'SampleTime':Ts,
              'K'         :500,
              'D'         :5.0,
              'draw'      :False,
              'anim'      :False,
              'interval'  :10,
              'H0'        :h_ini,
              'P0'        :p_ini,
              'H_desired' :h_desired,
              'P_desired' :p_desired,
              'L'         :L,
              'saveAnim'  :False,
              'controller':controller
            } 
    
    
    XP,YP,XH,YH,error,v_preys,v_hunters,converge = setup(N_hunters,M_preys,mode,config)
    found = True
        
    return XP,YP,XH,YH,error,v_preys,v_hunters,found,converge


