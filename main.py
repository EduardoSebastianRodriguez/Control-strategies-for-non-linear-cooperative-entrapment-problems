# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:32:59 2018

@author: Eduardo
"""
from setup import setup,f_0,f_1,f_2
import random
import numpy as np
from scipy.optimize import root
import control
import lqr
from numpy.linalg   import norm


"""

The idea is to implement, in a single function, the whole pack of perfomance of
the algoritm:
    
    1) We calculate, given a desired position of the preys, a desired hunters' 
    position which makes possible to linearalize the system, using the numeric 
    solver
    
    2) With both desired position sets, the control law is obteined
    
    3) The config dictionary is constructed and the setup is played

"""
def program(M0,N0,mode,h0,p0,noise_h,noise_p):

    if mode == 0:
        h  = root(f_0, h0, args = (p0,N0,M0), method = 'hybr')
    elif mode == 1:
        h  = root(f_1, h0, args = (p0,N0,M0), method = 'hybr')
    elif mode == 2:
        h  = root(f_2, h0, args = (p0,N0,M0), method = 'hybr')
    else:
        h  = root(f_0, h0, args = (p0,N0,M0), method = 'hybr')
              
    p  = p0
    
    if not (h.success):
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
    else:
        """
        Now is time to simulate the behavior from the positions calculated previously
        """    
        N_hunters = N0
        M_preys   = M0
        mode      = mode
        
        """ Place zero position of hunters and preys """
        h_desired = [[] for i in range(N_hunters)]
        h_ini     = [[] for i in range(N_hunters)]

        for i in range(N_hunters):
            h_desired[i].append( h.x[2*i] + 5 )  
            h_desired[i].append( h.x[2*i+1] + 5 )
            h_ini[i].append( h.x[2*i] + 5 + random.uniform(-noise_h,noise_h))  
            h_ini[i].append( h.x[2*i+1] + 5 + random.uniform(-noise_h,noise_h))
        
        p_desired = [[] for j in range(M_preys)]
        p_ini     = [[] for j in range(M_preys)]
        for j in range(M_preys):
            p_desired[j].append( p[2*j] + 5 )
            p_desired[j].append( p[2*j+1] + 5 ) 
            p_ini[j].append( p[2*j] + 5 + random.uniform(-noise_p,noise_p))
            p_ini[j].append( p[2*j+1] + 5 + random.uniform(-noise_p,noise_p)) 
                
###############################################################################
        """ Calculate the state-space system """
        A = np.zeros([2*M_preys, 2*M_preys])
        B = np.zeros([2*M_preys, 2*N_hunters]) 
        C = np.eye(2*M_preys)
        D = np.zeros([2*M_preys, 2*N_hunters])
        Ts = 0.0005
###############################################################################
######  MODE 0 CONTROL SYSTEM #################################################
###############################################################################
        if mode == 0:        
            for j in range(M_preys):
                
                a = np.zeros([2,2])
                b = np.zeros([2,2])
                I = np.matrix([[1,0],[0,1]])
                pj0 = np.matrix([p_desired[j][0],p_desired[j][1]])
                
                for i in range(N_hunters):
                    hi0 = np.matrix([h_desired[i][0],h_desired[i][1]])
                    
                    term = (I*norm(hi0-pj0)**3)
                    a += (term+3*(norm(hi0-pj0)*np.transpose(hi0-pj0)*(hi0-pj0)))/(norm(hi0-pj0)**5)
                    b = (-term+3*(norm(hi0-pj0)*np.transpose(hi0-pj0)*(hi0-pj0)))/(norm(hi0-pj0)**5)
                    
                    B[2*j,2*i]    = b[0,0]
                    B[2*j,2*i+1]  = b[0,1]
                    B[2*j+1,2*i]  = b[1,0]
                    B[2*j+1,2*i+1]= b[1,1]
                    
                A[2*j,2*j]    = a[0,0]
                A[2*j,2*j+1]  = a[0,1]
                A[2*j+1,2*j]  = a[1,0]
                A[2*j+1,2*j+1]= a[1,1]
 
###############################################################################
######  MODE 1 CONTROL SYSTEM #################################################
###############################################################################
           
        elif mode == 1:
            for j in range(M_preys):
            
                a = np.zeros([2,2])
                b = np.zeros([2,2])
                I = np.matrix([[1,0],[0,1]])
                pj0 = np.matrix([p_desired[j][0],p_desired[j][1]])
                
                for i in range(N_hunters):
                    hi0 = np.matrix([h_desired[i][0],h_desired[i][1]])
                    
                    term = np.exp(-(1/(50**2))*(pj0-hi0)*np.transpose(pj0-hi0))
#                    if norm(hi0-pj0) < 0.5:
                    a += -I*0.5*(term-term*(pj0-hi0)*(2/(50**2))*np.transpose(pj0-hi0))
                    b =  -I*0.5*(-term+term*(pj0-hi0)*(2/(50**2))*np.transpose(pj0-hi0))
#                    else:
#                        a += -I*0.05*(term-term*(pj0-hi0)*(2/(50**2))*np.transpose(pj0-hi0))
#                        b =  -I*0.05*(-term+term*(pj0-hi0)*(2/(50**2))*np.transpose(pj0-hi0))
                    
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
            
            for j in range(M_preys):
                
                a = np.zeros([2,2])
                b = np.zeros([2,2])
                I = np.matrix([[1,0],[0,1]])
                pj0 = np.matrix([p_desired[j][0],p_desired[j][1]])
                
                for i in range(N_hunters):
                    hi0 = np.matrix([h_desired[i][0],h_desired[i][1]])
                    
                    term = (I*norm(hi0-pj0)**3)
                    a +=(term+3*(norm(hi0-pj0)*np.transpose(hi0-pj0)*(hi0-pj0)))/(norm(hi0-pj0)**5)
                    b = (-term+3*(norm(hi0-pj0)*np.transpose(hi0-pj0)*(hi0-pj0)))/(norm(hi0-pj0)**5)
                    
                    B[2*j,2*i]    = b[0,0]
                    B[2*j,2*i+1]  = b[0,1]
                    B[2*j+1,2*i]  = b[1,0]
                    B[2*j+1,2*i+1]= b[1,1]
                
                a += np.matrix([[0.01,-0.001],[-0.005,0.03]])
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
                pj0 = np.matrix([p_desired[j][0],p_desired[j][1]])
                
                for i in range(N_hunters):
                    hi0 = np.matrix([h_desired[i][0],h_desired[i][1]])
                    
                    term = (I*norm(hi0-pj0)**3)
                    a += (term+3*(norm(pj0-hi0)*np.transpose(pj0-hi0)*(pj0-hi0)))/(norm(pj0-hi0)**5)
                    b = (-term+3*(norm(pj0-hi0)*np.transpose(pj0-hi0)*(pj0-hi0)))/(norm(pj0-hi0)**5)
                    
                    B[2*j,2*i]    = b[0,0]
                    B[2*j,2*i+1]  = b[0,1]
                    B[2*j+1,2*i]  = b[1,0]
                    B[2*j+1,2*i+1]= b[1,1]
                    
                A[2*j,2*j]    = a[0,0]
                A[2*j,2*j+1]  = a[0,1]
                A[2*j+1,2*j]  = a[1,0]
                A[2*j+1,2*j+1]= a[1,1]
                
            
            
        cont_system = control.StateSpace(A,B,C,D)
        disc_system = control.c2d(cont_system,Ts)
        
#        poles = np.exp(-1 * np.ones([N_hunters])*Ts)
        [L,S,e] = lqr.dlqr(disc_system.A,disc_system.B,1*np.eye(2*M_preys),2*np.eye(2*N_hunters))
#        L = control.place(disc_system.A,disc_system.B,poles)       

        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        config = {'SampleTime':Ts,
                  'K'         :1000,
                  'D'         :10,
                  'draw'      :False,
                  'anim'      :False,
                  'interval'  :10,
                  'H0'        :h_ini,
                  'P0'        :p_ini,
                  'H_desired' :h_desired,
                  'P_desired' :p_desired,
                  'L'         :L,
                  'saveAnim'  :False
                } 
        
        
        XP,YP,XH,YH,error,v_preys,v_hunters,converge = setup(N_hunters,M_preys,mode,config)
        found = True
        
    return XP,YP,XH,YH,error,v_preys,v_hunters,found,converge


