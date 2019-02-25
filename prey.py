# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:42:16 2018

@author: Eduardo
"""

import numpy as np
from numpy.linalg   import norm
from sklearn.preprocessing import normalize
"""
Class defining the behavior of a prey.
The main structure of the class defines the behavior of a standard prey: animal, 
fire focus, epidemic disease, ...
Thus, different dynamics can be selected by switching the flag MODE:
    
    *Mode 0: Pierson & Schwager
        
    *Mode 1: Licitra et al.
    
    *Mode 2: Browning & Wexler
    
"""
class prey:
    
    def __init__(self):
        
        self.newPosition = np.matrix([0.,0.])     #Present postion of the prey
        self.prePosition = np.matrix([0.,0.])     #Previous position of the prey
        self.orientation = 0.                     #Orientation of the prey in radians
        self.angryMode = False                    #Determines if the prey is angry or not
        self.repulsionVelocity =np.matrix([0.,0.])#Repulsion due to hunters and other preys
        self.radius = 0.3                         #Range in which if there is a hunter, the prey becomes angry
        self.label = 0.                           #Tag labelling the prey
        self.mode = 0                             #Flag selecting the mode of the prey
        self.gamma = 1                            #Gain parameter of the repulsion velocity
        self.alpha = 1                            #Gain parameter in mode 1
        self.beta = 1                             #Gain parameter in mode 1
        self.var = 50                             #Gain parameter in mode 1
        self.mu  = 0.2                            #Gain parameter in mode 2
        self.ohm = 0.0                            #Gain parameter in mode 2
        self.landax = np.random.randint(-10,10)/500 #Gain parameter in mode 2
        self.landay = np.random.randint(-10,10)/500 #Gain parameter in mode 2
        self.saturation = 100.5                   #Saturation level for velocity
        return
    
    def setLocalization(self 
                        ,p #new position of the prey
                        ,tita #new orientation of the prey
                        ):
        
        self.prePosition = p
        self.newPosition = p
        self.orientation = tita
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
                    ):
        
        """
        Reset the variables
        """
        self.repulsionVelocity = np.matrix([0.,0.])
        self.prePosition = self.newPosition
        
        """
        Mode 0: A standard artificial potential dynamic is implemented: a repulsion
                force decides the velocity (vector) of the prey.
        """
        self.gamma = 50
        self.alpha = 1
        self.beta  = 1
        
        if (self.mode == 0 or self.mode!=1 or self.mode!=2 or self.mode!=3):
            
            for i in range(len(H)):
                dist = np.sqrt((self.prePosition[0,0]-H[i].prePosition[0,0])**2 + (self.prePosition[0,1]-H[i].prePosition[0,1])**2)
                    
                self.repulsionVelocity = self.repulsionVelocity - self.gamma*(H[i].prePosition-self.prePosition)/(dist**3)    
    
            """
            Calculate the new position and orientation with respect to 
            a saturation norm
            """
            if (norm(self.repulsionVelocity)<self.saturation):
            
                self.repulsionVelocity = self.repulsionVelocity
                
            else:
                
                self.repulsionVelocity = np.matrix(normalize(self.repulsionVelocity))*self.saturation
                    
            self.newPosition[0,0]= self.prePosition[0,0] + T*self.repulsionVelocity[0,0]
            self.newPosition[0,1]= self.prePosition[0,1] + T*self.repulsionVelocity[0,1]
            self.orientation = np.arctan2(self.newPosition[0,1], self.newPosition[0,0])

       
        """
        Mode 1: A Gaussian standard artificial potential dynamic is implemented: 
                a repulsion force decides the velocity (vector) of the prey.
        """
        
        self.alpha = 0.05
        self.beta  = 2.0
        
        
        if (self.mode == 1):
            
            """
            Decide  if the prey is angry or not
            """
            i = 0
            while (self.angryMode==False and i < len(H)):
                
                dist = np.sqrt((self.prePosition[0,0]-H[i].prePosition[0,0])**2+(self.prePosition[0,1]-H[i].prePosition[0,1])**2)
                
                if (dist < self.radius):
                    self.angryMode = True
                    self.gamma = 1
                    
                if(dist>=self.radius and self.angryMode==False):
                    self.angryMode = False
                    self.gamma = 0.1
               
                i += 1
            """
            Depending on the angry state, the repulsion force is one or the other
            """
#            if self.angryMode:   
                
            for i in range(len(H)):
                X = (1/(self.var**2))*((self.prePosition-H[i].prePosition))*np.transpose(self.prePosition-H[i].prePosition)
                
                self.repulsionVelocity = self.repulsionVelocity - self.gamma*(self.alpha*(self.prePosition-H[i].prePosition))*np.exp(-X)
#            else:
#                
#                for i in range(len(H)):
#                    X = (1/(self.var**2))*((self.prePosition-H[i].prePosition))*np.transpose(self.prePosition-H[i].prePosition)
#                    
#                    self.repulsionVelocity = self.repulsionVelocity - (self.alpha*(self.prePosition-H[i].prePosition))*np.exp(-X)    
            
            
            if (norm(self.repulsionVelocity)<self.saturation):
            
                self.repulsionVelocity = self.repulsionVelocity
                
            else:
                
                self.repulsionVelocity = np.matrix(normalize(self.repulsionVelocity))*self.saturation
            """
            Calculate the new position and orientation
            """
            self.newPosition[0,0]= self.prePosition[0,0] + T*self.repulsionVelocity[0,0]
            self.newPosition[0,1]= self.prePosition[0,1] + T*self.repulsionVelocity[0,1]
            self.orientation = np.arctan2(self.newPosition[0,1], self.newPosition[0,0])
        
    
        """
        Mode 2: Browning & Wexler proposed a bi dimensional model for wind in 1968.
        Thus, and adapted by using a general velocity vector field, a fire focus 
        model is proposed: the focuses (preys) move following the wind field, except 
        when a fireman (hunter) is closed. In this case, the fire is repulsed with a
        force proporcional to the distance and with the opposite direction with respect
        to the fireman
        """
        
        if (self.mode == 2):
            
            self.ohm = 1.0/D
            self.gamma = 0.2
            self.alpha = 0.01                            
            self.beta = -0.005                             
            self.var = -0.001                            
            self.mu  = 0.03                           
            """
            Decide if the prey is within the radius of the fireman 
            """
            for i in range(len(H)):
                dist = np.sqrt((self.prePosition[0,0]-H[i].prePosition[0,0])**2 + (self.prePosition[0,1]-H[i].prePosition[0,1])**2)
                    
                self.repulsionVelocity = self.repulsionVelocity - self.gamma*(H[i].prePosition-self.prePosition)/(dist**3)    
            
            
            if (norm(self.repulsionVelocity)<self.saturation):
            
                self.repulsionVelocity = self.repulsionVelocity
                
            else:
                
                self.repulsionVelocity = np.matrix(normalize(self.repulsionVelocity))*self.saturation
            
            """ Vector field effect """
            self.repulsionVelocity[0,0] = self.repulsionVelocity[0,0] + (self.alpha*(self.prePosition[0,0]-D/2) + self.beta*(self.prePosition[0,1]-D/2))
            self.repulsionVelocity[0,1] = self.repulsionVelocity[0,1] + (self.var*(self.prePosition[0,0]-D/2) + self.mu*(self.prePosition[0,1]-D/2))
        
            """ New position calcs """
            self.newPosition[0,0] = self.newPosition[0,0] + T*self.repulsionVelocity[0,0]
            self.newPosition[0,1] = self.newPosition[0,1] + T*self.repulsionVelocity[0,1]

            
        """
        Limits the movements of the object: it is not possible to cross the preserve
        """
        if (self.newPosition[0,0] < 0.):
            self.newPosition[0,0] = 0
        
        if (self.newPosition[0,0] > D):
            self.newPosition[0,0] = D
        
        if (self.newPosition[0,1] < 0.):
            self.newPosition[0,1] = 0
        
        if (self.newPosition[0,1] > D):
            self.newPosition[0,1] = D
            
        if (self.prePosition[0,0] < 0.):
            self.prePosition[0,0] = 0
        
        if (self.prePosition[0,0] > D):
            self.prePosition[0,0] = D
        
        if (self.prePosition[0,1] < 0.):
            self.prePosition[0,1] = 0
        
        if (self.prePosition[0,1] > D):
            self.prePosition[0,1] = D
        
        v = norm(self.repulsionVelocity) 
        
        return v