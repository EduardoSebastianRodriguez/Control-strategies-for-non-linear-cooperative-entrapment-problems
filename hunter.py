# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 12:21:20 2018

@author: Eduardo
"""
import numpy as np
from numpy.linalg   import norm
from sklearn.preprocessing import normalize

"""
Class defining the behavior of a hunter
"""
class hunter:
    
    def __init__(self):
        
        self.newPosition = np.matrix([0.,0.])    #Present postion of the hunter
        self.prePosition = np.matrix([0.,0.])    #Previous position of the hunter
        self.orientation = 0                     #Orientation of the hunter in radians
        self.label = 0                           #Tag labelling the hunter
        self.velocity = np.matrix([0.0,0.0])     #Velocity of the hunter
        self.saturation = 100.5                 #Saturation level
        return
    
    def setLocalization(self 
                        ,p    #new position of the hunter
                        ,tita #new orientation of the hunter
                        ):
        
        self.prePosition = p
        self.newPosition = p
        self.orientation = tita
        return
    
    def setLabel(self
                 ,l #int that names the object
                 ):
    
        self.label = l
        return
        
    def interaction(self
                    ,p    #new position of the hunter
                    ,D    #limits of the preserve
                    ,T    #sample time of the system
                    ):
        
        self.prePosition = self.newPosition
        if (norm(p-self.prePosition)/T < self.saturation):
            
            self.newPosition = p
            v = norm(p-self.prePosition)/T 
            
        else:
            inc    = np.matrix(normalize((p-self.prePosition)/T))*self.saturation 
            self.newPosition += T*inc
            v = self.saturation
            
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
        
        return v