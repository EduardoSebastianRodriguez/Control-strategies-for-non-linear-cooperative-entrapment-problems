# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:06:08 2019

@author: Eduardo
"""

import numpy as np
import matplotlib.pyplot as plt


preys = np.zeros(100)
hunters = np.zeros(100)

plt.rcParams.update({'font.size': 20})


for i in range(10):
    for j in range(10):
        preys[i*10+j] = i+1
        hunters[i*10+j] = j+1
        
values = [100,1,0,1,1,2,0,0,0,0,
          100,100,1,0,2,0,1,1,0,2,
          73,98,81,1,1,0,2,3,7,8,
          80,76,98,34,8,3,4,1,8,14,
          84,78,89,94,18,8,9,13,20,22,
          76,88,80,91,88,20,17,17,19,31,
          77,81,86,91,90,95,40,36,30,38,
          87,84,83,89,89,95,96,40,53,49,
          81,86,91,92,87,90,96,97,60,61,
          84,89,91,90,90,96,96,99,100,65]


for i in range(len(values)):
    values[i] = 100-values[i]
"""Plot comprise info of final error wrt prey noise"""
         
fig = plt.figure()
ax  = plt.axes()

colors = ['C0','C1','C2','C3','C4','C5', 'C6','C7','C8','C9']
styles = ['-','--','-.',':','-','--','-.',':','-','--']
for i in range(10):
    plt.plot(hunters[0:9],values[10*i:10*i+9],colors[i],label=str(i+1)+" preys",linewidth=5,linestyle=styles[i])
    
ax.set_xlabel('Number of hunters')
ax.set_ylabel('Ratio of convergence of the solver (%)')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=False, fontsize='small')
plt.show()
okey = False
while not okey:
    try:
        plt.savefig('SUPERFIGURE')
        okey = True
    except:
        okey = False
        
        
preys = np.zeros(100)
hunters = np.zeros(100)

plt.rcParams.update({'font.size': 20})


for i in range(10):
    for j in range(10):
        preys[i*10+j] = i+1
        hunters[i*10+j] = j+1
        
values = [100,0,3,1,1,2,4,2,0,0,
          100,100,4,3,2,1,1,1,0,0,
          98,97,100,14,13,5,2,3,7,8,
          99,100,100,78,18,3,4,5,4,14,
          95,89,97,92,64,8,9,13,20,22,
          87,85,76,94,89,15,18,16,13,8,
          87,99,100,97,100,98,32,32,20,30,
          98,89,89,89,100,92,97,46,50,41,
          91,96,98,97,100,95,100,97,42,51,
          94,99,96,99,99,95,94,93,100,55]


for i in range(len(values)):
    values[i] = 100-values[i]
"""Plot comprise info of final error wrt prey noise"""
         
fig = plt.figure()
ax  = plt.axes()

colors = ['C0','C1','C2','C3','C4','C5', 'C6','C7','C8','C9']
styles = ['-','--','-.',':','-','--','-.',':','-','--']
for i in range(10):
    plt.plot(hunters[0:9],values[10*i:10*i+9],colors[i],label=str(i+1)+" preys",linewidth=5,linestyle=styles[i])
    
ax.set_xlabel('Number of hunters')
ax.set_ylabel('Ratio of convergence of the solver (%)')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=False, fontsize='small')
plt.show()
okey = False
while not okey:
    try:
        plt.savefig('SUPERFIGURE1')
        okey = True
    except:
        okey = False
        
        
preys = np.zeros(100)
hunters = np.zeros(100)

plt.rcParams.update({'font.size': 20})


for i in range(10):
    for j in range(10):
        preys[i*10+j] = i+1
        hunters[i*10+j] = j+1
        
values = [100,0,7,3,0,0,2,1,1,1,
          98,100,0,2,0,1,1,1,2,1,
          89,98,94,1,3,2,3,5,9,11,
          91,95,99,88,8,4,13,10,12,18,
          92,94,97,97,63,11,16,9,22,27,
          93,98,90,95,95,29,20,22,30,31,
          97,93,89,94,95,92,37,39,34,38,
          94,94,93,94,93,95,98,49,41,49,
          91,95,84,91,94,93,96,98,66,59,
          96,95,85,92,90,96,96,99,98,73]


for i in range(len(values)):
    values[i] = 100-values[i]
"""Plot comprise info of final error wrt prey noise"""
         
fig = plt.figure()
ax  = plt.axes()

colors = ['C0','C1','C2','C3','C4','C5', 'C6','C7','C8','C9']
styles = ['-','--','-.',':','-','--','-.',':','-','--']
for i in range(10):
    plt.plot(hunters[0:9],values[10*i:10*i+9],colors[i],label=str(i+1)+" preys",linewidth=5,linestyle=styles[i])
    
ax.set_xlabel('Number of hunters')
ax.set_ylabel('Ratio of convergence of the solver (%)')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=False, fontsize='small')
plt.show()
okey = False
while not okey:
    try:
        plt.savefig('SUPERFIGURE2')
        okey = True
    except:
        okey = False
        
        
preys = np.zeros(100)
hunters = np.zeros(100)

plt.rcParams.update({'font.size': 20})


for i in range(10):
    for j in range(10):
        preys[i*10+j] = i+1
        hunters[i*10+j] = j+1
        
values = [100,1,2,1,1,2,3,2,0,0,
          99,100,4,2,1,1,1,0,0,0,
          95,92,100,14,17,4,2,3,6,8,
          99,100,100,75,13,7,4,5,4,10,
          93,91,95,93,56,8,10,14,21,22,
          87,84,75,99,82,18,12,10,10,14,
          87,99,100,97,100,95,39,38,27,43,
          92,89,89,89,100,90,96,46,52,47,
          97,96,98,97,100,91,98,94,47,58,
          93,99,96,99,99,94,94,97,89,52]


for i in range(len(values)):
    values[i] = 100-values[i]
"""Plot comprise info of final error wrt prey noise"""
         
fig = plt.figure()
ax  = plt.axes()

colors = ['C0','C1','C2','C3','C4','C5', 'C6','C7','C8','C9']
styles = ['-','--','-.',':','-','--','-.',':','-','--']
for i in range(10):
    plt.plot(hunters[0:9],values[10*i:10*i+9],colors[i],label=str(i+1)+" preys",linewidth=5,linestyle=styles[i])
    
ax.set_xlabel('Number of hunters')
ax.set_ylabel('Ratio of convergence of the solver (%)')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=False, fontsize='small')
plt.show()
okey = False
while not okey:
    try:
        plt.savefig('SUPERFIGURE3')
        okey = True
    except:
        okey = False
        
        
dir = 'C:/Users/Eduardo/Desktop/STORE'+str(1)+"_"+str(4)+"_"+str(1)+'.obj'
okey = False
N = 7
M = 3
mode = 1

while not okey:
    try:
                
        dicdic1 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/COUNTERS_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        [counter_p,counter_h,counter_c,counter_r,counter_t,counter_x] = np.loadtxt(dicdic1)
        dicdic2 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fee_global_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fee_global=np.loadtxt(dicdic2)
        dicdic3 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fee_preys_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fee_preys=np.loadtxt(dicdic3)
        dicdic4 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fee_hunters_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fee_hunters=np.loadtxt(dicdic4)
        dicdic5 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fve_global_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fve_global=np.loadtxt(dicdic5)
        dicdic6 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fee_global_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fee_global=np.loadtxt(dicdic6)
        dicdic7 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fve_preys_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fve_preys=np.loadtxt(dicdic7)
        dicdic8 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fve_hunters_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fve_hunters=np.loadtxt(dicdic8)
        dicdic9 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fce_preys_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fce_preys=np.loadtxt(dicdic9)
        dicdic10 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fce_hunters_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fce_hunters=np.loadtxt(dicdic10)
        dicdic11 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fre_preys_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fre_preys=np.loadtxt(dicdic11)
        dicdic12 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fre_hunters_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fre_hunters=np.loadtxt(dicdic12)
        dicdic13 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fte_preys_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fte_preys=np.loadtxt(dicdic13)
        dicdic14 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/fte_hunters_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        fte_hunters=np.loadtxt(dicdic14)
        dicdic15 = 'G:/Unidades de equipo/EduardoSebastian/TFG/07_Datos/Nonlinear/superfinal_'+str(mode)+"_"+str(N)+"_"+str(M)+'.dat'      
        superfinal=np.loadtxt(dicdic15)
        okey = True
    except:
        okey = False