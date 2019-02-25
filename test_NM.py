# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:01:58 2018

@author: Eduardo
"""

from main import program
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
from plotObject import plot_mean_and_CI

plt.close('all')

"""

In order to test the performance of the algorithm, this script plays the 
program as many times as we want, using the mean features we need to compare 
and evaluate the performance

"""

""" Initial variables """
random.seed(1000000000000066600000000000001)

D                = 10
M_preys          = 0
N_hunters        = 0

mode             = 0

Sim_num          = 100

obj_jump         = 1
num_noise_p      = 10
num_noise_h      = 10
noise_level_p    = 0.5
noise_level_h    = 0.5

plot             = False

h0_found         = False

fee_global       = np.zeros([Sim_num*num_noise_p*num_noise_h,M_preys])
fee_preys        = np.zeros([num_noise_p, Sim_num*num_noise_h,M_preys])
fee_hunters      = np.zeros([num_noise_h,num_noise_p*Sim_num,M_preys])
fve_global       = np.zeros([Sim_num*num_noise_p*num_noise_h,M_preys])
fve_preys        = np.zeros([num_noise_p, Sim_num*num_noise_h,M_preys])
fve_hunters      = np.zeros([num_noise_h,num_noise_p*Sim_num,M_preys])

counter_g        = 0
counter_p        = 0
counter_h        = 0

for n in range(num_noise_p):
    
    M_preys += obj_jump
    N_hunters = 0
    counter_p = 0
    for m in range(num_noise_h):
        
        N_hunters += obj_jump
        counter_h = Sim_num*n
        
        for k in range (Sim_num):  
            
            resta = N_hunters - M_preys
            if resta > 0:
                p0 = []
                for j in range(M_preys):
                    p0.append(random.uniform(-2.0,2.0))
                    p0.append(random.uniform(-2.0,2.0))
                for i in range(N_hunters):
                    p0.append(0.0)
                    p0.append(0.0)
                    
            else:
                p0 = []
                for j in range(M_preys):
                    p0.append(random.uniform(-2.0,2.0))
                    p0.append(random.uniform(-2.0,2.0))
     
            
            while not (h0_found):
                
                nn = 2.0
                
                if resta < 0:
                    h0 = []
                    angle = 2*np.pi/N_hunters
                    for i in range(N_hunters):
                        h0.append(3.0*np.cos(angle)+random.uniform(-1.0,1.0))
                        h0.append(3.0*np.sin(angle)+random.uniform(-1.0,1.0))
                        angle += angle
                    for j in range(M_preys):
                        h0.append(0.0)
                        h0.append(0.0)
                        
                else:
                    h0 = []
                    angle = 2*np.pi/N_hunters
                    for i in range(N_hunters):
                        h0.append(3.0*np.cos(angle)+random.uniform(-1.0,1.0))
                        h0.append(3.0*np.sin(angle)+random.uniform(-1.0,1.0))
                
                XP,YP,XH,YH,error,v_preys,v_hunters,h0_found = program(M_preys,N_hunters,mode,h0,p0,noise_level_h,noise_level_p)
                
            if plot:
                """Plot error"""    
                fig = plt.figure()
                ax = plt.axes()
                ax.set_title('error metric: distance of each prey from its desired position')
                ax.set_xlabel('Samples')
                ax.set_ylabel('Error')
                for e in range(M_preys):
                    ax.plot(error[:,e],label='Prey '+str(e))
                legend = ax.legend(loc='best', shadow=False, fontsize='medium')
                legend.get_frame().set_facecolor('None')
                plt.show()
                
                """Plot v preys """
                fig = plt.figure()
                ax = plt.axes()
                ax.set_title('error metric: velocity of each prey')
                ax.set_xlabel('Samples')
                ax.set_ylabel('Velocity')
                for e in range(M_preys):
                    ax.plot(v_preys[:,e],label='Prey '+str(e))
                legend = ax.legend(loc='best', shadow=False, fontsize='medium')
                legend.get_frame().set_facecolor('None')
                plt.show()
            
            """Save final error"""
            for e in range(M_preys):
                fee_global[counter_g,e]    = error[-1,e]
                fee_preys[n,counter_p,e]   = error[-1,e]
                fee_hunters[m,counter_h,e] = error[-1,e]
                fve_global[counter_g,e]    = v_preys[-1,e]
                fve_preys[n,counter_p,e]   = v_preys[-1,e]
                fve_hunters[m,counter_h,e] = v_preys[-1,e]
                
            """Reset variables"""
            h0_found = False
            print(counter_g)
            counter_g += 1
            counter_p += 1
            counter_h += 1



###############################################################################
"""Global error Histogram"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Global Error Histogram')
ax.set_xlabel('Final Error')
ax.set_ylabel('Frecuency')
  
plt.hist((fee_global[:,0]+fee_global[:,1])*0.5,bins,alpha=0.5, label = 'Prey '+str(e))
    
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()

###############################################################################
"""Global Convergence Histogram"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 10, 100)

ax.set_title('Global Convergence Histogram')
ax.set_xlabel('Final Velocity')
ax.set_ylabel('Frecuency')
  
plt.hist((fve_global[:,0]+fve_global[:,1])*0.5,bins,alpha=0.5, label = 'Prey '+str(e))
    
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()

###############################################################################
"""Histograms of the influence of the number of hunters in error"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Histograms of the influence of the number of hunters in error')
ax.set_xlabel('Final Error')
ax.set_ylabel('Frecuency')
  
"""One for each number of preys"""

for n in range(num_noise_p):
    media = 0*fee_preys[n,:,0]
    for e in range(M_preys):
        media += fee_preys[n,:,e]
    media = media/M_preys
    plt.hist(media,bins,alpha=0.5, label = 'Number of preys is '+str(n/num_noise_p))
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()


###############################################################################
"""Histograms of the influence of the number of preys in error"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Histograms of the influence of the number of preys in error')
ax.set_xlabel('Final Error')
ax.set_ylabel('Frecuency')
  
"""One for each number of hunters"""

for m in range(num_noise_h):
    media = 0*fee_hunters[m,:,0]
    for e in range(M_preys):
        media += fee_hunters[m,:,e]
    media = media/M_preys
    plt.hist(media,bins,alpha=0.5, label = 'Number of hunters is '+str(m/num_noise_h))
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()


###############################################################################
"""Histograms of the influence of the number of hunters in convergence"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Histograms of the influence of the number of hunters in convergence')
ax.set_xlabel('Final Velocity')
ax.set_ylabel('Frecuency')
  
"""One for each number of preys"""

for n in range(num_noise_p):
    media = 0*fve_preys[n,:,0]
    for e in range(M_preys):
        media += fve_preys[n,:,e]
    media = media/M_preys
    plt.hist(media,bins,alpha=0.5, label = 'Number of preys is '+str(n/num_noise_p))
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()


###############################################################################
"""Histograms of the influence of the number of preys in convergence"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Histograms of the influence of the number of preys in convergence')
ax.set_xlabel('Final Velocity')
ax.set_ylabel('Frecuency')
  
"""One for each number of hunters"""

for m in range(num_noise_h):
    media = 0*fve_hunters[m,:,0]
    for e in range(M_preys):
        media += fve_hunters[m,:,e]
    media = media/M_preys
    plt.hist(media,bins,alpha=0.5, label = 'Number of hunters is '+str(m/num_noise_h))
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()

###############################################################################
# MÉTRICAS COMPACTAS
###############################################################################
mean_and_cov_e = np.zeros([num_noise_p,num_noise_h,2])
mean_and_cov_v = np.zeros([num_noise_p,num_noise_h,2])

for n in range(num_noise_p):
    media_e = 0
    covarianza_e = 0
    media_v = 0
    covarianza_v = 0 
    counter = 0
    
    for m in range(num_noise_h*Sim_num):
        
        for e in range(M_preys):
        
            media_e += fee_preys[n,m,e]
            media_v += fve_preys[n,m,e]
    
        if ((m+1)%(Sim_num) == 0):
            mean_and_cov_e[n,counter,0] = media_e/(M_preys*Sim_num)
            mean_and_cov_v[n,counter,0] = media_v/(M_preys*Sim_num)
            media_e = 0
            media_v = 0
            counter += 1
    
    counter = 0
    
    for m in range(num_noise_h*Sim_num):
        
        for e in range(M_preys):
            
            covarianza_e += (fee_preys[n,m,e] - mean_and_cov_e[n,counter,0])**2
            covarianza_v += (fve_preys[n,m,e] - mean_and_cov_v[n,counter,0])**2

        if ((m+1)%(Sim_num) == 0):
            mean_and_cov_e[n,counter,1] = np.sqrt(covarianza_e/(M_preys*Sim_num))
            mean_and_cov_v[n,counter,1] = np.sqrt(covarianza_v/(M_preys*Sim_num))
            covarianza_e = 0
            covarianza_v = 0
            counter += 1
            
###############################################################################
"""Plot comprise info of final error wrt preys number"""
 
fig = plt.figure()
ax  = plt.axes()

ax_leg = ['0.0']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
for i in range(num_noise_p):
    
    ax_leg.append(str(i/num_noise_h))    
    mean = mean_and_cov_e[i,:,0]
    ub = mean_and_cov_e[i,:,0] + mean_and_cov_e[i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    lb = mean_and_cov_e[i,:,0] - mean_and_cov_e[i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = 'Number of preys is '+str(i/num_noise_h))

ax.set_title('Comprise info of the influence of number of preys in final error')
ax.set_xlabel('Number of hunters')
ax.set_ylabel('Final error')
ax.set_xticklabels(ax_leg)
ax.locator_params(tight=True, nbins=len(ax_leg))
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()



###############################################################################
"""Plot comprise info of convergence wrt prey noise"""


fig = plt.figure()
ax  = plt.axes()

ax_leg = ['0.0']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

for i in range(num_noise_p):
    
    ax_leg.append(str(i/num_noise_h))
    mean = mean_and_cov_v[i,:,0]
    ub = mean_and_cov_v[i,:,0] + mean_and_cov_v[i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    lb = mean_and_cov_v[i,:,0] - mean_and_cov_v[i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = 'Preys noise level '+str(i/num_noise_h))

ax.set_title('Comprise info of the influence of noise in convergence')
ax.set_xlabel('Level of noise in hunters')
ax.set_ylabel('Final velocity')
ax.set_xticklabels(ax_leg)
ax.locator_params(tight=True, nbins=len(ax_leg))
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()



###############################################################################
""" SAVE """
csvFile = open('Metrics.csv', 'a')
csvWriter = csv.writer(csvFile)
csvWriter.writerow([error])
csvWriter.writerow([v_preys])
csvWriter.writerow([v_hunters])
csvFile.close() 




