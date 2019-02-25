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
M_preys          = 2
N_hunters        = 4
mode             = 0

Sim_num          = 100

noise_jump       = 0.1
num_noise_p      = 10
num_noise_h      = 10
noise_level_p    = 0.0
noise_level_h    = 0.0

plot             = False

h0_found         = False
found_counter    = 0

fee_global       = np.zeros([Sim_num*num_noise_p*num_noise_h,M_preys])
fee_preys        = np.zeros([num_noise_p, Sim_num*num_noise_h,M_preys])
fee_hunters      = np.zeros([num_noise_h,num_noise_p*Sim_num,M_preys])
fve_global       = np.zeros([Sim_num*num_noise_p*num_noise_h,M_preys])
fve_preys        = np.zeros([num_noise_p, Sim_num*num_noise_h,M_preys])
fve_hunters      = np.zeros([num_noise_h,num_noise_p*Sim_num,M_preys])
fce_preys        = np.zeros([num_noise_p])
fce_hunters      = np.zeros([num_noise_h])
converge         = False

counter_g        = 0
counter_p        = 0
counter_h        = 0
counter_c        = 0

for n in range(num_noise_p):
    
    noise_level_p += noise_jump
    noise_level_h = 0.0
    counter_p = 0
    for m in range(num_noise_h):
        
        noise_level_h += noise_jump
        counter_h = Sim_num*n
        
        for k in range (Sim_num):  
            
            if N_hunters > M_preys:
                p0 = []
                for j in range(M_preys):
                    p0.append(random.uniform(-2.0,2.0))
                    p0.append(random.uniform(-2.0,2.0))
                for i in range(N_hunters-M_preys):
                    p0.append(0.0)
                    p0.append(0.0)
                    
            else:
                p0 = []
                for j in range(M_preys):
                    p0.append(random.uniform(-2.0,2.0))
                    p0.append(random.uniform(-2.0,2.0)) 
                    
            if found_counter == 100:
                print("next config")
                
            found_counter = 0
            while not (h0_found and found_counter<100):
                
                found_counter += 1
                nn = 1.0
                
                if N_hunters < M_preys:
                    h0 = []
                    angle = 2*np.pi/N_hunters 
                    for i in range(N_hunters):
                        h0.append(4.4*np.cos(i*angle)+random.uniform(-nn,nn))
                        h0.append(4.4*np.sin(i*angle)+random.uniform(-nn,nn))
                    for j in range(M_preys-N_hunters):
                        h0.append(0.0)
                        h0.append(0.0)
                        
                else:
                    h0 = []
                    angle = 2*np.pi/N_hunters
                    for i in range(N_hunters):
                        h0.append(4.4*np.cos(i*angle)+random.uniform(-nn,nn))
                        h0.append(4.4*np.sin(i*angle)+random.uniform(-nn,nn)) 
                
                XP,YP,XH,YH,error,v_preys,v_hunters,h0_found,converge = program(M_preys,N_hunters,mode,h0,p0,noise_level_h,noise_level_p)

            if converge:
                counter_c += 1
                fce_preys[n]        += 1
                fce_hunters[m]      += 1
                
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
""" Convergence """

print("Ratio of system convergences is: "+str(counter_c/(num_noise_p*num_noise_h*Sim_num)*100))
print("Ratio of system completes K simulations is: "+str((1-(counter_c/(num_noise_p*num_noise_h*Sim_num)))*100))


###############################################################################
"""Convergence wrt prey noise"""
 
fig = plt.figure()
ax  = plt.axes()

ax_leg = ['0.0']
for i in range(num_noise_p):
    ax_leg.append(str(i/num_noise_p))
    ax_leg.append(str(i/num_noise_p))
    ax_leg.append(str(i/num_noise_p))
    
    plt.plot(fce_preys/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
   
ax.set_title('Convergence wrt prey noise')
ax.set_xlabel('Level of noise in preys')
ax.set_ylabel('Frecuency')
ax.set_xticklabels(ax_leg)
ax.locator_params(tight=True, nbins=len(ax_leg))
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()

###############################################################################
"""Convergence wrt hunters noise"""
 
fig = plt.figure()
ax  = plt.axes()

ax_leg = ['0.0']
for i in range(num_noise_h):
    ax_leg.append(str(i/num_noise_h))
    ax_leg.append(str(i/num_noise_h))
    ax_leg.append(str(i/num_noise_h))
    
    plt.plot(fce_hunters/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
   
ax.set_title('Convergence wrt hunter noise')
ax.set_xlabel('Level of noise in hunters')
ax.set_ylabel('Frecuency')
ax.set_xticklabels(ax_leg)
ax.locator_params(tight=True, nbins=len(ax_leg))
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()

###############################################################################
"""Global error Histogram"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Global Error Histogram')
ax.set_xlabel('Final Error')
ax.set_ylabel('Frecuency')
  
"""One for each prey"""
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
  
"""One for each prey"""
plt.hist((fve_global[:,0]+fve_global[:,1])*0.5,bins,alpha=0.5, label = 'Prey '+str(e))
    
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()

###############################################################################
"""Histograms of the influence of the hunters' noise in error"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Histograms of the influence of the hunters noise in error')
ax.set_xlabel('Final Error')
ax.set_ylabel('Frecuency')
  
"""One for each noise preys' level"""

for n in range(num_noise_p):
    media = 0*fee_preys[n,:,0]
    for e in range(M_preys):
        media += fee_preys[n,:,e]
    media = media/M_preys
    plt.hist(media,bins,alpha=0.5, label = 'Level of preys noise is '+str(n/num_noise_p))
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()


###############################################################################
"""Histograms of the influence of the preys' noise in error"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Histograms of the influence of the preys noise in error')
ax.set_xlabel('Final Error')
ax.set_ylabel('Frecuency')
  
"""One for each noise hunters' level"""

for m in range(num_noise_h):
    media = 0*fee_hunters[m,:,0]
    for e in range(M_preys):
        media += fee_hunters[m,:,e]
    media = media/M_preys
    plt.hist(media,bins,alpha=0.5, label = 'Level of hunters noise is '+str(m/num_noise_h))
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()


###############################################################################
"""Histograms of the influence of the hunters' noise in convergence"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Histograms of the influence of the hunters noise in convergence')
ax.set_xlabel('Final Velocity')
ax.set_ylabel('Frecuency')
  
"""One for each noise preys' level"""

for n in range(num_noise_p):
    media = 0*fve_preys[n,:,0]
    for e in range(M_preys):
        media += fve_preys[n,:,e]
    media = media/M_preys
    plt.hist(media,bins,alpha=0.5, label = 'Level of preys noise is '+str(n/num_noise_p))
    
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()


###############################################################################
"""Histograms of the influence of the preys' noise in convergence"""
histogram=plt.figure()
ax       =plt.axes()

bins = np.linspace(0, 3, 100)

ax.set_title('Histograms of the influence of the preys noise in convergence')
ax.set_xlabel('Final Velocity')
ax.set_ylabel('Frecuency')
  
"""One for each noise hunters' level"""

for m in range(num_noise_h):
    media = 0*fve_hunters[m,:,0]
    for e in range(M_preys):
        media += fve_hunters[m,:,e]
    media = media/M_preys
    plt.hist(media,bins,alpha=0.5, label = 'Level of hunters noise is '+str(m/num_noise_h))
    
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
"""Plot comprise info of final error wrt prey noise"""
 
fig = plt.figure()
ax  = plt.axes()

ax_leg = ['0.0']
colors = ['C0', 'C3', 'C6', 'C9']
for i in range(4):
    ax_leg.append(str((3*i+0)/num_noise_p))
    ax_leg.append(str((3*i+1)/num_noise_p))
    ax_leg.append(str((3*i+2)/num_noise_p))
    
    mean = mean_and_cov_e[i*3,:,0]
    ub = mean_and_cov_e[i*3,:,0] + mean_and_cov_e[i*3,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    lb = mean_and_cov_e[i*3,:,0] - mean_and_cov_e[i*3,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = 'Preys noise level '+str(i*3/num_noise_h))

ax.set_title('Comprise info of the influence of noise in final error')
ax.set_xlabel('Level of noise in hunters')
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
colors = ['C0','C3', 'C6','C9']

for i in range(4):
    
    ax_leg.append(str((3*i+0)/num_noise_p))
    ax_leg.append(str((3*i+1)/num_noise_p))
    ax_leg.append(str((3*i+2)/num_noise_p))
    
    mean = mean_and_cov_v[3*i,:,0]
    ub = mean_and_cov_v[3*i,:,0] + mean_and_cov_v[3*i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    lb = mean_and_cov_v[3*i,:,0] - mean_and_cov_v[3*i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = 'Preys noise level '+str(3*i/num_noise_h))

ax.set_title('Comprise info of the influence of noise in convergence')
ax.set_xlabel('Level of noise in hunters')
ax.set_ylabel('Final velocity')
ax.set_xticklabels(ax_leg)
ax.locator_params(tight=True, nbins=len(ax_leg))
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()


###############################################################################
# MÉTRICAS COMPACTAS
###############################################################################
mean_and_cov_e = np.zeros([num_noise_h,num_noise_p,2])
mean_and_cov_v = np.zeros([num_noise_h,num_noise_p,2])

for n in range(num_noise_h):
    media_e = 0
    covarianza_e = 0
    media_v = 0
    covarianza_v = 0 
    counter = 0
    
    for m in range(num_noise_p*Sim_num):
        
        for e in range(M_preys):
        
            media_e += fee_preys[m,n,e]
            media_v += fve_preys[m,n,e]
    
        if ((m+1)%(Sim_num) == 0):
            mean_and_cov_e[n,counter,0] = media_e/(M_preys*Sim_num)
            mean_and_cov_v[n,counter,0] = media_v/(M_preys*Sim_num)
            media_e = 0
            media_v = 0
            counter += 1
    
    counter = 0
    
    for m in range(num_noise_p*Sim_num):
        
        for e in range(M_preys):
            
            covarianza_e += (fee_preys[m,n,e] - mean_and_cov_e[n,counter,0])**2
            covarianza_v += (fve_preys[m,n,e] - mean_and_cov_v[n,counter,0])**2

        if ((m+1)%(Sim_num) == 0):
            mean_and_cov_e[n,counter,1] = np.sqrt(covarianza_e/(M_preys*Sim_num))
            mean_and_cov_v[n,counter,1] = np.sqrt(covarianza_v/(M_preys*Sim_num))
            covarianza_e = 0
            covarianza_v = 0
            counter += 1


###############################################################################
"""Plot comprise info of final error wrt hunter noise"""
 
fig = plt.figure()
ax  = plt.axes()

ax_leg = ['0.0']
colors = ['C0', 'C3', 'C6', 'C9']
for i in range(4):
    
    ax_leg.append(str((3*i+0)/num_noise_p))
    ax_leg.append(str((3*i+1)/num_noise_p))
    ax_leg.append(str((3*i+2)/num_noise_p))
    
    mean = mean_and_cov_e[i*3,:,0]
    ub = mean_and_cov_e[i*3,:,0] + mean_and_cov_e[i*3,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    lb = mean_and_cov_e[i*3,:,0] - mean_and_cov_e[i*3,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = 'Hunters noise level '+str(i*3/num_noise_p))

ax.set_title('Comprise info of the influence of noise in final error')
ax.set_xlabel('Level of noise in preys')
ax.set_ylabel('Final error')
ax.set_xticklabels(ax_leg)
ax.locator_params(tight=True, nbins=len(ax_leg))
plt.legend(loc='best', shadow=False, fontsize='medium')
plt.show()

###############################################################################
"""Plot comprise info of convergence wrt hunter noise"""


fig = plt.figure()
ax  = plt.axes()

ax_leg = ['0.0']
colors = ['C0','C3', 'C6','C9']

for i in range(4):
    
    ax_leg.append(str((3*i+0)/num_noise_p))
    ax_leg.append(str((3*i+1)/num_noise_p))
    ax_leg.append(str((3*i+2)/num_noise_p))
    
    mean = mean_and_cov_v[3*i,:,0]
    ub = mean_and_cov_v[3*i,:,0] + mean_and_cov_v[3*i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    lb = mean_and_cov_v[3*i,:,0] - mean_and_cov_v[3*i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
    plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = 'Hunter noise level '+str(3*i/num_noise_p))

ax.set_title('Comprise info of the influence of noise in convergence')
ax.set_xlabel('Level of noise in preys')
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




