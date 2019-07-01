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

plt.close('all')

"""

In order to test the performance of the algorithm, this script plays the 
program as many times as we want, using the mean features we need to compare 
and evaluate the performance

"""

modes = [1,2,3,4]

for w in range(len(modes)):

    """ Initial variables """
    random.seed(1000000000000066600000000000001)
    
    D                = 10
    M_preys          =0
    N_hunters        =0
    dic_name         = "G:/Unidades de equipo/EduardoSebastian/TFG/04_Imagenes/Modelo "+str(modes[w])+"/N_M/"
    mode             = modes[w]-1
    
    Sim_num          = 100
    
    obj_jump         = 1
    num_noise_p      = 10
    num_noise_h      = 10
    noise_level_p    = 0.4
    noise_level_h    = 0.4
    
    plot             = False
    
    h0_found         = False
    found_counter    = 0
    
    fce_preys        = np.zeros([num_noise_p])
    fce_hunters      = np.zeros([num_noise_h])
    fre_preys        = np.zeros([num_noise_p])
    fre_hunters      = np.zeros([num_noise_h])
    fte_preys        = np.zeros([num_noise_p])
    fte_hunters      = np.zeros([num_noise_h])
    fee_preys        = np.zeros([num_noise_p])
    fee_hunters      = np.zeros([num_noise_h])
    
    supermetric      = np.zeros([num_noise_p,num_noise_h,4])

    converge         = False
    med_error        = 0.0
    
    counter_g        = 0
    counter_p        = 0
    counter_h        = 0
    counter_c        = 0
    counter_r        = 0
    counter_t        = 0
    counter_e        = 0
    
    for n in range(num_noise_p):
        
        M_preys += obj_jump
        N_hunters = 0
        counter_p = 0
        for m in range(num_noise_h):
            
            N_hunters += obj_jump
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
                        
                found_counter = 0
                while (not h0_found and found_counter<10):
                    
                    found_counter += 1
                    nn = 1.0
                    
                    if N_hunters < M_preys:
                        h0 = []
                        angle = 2*np.pi/N_hunters 
                        for i in range(N_hunters):
                            h0.append(2.4*np.cos(i*angle)+random.uniform(-nn,nn))
                            h0.append(2.4*np.sin(i*angle)+random.uniform(-nn,nn))
                        for j in range(M_preys-N_hunters):
                            h0.append(2.0)
                            h0.append(2.0)
                            
                    else:
                        h0 = []
                        angle = 2*np.pi/N_hunters
                        for i in range(N_hunters):
                            h0.append(2.4*np.cos(i*angle)+random.uniform(-nn,nn))
                            h0.append(2.4*np.sin(i*angle)+random.uniform(-nn,nn))
                    
                    XP,YP,XH,YH,error,v_preys,v_hunters,h0_found,converge = program(M_preys,N_hunters,mode,h0,p0,noise_level_h,noise_level_p)
               
                    if found_counter == 10:
                        print("next config")
                        counter_e += 1
                        fee_preys[n]        += 1
                        fee_hunters[m]      += 1
                        supermetric[n,m,0]  += 1
                
                if found_counter!=10 and not isinstance(error,int):
                    if converge:
                        counter_c += 1
                        fce_preys[n]        += 1
                        fce_hunters[m]      += 1
                        supermetric[n,m,1]  += 1
                    
                    for j in range (M_preys):
                         med_error += error[-1,j]
                         
                    if (med_error/M_preys)<0.2:
                        counter_r += 1
                        fre_preys[n] += 1
                        fre_hunters[m] += 1
                        supermetric[n,m,2]  += 1
                    
                    if converge and (med_error/M_preys)<0.2:
                        counter_t += 1
                        fte_preys[n] += 1
                        fte_hunters[m] += 1
                        supermetric[n,m,3]  += 1
                    
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
                    
       
                """Reset variables"""
                h0_found = False
                if counter_g%100==0:
                    print("config " + str(modes[w])+"/"+str(N_hunters)+"_"+str(M_preys))
                    print(counter_g)
                counter_g += 1
                counter_p += 1
                counter_h += 1
                med_error = 0.0
                
                    
    ###############################################################################
    """ Convergence: means preys has reached velocity equals to zero """
    
    print("Ratio of system convergences is: "+str(counter_c/(num_noise_p*num_noise_h*Sim_num)*100)+"%")
    print("Ratio of system completes K simulations is: "+str((1-(counter_c/(num_noise_p*num_noise_h*Sim_num)))*100)+"%")
    
    ###############################################################################
    """ Final error """
    
    print("Ratio of system reaches desired set is: "+str(counter_r/(num_noise_p*num_noise_h*Sim_num)*100)+"%")
    print("Ratio of system does not reach desired set is: "+str((1-(counter_r/(num_noise_p*num_noise_h*Sim_num)))*100)+"%")
    
    ###############################################################################
    """ Total objective: convergence + final error """
    
    print("Ratio of system success is: "+str(counter_t/(num_noise_p*num_noise_h*Sim_num)*100)+"%")
    print("Ratio of system no success is: "+str((1-(counter_t/(num_noise_p*num_noise_h*Sim_num)))*100)+"%")
    
    ###############################################################################
    """ Solver efficiency """
    
    print("Solver efficiency is: "+str(counter_e/(num_noise_p*num_noise_h*Sim_num)*100)+"%")

    ###############################################################################
    okey = False
    while not okey:
        try:
            with open(dic_name+"Performance.txt", "w") as text_file:
                print(f"Ratio of system convergences is: "+str(counter_c/(num_noise_p*num_noise_h*Sim_num)*100)+"%", file=text_file)
                print(f"Ratio of system completes K simulations is: "+str((1-(counter_c/(num_noise_p*num_noise_h*Sim_num)))*100)+"%", file=text_file)
                print(f"Ratio of system reaches desired set is: "+str(counter_r/(num_noise_p*num_noise_h*Sim_num)*100)+"%", file=text_file)
                print(f"Ratio of system does not reach desired set is: "+str((1-(counter_r/(num_noise_p*num_noise_h*Sim_num)))*100)+"%", file=text_file)
                print(f"Ratio of system success is: "+str(counter_t/(num_noise_p*num_noise_h*Sim_num)*100)+"%", file=text_file)
                print(f"Ratio of system no success is: "+str((1-(counter_t/(num_noise_p*num_noise_h*Sim_num)))*100)+"%", file=text_file)
                print(f"Solver efficiency is: "+str(counter_e/(num_noise_p*num_noise_h*Sim_num)*100)+"%", file=text_file)

            okey = True
        except:
            okey = False

    ###############################################################################
    """Convergence wrt prey number"""
     
    fig = plt.figure()
    ax  = plt.axes()
    
    ax_leg = ['0']
    
    for i in range(num_noise_p):
        ax_leg.append(str(i+1+1))
        plt.plot(fce_preys/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
       
    ax.set_title("Convergence wrt prey's number")
    ax.set_xlabel('Number of preys (units)')
    ax.set_ylabel('Frecuency (%)')
    ax.set_ylim([0,101])
    ax.set_xticklabels(ax_leg)
    ax.locator_params(tight=True, nbins=len(ax_leg))
    plt.legend(loc='best', shadow=False, fontsize='medium')
    plt.show()
    okey = False
    while not okey:
        try:
            plt.savefig(dic_name+'Figure_1')
            okey = True
        except:
            okey = False
    
    ###############################################################################
    """Convergence wrt hunters number"""
     
    fig = plt.figure()
    ax  = plt.axes()
    
    ax_leg = ['0']
    for i in range(num_noise_h):
        ax_leg.append(str(i+1+1))
        plt.plot(fce_hunters/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
       
    ax.set_title("Convergence wrt hunter's number")
    ax.set_xlabel('Number of hunters (units)')
    ax.set_ylabel('Frecuency (%)')
    ax.set_ylim([0,101])
    ax.set_xticklabels(ax_leg)
    ax.locator_params(tight=True, nbins=len(ax_leg))
    plt.legend(loc='best', shadow=False, fontsize='medium')
    plt.show()
    okey = False
    while not okey:
        try:
            plt.savefig(dic_name+'Figure_2')
            okey = True
        except:
            okey = False
    
    ###############################################################################
    """Final error wrt prey number"""
     
    fig = plt.figure()
    ax  = plt.axes()
    
    ax_leg = ['0']
    
    for i in range(num_noise_p):
        ax_leg.append(str(i+1+1))
        plt.plot(fre_preys/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
       
    ax.set_title("Final error wrt prey's number")
    ax.set_xlabel('Number of preys (units)')
    ax.set_ylabel('Frecuency (%)')
    ax.set_ylim([0,101])
    ax.set_xticklabels(ax_leg)
    ax.locator_params(tight=True, nbins=len(ax_leg))
    plt.legend(loc='best', shadow=False, fontsize='medium')
    plt.show()
    okey = False
    while not okey:
        try:
            plt.savefig(dic_name+'Figure_3')
            okey = True
        except:
            okey = False
    
    ###############################################################################
    """Final error wrt hunters number"""
     
    fig = plt.figure()
    ax  = plt.axes()
    
    ax_leg = ['0']
    for i in range(num_noise_h):
        ax_leg.append(str(i+1+1))
        plt.plot(fre_hunters/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
       
    ax.set_title("Final error wrt hunter's number")
    ax.set_xlabel('Number of hunters (units)')
    ax.set_ylabel('Frecuency (%)')
    ax.set_ylim([0,101])
    ax.set_xticklabels(ax_leg)
    ax.locator_params(tight=True, nbins=len(ax_leg))
    plt.legend(loc='best', shadow=False, fontsize='medium')
    plt.show()
    okey = False
    while not okey:
        try:
            plt.savefig(dic_name+'Figure_4')
            okey = True
        except:
            okey = False
    ###############################################################################
    """Success wrt prey number"""
     
    fig = plt.figure()
    ax  = plt.axes()
    
    ax_leg = ['0']
    
    for i in range(num_noise_p):
        ax_leg.append(str(i+1+1))
        plt.plot(fte_preys/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
       
    ax.set_title("Success wrt prey's number")
    ax.set_xlabel('Number of preys (units)')
    ax.set_ylabel('Frecuency (%)')
    ax.set_ylim([0,101])
    ax.set_xticklabels(ax_leg)
    ax.locator_params(tight=True, nbins=len(ax_leg))
    plt.legend(loc='best', shadow=False, fontsize='medium')
    plt.show()
    okey = False
    while not okey:
        try:
            plt.savefig(dic_name+'Figure_5')
            okey = True
        except:
            okey = False
    
    ###############################################################################
    """Success wrt hunters number"""
     
    fig = plt.figure()
    ax  = plt.axes()
    
    ax_leg = ['0']
    for i in range(num_noise_h):
        ax_leg.append(str(i+1+1))
        plt.plot(fte_hunters/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
       
    ax.set_title("Success wrt hunter's number")
    ax.set_xlabel('Number of hunters (units)')
    ax.set_ylabel('Frecuency (%)')
    ax.set_ylim([0,101])
    ax.set_xticklabels(ax_leg)
    ax.locator_params(tight=True, nbins=len(ax_leg))
    plt.legend(loc='best', shadow=False, fontsize='medium')
    plt.show()
    okey = False
    while not okey:
        try:
            plt.savefig(dic_name+'Figure_6')
            okey = True
        except:
            okey = False
  
    ###############################################################################
    """Solver efficiency wrt hunter's number"""
     
    fig = plt.figure()
    ax  = plt.axes()
    
    ax_leg = ['0']
    for i in range(num_noise_h):
        ax_leg.append(str(i+1+1))
        plt.plot(fee_preys/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
       
    ax.set_title("Efficiency wrt hunter's number")
    ax.set_xlabel('Number of hunters (units)')
    ax.set_ylabel('Frecuency (%)')
    ax.set_ylim([0,101])
    ax.set_xticklabels(ax_leg)
    ax.locator_params(tight=True, nbins=len(ax_leg))
    plt.legend(loc='best', shadow=False, fontsize='medium')
    plt.show()
    okey = False
    while not okey:
        try:
            plt.savefig(dic_name+'Figure_7')
            okey = True
        except:
            okey = False
    
    ###############################################################################
    """Solver efficiency wrt prey's number"""
     
    fig = plt.figure()
    ax  = plt.axes()
    
    ax_leg = ['0']
    for i in range(num_noise_h):
        ax_leg.append(str(i+1+1))
        plt.plot(fee_hunters/(num_noise_p*num_noise_h*Sim_num)*100, 'b')
       
    ax.set_title("Efficiency wrt prey's number")
    ax.set_xlabel('Number of preys (units)')
    ax.set_ylabel('Frecuency (%)')
    ax.set_ylim([0,101])
    ax.set_xticklabels(ax_leg)
    ax.locator_params(tight=True, nbins=len(ax_leg))
    plt.legend(loc='best', shadow=False, fontsize='medium')
    plt.show()
    okey = False
    while not okey:
        try:
            plt.savefig(dic_name+'Figure_8')
            okey = True
        except:
            okey = False
    
    ###############################################################################
    """ SAVE """
    csvFile = open('MetricsNM.csv', 'a')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow([error])
    csvWriter.writerow([v_preys])
    csvWriter.writerow([v_hunters])
    csvFile.close() 

    csvFile = open('superMetrics'+str(modes[w])+'.csv', 'a')
    csvWriter = csv.writer(csvFile)
    for i in range(num_noise_p):
        for j in range(num_noise_h):        
            csvWriter.writerow([i+1,j+1,supermetric[i,j,:]])
    csvFile.close()
    
    plt.close('all')


