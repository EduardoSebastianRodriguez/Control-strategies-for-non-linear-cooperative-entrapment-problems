# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:01:58 2018

@author: Eduardo
"""

from main import program
import matplotlib.pyplot as plt
import numpy as np
import random
from functions import plot_mean_and_CI

plt.close('all')
plt.rcParams.update({'font.size': 15})

"""

In order to test the performance of the algorithm, this script plays the 
program as many times as we want, using the mean features we need to compare 
and evaluate the performance

"""

examples = [[2,1],[4,2],[7,3]]
modes    = [1,2,3,4] 
controller = "non-linear"

for w in range (len(modes)):
    for h in range (len(examples)):

        """ Initial variables """
        random.seed(1000000000000066600000000000001)
        
        D                = 5.0
        M_preys          = examples[h][1]
        N_hunters        = examples[h][0]
        mode             = modes[w]
        dic_name         = "G:/Unidades de equipo/EduardoSebastian/TFG/09_Stuff/misres/Modelo_ "+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+"_"
        
        Sim_num          = 1
        
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
        fre_preys        = np.zeros([num_noise_p])
        fre_hunters      = np.zeros([num_noise_h])
        fte_preys        = np.zeros([num_noise_p])
        fte_hunters      = np.zeros([num_noise_h])
        superfinal       = np.zeros([num_noise_p,num_noise_h])
        converge         = False
        med_error        = 0.0
        
        counter_g        = 0
        counter_p        = 0
        counter_h        = 0
        counter_c        = 0
        counter_r        = 0
        counter_t        = 0
        counter_x        = 0
        
        if controller == "linear":
            RR = 2.4
        else:
            RR = 4.5
        
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
                            
                    found_counter = 0
                    while (not h0_found and found_counter<10):
                        
                        found_counter += 1
                        nn = 0.0
                        
                        if N_hunters < M_preys:
                            h0 = []
                            angle = 2*np.pi/N_hunters 
                            for i in range(N_hunters):
                                h0.append(RR*np.cos(i*angle)+random.uniform(-nn,nn))
                                h0.append(RR*np.sin(i*angle)+random.uniform(-nn,nn))
                            for j in range(M_preys-N_hunters):
                                h0.append(0.0)
                                h0.append(0.0)
                                
                        else:
                            h0 = []
                            angle = 2*np.pi/N_hunters
                            for i in range(N_hunters):
                                h0.append(RR*np.cos(i*angle)+random.uniform(-nn,nn))
                                h0.append(RR*np.sin(i*angle)+random.uniform(-nn,nn)) 
                         
                        XP,YP,XH,YH,error,v_preys,v_hunters,h0_found,converge = program(M_preys,N_hunters,mode,h0,p0,noise_level_h,noise_level_p,controller)
                        
                        if found_counter == 10:
                            print("next config")
                                                
                    if found_counter!=10 and not isinstance(error,int):    
                        if converge:
                            fce_preys[n]        += 1
                            fce_hunters[m]      += 1
                        
                        for j in range (M_preys):
                             med_error += error[-1,j]
                             
                        if (med_error*(1.0/M_preys))<0.2:
                            fre_preys[n] += 1
                            fre_hunters[m] += 1
                        
                        if converge and (med_error*(1.0/M_preys))<0.2:
                            counter_t += 1
                            fte_preys[n] += 1
                            fte_hunters[m] += 1
                            superfinal[n,m] += 1
                            
                        if (converge and ((med_error*(1.0/M_preys))>=0.2)):
                            counter_c += 1
                        if (not converge and (med_error*(1.0/M_preys))<0.2):
                            counter_r += 1
                        if (not converge and ((med_error*(1.0/M_preys))>0.2)):
                            counter_x += 1
                        
                    if plot:
                        """Plot error"""    
                        fig = plt.figure()
                        ax = plt.axes()
                        ax.set_title('error metric: distance of each prey from its desired position (m)')
                        ax.set_xlabel('Samples')
                        ax.set_ylabel('Error (m)')
                        for e in range(M_preys):
                            ax.plot(error[:,e],label='Prey '+str(e))
                        legend = ax.legend(loc='best', shadow=False, fontsize='medium')
                        legend.get_frame().set_facecolor('None')
                        plt.show()
                        
                        """Plot v preys """
                        fig = plt.figure()
                        ax = plt.axes()
                        ax.set_title('error metric: velocity of each prey (m/s)')
                        ax.set_xlabel('Samples')
                        ax.set_ylabel('Velocity (m/s)')
                        for e in range(M_preys):
                            ax.plot(v_preys[:,e],label='Prey '+str(e))
                        legend = ax.legend(loc='best', shadow=False, fontsize='medium')
                        legend.get_frame().set_facecolor('None')
                        plt.show()
                    
                    """Save final error"""
                    if (found_counter != 10):
                        for e in range(M_preys):
                            fee_global[counter_g,e]    = error[-1,e]
                            fee_preys[n,counter_p,e]   = error[-1,e]
                            fee_hunters[m,counter_h,e] = error[-1,e]
                            fve_global[counter_g,e]    = v_preys[-1,e]
                            fve_preys[n,counter_p,e]   = v_preys[-1,e]
                            fve_hunters[m,counter_h,e] = v_preys[-1,e]
           
                    """Reset variables"""
                    h0_found = False
                    if counter_g%100==0:
                        print("config " + str(modes[w])+"/"+str(N_hunters)+"_"+str(M_preys))
                        print(counter_g)
                    counter_g += 1
                    counter_p += 1
                    counter_h += 1
                    med_error = 0 
                    
        
        ###############################################################################
        """ Convergence: means preys has reached velocity equals to zero """
        
        print("Ratio of system only converges is: "+str(counter_c/(num_noise_p*num_noise_h*Sim_num)*100)+"%")
        
        ###############################################################################
        """ Final error """
        
        print("Ratio of system only reaches desired set is: "+str(counter_r/(num_noise_p*num_noise_h*Sim_num)*100)+"%")
        
        ###############################################################################
        """ Total objective: convergence + final error """
        
        print("Ratio of system success is: "+str(counter_t/(num_noise_p*num_noise_h*Sim_num)*100)+"%")
        
        ###############################################################################
        """ Fatality: everything goes wrong """
        
        print("Ratio of fatal system is: "+str(counter_x/(num_noise_p*num_noise_h*Sim_num)*100)+"%")
        

        ###############################################################################
#        okey = False
#        while not okey:
#            try:
#                with open(dic_name+"Performance.txt", "w") as text_file:
#                    print(f"Ratio of system only convergences is: "+str(counter_c/(num_noise_p*num_noise_h*Sim_num)*100)+"%", file=text_file)
#                    print(f"Ratio of system only reaches desired set is: "+str(counter_r/(num_noise_p*num_noise_h*Sim_num)*100)+"%", file=text_file)
#                    print(f"Ratio of system success is: "+str(counter_t/(num_noise_p*num_noise_h*Sim_num)*100)+"%", file=text_file)
#                    print(f"Ratio of fatal system is: "+str(counter_x/(num_noise_p*num_noise_h*Sim_num)*100)+"%", file=text_file)
#
#                okey = True
#            except:
#                okey = False    
#        ###############################################################################
#        """Convergence wrt prey noise"""
#         
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        for i in range(num_noise_p):
#            ax_leg.append(str(i/num_noise_p))    
#            plt.plot(fce_preys/(num_noise_h*Sim_num)*100, 'b')
#           
#        ax.set_xlabel('Radius of attraction region (m)')
#        ax.set_ylabel('Frecuency (%)')
#        ax.set_ylim([0,101])
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_1')
#                okey = True
#            except:
#                okey = False
#        ###############################################################################
#        """Convergence wrt hunters noise"""
#         
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        for i in range(num_noise_h):
#            ax_leg.append(str(i/num_noise_h))    
#            plt.plot(fce_hunters/(num_noise_h*Sim_num)*100, 'b')
#           
#        ax.set_xlabel('Radius of attraction region (m)')
#        ax.set_ylabel('Frecuency (%)')
#        ax.set_ylim([0,101])
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_2')
#                okey = True
#            except:
#                okey = False
#        
#        ###############################################################################
#        """Final error wrt prey noise"""
#         
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        for i in range(num_noise_p):
#            ax_leg.append(str(i/num_noise_p))    
#            plt.plot(fre_preys/(num_noise_h*Sim_num)*100, 'b')
#           
#        ax.set_xlabel('Radius of attraction region (m)')
#        ax.set_ylabel('Frecuency (%)')
#        ax.set_ylim([0,101])
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_3')
#                okey = True
#            except:
#                okey = False
#        ###############################################################################
#        """Final error wrt hunters noise"""
#         
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        for i in range(num_noise_h):
#            ax_leg.append(str(i/num_noise_h))    
#            plt.plot(fre_hunters/(num_noise_h*Sim_num)*100, 'b')
#           
#        ax.set_xlabel('Radius of attraction region (m)')
#        ax.set_ylabel('Frecuency (%)')
#        ax.set_ylim([0,101])
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_4')
#                okey = True
#            except:
#                okey = False
#        
#        ###############################################################################
#        """Success wrt prey noise"""
#         
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        for i in range(num_noise_p):
#            ax_leg.append(str(i/num_noise_p))    
#            plt.plot(fte_preys/(num_noise_h*Sim_num)*100, 'b')
#           
#        ax.set_xlabel('Radius of attraction region (m)')
#        ax.set_ylabel('Frecuency (%)')
#        ax.set_ylim([0,101])
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_5')
#                okey = True
#            except:
#                okey = False
#        ###############################################################################
#        """Success wrt hunters noise"""
#         
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        for i in range(num_noise_h):
#            ax_leg.append(str(i/num_noise_h))    
#            plt.plot(fte_hunters/(num_noise_h*Sim_num)*100, 'b')
#           
#        ax.set_xlabel('Radius of attraction region (m)')
#        ax.set_ylabel('Frecuency (%)')
#        ax.set_ylim([0,101])
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_6')
#                okey = True
#            except:
#                okey = False
#        
#        ###############################################################################
#        """Global error Histogram"""
#        histogram=plt.figure()
#        ax       =plt.axes()
#        
#        bins = np.linspace(0, 3, 100)
#        
#        ax.set_title('Global Error Histogram')
#        ax.set_xlabel('Final Error (m)')
#        ax.set_ylabel('Frecuency (ocurrencies)')
#          
#        """One for each prey"""
#        media = np.zeros([Sim_num*num_noise_p*num_noise_h])
#        for j in range (M_preys):
#            media += fee_global[:,j]
#            
#        media = media*(1.0/M_preys)
#        plt.hist(media,bins,alpha=0.5)
#            
#            
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_7')
#                okey = True
#            except:
#                okey = False
#        ###############################################################################
#        """Global Convergence Histogram"""
#        histogram=plt.figure()
#        ax       =plt.axes()
#        
#        bins = np.linspace(0, 10, 100)
#        
#        ax.set_title('Global Convergence Histogram')
#        ax.set_xlabel('Final Velocity (m/s)')
#        ax.set_ylabel('Frecuency (ocurrencies)')
#          
#        """One for each prey"""
#        media = np.zeros([Sim_num*num_noise_p*num_noise_h])
#        for j in range (M_preys):
#            media += fve_global[:,j]
#            
#        plt.hist(media*(1/M_preys),bins,alpha=0.5)
#        
#            
#            
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_8')
#                okey = True
#            except:
#                okey = False
#        ###############################################################################
#        """Histograms of the influence of the hunters' noise in error"""
#        histogram=plt.figure()
#        ax       =plt.axes()
#        
#        bins = np.linspace(0, 3, 100)
#        
#        ax.set_title("Histograms of the influence of hunter's radius of attraction region (m) in error")
#        ax.set_xlabel('Final Error (m)')
#        ax.set_ylabel('Frecuency (times)')
#          
#        """One for each noise preys' level"""
#        
#        for n in range(num_noise_p):
#            media = 0*fee_preys[n,:,0]
#            for e in range(M_preys):
#                media += fee_preys[n,:,e]
#            media = media/M_preys
#            plt.hist(media,bins,alpha=0.5, label = "Radius of attraction region of preys (m) is "+str(n/num_noise_p))
#            
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_9')
#                okey = True
#            except:
#                okey = False
#        
#        ###############################################################################
#        """Histograms of the influence of the preys' noise in error"""
#        histogram=plt.figure()
#        ax       =plt.axes()
#        
#        bins = np.linspace(0, 3, 100)
#        
#        ax.set_title("Histograms of the influence of prey's radius of attraction region (m) in error")
#        ax.set_xlabel('Final Error (m)')
#        ax.set_ylabel('Frecuency (ocurrencies)')
#          
#        """One for each noise hunters' level"""
#        
#        for m in range(num_noise_h):
#            media = 0*fee_hunters[m,:,0]
#            for e in range(M_preys):
#                media += fee_hunters[m,:,e]
#            media = media/M_preys
#            plt.hist(media,bins,alpha=0.5, label = "Radius of attraction region of hunters (m) is "+str(m/num_noise_h))
#            
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_10')
#                okey = True
#            except:
#                okey = False
#        
#        ###############################################################################
#        """Histograms of the influence of the hunters' noise in convergence"""
#        histogram=plt.figure()
#        ax       =plt.axes()
#        
#        bins = np.linspace(0, 3, 100)
#        
#        ax.set_title("Histograms of the influence of hunter's radius of attraction region (m) in convergence")
#        ax.set_xlabel('Final Velocity (m/s)')
#        ax.set_ylabel('Frecuency (ocurrencies)')
#          
#        """One for each noise preys' level"""
#        
#        for n in range(num_noise_p):
#            media = 0*fve_preys[n,:,0]
#            for e in range(M_preys):
#                media += fve_preys[n,:,e]
#            media = media/M_preys
#            plt.hist(media,bins,alpha=0.5, label = "Radius of attraction region of preys (m) is "+str(n/num_noise_p))
#            
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_11')
#                okey = True
#            except:
#                okey = False
#        
#        ###############################################################################
#        """Histograms of the influence of the preys' noise in convergence"""
#        histogram=plt.figure()
#        ax       =plt.axes()
#        
#        bins = np.linspace(0, 3, 100)
#        
#        ax.set_title("Histograms of the influence of prey's radius of attraction region (m) in convergence")
#        ax.set_xlabel('Final Velocity (m/s)')
#        ax.set_ylabel('Frecuency (ocurrencies)')
#          
#        """One for each noise hunters' level"""
#        
#        for m in range(num_noise_h):
#            media = 0*fve_hunters[m,:,0]
#            for e in range(M_preys):
#                media += fve_hunters[m,:,e]
#            media = media/M_preys
#            plt.hist(media,bins,alpha=0.5, label = "Radius of attraction region of hunters (m) is "+str(m/num_noise_h))
#            
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_12')
#                okey = True
#            except:
#                okey = False
#        
#        ###############################################################################
#        # MÉTRICAS COMPACTAS
#        ###############################################################################
#        mean_and_cov_e = np.zeros([num_noise_p,num_noise_h,2])
#        mean_and_cov_v = np.zeros([num_noise_p,num_noise_h,2])
#        
#        for n in range(num_noise_p):
#            media_e = 0
#            covarianza_e = 0
#            media_v = 0
#            covarianza_v = 0 
#            counter = 0
#            
#            for m in range(num_noise_h*Sim_num):
#                
#                for e in range(M_preys):
#                
#                    media_e += fee_preys[n,m,e]
#                    media_v += fve_preys[n,m,e]
#            
#                if ((m+1)%(Sim_num) == 0):
#                    mean_and_cov_e[n,counter,0] = media_e/(M_preys*Sim_num)
#                    mean_and_cov_v[n,counter,0] = media_v/(M_preys*Sim_num)
#                    media_e = 0
#                    media_v = 0
#                    counter += 1
#            
#            counter = 0
#            
#            for m in range(num_noise_h*Sim_num):
#                
#                for e in range(M_preys):
#                    
#                    covarianza_e += (fee_preys[n,m,e] - mean_and_cov_e[n,counter,0])**2
#                    covarianza_v += (fve_preys[n,m,e] - mean_and_cov_v[n,counter,0])**2
#        
#                if ((m+1)%(Sim_num) == 0):
#                    mean_and_cov_e[n,counter,1] = np.sqrt(covarianza_e/(M_preys*Sim_num))
#                    mean_and_cov_v[n,counter,1] = np.sqrt(covarianza_v/(M_preys*Sim_num))
#                    covarianza_e = 0
#                    covarianza_v = 0
#                    counter += 1
#                    
#        ###############################################################################
#        """Plot comprise info of final error wrt prey noise"""
#         
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        colors = ['C0', 'C3', 'C6', 'C9']
#        for i in range(4):
#            ax_leg.append(str((3*i+0)/num_noise_p))
#            ax_leg.append(str((3*i+1)/num_noise_p))
#            ax_leg.append(str((3*i+2)/num_noise_p))
#            
#            mean = mean_and_cov_e[i*3,:,0]
#            ub = mean_and_cov_e[i*3,:,0] + mean_and_cov_e[i*3,:,1]*1.96/np.sqrt(Sim_num*M_preys)
#            lb = mean_and_cov_e[i*3,:,0] - mean_and_cov_e[i*3,:,1]*1.96/np.sqrt(Sim_num*M_preys)
#            plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = "Radius is "+str(i*3/num_noise_h)+' m')
#        
#        ax.set_xlabel('Radius of attraction region in hunters (m)')
#        ax.set_ylabel('Final error (m)')
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_13')
#                okey = True
#            except:
#                okey = False
#        
#        
#        ###############################################################################
#        """Plot comprise info of convergence wrt prey noise"""
#        
#        
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        colors = ['C0','C3', 'C6','C9']
#        
#        for i in range(4):
#            
#            ax_leg.append(str((3*i+0)/num_noise_p))
#            ax_leg.append(str((3*i+1)/num_noise_p))
#            ax_leg.append(str((3*i+2)/num_noise_p))
#            
#            mean = mean_and_cov_v[3*i,:,0]
#            ub = mean_and_cov_v[3*i,:,0] + mean_and_cov_v[3*i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
#            lb = mean_and_cov_v[3*i,:,0] - mean_and_cov_v[3*i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
#            plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = "Radius is "+str(3*i/num_noise_h)+' m')
#        
#        ax.set_xlabel('Radius of attraction region in hunters (m)')
#        ax.set_ylabel('Final velocity (m/s)')
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_14')
#                okey = True
#            except:
#                okey = False
#        
#        ###############################################################################
#        # MÉTRICAS COMPACTAS
#        ###############################################################################
#        mean_and_cov_e = np.zeros([num_noise_h,num_noise_p,2])
#        mean_and_cov_v = np.zeros([num_noise_h,num_noise_p,2])
#        
#        for m in range(num_noise_h):
#            media_e = 0
#            covarianza_e = 0
#            media_v = 0
#            covarianza_v = 0 
#            counter = 0
#            
#            for n in range(num_noise_p*Sim_num):
#                
#                for e in range(M_preys):
#                
#                    media_e += fee_hunters[m,n,e]
#                    media_v += fve_hunters[m,n,e]
#            
#                if ((n+1)%(Sim_num) == 0):
#                    mean_and_cov_e[m,counter,0] = media_e/(M_preys*Sim_num)
#                    mean_and_cov_v[m,counter,0] = media_v/(M_preys*Sim_num)
#                    media_e = 0
#                    media_v = 0
#                    counter += 1
#            
#            counter = 0
#            
#            for n in range(num_noise_p*Sim_num):
#                
#                for e in range(M_preys):
#                    
#                    covarianza_e += (fee_preys[m,n,e] - mean_and_cov_e[m,counter,0])**2
#                    covarianza_v += (fve_preys[m,n,e] - mean_and_cov_v[m,counter,0])**2
#        
#                if ((n+1)%(Sim_num) == 0):
#                    mean_and_cov_e[m,counter,1] = np.sqrt(covarianza_e/(M_preys*Sim_num))
#                    mean_and_cov_v[m,counter,1] = np.sqrt(covarianza_v/(M_preys*Sim_num))
#                    covarianza_e = 0
#                    covarianza_v = 0
#                    counter += 1
#        
#        
#        ###############################################################################
#        """Plot comprise info of final error wrt hunter noise"""
#         
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        colors = ['C0', 'C3', 'C6', 'C9']
#        for i in range(4):
#            
#            ax_leg.append(str((3*i+0)/num_noise_p))
#            ax_leg.append(str((3*i+1)/num_noise_p))
#            ax_leg.append(str((3*i+2)/num_noise_p))
#            
#            mean = mean_and_cov_e[i*3,:,0]
#            ub = mean_and_cov_e[i*3,:,0] + mean_and_cov_e[i*3,:,1]*1.96/np.sqrt(Sim_num*M_preys)
#            lb = mean_and_cov_e[i*3,:,0] - mean_and_cov_e[i*3,:,1]*1.96/np.sqrt(Sim_num*M_preys)
#            plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = "Radius is "+str(i*3/num_noise_p)+' m')
#        
#        ax.set_xlabel('Radius of attraction region in preys (m)')
#        ax.set_ylabel('Final error (m)')
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_15')
#                okey = True
#            except:
#                okey = False
#        
#        ###############################################################################
#        """Plot comprise info of convergence wrt hunter noise"""
#        
#        
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        colors = ['C0','C3', 'C6','C9']
#        
#        for i in range(4):
#            
#            ax_leg.append(str((3*i+0)/num_noise_p))
#            ax_leg.append(str((3*i+1)/num_noise_p))
#            ax_leg.append(str((3*i+2)/num_noise_p))
#            
#            mean = mean_and_cov_v[3*i,:,0]
#            ub = mean_and_cov_v[3*i,:,0] + mean_and_cov_v[3*i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
#            lb = mean_and_cov_v[3*i,:,0] - mean_and_cov_v[3*i,:,1]*1.96/np.sqrt(Sim_num*M_preys)
#            plot_mean_and_CI(mean, ub, lb, color_mean=colors[i], color_shading=colors[i], legend = "Radius is "+str(3*i/num_noise_p)+' m')
#        
#        ax.set_xlabel('Radius of attraction region in preys (m)')
#        ax.set_ylabel('Final velocity (m/s)')
#        ax.set_xticklabels(ax_leg)
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.legend(loc='best', shadow=False, fontsize='medium')
#        plt.show()
#        okey = False
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_16')
#                okey = True
#            except:
#                okey = False
#        
#        #######################################################################
#        """SUPERMETRIC PLOT"""
#        
#        
#        fig = plt.figure()
#        ax  = plt.axes()
#        
#        ax_leg = ['0.0']
#        colors = ['C0','C1', 'C2','C3']
#        styles = ['-','--','-.',':']
#        
#        for i in range(4):
#            
#            ax_leg.append(str((3*i+0)/num_noise_p))
#            ax_leg.append(str((3*i+1)/num_noise_p))
#            ax_leg.append(str((3*i+2)/num_noise_p))
#            
#            plt.plot(superfinal[3*i,:], colors[i],label = "Radius is "+str(i*3/num_noise_p)+' m',linewidth=5,linestyle=styles[i])   
#            
#        ax.set_xlabel('Radius of attraction region of hunters (m)')
#        ax.set_ylabel('Ratio of success (%)')
#        ax.set_xticklabels(ax_leg)
#        box = ax.get_position()
#        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=False, fontsize='small')
#        ax.locator_params(tight=True, nbins=len(ax_leg))
#        plt.show()
#        okey = False
#        
#        while not okey:
#            try:
#                plt.savefig(dic_name+'Figure_17')
#                okey = True
#            except:
#                okey = False
#        
#        plt.close('all')
        ###############################################################################
        """ SAVE """
#        okey = False
#        while not okey:
#            try:
        dicdic1 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/COUNTERS_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic1, [counter_p,counter_h,counter_c,counter_r,counter_t,counter_x])
        dicdic2 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fee_global_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic2, fee_global)
#        dicdic3 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fee_preys_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
#        np.savetxt(dicdic3, fee_preys)
#        dicdic4 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fee_hunters_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
#        np.savetxt(dicdic4, fee_hunters)
        dicdic5 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fve_global_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic5, fve_global)
#        dicdic7 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fve_preys_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
#        np.savetxt(dicdic7, fve_preys)
#        dicdic8 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fve_hunters_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
#        np.savetxt(dicdic8, fve_hunters)
        dicdic9 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fce_preys_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic9, fce_preys)
        dicdic10 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fce_hunters_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic10, fce_hunters)
        dicdic11 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fre_preys_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic11, fre_preys)
        dicdic12 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fre_hunters_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic12, fre_hunters)
        dicdic13 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fte_preys_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic13, fte_preys)
        dicdic14 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/fte_hunters_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic14, fte_hunters)
        dicdic15 = 'G:/Unidades compartidas/EduardoSebastian/TFG/09_Stuff/misres/superfinal_'+str(modes[w])+"_"+str(N_hunters)+"_"+str(M_preys)+'.dat'      
        np.savetxt(dicdic15, superfinal)
#                okey = True
#            except e:
#                    print(e)
#                    okey = False
                    
        
        
#        errores = []
#        velocidades = []
#        media = 0
#        desv = 0
#        num = 0
#        for i in range(len(fee_global)):
#            MAE = 0
#            for j in range(3):
#                MAE += fee_global[i,j]
#            MAE = MAE/3
#            errores.append(MAE)
#        
#        for i in range(len(fve_global)):
#            V = 0
#            for j in range(3):
#                V += fve_global[i,j]
#            V = V/3
#            velocidades.append(V)
#        
#        for i in range(len(errores)):
#            if errores[i] < 0.2 and velocidades[i] < 0.5:
#                media += errores[i]
#                num += 1
#                
#        media = media/num
#        print(media)
#        
#        for i in range(len(errores)):
#            if errores[i] < 0.2 and velocidades[i] < 0.5:
#                desv += (errores[i]-media)*(errores[i]-media)
#                
#        desv = np.sqrt(desv/num)
#        print(desv)
#        
#        errores = []
#        velocidades = []
#        media = 0
#        desv = 0
#        num = 0
#        for i in range(len(fee_global)):
#            MAE = 0
#            for j in range(3):
#                MAE += fee_global[i,j]
#            MAE = MAE/3
#            errores.append(MAE)
#        
#        for i in range(len(fve_global)):
#            V = 0
#            for j in range(3):
#                V += fve_global[i,j]
#            V = V/3
#            velocidades.append(V)
#        
#        for i in range(len(errores)):
#            if errores[i] < 0.2 and velocidades[i] < 0.25:
#                media += errores[i]
#                num += 1
#                
#        media = media/num
#        print(media)
#        
#        for i in range(len(errores)):
#            if errores[i] < 0.2 and velocidades[i] < 0.25:
#                desv += (errores[i]-media)*(errores[i]-media)
#                
#        desv = np.sqrt(desv/num)
#        print(desv)
                
            
        
                        

                
        




