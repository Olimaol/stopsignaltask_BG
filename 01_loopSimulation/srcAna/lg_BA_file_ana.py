"""
Created on Fri Oct 16 2020

@authors: Oliver Maith, Lorenz Goenner, ilko
"""

from ANNarchy import*
import pylab 
import random
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import scipy.stats as st
from scipy import stats
from scipy.signal import argrelmin

from BGmodelSST.analysis import custom_zscore, custom_zscore_start0, \
    plot_zscore_stopVsGo, plot_zscore_stopVsGo_five, plot_zscore_stopVsGo_NewData, plot_zscore_stopVsGo_five_NewData, plot_correl_rates_Intmax, plot_meanrate_All_FailedVsCorrectStop, plot_meanrate_All_FailedStopVsCorrectGo, calc_KW_stats_all, rate_plot, \
    calc_meanrate_std_failed_correct, get_rates_failed_correct, get_poprate_aligned_onset, get_peak_response_time, custom_poprate, calc_meanrate_std_Fast_Slow, get_fast_slow_go_trials, get_rates_allGo_fastGo_slowGo
from BGmodelSST.plotting import get_and_plot_syn_mon    
from BGmodelSST.sim_params import params
from BGmodelSST.init import init_neuronmodels
import matplotlib.lines as mlines



#setup(dt=1)
setup(dt=0.1) # Extremely different results!!!
setup(num_threads=1)
# TODO: Adapt all plots to dt=0.1 !!!

#setup(seed=0)
#np.random.seed(0)

from BGmodelSST.neuronmodels import Izhikevich_neuron, Izhikevich_STR_neuron,STR_FSI_neuron, Integrator_neuron, Poisson_neuron, FixedSynapse
from BGmodelSST.populations import Stoppinput1, Cortex_S, Cortex_G, STR_D1, STR_D2, STN, SNr, GPe_Proto, Thal, Integrator, IntegratorStop, GPeE, SNrE, STNE, STR_FSI, STRE, GPe_Arky, TestThalnoise, population_size, GPe_Proto2
from BGmodelSST.projections import Stoppinput1STN, Cortex_GSTR_D1, Cortex_GSTR_D2, Cortex_GSTR_FSI, Cortex_GThal, Cortex_SGPe_Arky, STR_D1SNr, STR_D2GPe_Proto, STNSNr, \
                                    STNGPe_Proto, GPe_ProtoSTN, GPe_ProtoSNr, SNrThal, ThalIntegrator, GPeEGPe_Proto, GPEGPe_Arky, SNrESNr, STNESTN, \
                                    STR_FSISTR_D1, STR_FSISTR_D2, STRESTR_D1, STRESTR_D2, GPe_ArkySTR_D1, GPe_ArkySTR_D2, TestThalnoiseThal,STRESTR_FSI, \
                                    STR_FSISTR_FSI, GPe_ArkySTR_FSI, GPe_ArkyGPe_Proto, GPe_ProtoGPe_Arky, STR_D2GPe_Arky, \
                                    GPe_ProtoSTR_FSI, STR_D1STR_D1, STR_D1STR_D2, STR_D2STR_D1, STR_D2STR_D2, Cortex_GSTR_D2, Cortex_SGPe_Proto, STNGPe_Arky, GPe_ArkyCortex_G, \
                                    ThalSD1, ThalSD2, ThalFSI, \
                                    GPe_ProtoGPe_Proto2, GPe_Proto2GPe_Proto, STR_D2GPe_Proto2, STR_D1GPe_Proto2, STNGPe_Proto2, Cortex_SGPe_Proto2, GPe_ArkyGPe_Proto2, GPe_Proto2STR_D1, GPe_Proto2STR_D2, GPe_Proto2STR_FSI, GPe_Proto2GPe_Arky, GPe_Proto2IntegratorStop, EProto1GPe_Proto, EProto2GPe_Proto2, EArkyGPe_Arky, \
                                    Cortex_SGPe_Arky2, STR_D2GPe_Arky2, GPe_ProtoGPe_Arky2, STNGPe_Arky2, GPe_Proto2GPe_Arky2, EArkyGPe_Arky2, GPe_Arky2STR_D1, GPe_Arky2STR_D2, GPe_Arky2STR_FSI





##############################################################################################
if abfrage_loop == 'STR': 
    #plt.ion()

    param_id = params['general_id']
    t_init = params['t_init'] # 100
    t_SSD = params['t_SSD'] # 170   
    #t_min = int((t_init + t_SSD - 600) / dt())
    t_min = int((t_init + t_SSD - 200) / dt())    
    t_max = int((t_init + t_SSD + 400) / dt())    
    
    fsize = 15    
    reset(populations=True, projections=False, synapses=False, net_id=0)           

    STR_D1_Stop_mean, STR_D1_Stop_sem = np.load('data/SD1_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')    
    STR_D1_Go_mean, STR_D1_Go_sem = np.load('data/SD1_rate_Go_mean_std'+str(i_netw_rep)+'.npy')    
    STR_D2_Stop_mean, STR_D2_Stop_sem = np.load('data/SD2_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')    
    STR_D2_Go_mean, STR_D2_Go_sem = np.load('data/SD2_rate_Go_mean_std'+str(i_netw_rep)+'.npy')    
    GPe_Arky_mean_Stop, GPe_Arky_sem_Stop = np.load('data/GPeArky_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')    
    GPe_Proto_mean_Stop, GPe_Proto_sem_Stop = np.load('data/GPeProto_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')
    GPe_Proto2_mean_Stop, GPe_Proto2_sem_Stop = np.load('data/GPeProto2_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')    
    STR_FSI_Stop_mean, STR_FSI_Stop_sem = np.load('data/FSI_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')  
    STR_FSI_Go_mean, STR_FSI_Go_sem = np.load('data/FSI_rate_Go_mean_std'+str(i_netw_rep)+'.npy')    
    STN_mean_Stop, STN_sem_Stop = np.load('data/STN_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')    
    SNr_mean_Stop, SNr_sem_Stop = np.load('data/SNr_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')        
    STR_D1_Stop_mean = STR_D1_Stop_mean[ : -100] # exclude artefact in last time steps?
    STR_D1_Go_mean = STR_D1_Go_mean[ : -100]    
    STR_D2_Stop_mean = STR_D2_Stop_mean[ : -100] # exclude artefact in last time steps?
    STR_D2_Go_mean = STR_D2_Go_mean[ : -100]    
    STR_FSI_Stop_mean = STR_FSI_Stop_mean[ : -100] # exclude artefact in last time steps?
    STR_FSI_Go_mean = STR_FSI_Go_mean[ : -100]        

    GPe_Arky_mean_Go, GPe_Arky_sem_Go = np.load('data/GPeArky_rate_Go_mean_std'+str(i_netw_rep)+'.npy')    
    GPe_Proto_mean_Go, GPe_Proto_sem_Go = np.load('data/GPeProto_rate_Go_mean_std'+str(i_netw_rep)+'.npy')    
    GPe_Proto2_mean_Go, GPe_Proto2_sem_Go = np.load('data/GPeProto2_rate_Go_mean_std'+str(i_netw_rep)+'.npy')        
    STN_mean_Go, STN_sem_Go = np.load('data/STN_rate_Go_mean_std'+str(i_netw_rep)+'.npy')        
    SNr_mean_Go, SNr_sem_Go = np.load('data/SNr_rate_Go_mean_std'+str(i_netw_rep)+'.npy')            
    Thal_mean_Go, Thal_sem_Go = np.load('data/Thal_rate_Go_mean_std'+str(i_netw_rep)+'.npy')       
    Thal_mean_Stop, Thal_sem_Stop = np.load('data/Thal_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')           

    Cortex_G_mean_Stop, Cortex_G_std_Stop =  np.load('data/Cortex_G_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')
    Cortex_S_mean_Stop, Cortex_S_std_Stop =  np.load('data/Cortex_S_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')

    SD1_mean_FailedStop, SD1_std_FailedStop, SD1_mean_CorrectStop, SD1_std_CorrectStop = np.load('data/SD1_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    SD2_mean_FailedStop, SD2_std_FailedStop, SD2_mean_CorrectStop, SD2_std_CorrectStop = np.load('data/SD2_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    
    FSI_mean_FailedStop, FSI_std_FailedStop, FSI_mean_CorrectStop, FSI_std_CorrectStop = np.load('data/FSI_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')        
    STN_mean_FailedStop, STN_std_FailedStop, STN_mean_CorrectStop, STN_std_CorrectStop = np.load('data/STN_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    GPe_Proto_mean_FailedStop, GPe_Proto_std_FailedStop, GPe_Proto_mean_CorrectStop, GPe_Proto_std_CorrectStop = np.load('data/Proto_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    
    GPe_Arky_mean_FailedStop, GPe_Arky_std_FailedStop, GPe_Arky_mean_CorrectStop, GPe_Arky_std_CorrectStop = np.load('data/Arky_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')        
    SNr_mean_FailedStop, SNr_std_FailedStop, SNr_mean_CorrectStop, SNr_std_CorrectStop = np.load('data/SNr_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')

    StrD1_meanrate_FastGo,      StrD1_std_FastGo,       StrD1_meanrate_SlowGo,      StrD1_std_SlowGo        = np.load('data/SD1_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')            
    StrD2_meanrate_FastGo,      StrD2_std_FastGo,       StrD2_meanrate_SlowGo,      StrD2_std_SlowGo        = np.load('data/SD2_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')                        
    StrFSI_meanrate_FastGo,     StrFSI_std_FastGo,      StrFSI_meanrate_SlowGo,     StrFSI_std_SlowGo       = np.load('data/FSI_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo     = np.load('data/GPe_Arky_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo    = np.load('data/GPe_Proto_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
                         

    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()      
    rate_plot(STR_D1_Stop_mean, STR_D1_Stop_sem, 'STR_D1_Stop', trials,  'r', plot_sem = False, pltlabel='D1 Stop')     
    rate_plot(STR_D1_Go_mean, STR_D1_Go_sem, 'STR_D1_Go', trials, 'g', plot_sem = False, pltlabel='D1 Go')         
    rate_plot(GPe_Arky_mean_Stop, GPe_Arky_sem_Stop, 'GPe_Arky', trials, 'cyan', plot_sem = False, pltlabel='Arky Stop')         
    rate_plot(GPe_Proto_mean_Stop, GPe_Proto_sem_Stop, 'GPe_Proto', trials, 'blue', plot_sem = False, pltlabel='Proto Stop')    
    plt.plot((t_init + t_SSD)/dt() * np.ones(2), [0, plt.axis()[3]], 'k--', lw=1.5, label='Stop cue')
    ax.axis([t_min, t_max, max(0, ax.axis()[2]), 80]) # dt=0.1
    plt.xlabel('t [msec]')
    plt.ylabel('Firing rate [spk/s]', fontsize=fsize)
    ax.set_xticks([(t_init + t_SSD - 200)/dt(),(t_init + t_SSD)/dt(), (t_init + t_SSD + 400)/dt()]) # LG
    ax.set_xticklabels([- 0.2, 0, 0.4]) # LG
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.legend(fontsize=6)    
    plt.tight_layout()    
    plt.savefig(saveFolderPlots+'mean_rate_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_STRD1_GPeStop.png') #             
    
    plt.figure(figsize=(3.5,4), dpi=300)
    rate_plot(STR_D2_Stop_mean, STR_D2_Stop_sem, 'STR_D2_Stop', trials, 'pink', plot_sem = False, pltlabel='D2 Stop')     
    rate_plot(STR_D2_Go_mean, STR_D2_Go_sem, 'STR_D2_Go', trials, 'purple', plot_sem = False, pltlabel='D2 Go')            
    rate_plot(STR_FSI_Stop_mean, STR_FSI_Stop_sem, 'STR_FSI_Stop', trials, 'k', plot_sem = False, pltlabel='FSI Stop')     
    rate_plot(STR_FSI_Go_mean, STR_FSI_Go_sem, 'STR_FSI_Go', trials, '0.6', plot_sem = False, pltlabel='FSI Go') 
    plt.ion()
    ax=plt.gca()
    #plt.title('Population-averaged firing rate, mean across '+str(trials)+' trials') # STR D1
    #t0 = 50 # 50ms simulation for initialization is not recorded, see above
    plt.plot((t_init + t_SSD)/dt() * np.ones(2), [0, plt.axis()[3]], 'k--', lw=1.5, label='Stop cue')
    ax.axis([t_min, t_max, max(0, ax.axis()[2]), 80]) # dt=0.1
    plt.xlabel('t [msec]')
    plt.ylabel('Firing rate [spk/s]', fontsize=fsize)
    ax.set_xticks([(t_init + t_SSD - 200)/dt(),(t_init + t_SSD)/dt(), (t_init + t_SSD + 400)/dt()]) # LG
    ax.set_xticklabels([- 0.2, 0, 0.4]) # LG
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #plt.legend(('D1 Stop', 'D1 Go', 'D2 Stop', 'D2 Go', 'GPe Arky Stop', 'GPe Proto Stop', 'FSI Stop', 'FSI Go', 'Stop cue'), fontsize=6) # 10)
    plt.legend(fontsize=6)    
    plt.tight_layout()
    plt.savefig(saveFolderPlots+'mean_rate_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_STR_D2FSI.png') #             
    #plt.ioff()    
    #plt.show()
    
    toRGB = {'blue':np.array([3,67,223])/255., 'cyan':np.array([0,255,255])/255., 'gold':np.array([219,180,12])/255., 'orange':np.array([249,115,6])/255., 'red':np.array([229,0,0])/255., 'purple':np.array([126,30,156])/255., 'grey':np.array([146,149,145])/255., 'light brown':np.array([173,129,80])/255., 'lime':np.array([170,255,50])/255., 'green':np.array([21,176,26])/255., 'yellow':np.array([255,255,20])/255., 'lightgrey':np.array([216,220,214])/255.}
    STRzscorePlotsMaxT = 350

    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()    
    STRD2_all_mean = StrD2_meanrate_SlowGo#0.5*(SD2_mean_CorrectStop + StrD2_meanrate_SlowGo)
    STRFSI_all_mean = StrFSI_meanrate_SlowGo
    #STR_all_mean = 1.0/2.0*(SD1_mean_CorrectStop + SD2_mean_CorrectStop)# + FSI_mean_CorrectStop)    
    plt.plot(custom_zscore_start0(SD2_mean_CorrectStop, (t_init + t_SSD - 200)/dt(), STRD2_all_mean), color=toRGB['purple'], lw=3)
    plt.plot(custom_zscore_start0(StrD2_meanrate_SlowGo, (t_init + t_SSD - 200)/dt(), STRD2_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['purple']*0.7, lw=3) 
    plt.plot(custom_zscore_start0(FSI_mean_CorrectStop, (t_init + t_SSD - 200)/dt(), STRFSI_all_mean), color=toRGB['grey'], lw=3)     
    plt.plot(custom_zscore_start0(StrFSI_meanrate_SlowGo, (t_init + t_SSD - 200)/dt(), STRFSI_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['grey']*0.7, lw=3) 
    plt.axvline((t_init + t_SSD)/dt(), color='grey', lw=0.5)
    plt.ylabel('$\Delta$Firing rate (z-score)', fontsize=fsize)
    ax.axis([(t_init + t_SSD - 200)/dt(), (t_init + t_SSD + STRzscorePlotsMaxT)/dt(), -2.2, 3.6]) # 
    ax.set_xticks([(t_init + t_SSD - 200)/dt(), (t_init + t_SSD)/dt(), (t_init + t_SSD + 300)/dt()]) # LG
    ax.set_xticklabels([-0.2, 0, 0.3]) # LG
    ax.set_yticks(range(-2,4)) # 4
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #legend
    normalLine = mlines.Line2D([], [], color='k', label='Stop', lw=3)
    dashedLine = mlines.Line2D([], [], color='k', label='slow Go', dash_capstyle='round', dashes=(0.05,2), lw=3)
    purpleLine = mlines.Line2D([], [], color=toRGB['purple'], label='StrD2', lw=3)
    greyLine = mlines.Line2D([], [], color=toRGB['grey'], label='StrFSI', lw=3)
    plt.legend(handles=[dashedLine, normalLine, purpleLine, greyLine], bbox_to_anchor=(-0.02,-0.015), bbox_transform=plt.gca().transAxes, fontsize=10, loc='lower left')
    #save
    plt.tight_layout()
    plt.savefig(saveFolderPlots+'zscore_StopSTRD2FSI_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.png', dpi=300)
    

    plt.figure()
    plt.subplot(111)
    #plt.plot(STR_D1_Go_mean,label='1')
    plt.plot(StrD1_meanrate_SlowGo,label='2')
    plt.plot(StrD1_meanrate_FastGo,label='3')
    plt.legend()
    plt.savefig('TESTFIGURE.png')


    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()
    STR_all_mean        = StrD1_meanrate_SlowGo#0.5*(SD1_mean_CorrectStop + StrD1_meanrate_SlowGo)       
    GPe_Arky_all_mean   = GPe_Arky_meanrate_SlowGo
    GPe_Proto_all_mean  = GPe_Proto_meanrate_SlowGo
    plt.plot(custom_zscore_start0(SD1_mean_CorrectStop, (t_init + t_SSD - 200)/dt(), STR_all_mean), color=toRGB['red'], lw=3)     
    plt.plot(custom_zscore_start0(StrD1_meanrate_SlowGo, (t_init + t_SSD - 200)/dt(), STR_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['red']*0.7, lw=3)         
    plt.plot(custom_zscore_start0(GPe_Arky_mean_CorrectStop, (t_init + t_SSD - 200)/dt(), GPe_Arky_all_mean), color=toRGB['cyan'], lw=3)
    plt.plot(custom_zscore_start0(GPe_Arky_meanrate_SlowGo, (t_init + t_SSD - 200)/dt(), GPe_Arky_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['cyan']*0.7, lw=3)
    plt.plot(custom_zscore_start0(GPe_Proto_mean_CorrectStop, (t_init + t_SSD - 200)/dt(), GPe_Proto_all_mean), color=toRGB['blue'], lw=3) 
    plt.plot(custom_zscore_start0(GPe_Proto_meanrate_SlowGo, (t_init + t_SSD - 200)/dt(), GPe_Proto_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['blue']*0.7, lw=3)
    plt.axvline((t_init + t_SSD)/dt(), color='grey', lw=0.5)
    plt.ylabel('$\Delta$Firing rate (z-score)', fontsize=fsize)
    ax.axis([(t_init + t_SSD - 200)/dt(), (t_init + t_SSD + STRzscorePlotsMaxT)/dt(), -2.2, 3.6]) # 
    ax.set_xticks([(t_init + t_SSD - 200)/dt(), (t_init + t_SSD)/dt(), (t_init + t_SSD + 300)/dt()]) # LG
    ax.set_xticklabels([-0.2, 0, 0.3]) # LG
    ax.set_yticks(range(-2,4)) # 4
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #legend
    normalLine = mlines.Line2D([], [], color='k', label='Stop', lw=3)
    dashedLine = mlines.Line2D([], [], color='k', label='slow Go', dash_capstyle='round', dashes=(0.05,2), lw=3)
    redLine = mlines.Line2D([], [], color=toRGB['red'], label='StrD1', lw=3)
    cyanLine = mlines.Line2D([], [], color=toRGB['cyan'], label='GPe-Arky', lw=3)
    blueLine = mlines.Line2D([], [], color=toRGB['blue'], label='GPe-Proto', lw=3)
    plt.legend(handles=[dashedLine, normalLine, redLine, cyanLine, blueLine], bbox_to_anchor=(-0.02,1.015), bbox_transform=plt.gca().transAxes, fontsize=10, loc='upper left')
    #save
    plt.tight_layout()
    plt.savefig(saveFolderPlots+'zscore_StopSTR_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.png', dpi=300)



    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()
    rate_plot(STN_mean_Stop, STN_sem_Stop, 'STN', trials, 'orange', plot_sem = False, pltlabel='STN Stop')     
    rate_plot(SNr_mean_Stop, SNr_sem_Stop, 'SNr', trials, 'tomato', plot_sem = False, pltlabel='SNr Stop')     
    plt.plot(GPe_Arky_mean_Stop, color='cyan', lw=3) 
    #ax.fill_between(range(len(GPe_Arky_mean_Stop)), GPe_Arky_mean_Stop - GPe_Arky_sem_Stop, GPe_Arky_mean_Stop + GPe_Arky_sem_Stop, color='cyan', alpha=0.4, edgecolor='None')
    plt.plot(GPe_Proto_mean_Stop, color='blue', lw=3) 
    #ax.fill_between(range(len(GPe_Proto_mean_Stop)), GPe_Proto_mean_Stop - GPe_Proto_sem_Stop, GPe_Proto_mean_Stop + GPe_Proto_sem_Stop, color='blue', alpha=0.4, edgecolor='None')  
    plt.axvline((t_init + t_SSD)/dt(), color='grey', lw=0.5)
    fsize = 15 # 10
    plt.ylabel('Firing rate [spk/s]', fontsize=fsize)
    #ax.axis([t_init + t_SSD - 50, t_init + t_SSD + 150, max(0, ax.axis()[2]), 350])
    ax.axis([(t_init + t_SSD - 50)/dt(), (t_init + t_SSD + 250)/dt(), max(0, ax.axis()[2]), 100])    
    ax.set_xticks([(t_init + t_SSD - 50)/dt(), (t_init + t_SSD)/dt(), (t_init + t_SSD + 50)/dt(), (t_init + t_SSD + 100)/dt()])
    ax.set_xticklabels([-0.05, 0, 0.05, 0.1])  
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.legend(('STN-Stop', 'SNr-Stop', 'Arky-Stop', 'Proto-Stop'), fontsize=10) # , 'Stop cue'
    plt.tight_layout()
    plt.savefig(saveFolderPlots+'mean_rate_StopDetail_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.png')
    #plt.show()

    plt.figure(figsize=(3.5,4), dpi=300)
    #fsize = 6
    ax=plt.gca()
    #plt.plot(custom_zscore(STN_mean_Stop, STN_mean_Stop), color='orange', lw=3) # Old - not aligned to zero 
    plt.plot(custom_zscore_start0(STN_mean_CorrectStop, (t_init + t_SSD - 50)/dt(), STN_mean_CorrectStop), color=toRGB['gold'], lw=3) 
    plt.plot(custom_zscore_start0(SNr_mean_CorrectStop, (t_init + t_SSD - 50)/dt(), SNr_mean_CorrectStop), color=toRGB['orange'], lw=3) 
    plt.plot(custom_zscore_start0(GPe_Arky_mean_CorrectStop, (t_init + t_SSD - 50)/dt(), GPe_Arky_mean_CorrectStop), color=toRGB['cyan'], lw=3) 
    plt.plot(custom_zscore_start0(GPe_Proto_mean_CorrectStop, (t_init + t_SSD - 50)/dt(), GPe_Proto_mean_CorrectStop), color=toRGB['blue'], lw=3) 
    plt.plot((t_init + t_SSD)/dt() * np.ones(2), [plt.axis()[2]+0.1, plt.axis()[3]], color='grey', lw=0.5) # 'k--', lw=1.5)
    plt.ylabel('$\Delta$Firing rate (z-score)', fontsize=fsize)
    ax.axis([(t_init + t_SSD - 50)/dt(), (t_init + t_SSD + 200)/dt(), -2.2, 3.6])
    #ax.set_xticks([(t_init + t_SSD - 50)/dt(), (t_init + t_SSD)/dt(), (t_init + t_SSD + 50)/dt(), (t_init + t_SSD + 100)/dt(), (t_init + t_SSD + 150)/dt()])
    #ax.set_xticklabels([-0.05, 0, 0.05, 0.1, 0.15])
    xtickList=[0,0.1,0.2]  
    ax.set_xticks([(t_init + t_SSD + i*1000)/dt() for i in xtickList])
    ax.set_xticklabels(xtickList)
    ax.set_yticks(range(-2,4))
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.legend(('STN Stop', 'SNr Stop', 'GPe-Arky Stop', 'GPe-Proto Stop'), bbox_to_anchor=(-0.02,-0.015), bbox_transform=plt.gca().transAxes, fontsize=10, loc='lower left') # , 'Stop cue'
    plt.tight_layout()
    plt.savefig(saveFolderPlots+'zscore_StopDetail_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.png', dpi=300)
    #plt.show()


    plt.figure(figsize=(3.5, 4), dpi=300)
    ax=plt.gca()
    rate_plot(STN_mean_Go, STN_sem_Go, 'STN', trials, 'orange', plot_sem = False, pltlabel='STN Go')
    rate_plot(SNr_mean_Go, SNr_sem_Go, 'SNr', trials, 'tomato', plot_sem = False, pltlabel='SNr Go')         
    rate_plot(GPe_Arky_mean_Go, GPe_Arky_sem_Go, 'GPe_Arky', trials, 'cyan', plot_sem = False, pltlabel='Arky Go')         
    rate_plot(GPe_Proto_mean_Go, GPe_Proto_sem_Go, 'GPe_Proto', trials, 'blue', plot_sem = False, pltlabel='Proto Go')    
    plt.close()

    #np.save('plots/plotdata/STR_D1_Stop_mean_id'+str(int(param_id))+'.npy', STR_D1_Stop_mean)

    plot_zscore_stopVsGo(STN_mean_Stop, STN_mean_Go, SNr_mean_Stop, SNr_mean_Go, GPe_Arky_mean_Stop, GPe_Arky_mean_Go, GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, param_id, trials, dt(), saveFolderPlots)
    plot_zscore_stopVsGo_five(STR_D1_Stop_mean, STR_D1_Go_mean, STR_D2_Stop_mean, STR_D2_Go_mean, STR_FSI_Stop_mean, STR_FSI_Go_mean, GPe_Proto2_mean_Stop, GPe_Proto2_mean_Go, Thal_mean_Stop, Thal_mean_Go, t_init, t_SSD, param_id, trials, dt(), saveFolderPlots, \
                              labels=['StrD1', 'StrD2', 'StrFSI', 'GPe Cortex-projecting', 'Thalamus'], linecol=[['red', 'green'], ['purple', 'pink'], ['black','grey'], ['brown', 'olive'], ['lime','teal']])

    ### same plot (Figure 5) with different data
    #load
    SD1_mean_FailedStop, SD1_std_FailedStop, SD1_mean_CorrectStop, SD1_std_CorrectStop = np.load('data/SD1_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    SD2_mean_FailedStop, SD2_std_FailedStop, SD2_mean_CorrectStop, SD2_std_CorrectStop = np.load('data/SD2_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    
    FSI_mean_FailedStop, FSI_std_FailedStop, FSI_mean_CorrectStop, FSI_std_CorrectStop = np.load('data/FSI_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')        
    STN_mean_FailedStop, STN_std_FailedStop, STN_mean_CorrectStop, STN_std_CorrectStop = np.load('data/STN_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    GPe_Proto_mean_FailedStop, GPe_Proto_std_FailedStop, GPe_Proto_mean_CorrectStop, GPe_Proto_std_CorrectStop = np.load('data/Proto_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    GPe_Proto2_mean_FailedStop, GPe_Proto2_std_FailedStop, GPe_Proto2_mean_CorrectStop, GPe_Proto2_std_CorrectStop = np.load('data/Proto2_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    
    GPe_Arky_mean_FailedStop, GPe_Arky_std_FailedStop, GPe_Arky_mean_CorrectStop, GPe_Arky_std_CorrectStop = np.load('data/Arky_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')        
    SNr_mean_FailedStop, SNr_std_FailedStop, SNr_mean_CorrectStop, SNr_std_CorrectStop = np.load('data/SNr_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    Thal_mean_FailedStop, Thal_std_FailedStop, Thal_mean_CorrectStop, Thal_std_CorrectStop = np.load('data/Thal_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    #plot
    plot_zscore_stopVsGo_NewData(STN_mean_CorrectStop, STN_mean_CorrectStop, SNr_mean_CorrectStop, SNr_mean_CorrectStop, GPe_Arky_mean_CorrectStop, GPe_Arky_mean_CorrectStop, GPe_Proto_mean_CorrectStop, GPe_Proto_mean_CorrectStop, t_init, t_SSD, param_id, trials, dt(), saveFolderPlots, labels = ['STN', 'SNr', 'GPe-Arky', 'GPe-Proto'], linecol = [[toRGB['gold'], toRGB['gold']*0.7], [toRGB['orange'], toRGB['orange']*0.7], [toRGB['cyan'],toRGB['cyan']*0.7], [toRGB['blue'],toRGB['blue']*0.7]])
    plot_zscore_stopVsGo_five_NewData(SD1_mean_CorrectStop, SD1_mean_CorrectStop, SD2_mean_CorrectStop, SD2_mean_CorrectStop, FSI_mean_CorrectStop, FSI_mean_CorrectStop, GPe_Proto2_mean_CorrectStop, GPe_Proto2_mean_CorrectStop, Thal_mean_CorrectStop, Thal_mean_CorrectStop, t_init, t_SSD, param_id, trials, dt(), saveFolderPlots, labels=['StrD1', 'StrD2', 'StrFSI', 'GPe-Cp', 'Thalamus'], linecol=[[toRGB['red'], toRGB['red']*0.7], [toRGB['purple'], toRGB['purple']*0.7], [toRGB['grey'], toRGB['grey']*0.7], [toRGB['light brown'], toRGB['light brown']*0.7], [toRGB['lime'], toRGB['lime']*0.7]])                             


##############################################################################################
if abfrage_loop == 'extra': 
    plt.ion()
    param_id = params['general_id']
    t_init = params['t_init'] # 100
    t_SSD = params['t_SSD'] # 170
    t_stopCue = int(t_init + t_SSD)    
    fsize = 6    

    #'''#
    plt.figure(figsize=(3.5,4), dpi=300)
    #mInt_stop = np.load('data/Integrator_ampa_Stop_id'+str(int(param_id))+'.npy')
    mInt_stop = np.load('data/Integrator_ampa_Stop_'+str(i_netw_rep)+'_id'+str(int(params['general_id']))+'.npy')
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(trials)), trials], order='F') # order seems correct
    mInt_maxpertrial = np.nanmax(rsp_mInt_stop, 0)
    nz_FailedStop = np.nonzero(mInt_maxpertrial >= Integrator.threshold)[0] # This seems correct
    nz_CorrectStop = np.nonzero(mInt_maxpertrial < Integrator.threshold)[0]

    SD1_mean_FailedStop, SD1_std_FailedStop, SD1_mean_CorrectStop, SD1_std_CorrectStop = np.load('data/SD1_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    SD2_mean_FailedStop, SD2_std_FailedStop, SD2_mean_CorrectStop, SD2_std_CorrectStop = np.load('data/SD2_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    
    FSI_mean_FailedStop, FSI_std_FailedStop, FSI_mean_CorrectStop, FSI_std_CorrectStop = np.load('data/FSI_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')        
    STN_mean_FailedStop, STN_std_FailedStop, STN_mean_CorrectStop, STN_std_CorrectStop = np.load('data/STN_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    GPe_Proto_mean_FailedStop, GPe_Proto_std_FailedStop, GPe_Proto_mean_CorrectStop, GPe_Proto_std_CorrectStop = np.load('data/Proto_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    
    GPe_Proto2_mean_FailedStop, GPe_Proto2_std_FailedStop, GPe_Proto2_mean_CorrectStop, GPe_Proto2_std_CorrectStop = np.load('data/Proto2_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')        
    GPe_Arky_mean_FailedStop, GPe_Arky_std_FailedStop, GPe_Arky_mean_CorrectStop, GPe_Arky_std_CorrectStop = np.load('data/Arky_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')        
    SNr_mean_FailedStop, SNr_std_FailedStop, SNr_mean_CorrectStop, SNr_std_CorrectStop = np.load('data/SNr_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    Thal_mean_FailedStop, Thal_std_FailedStop, Thal_mean_CorrectStop, Thal_std_CorrectStop = np.load('data/Thal_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    
    SNrE_mean_FailedStop, SNrE_std_FailedStop, SNrE_mean_CorrectStop, SNrE_std_CorrectStop = np.load('data/SNrE_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')
    Cortex_G_mean_FailedStop, Cortex_G_std_FailedStop, Cortex_G_mean_CorrectStop, Cortex_G_std_CorrectStop = np.load('data/Cortex_G_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    
    Cortex_S_mean_FailedStop, Cortex_S_std_FailedStop, Cortex_S_mean_CorrectStop, Cortex_S_std_CorrectStop = np.load('data/Cortex_S_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    
    PauseInput_mean_FailedStop, PauseInput_std_FailedStop, PauseInput_mean_CorrectStop, PauseInput_std_CorrectStop = np.load('data/Stoppinput1_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy')    

    #t_simend = int( (840 + (t_init-100)) / dt() ) # 

    #pvalue_list = []
    #pvalue_times = []    
    pvalue_list, pvalue_times = np.load('data/p_value_list_times_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy', allow_pickle=True)

    plt.close()

    plt.figure()
    Cortex_G_mean_Stop, Cortex_G_std_Stop =  np.load('data/Cortex_G_rate_Stop_mean_std'+str(i_netw_rep)+'.npy')
    plt.subplot(111)
    plt.plot(Cortex_G_mean_Stop,label='1')
    plt.plot(Cortex_G_mean_CorrectStop,label='2')
    plt.plot(Cortex_G_mean_FailedStop,label='3')
    plt.legend()
    plt.savefig('TESTFIGUREextra.png')


    tmin = t_init + t_SSD # Time of Stop cue presentation
    tmax = t_init + t_SSD + 100 # 200
    STR_D1_ratepertrial_Stop = np.load('data/SD1_rate_Stop_tempmean_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy')
    STR_D2_ratepertrial_Stop = np.load('data/SD2_rate_Stop_tempmean_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy')
    STN_ratepertrial_Stop = np.load('data/STN_rate_Stop_tempmean_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy')
    GPe_Proto_ratepertrial_Stop = np.load('data/GPeProto_rate_Stop_tempmean_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy')        
    GPe_Arky_ratepertrial_Stop = np.load('data/GPeArky_rate_Stop_tempmean_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy')        
    SNr_ratepertrial_Stop = np.load('data/SNr_rate_Stop_tempmean_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy')
    Thal_ratepertrial_Stop = np.load('data/Thal_rate_Stop_tempmean_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy')
    Cortex_G_ratepertrial_Stop = np.load('data/Cortex_G_rate_Stop_tempmean_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy')        
    Cortex_S_ratepertrial_Stop = np.load('data/Cortex_S_rate_Stop_tempmean_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy')                

    #plot_correl_rates_Intmax(GPe_Arky_ratepertrial_Stop, GPe_Proto_ratepertrial_Stop, STR_D1_ratepertrial_Stop,   STR_D2_ratepertrial_Stop,   STN_ratepertrial_Stop, \
    #                         SNr_ratepertrial_Stop,      Thal_ratepertrial_Stop,      Cortex_G_ratepertrial_Stop, Cortex_S_ratepertrial_Stop, mInt_maxpertrial, param_id, trials)



    #'''#
    plot_meanrate_All_FailedVsCorrectStop(saveFolderPlots, \
                                          GPe_Arky_mean_FailedStop, GPe_Arky_mean_CorrectStop, GPe_Arky_std_FailedStop, GPe_Arky_std_CorrectStop, \
                                          GPe_Proto_mean_FailedStop, GPe_Proto_mean_CorrectStop, \
                                          SD1_mean_FailedStop, SD1_mean_CorrectStop, \
                                          SD2_mean_FailedStop, SD2_mean_CorrectStop, \
                                          STN_mean_FailedStop, STN_mean_CorrectStop, \
                                          SNr_mean_FailedStop, SNr_mean_CorrectStop, \
                                          Thal_mean_FailedStop, Thal_mean_CorrectStop, \
                                          rsp_mInt_stop, nz_FailedStop, nz_CorrectStop, \
                                          Integrator.threshold, \
                                          Cortex_G_mean_FailedStop, Cortex_G_mean_CorrectStop, \
                                          GPe_Proto2_mean_FailedStop, GPe_Proto2_mean_CorrectStop, \
                                          PauseInput_mean_FailedStop, PauseInput_mean_CorrectStop, \
                                          Cortex_S_mean_FailedStop, Cortex_S_mean_CorrectStop, \
                                          SNrE_mean_FailedStop, SNrE_mean_CorrectStop, \
                                          FSI_mean_FailedStop, FSI_mean_CorrectStop, \
                                          t_init, t_SSD, param_id, trials, dt(), pvalue_list, pvalue_times) # FSI_mean
    #'''

    ### load all correct go, slow go and fast go data for all nuclei
    # all Go
    GPe_Arky_mean_Go,   GPe_Arky_std_Go     = np.load('data/GPeArky_rate_Go_mean_std'+str(i_netw_rep)+'.npy')
    GPe_Proto_mean_Go,  GPe_Proto_std_Go    = np.load('data/GPeProto_rate_Go_mean_std'+str(i_netw_rep)+'.npy')  
    STR_D1_mean_Go,     STR_D1_std_Go       = np.load('data/SD1_rate_Go_mean_std'+str(i_netw_rep)+'.npy')     
    STR_D2_mean_Go,     STR_D2_std_Go       = np.load('data/SD2_rate_Go_mean_std'+str(i_netw_rep)+'.npy')        
    STN_mean_Go,        STN_std_Go          = np.load('data/STN_rate_Go_mean_std'+str(i_netw_rep)+'.npy')
    SNr_mean_Go,        SNr_std_Go          = np.load('data/SNr_rate_Go_mean_std'+str(i_netw_rep)+'.npy')            
    Thal_mean_Go,       Thal_std_Go         = np.load('data/Thal_rate_Go_mean_std'+str(i_netw_rep)+'.npy')
    Cortex_G_mean_Go,   Cortex_G_std_Go     = np.load('data/Cortex_G_rate_Go_mean_std'+str(i_netw_rep)+'.npy')
    GPe_Proto2_mean_Go, GPe_Proto2_std_Go   = np.load('data/GPeProto2_rate_Go_mean_std'+str(i_netw_rep)+'.npy')
    PauseInput_mean_Go, PauseInput_std_Go   = np.load('data/Stoppinput1_rate_Go_mean_std'+str(i_netw_rep)+'.npy')
    Cortex_S_mean_Go,   Cortex_S_std_Go     = np.load('data/Cortex_S_rate_Go_mean_std'+str(i_netw_rep)+'.npy')
    STR_FSI_mean_Go,    STR_FSI_std_Go      = np.load('data/FSI_rate_Go_mean_std'+str(i_netw_rep)+'.npy')            

    # fast and slow Go
    GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo     = np.load('data/GPe_Arky_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo    = np.load('data/GPe_Proto_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    STR_D1_meanrate_FastGo,     STR_D1_std_FastGo,      STR_D1_meanrate_SlowGo,     STR_D1_std_SlowGo       = np.load('data/SD1_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    STR_D2_meanrate_FastGo,     STR_D2_std_FastGo,      STR_D2_meanrate_SlowGo,     STR_D2_std_SlowGo       = np.load('data/SD2_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')            
    STN_meanrate_FastGo,        STN_std_FastGo,         STN_meanrate_SlowGo,        STN_std_SlowGo          = np.load('data/STN_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    SNr_meanrate_FastGo,        SNr_std_FastGo,         SNr_meanrate_SlowGo,        SNr_std_SlowGo          = np.load('data/SNr_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    Thal_meanrate_FastGo,       Thal_std_FastGo,        Thal_meanrate_SlowGo,       Thal_std_SlowGo         = np.load('data/Thal_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    Cortex_G_meanrate_FastGo,   Cortex_G_std_FastGo,    Cortex_G_meanrate_SlowGo,   Cortex_G_std_SlowGo     = np.load('data/Cortex_G_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    GPe_Proto2_meanrate_FastGo, GPe_Proto2_std_FastGo,  GPe_Proto2_meanrate_SlowGo, GPe_Proto2_std_SlowGo   = np.load('data/GPe_Proto2_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    PauseInput_meanrate_FastGo, PauseInput_std_FastGo,  PauseInput_meanrate_SlowGo, PauseInput_std_SlowGo   = np.load('data/PauseInput_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')
    Cortex_S_meanrate_FastGo,   Cortex_S_std_FastGo,    Cortex_S_meanrate_SlowGo,   Cortex_S_std_SlowGo     = np.load('data/Cortex_S_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy')                        
    STR_FSI_meanrate_FastGo,    STR_FSI_std_FastGo,     STR_FSI_meanrate_SlowGo,    STR_FSI_std_SlowGo      = np.load('data/FSI_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy') 

    # load further things
    mInt_Go = np.load('data/Integrator_ampa_Go_'+str(i_netw_rep)+'_id'+str(int(params['general_id']))+'.npy')
    pvalue_list = {}
    nameList = ['failedStop_vs_allGo', 'failedStop_vs_fastGo', 'failedStop_vs_slowGo']
    for name in nameList:
        pvalue_list[name], pvalue_times = np.load('data/p_value_list_'+name+'_times_'+str(int(params['general_id']))+str(i_netw_rep)+'.npy', allow_pickle=True)

    plot_meanrate_All_FailedStopVsCorrectGo(saveFolderPlots, \
                                          GPe_Arky_mean_FailedStop,     GPe_Arky_mean_Go,   GPe_Arky_meanrate_FastGo,   GPe_Arky_meanrate_SlowGo,  \
                                          GPe_Proto_mean_FailedStop,    GPe_Proto_mean_Go,  GPe_Proto_meanrate_FastGo,  GPe_Proto_meanrate_SlowGo,  \
                                          SD1_mean_FailedStop,          STR_D1_mean_Go,     STR_D1_meanrate_FastGo,     STR_D1_meanrate_SlowGo,  \
                                          SD2_mean_FailedStop,          STR_D2_mean_Go,     STR_D2_meanrate_FastGo,     STR_D2_meanrate_SlowGo,  \
                                          STN_mean_FailedStop,          STN_mean_Go,        STN_meanrate_FastGo,        STN_meanrate_SlowGo,  \
                                          SNr_mean_FailedStop,          SNr_mean_Go,        SNr_meanrate_FastGo,        SNr_meanrate_SlowGo,  \
                                          Thal_mean_FailedStop,         Thal_mean_Go,       Thal_meanrate_FastGo,       Thal_meanrate_SlowGo,  \
                                          Cortex_G_mean_FailedStop,     Cortex_G_mean_Go,   Cortex_G_meanrate_FastGo,   Cortex_G_meanrate_SlowGo,  \
                                          GPe_Proto2_mean_FailedStop,   GPe_Proto2_mean_Go, GPe_Proto2_meanrate_FastGo, GPe_Proto2_meanrate_SlowGo,  \
                                          Cortex_S_mean_FailedStop,     Cortex_S_mean_Go,   Cortex_S_meanrate_FastGo,   Cortex_S_meanrate_SlowGo,  \
                                          FSI_mean_FailedStop,          STR_FSI_mean_Go,    STR_FSI_meanrate_FastGo,    STR_FSI_meanrate_SlowGo,  \
                                          nz_FailedStop, nz_CorrectStop, \
                                          mInt_Go, mInt_stop, \
                                          Integrator.threshold, \
                                          t_init, t_SSD, param_id, trials, dt(), pvalue_list, pvalue_times, GO_Mode='fast')


    #rateperneuron_GPeArky_allStoptrials = np.load('data/GPeArky_rateperneuron_allStoptrials_'+str(int(params['general_id']))+'.npy')
    STN_poprate_Stop_alltrials = np.load('data/STN_rate_Stop_alltrials'+str(i_netw_rep)+'.npy')
    SNr_poprate_Stop_alltrials = np.load('data/SNr_rate_Stop_alltrials'+str(i_netw_rep)+'.npy')
    GPe_Proto_poprate_Stop_alltrials = np.load('data/GPeProto_rate_Stop_alltrials'+str(i_netw_rep)+'.npy')
    GPe_Arky_poprate_Stop_alltrials = np.load('data/GPeArky_rate_Stop_alltrials'+str(i_netw_rep)+'.npy')        
    #get_peak_response_time(STN_poprate_Stop_alltrials, SNr_poprate_Stop_alltrials, GPe_Proto_poprate_Stop_alltrials, GPe_Arky_poprate_Stop_alltrials, t_init + t_SSD, t_init + t_SSD + 200, dt(), param_id, trials, param_id, trials, paramname)    


##############################################################################################
if abfrage_loop=='RT':
    print("Calculating reaction times...")                  

    param_id = params['general_id']
    t_init = params['t_init'] # 100
    i_cycle=0

    #'''#
    spike_times, ranks = np.load('./data/Integrator_spike_Go'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'.npy')
    spike_times_stop, ranks_stop = np.load('./data/Integrator_spike_Stop'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'.npy')
    mInt_stop = np.load('data/Integrator_ampa_Stop_'+str(i_netw_rep)+'_id'+str(int(params['general_id']))+'.npy')
    mInt_go = np.load('data/Integrator_ampa_Go_'+str(i_netw_rep)+'_id'+str(int(params['general_id']))+'.npy')            
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(trials)), trials], order='F')
    rsp_mInt_go = np.reshape(mInt_go, [int(mInt_go.shape[0] / float(trials)), trials], order='F')
    mInt_maxpertrial = np.nanmax(rsp_mInt_go, 0)
    nz_FailedGo = np.nonzero(mInt_maxpertrial < Integrator.threshold)[0]
    nz_CorrectGo = np.nonzero(mInt_maxpertrial >= Integrator.threshold)[0]
    mInt_maxpertrial = np.nanmax(rsp_mInt_stop, 0)
    nz_FailedStop = np.nonzero(mInt_maxpertrial >= Integrator.threshold)[0]
    nz_CorrectStop = np.nonzero(mInt_maxpertrial < Integrator.threshold)[0]

    RT_Go = np.nan * np.ones(trials)
    RT_Stop = np.nan * np.ones(trials)
    for i_trial in range(trials):
        if np.nanmax(rsp_mInt_go[:, i_trial]) >= Integrator.threshold: 
            RT_Go[i_trial] = np.nonzero(rsp_mInt_go[:, i_trial] >= Integrator.threshold)[0][0]
            #RT_Go[i_trial] = np.nonzero(rsp_mInt_go[:, i_trial] >= Integrator.threshold)[0][0] * dt()            
        if np.nanmax(rsp_mInt_stop[:, i_trial]) >= Integrator.threshold: 
            RT_Stop[i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= Integrator.threshold)[0][0] # Correct
            #RT_Stop[i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= Integrator.threshold)[0][0] * dt() # Correct                

    #plt.ion()
    plt.figure(figsize=(3.5,4), dpi=300)
    fsize = 12 # 15
    nz_Go = np.nonzero(np.isnan(RT_Go)==False)
    nz_Stop = np.nonzero(np.isnan(RT_Stop)==False)
    binSize=15
    counts_Go, bins_Go = np.histogram(RT_Go[nz_Go] * dt(), np.arange(t_init,t_init+501,binSize)) #
    counts_Stop, bins_Stop = np.histogram(RT_Stop[nz_Stop] * dt(), np.arange(t_init,t_init+501,binSize)) # 
    if counts_Go.max() > 0:
        #plt.bar(bins_Go[:-1], np.array(counts_Go) * (1.0/counts_Go.max()), width=np.diff(bins_Go)[0], alpha=0.5, color='b') #
        plt.bar(bins_Go[:-1], counts_Go/counts_Go.max(), width=binSize, alpha=1, color=np.array([107,139,164])/255.) #
    if counts_Stop.max() > 0: 
        #plt.bar(bins_Stop[:-1], np.array(counts_Stop) * (1.0/counts_Stop.max()), width=np.diff(bins_Stop)[0], alpha=0.5, color='g') #
        plt.bar(bins_Stop[:-1], counts_Stop/counts_Stop.max(), width=binSize, alpha=0.5, color='tomato') #
    mean_CorrectGo = np.round( (np.nanmean(RT_Go[nz_Go]) - t_init/dt())*dt(), 1)
    mean_FailedStop = np.round( (np.nanmean(RT_Stop[nz_Stop]) - t_init/dt())*dt(), 1)
    plt.legend(('correct Go, mean RT='+str(mean_CorrectGo),'failed Stop, mean RT='+str(mean_FailedStop)), fontsize=fsize-2)
    plt.xlabel('Reaction time [ms]', fontsize=fsize)    
    plt.ylabel('Normalized trial count', fontsize=fsize)
    ax = plt.gca()
    ax.set_yticks([0.0,0.5,1.0])
    #ax.axis([t_init, 400 + t_init, ax.axis()[2], ax.axis()[3]])
    ax.axis([t_init, (500 + t_init), ax.axis()[2], 1.22])                
    #ax.axis([t_init/dt(), (500 + t_init)/dt(), ax.axis()[2], ax.axis()[3]])
    xtks = ax.get_xticks()
    xlabels = []
    for i in range(len(xtks)):
        xlabels.append(str( int( xtks[i]-t_init ) ))
    ax.set_xticklabels(xlabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.tight_layout()
    #plt.ioff()
    plt.savefig(saveFolderPlots+'Reaction_times_paramsid'+str(int(param_id))+'.png')
    #plt.close()
    #plt.show()
    #'''

    results_RT = {}
    results_RT['nFailedGoTrials'] = len(nz_FailedGo)
    results_RT['nCorrectGoTrials'] = len(nz_CorrectGo)
    results_RT['nFailedStopTrials'] = len(nz_FailedStop)
    results_RT['nCorrectStopTrials'] = len(nz_CorrectStop)
    results_RT['meanRT_CorrectGo'] = mean_CorrectGo
    results_RT['meanRT_FailedStop'] = mean_FailedStop

    #np.save('data/resultsRT_id'+str(int(param_id))+'.npy', results_RT)
    #np.save('data/resultsRT_'+str(i_netw_rep)+'id'+str(int(param_id))+'.npy', results_RT) # 
    #np.save('data/resultsRT_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', results_RT) 
    params_loaded = np.load('data/paramset_id'+str(int(param_id))+str(i_netw_rep)+'.npy', allow_pickle=True)
    print("params_loaded = ", params)


    t_init = params['t_init'] #100
    t_SSD = params['t_SSD'] # 170 # params_loaded['t_SSD'] # 170
    thresh = Integrator.threshold
    #plt.ion()
    plt.figure(figsize=(3.5,4), dpi=300)
    plt.plot(rsp_mInt_go[:, :], color=np.array([107,139,164])/255., lw=1, label='correct Go')
    if len(nz_FailedStop) > 0:
        plt.plot(rsp_mInt_stop[:, nz_FailedStop], color='tomato', lw=1, label = 'failed Stop', linestyle='--')
    if len(nz_CorrectStop) > 0:
        plt.plot(rsp_mInt_stop[:, nz_CorrectStop], color='purple', lw=1, label = 'correct Stop')
    #gocue
    plt.axvline(t_init / dt(), color='green', linestyle=':')
    GoCueHandle = mlines.Line2D([], [], color='green', label='Go cue', linestyle=':')
    #stopcue
    plt.axvline((t_init + t_SSD) / dt(), color='orange', linestyle=':')
    StopCueHandle = mlines.Line2D([], [], color='orange', label='Stop cue', linestyle=':')
    #threshold
    plt.axhline(thresh, color='grey', linestyle=':')
    ThresholdHandle = mlines.Line2D([], [], color='grey', label='Threshold', linestyle=':')
    #other handles
    CorrectGoHandle = mlines.Line2D([], [], color=np.array([107,139,164])/255., label='correct Go', linestyle='-')
    FailedStopHandle = mlines.Line2D([], [], color='tomato', label='failed Stop', linestyle='--')
    CorrectStopHandle = mlines.Line2D([], [], color='purple', label='correct Stop', linestyle='-')
    plt.legend(handles = [ThresholdHandle, GoCueHandle, StopCueHandle, CorrectGoHandle, FailedStopHandle, CorrectStopHandle], loc='upper left', fontsize=fsize-2)
    plt.title(str(100 * len(nz_CorrectStop) / (len(nz_CorrectStop) + len(nz_FailedStop)) )+'% correct Stop trials' , fontsize=fsize) #fsize)
    with open(saveFolderPlots+'stops_and_gos_'+str(trials)+'trials_paramsid'+str(int(param_id))+'.txt', 'w') as f:
        mInt_Go = np.load('data/Integrator_ampa_Go_'+str(i_netw_rep)+'_id'+str(int(params['general_id']))+'.npy')
        mInt_Stop = np.load('data/Integrator_ampa_Stop_'+str(i_netw_rep)+'_id'+str(int(params['general_id']))+'.npy')
        nz_FastGo, nz_SlowGo = get_fast_slow_go_trials(mInt_Go, mInt_Stop, Integrator.threshold, trials)
        print('correct Stops:',len(nz_CorrectStop),file=f)
        print('failed Stops:',len(nz_FailedStop),file=f)
        print('correct Gos:',len(nz_CorrectGo),file=f)
        print('failed Gos:',len(nz_FailedGo),file=f)
        print('slow Gos:',len(nz_SlowGo),file=f)
        print('fast Gos:',len(nz_FastGo),file=f)
    plt.xlabel('Time from Go cue [ms]', fontsize=fsize)    
    plt.ylabel('Integrator value', fontsize=fsize) # ampa
    ax = plt.gca()
    ax.set_yticks([0.0,0.1,0.2,0.3,0.4])
    ax.axis([(t_init-200)/dt(), (t_init+600)/dt(), 0, ax.axis()[3]])        
    ax.set_xticks([t_init/dt(), (t_init+200)/dt(), (t_init+400)/dt(), (t_init+600)/dt()])
    xtks = ax.get_xticks()
    xlabels = []
    for i in range(len(xtks)):
        xlabels.append(str(int(dt()*xtks[i]-t_init)))
    ax.set_xticklabels(xlabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.tight_layout()
    #plt.ioff()
    plt.savefig(saveFolderPlots+'Integrator_ampa_Stop_'+str(trials)+'trials_paramsid'+str(int(param_id))+'.png')
    #plt.close()
    #plt.show()
