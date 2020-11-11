"""
Created on Fri Oct 16 2020

@authors: Oliver Maith, Lorenz Goenner, ilko
"""

#from ANNarchy import*
import pylab as plt
#import random
import os
import sys
import numpy as np
#import math
#import matplotlib.pyplot as plt
#import matplotlib.patches as patch
#import scipy.stats as st
#from scipy import stats
#from scipy.signal import argrelmin

from BGmodelSST.analysis import custom_zscore, custom_zscore_start0, \
    plot_zscore_stopVsGo, plot_zscore_stopVsGo_five, plot_zscore_stopVsGo_NewData, plot_zscore_stopVsGo_five_NewData, plot_correl_rates_Intmax, plot_meanrate_All_FailedVsCorrectStop, plot_meanrate_All_FailedStopVsCorrectGo, calc_KW_stats_all, rate_plot, \
    calc_meanrate_std_failed_correct, get_rates_failed_correct, get_poprate_aligned_onset, get_peak_response_time, custom_poprate, calc_meanrate_std_Fast_Slow, get_fast_slow_go_trials, get_rates_allGo_fastGo_slowGo
#from BGmodelSST.plotting import get_and_plot_syn_mon    
#from BGmodelSST.sim_params import params
#from BGmodelSST.init import init_neuronmodels
import matplotlib.lines as mlines


"""from BGmodelSST.neuronmodels import Izhikevich_neuron, Izhikevich_STR_neuron,STR_FSI_neuron, Integrator_neuron, Poisson_neuron, FixedSynapse
from BGmodelSST.populations import Stoppinput1, Cortex_S, Cortex_G, STR_D1, STR_D2, STN, SNr, GPe_Proto, Thal, Integrator, IntegratorStop, GPeE, SNrE, STNE, STR_FSI, STRE, GPe_Arky, TestThalnoise, population_size, GPe_Proto2
from BGmodelSST.projections import Stoppinput1STN, Cortex_GSTR_D1, Cortex_GSTR_D2, Cortex_GSTR_FSI, Cortex_GThal, Cortex_SGPe_Arky, STR_D1SNr, STR_D2GPe_Proto, STNSNr, \
                                    STNGPe_Proto, GPe_ProtoSTN, GPe_ProtoSNr, SNrThal, ThalIntegrator, GPeEGPe_Proto, GPEGPe_Arky, SNrESNr, STNESTN, \
                                    STR_FSISTR_D1, STR_FSISTR_D2, STRESTR_D1, STRESTR_D2, GPe_ArkySTR_D1, GPe_ArkySTR_D2, TestThalnoiseThal,STRESTR_FSI, \
                                    STR_FSISTR_FSI, GPe_ArkySTR_FSI, GPe_ArkyGPe_Proto, GPe_ProtoGPe_Arky, STR_D2GPe_Arky, \
                                    GPe_ProtoSTR_FSI, STR_D1STR_D1, STR_D1STR_D2, STR_D2STR_D1, STR_D2STR_D2, Cortex_GSTR_D2, Cortex_SGPe_Proto, STNGPe_Arky, GPe_ArkyCortex_G, \
                                    ThalSD1, ThalSD2, ThalFSI, \
                                    GPe_ProtoGPe_Proto2, GPe_Proto2GPe_Proto, STR_D2GPe_Proto2, STR_D1GPe_Proto2, STNGPe_Proto2, Cortex_SGPe_Proto2, GPe_ArkyGPe_Proto2, GPe_Proto2STR_D1, GPe_Proto2STR_D2, GPe_Proto2STR_FSI, GPe_Proto2GPe_Arky, GPe_Proto2IntegratorStop, EProto1GPe_Proto, EProto2GPe_Proto2, EArkyGPe_Arky, \
                                    Cortex_SGPe_Arky2, STR_D2GPe_Arky2, GPe_ProtoGPe_Arky2, STNGPe_Arky2, GPe_Proto2GPe_Arky2, EArkyGPe_Arky2, GPe_Arky2STR_D1, GPe_Arky2STR_D2, GPe_Arky2STR_FSI
"""
if len(sys.argv) <= 1:
    print('Plot modus missing')
    quit()
else:
    abfrage_loop = str(sys.argv[1])


loadFolder = 'testSim'
netNr = '1'
params  = np.load('../data/'+loadFolder+'/paramset_id8007'+netNr+'.npy', allow_pickle=True)[0]
paramsS = np.load('../data/'+loadFolder+'/paramSset_id8007'+netNr+'.npy', allow_pickle=True)[0]

resultsDir='../results/'+loadFolder
try:
    os.makedirs(resultsDir)
except:
    if os.path.isdir(resultsDir):
        print(resultsDir+' already exists')
    else:
        print('could not create '+resultsDir+' folder')
        quit()

##############################################################################################
if abfrage_loop == 'STR':

    """#t_min = int((params['t_init'] + params['t_SSD'] - 600) / params['general_dt'])
    t_min = int((params['t_init'] + params['t_SSD'] - 200) / params['general_dt'])    
    t_max = int((params['t_init'] + params['t_SSD'] + 400) / params['general_dt'])"""  
    
    fsize = 15    

    ### LOAD STOP RATES
    Cortex_G_mean_Stop, Cortex_G_std_Stop     = np.load('../data/'+loadFolder+'/Cortex_G_rate_Stop_mean_std'+netNr+'.npy')
    Cortex_S_mean_Stop, Cortex_S_std_Stop     = np.load('../data/'+loadFolder+'/Cortex_S_rate_Stop_mean_std'+netNr+'.npy')
    STR_D1_Stop_mean, STR_D1_Stop_sem         = np.load('../data/'+loadFolder+'/SD1_rate_Stop_mean_std'+netNr+'.npy')
    STR_D2_Stop_mean, STR_D2_Stop_sem         = np.load('../data/'+loadFolder+'/SD2_rate_Stop_mean_std'+netNr+'.npy')
    GPe_Arky_mean_Stop, GPe_Arky_sem_Stop     = np.load('../data/'+loadFolder+'/GPeArky_rate_Stop_mean_std'+netNr+'.npy')
    GPe_Proto_mean_Stop, GPe_Proto_sem_Stop   = np.load('../data/'+loadFolder+'/GPeProto_rate_Stop_mean_std'+netNr+'.npy')
    GPe_Proto2_mean_Stop, GPe_Proto2_sem_Stop = np.load('../data/'+loadFolder+'/GPeProto2_rate_Stop_mean_std'+netNr+'.npy')
    STR_FSI_Stop_mean, STR_FSI_Stop_sem       = np.load('../data/'+loadFolder+'/FSI_rate_Stop_mean_std'+netNr+'.npy')
    STN_mean_Stop, STN_sem_Stop               = np.load('../data/'+loadFolder+'/STN_rate_Stop_mean_std'+netNr+'.npy')
    SNr_mean_Stop, SNr_sem_Stop               = np.load('../data/'+loadFolder+'/SNr_rate_Stop_mean_std'+netNr+'.npy')
    Thal_mean_Stop, Thal_sem_Stop             = np.load('../data/'+loadFolder+'/Thal_rate_Stop_mean_std'+netNr+'.npy')
    STR_D1_Stop_mean                          = STR_D1_Stop_mean[ : -100] # exclude artefact in last time steps?
    STR_D2_Stop_mean                          = STR_D2_Stop_mean[ : -100] # exclude artefact in last time steps?
    STR_FSI_Stop_mean                         = STR_FSI_Stop_mean[ : -100] # exclude artefact in last time steps?
    
    ### LOAD GO RATES
    STR_D1_Go_mean, STR_D1_Go_sem             = np.load('../data/'+loadFolder+'/SD1_rate_Go_mean_std'+netNr+'.npy')
    STR_D2_Go_mean, STR_D2_Go_sem             = np.load('../data/'+loadFolder+'/SD2_rate_Go_mean_std'+netNr+'.npy')
    GPe_Arky_mean_Go, GPe_Arky_sem_Go         = np.load('../data/'+loadFolder+'/GPeArky_rate_Go_mean_std'+netNr+'.npy')
    GPe_Proto_mean_Go, GPe_Proto_sem_Go       = np.load('../data/'+loadFolder+'/GPeProto_rate_Go_mean_std'+netNr+'.npy')
    GPe_Proto2_mean_Go, GPe_Proto2_sem_Go     = np.load('../data/'+loadFolder+'/GPeProto2_rate_Go_mean_std'+netNr+'.npy')
    STR_FSI_Go_mean, STR_FSI_Go_sem           = np.load('../data/'+loadFolder+'/FSI_rate_Go_mean_std'+netNr+'.npy')
    STN_mean_Go, STN_sem_Go                   = np.load('../data/'+loadFolder+'/STN_rate_Go_mean_std'+netNr+'.npy')
    SNr_mean_Go, SNr_sem_Go                   = np.load('../data/'+loadFolder+'/SNr_rate_Go_mean_std'+netNr+'.npy')
    Thal_mean_Go, Thal_sem_Go                 = np.load('../data/'+loadFolder+'/Thal_rate_Go_mean_std'+netNr+'.npy')
    STR_D1_Go_mean                            = STR_D1_Go_mean[ : -100]    
    STR_D2_Go_mean                            = STR_D2_Go_mean[ : -100]    
    STR_FSI_Go_mean                           = STR_FSI_Go_mean[ : -100]        

    ### LOAD FAILED/CORRECT STOP RATES
    SD1_mean_FailedStop,        SD1_std_FailedStop,        SD1_mean_CorrectStop,        SD1_std_CorrectStop        = np.load('../data/'+loadFolder+'/SD1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    SD2_mean_FailedStop,        SD2_std_FailedStop,        SD2_mean_CorrectStop,        SD2_std_CorrectStop        = np.load('../data/'+loadFolder+'/SD2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    FSI_mean_FailedStop,        FSI_std_FailedStop,        FSI_mean_CorrectStop,        FSI_std_CorrectStop        = np.load('../data/'+loadFolder+'/FSI_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    STN_mean_FailedStop,        STN_std_FailedStop,        STN_mean_CorrectStop,        STN_std_CorrectStop        = np.load('../data/'+loadFolder+'/STN_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    GPe_Proto_mean_FailedStop,  GPe_Proto_std_FailedStop,  GPe_Proto_mean_CorrectStop,  GPe_Proto_std_CorrectStop  = np.load('../data/'+loadFolder+'/Proto_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    GPe_Proto2_mean_FailedStop, GPe_Proto2_std_FailedStop, GPe_Proto2_mean_CorrectStop, GPe_Proto2_std_CorrectStop = np.load('../data/'+loadFolder+'/Proto2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    GPe_Arky_mean_FailedStop,   GPe_Arky_std_FailedStop,   GPe_Arky_mean_CorrectStop,   GPe_Arky_std_CorrectStop   = np.load('../data/'+loadFolder+'/Arky_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    SNr_mean_FailedStop,        SNr_std_FailedStop,        SNr_mean_CorrectStop,        SNr_std_CorrectStop        = np.load('../data/'+loadFolder+'/SNr_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    Thal_mean_FailedStop,       Thal_std_FailedStop,       Thal_mean_CorrectStop,       Thal_std_CorrectStop       = np.load('../data/'+loadFolder+'/Thal_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')

    ### LOAD FAST/SLOW GO RATES
    StrD1_meanrate_FastGo,      StrD1_std_FastGo,       StrD1_meanrate_SlowGo,      StrD1_std_SlowGo        = np.load('../data/'+loadFolder+'/SD1_meanrate_std_Fast-Slow_Go'+netNr+'.npy')            
    StrD2_meanrate_FastGo,      StrD2_std_FastGo,       StrD2_meanrate_SlowGo,      StrD2_std_SlowGo        = np.load('../data/'+loadFolder+'/SD2_meanrate_std_Fast-Slow_Go'+netNr+'.npy')                        
    StrFSI_meanrate_FastGo,     StrFSI_std_FastGo,      StrFSI_meanrate_SlowGo,     StrFSI_std_SlowGo       = np.load('../data/'+loadFolder+'/FSI_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo     = np.load('../data/'+loadFolder+'/GPe_Arky_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo    = np.load('../data/'+loadFolder+'/GPe_Proto_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
        

            
                
    
    

                 

    """plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()      
    rate_plot(STR_D1_Stop_mean, STR_D1_Stop_sem, 'STR_D1_Stop', paramsS['trials'],  'r', plot_sem = False, pltlabel='D1 Stop')     
    rate_plot(STR_D1_Go_mean, STR_D1_Go_sem, 'STR_D1_Go', paramsS['trials'], 'g', plot_sem = False, pltlabel='D1 Go')         
    rate_plot(GPe_Arky_mean_Stop, GPe_Arky_sem_Stop, 'GPe_Arky', paramsS['trials'], 'cyan', plot_sem = False, pltlabel='Arky Stop')         
    rate_plot(GPe_Proto_mean_Stop, GPe_Proto_sem_Stop, 'GPe_Proto', paramsS['trials'], 'blue', plot_sem = False, pltlabel='Proto Stop')    
    plt.plot((params['t_init'] + params['t_SSD'])/params['general_dt'] * np.ones(2), [0, plt.axis()[3]], 'k--', lw=1.5, label='Stop cue')
    ax.axis([t_min, t_max, max(0, ax.axis()[2]), 80]) # dt=0.1
    plt.xlabel('t [msec]')
    plt.ylabel('Firing rate [spk/s]', fontsize=fsize)
    ax.set_xticks([(params['t_init'] + params['t_SSD'] - 200)/params['general_dt'],(params['t_init'] + params['t_SSD'])/params['general_dt'], (params['t_init'] + params['t_SSD'] + 400)/params['general_dt']]) # LG
    ax.set_xticklabels([- 0.2, 0, 0.4]) # LG
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.legend(fontsize=6)    
    plt.tight_layout()    
    plt.savefig(resultsDir+'/mean_rate_paramsid'+str(params['general_id'])+'_'+str(paramsS['trials'])+'trials_STRD1_GPeStop.png') #             
    
    plt.figure(figsize=(3.5,4), dpi=300)
    rate_plot(STR_D2_Stop_mean, STR_D2_Stop_sem, 'STR_D2_Stop', paramsS['trials'], 'pink', plot_sem = False, pltlabel='D2 Stop')     
    rate_plot(STR_D2_Go_mean, STR_D2_Go_sem, 'STR_D2_Go', paramsS['trials'], 'purple', plot_sem = False, pltlabel='D2 Go')            
    rate_plot(STR_FSI_Stop_mean, STR_FSI_Stop_sem, 'STR_FSI_Stop', paramsS['trials'], 'k', plot_sem = False, pltlabel='FSI Stop')     
    rate_plot(STR_FSI_Go_mean, STR_FSI_Go_sem, 'STR_FSI_Go', paramsS['trials'], '0.6', plot_sem = False, pltlabel='FSI Go') 
    plt.ion()
    ax=plt.gca()
    #plt.title('Population-averaged firing rate, mean across '+str(paramsS['trials'])+' trials') # STR D1
    #t0 = 50 # 50ms simulation for initialization is not recorded, see above
    plt.plot((params['t_init'] + params['t_SSD'])/params['general_dt'] * np.ones(2), [0, plt.axis()[3]], 'k--', lw=1.5, label='Stop cue')
    ax.axis([t_min, t_max, max(0, ax.axis()[2]), 80]) # dt=0.1
    plt.xlabel('t [msec]')
    plt.ylabel('Firing rate [spk/s]', fontsize=fsize)
    ax.set_xticks([(params['t_init'] + params['t_SSD'] - 200)/params['general_dt'],(params['t_init'] + params['t_SSD'])/params['general_dt'], (params['t_init'] + params['t_SSD'] + 400)/params['general_dt']]) # LG
    ax.set_xticklabels([- 0.2, 0, 0.4]) # LG
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #plt.legend(('D1 Stop', 'D1 Go', 'D2 Stop', 'D2 Go', 'GPe Arky Stop', 'GPe Proto Stop', 'FSI Stop', 'FSI Go', 'Stop cue'), fontsize=6) # 10)
    plt.legend(fontsize=6)    
    plt.tight_layout()
    plt.savefig(resultsDir+'/mean_rate_paramsid'+str(params['general_id'])+'_'+str(paramsS['trials'])+'trials_STR_D2FSI.png') #             
    #plt.ioff()    
    #plt.show()"""
    
    toRGB = {'blue':np.array([3,67,223])/255., 'cyan':np.array([0,255,255])/255., 'gold':np.array([219,180,12])/255., 'orange':np.array([249,115,6])/255., 'red':np.array([229,0,0])/255., 'purple':np.array([126,30,156])/255., 'grey':np.array([146,149,145])/255., 'light brown':np.array([173,129,80])/255., 'lime':np.array([170,255,50])/255., 'green':np.array([21,176,26])/255., 'yellow':np.array([255,255,20])/255., 'lightgrey':np.array([216,220,214])/255.}
    STRzscorePlotsMaxT = 350


    ### FIGURE: Z-SCORES, STOP CUE RESPONSE OF STRD2 AND STRFSI CORRECT_STOP/SLOW_GO
    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()    
    STRD2_all_mean = StrD2_meanrate_SlowGo
    STRFSI_all_mean = StrFSI_meanrate_SlowGo
    ## PLOT MEAN RATES
    plt.plot(custom_zscore_start0(SD2_mean_CorrectStop, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], STRD2_all_mean), color=toRGB['purple'], lw=3)
    plt.plot(custom_zscore_start0(StrD2_meanrate_SlowGo, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], STRD2_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['purple']*0.7, lw=3) 
    plt.plot(custom_zscore_start0(FSI_mean_CorrectStop, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], STRFSI_all_mean), color=toRGB['grey'], lw=3)     
    plt.plot(custom_zscore_start0(StrFSI_meanrate_SlowGo, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], STRFSI_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['grey']*0.7, lw=3) 
    ## STOP CUE MARKER
    plt.axvline((params['t_init'] + params['t_SSD'])/params['general_dt'], color='grey', lw=0.5)
    ## LABELS
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    plt.ylabel('$\Delta$Firing rate (z-score)', fontsize=fsize)
    ## LIMITS AND TICKS
    ax.axis([(params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], (params['t_init'] + params['t_SSD'] + STRzscorePlotsMaxT)/params['general_dt'], -2.2, 3.6])
    ax.set_xticks([(params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], (params['t_init'] + params['t_SSD'])/params['general_dt'], (params['t_init'] + params['t_SSD'] + 300)/params['general_dt']])
    ax.set_xticklabels([-0.2, 0, 0.3])
    ax.set_yticks(range(-2,4))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ## LEGEND
    normalLine = mlines.Line2D([], [], color='k', label='Stop', lw=3)
    dashedLine = mlines.Line2D([], [], color='k', label='slow Go', dash_capstyle='round', dashes=(0.05,2), lw=3)
    purpleLine = mlines.Line2D([], [], color=toRGB['purple'], label='StrD2', lw=3)
    greyLine = mlines.Line2D([], [], color=toRGB['grey'], label='StrFSI', lw=3)
    plt.legend(handles=[dashedLine, normalLine, purpleLine, greyLine], bbox_to_anchor=(-0.02,-0.015), bbox_transform=plt.gca().transAxes, fontsize=10, loc='lower left')
    ## SAVE
    plt.tight_layout()
    plt.savefig(resultsDir+'/zscore_StopSTRD2FSI_paramsid'+str(params['general_id'])+'_'+str(paramsS['trials'])+'trials.png', dpi=300)
    

    ### FIGURE: Z-SCORES, STOP CUE RESPONSE OF STRD1, GPE_Proto AND GPE_ARKY CORRECT_STOP/SLOW_GO
    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()
    STR_all_mean        = StrD1_meanrate_SlowGo
    GPe_Arky_all_mean   = GPe_Arky_meanrate_SlowGo
    GPe_Proto_all_mean  = GPe_Proto_meanrate_SlowGo
    ## PLOT MEAN RATES
    plt.plot(custom_zscore_start0(SD1_mean_CorrectStop, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], STR_all_mean), color=toRGB['red'], lw=3)     
    plt.plot(custom_zscore_start0(StrD1_meanrate_SlowGo, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], STR_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['red']*0.7, lw=3)         
    plt.plot(custom_zscore_start0(GPe_Arky_mean_CorrectStop, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], GPe_Arky_all_mean), color=toRGB['cyan'], lw=3)
    plt.plot(custom_zscore_start0(GPe_Arky_meanrate_SlowGo, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], GPe_Arky_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['cyan']*0.7, lw=3)
    plt.plot(custom_zscore_start0(GPe_Proto_mean_CorrectStop, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], GPe_Proto_all_mean), color=toRGB['blue'], lw=3) 
    plt.plot(custom_zscore_start0(GPe_Proto_meanrate_SlowGo, (params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], GPe_Proto_all_mean), dash_capstyle='round', dashes=(0.05,2), color=toRGB['blue']*0.7, lw=3)
    ## STOP CUE MARKER
    plt.axvline((params['t_init'] + params['t_SSD'])/params['general_dt'], color='grey', lw=0.5)
    ## LABELS
    plt.ylabel('$\Delta$Firing rate (z-score)', fontsize=fsize)
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    ## LIMITS AND TICKS
    ax.axis([(params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], (params['t_init'] + params['t_SSD'] + STRzscorePlotsMaxT)/params['general_dt'], -2.2, 3.6])
    ax.set_xticks([(params['t_init'] + params['t_SSD'] - 200)/params['general_dt'], (params['t_init'] + params['t_SSD'])/params['general_dt'], (params['t_init'] + params['t_SSD'] + 300)/params['general_dt']])
    ax.set_xticklabels([-0.2, 0, 0.3])
    ax.set_yticks(range(-2,4))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ## LEGEND
    normalLine = mlines.Line2D([], [], color='k', label='Stop', lw=3)
    dashedLine = mlines.Line2D([], [], color='k', label='slow Go', dash_capstyle='round', dashes=(0.05,2), lw=3)
    redLine = mlines.Line2D([], [], color=toRGB['red'], label='StrD1', lw=3)
    cyanLine = mlines.Line2D([], [], color=toRGB['cyan'], label='GPe-Arky', lw=3)
    blueLine = mlines.Line2D([], [], color=toRGB['blue'], label='GPe-Proto', lw=3)
    plt.legend(handles=[dashedLine, normalLine, redLine, cyanLine, blueLine], bbox_to_anchor=(-0.02,1.015), bbox_transform=plt.gca().transAxes, fontsize=10, loc='upper left')
    ## SAVE
    plt.tight_layout()
    plt.savefig(resultsDir+'/zscore_StopSTR_paramsid'+str(params['general_id'])+'_'+str(paramsS['trials'])+'trials.png', dpi=300)



    """plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()
    rate_plot(STN_mean_Stop, STN_sem_Stop, 'STN', paramsS['trials'], 'orange', plot_sem = False, pltlabel='STN Stop')     
    rate_plot(SNr_mean_Stop, SNr_sem_Stop, 'SNr', paramsS['trials'], 'tomato', plot_sem = False, pltlabel='SNr Stop')     
    plt.plot(GPe_Arky_mean_Stop, color='cyan', lw=3) 
    #ax.fill_between(range(len(GPe_Arky_mean_Stop)), GPe_Arky_mean_Stop - GPe_Arky_sem_Stop, GPe_Arky_mean_Stop + GPe_Arky_sem_Stop, color='cyan', alpha=0.4, edgecolor='None')
    plt.plot(GPe_Proto_mean_Stop, color='blue', lw=3) 
    #ax.fill_between(range(len(GPe_Proto_mean_Stop)), GPe_Proto_mean_Stop - GPe_Proto_sem_Stop, GPe_Proto_mean_Stop + GPe_Proto_sem_Stop, color='blue', alpha=0.4, edgecolor='None')  
    plt.axvline((params['t_init'] + params['t_SSD'])/params['general_dt'], color='grey', lw=0.5)
    fsize = 15 # 10
    plt.ylabel('Firing rate [spk/s]', fontsize=fsize)
    #ax.axis([params['t_init'] + params['t_SSD'] - 50, params['t_init'] + params['t_SSD'] + 150, max(0, ax.axis()[2]), 350])
    ax.axis([(params['t_init'] + params['t_SSD'] - 50)/params['general_dt'], (params['t_init'] + params['t_SSD'] + 250)/params['general_dt'], max(0, ax.axis()[2]), 100])    
    ax.set_xticks([(params['t_init'] + params['t_SSD'] - 50)/params['general_dt'], (params['t_init'] + params['t_SSD'])/params['general_dt'], (params['t_init'] + params['t_SSD'] + 50)/params['general_dt'], (params['t_init'] + params['t_SSD'] + 100)/params['general_dt']])
    ax.set_xticklabels([-0.05, 0, 0.05, 0.1])  
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.legend(('STN-Stop', 'SNr-Stop', 'Arky-Stop', 'Proto-Stop'), fontsize=10) # , 'Stop cue'
    plt.tight_layout()
    plt.savefig(resultsDir+'/mean_rate_StopDetail_paramsid'+str(params['general_id'])+'_'+str(paramsS['trials'])+'trials.png')
    #plt.show()"""
    

    ### FIGURE: Z-SCORES, STOP CUE RESPONSE OF STN, SNR, GPE_PROTO AND GPE_ARKY CORRECT_STOP
    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()
    ## PLOT MEAN RATES
    plt.plot(custom_zscore_start0(STN_mean_CorrectStop, (params['t_init'] + params['t_SSD'] - 50)/params['general_dt'], STN_mean_CorrectStop), color=toRGB['gold'], lw=3) 
    plt.plot(custom_zscore_start0(SNr_mean_CorrectStop, (params['t_init'] + params['t_SSD'] - 50)/params['general_dt'], SNr_mean_CorrectStop), color=toRGB['orange'], lw=3) 
    plt.plot(custom_zscore_start0(GPe_Arky_mean_CorrectStop, (params['t_init'] + params['t_SSD'] - 50)/params['general_dt'], GPe_Arky_mean_CorrectStop), color=toRGB['cyan'], lw=3) 
    plt.plot(custom_zscore_start0(GPe_Proto_mean_CorrectStop, (params['t_init'] + params['t_SSD'] - 50)/params['general_dt'], GPe_Proto_mean_CorrectStop), color=toRGB['blue'], lw=3) 
    ## STOP CUE MARKER
    plt.axvline((params['t_init'] + params['t_SSD'])/params['general_dt'], color='grey', lw=0.5)
    ## LABELS
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    plt.ylabel('$\Delta$Firing rate (z-score)', fontsize=fsize)
    ## LIMITS AND TICKS
    ax.axis([(params['t_init'] + params['t_SSD'] - 50)/params['general_dt'], (params['t_init'] + params['t_SSD'] + 200)/params['general_dt'], -2.2, 3.6])
    xtickList=[0,0.1,0.2]  
    ax.set_xticks([(params['t_init'] + params['t_SSD'] + i*1000)/params['general_dt'] for i in xtickList])
    ax.set_xticklabels(xtickList)
    ax.set_yticks(range(-2,4))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ## LEGEND
    plt.legend(('STN Stop', 'SNr Stop', 'GPe-Arky Stop', 'GPe-Proto Stop'), bbox_to_anchor=(-0.02,-0.015), bbox_transform=plt.gca().transAxes, fontsize=10, loc='lower left') # , 'Stop cue'
    ## SAVE
    plt.tight_layout()
    plt.savefig(resultsDir+'/zscore_StopDetail_paramsid'+str(params['general_id'])+'_'+str(paramsS['trials'])+'trials.png', dpi=300)


    """plt.figure(figsize=(3.5, 4), dpi=300)
    ax=plt.gca()
    rate_plot(STN_mean_Go, STN_sem_Go, 'STN', paramsS['trials'], 'orange', plot_sem = False, pltlabel='STN Go')
    rate_plot(SNr_mean_Go, SNr_sem_Go, 'SNr', paramsS['trials'], 'tomato', plot_sem = False, pltlabel='SNr Go')         
    rate_plot(GPe_Arky_mean_Go, GPe_Arky_sem_Go, 'GPe_Arky', paramsS['trials'], 'cyan', plot_sem = False, pltlabel='Arky Go')         
    rate_plot(GPe_Proto_mean_Go, GPe_Proto_sem_Go, 'GPe_Proto', paramsS['trials'], 'blue', plot_sem = False, pltlabel='Proto Go')    
    plt.close()"""


    """plot_zscore_stopVsGo(STN_mean_Stop, STN_mean_Go, SNr_mean_Stop, SNr_mean_Go, GPe_Arky_mean_Stop, GPe_Arky_mean_Go, GPe_Proto_mean_Stop, GPe_Proto_mean_Go, params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], resultsDir+'/')
    plot_zscore_stopVsGo_five(STR_D1_Stop_mean, STR_D1_Go_mean, STR_D2_Stop_mean, STR_D2_Go_mean, STR_FSI_Stop_mean, STR_FSI_Go_mean, GPe_Proto2_mean_Stop, GPe_Proto2_mean_Go, Thal_mean_Stop, Thal_mean_Go, params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], resultsDir+'/', \
                              labels=['StrD1', 'StrD2', 'StrFSI', 'GPe Cortex-projecting', 'Thalamus'], linecol=[['red', 'green'], ['purple', 'pink'], ['black','grey'], ['brown', 'olive'], ['lime','teal']])
    """
    
    
    ### FIGURE: Z-SCORES, GO AND STOP CUE RESPONSES OF MULTIPLE POPS DURING CORRECT_STOP
    plot_zscore_stopVsGo_NewData(STN_mean_CorrectStop, STN_mean_CorrectStop, SNr_mean_CorrectStop, SNr_mean_CorrectStop, GPe_Arky_mean_CorrectStop, GPe_Arky_mean_CorrectStop, GPe_Proto_mean_CorrectStop, GPe_Proto_mean_CorrectStop, params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], resultsDir+'/', labels = ['STN', 'SNr', 'GPe-Arky', 'GPe-Proto'], linecol = [[toRGB['gold'], toRGB['gold']*0.7], [toRGB['orange'], toRGB['orange']*0.7], [toRGB['cyan'],toRGB['cyan']*0.7], [toRGB['blue'],toRGB['blue']*0.7]])
    plot_zscore_stopVsGo_five_NewData(SD1_mean_CorrectStop, SD1_mean_CorrectStop, SD2_mean_CorrectStop, SD2_mean_CorrectStop, FSI_mean_CorrectStop, FSI_mean_CorrectStop, GPe_Proto2_mean_CorrectStop, GPe_Proto2_mean_CorrectStop, Thal_mean_CorrectStop, Thal_mean_CorrectStop, params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], resultsDir+'/', labels=['StrD1', 'StrD2', 'StrFSI', 'GPe-Cp', 'Thalamus'], linecol=[[toRGB['red'], toRGB['red']*0.7], [toRGB['purple'], toRGB['purple']*0.7], [toRGB['grey'], toRGB['grey']*0.7], [toRGB['light brown'], toRGB['light brown']*0.7], [toRGB['lime'], toRGB['lime']*0.7]])                             


##############################################################################################
if abfrage_loop == 'extra':
    t_stopCue = int(params['t_init'] + params['t_SSD'])    
    fsize = 6    

    #'''#
    plt.figure(figsize=(3.5,4), dpi=300)
    #mInt_stop = np.load('../data/'+loadFolder+'/Integrator_ampa_Stop_id'+str(params['general_id'])+'.npy')
    mInt_stop = np.load('../data/'+loadFolder+'/Integrator_ampa_Stop_'+netNr+'_id'+str(params['general_id'])+'.npy')
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(paramsS['trials'])), paramsS['trials']], order='F') # order seems correct
    mInt_maxpertrial = np.nanmax(rsp_mInt_stop, 0)
    nz_FailedStop = np.nonzero(mInt_maxpertrial >= Integrator.threshold)[0] # This seems correct
    nz_CorrectStop = np.nonzero(mInt_maxpertrial < Integrator.threshold)[0]

    SD1_mean_FailedStop, SD1_std_FailedStop, SD1_mean_CorrectStop, SD1_std_CorrectStop = np.load('../data/'+loadFolder+'/SD1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    SD2_mean_FailedStop, SD2_std_FailedStop, SD2_mean_CorrectStop, SD2_std_CorrectStop = np.load('../data/'+loadFolder+'/SD2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    FSI_mean_FailedStop, FSI_std_FailedStop, FSI_mean_CorrectStop, FSI_std_CorrectStop = np.load('../data/'+loadFolder+'/FSI_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    STN_mean_FailedStop, STN_std_FailedStop, STN_mean_CorrectStop, STN_std_CorrectStop = np.load('../data/'+loadFolder+'/STN_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    GPe_Proto_mean_FailedStop, GPe_Proto_std_FailedStop, GPe_Proto_mean_CorrectStop, GPe_Proto_std_CorrectStop = np.load('../data/'+loadFolder+'/Proto_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    GPe_Proto2_mean_FailedStop, GPe_Proto2_std_FailedStop, GPe_Proto2_mean_CorrectStop, GPe_Proto2_std_CorrectStop = np.load('../data/'+loadFolder+'/Proto2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    GPe_Arky_mean_FailedStop, GPe_Arky_std_FailedStop, GPe_Arky_mean_CorrectStop, GPe_Arky_std_CorrectStop = np.load('../data/'+loadFolder+'/Arky_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    SNr_mean_FailedStop, SNr_std_FailedStop, SNr_mean_CorrectStop, SNr_std_CorrectStop = np.load('../data/'+loadFolder+'/SNr_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    Thal_mean_FailedStop, Thal_std_FailedStop, Thal_mean_CorrectStop, Thal_std_CorrectStop = np.load('../data/'+loadFolder+'/Thal_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    SNrE_mean_FailedStop, SNrE_std_FailedStop, SNrE_mean_CorrectStop, SNrE_std_CorrectStop = np.load('../data/'+loadFolder+'/SNrE_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    Cortex_G_mean_FailedStop, Cortex_G_std_FailedStop, Cortex_G_mean_CorrectStop, Cortex_G_std_CorrectStop = np.load('../data/'+loadFolder+'/Cortex_G_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    Cortex_S_mean_FailedStop, Cortex_S_std_FailedStop, Cortex_S_mean_CorrectStop, Cortex_S_std_CorrectStop = np.load('../data/'+loadFolder+'/Cortex_S_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    PauseInput_mean_FailedStop, PauseInput_std_FailedStop, PauseInput_mean_CorrectStop, PauseInput_std_CorrectStop = np.load('../data/'+loadFolder+'/Stoppinput1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    

    #t_simend = int( (840 + (params['t_init']-100)) / params['general_dt'] ) # 

    #pvalue_list = []
    #pvalue_times = []    
    pvalue_list, pvalue_times = np.load('../data/'+loadFolder+'/p_value_list_times_'+str(params['general_id'])+netNr+'.npy', allow_pickle=True)

    plt.close()

    plt.figure()
    Cortex_G_mean_Stop, Cortex_G_std_Stop =  np.load('../data/'+loadFolder+'/Cortex_G_rate_Stop_mean_std'+netNr+'.npy')
    plt.subplot(111)
    plt.plot(Cortex_G_mean_Stop,label='1')
    plt.plot(Cortex_G_mean_CorrectStop,label='2')
    plt.plot(Cortex_G_mean_FailedStop,label='3')
    plt.legend()
    plt.savefig('TESTFIGUREextra.png')


    tmin = params['t_init'] + params['t_SSD'] # Time of Stop cue presentation
    tmax = params['t_init'] + params['t_SSD'] + 100 # 200
    STR_D1_ratepertrial_Stop = np.load('../data/'+loadFolder+'/SD1_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy')
    STR_D2_ratepertrial_Stop = np.load('../data/'+loadFolder+'/SD2_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy')
    STN_ratepertrial_Stop = np.load('../data/'+loadFolder+'/STN_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy')
    GPe_Proto_ratepertrial_Stop = np.load('../data/'+loadFolder+'/GPeProto_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy')        
    GPe_Arky_ratepertrial_Stop = np.load('../data/'+loadFolder+'/GPeArky_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy')        
    SNr_ratepertrial_Stop = np.load('../data/'+loadFolder+'/SNr_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy')
    Thal_ratepertrial_Stop = np.load('../data/'+loadFolder+'/Thal_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy')
    Cortex_G_ratepertrial_Stop = np.load('../data/'+loadFolder+'/Cortex_G_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy')        
    Cortex_S_ratepertrial_Stop = np.load('../data/'+loadFolder+'/Cortex_S_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy')                

    #plot_correl_rates_Intmax(GPe_Arky_ratepertrial_Stop, GPe_Proto_ratepertrial_Stop, STR_D1_ratepertrial_Stop,   STR_D2_ratepertrial_Stop,   STN_ratepertrial_Stop, \
    #                         SNr_ratepertrial_Stop,      Thal_ratepertrial_Stop,      Cortex_G_ratepertrial_Stop, Cortex_S_ratepertrial_Stop, mInt_maxpertrial, params['general_id'], paramsS['trials'])



    #'''#
    plot_meanrate_All_FailedVsCorrectStop(resultsDir+'/', \
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
                                          params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], pvalue_list, pvalue_times) # FSI_mean
    #'''

    ### load all correct go, slow go and fast go data for all nuclei
    # all Go
    GPe_Arky_mean_Go,   GPe_Arky_std_Go     = np.load('../data/'+loadFolder+'/GPeArky_rate_Go_mean_std'+netNr+'.npy')
    GPe_Proto_mean_Go,  GPe_Proto_std_Go    = np.load('../data/'+loadFolder+'/GPeProto_rate_Go_mean_std'+netNr+'.npy')  
    STR_D1_mean_Go,     STR_D1_std_Go       = np.load('../data/'+loadFolder+'/SD1_rate_Go_mean_std'+netNr+'.npy')     
    STR_D2_mean_Go,     STR_D2_std_Go       = np.load('../data/'+loadFolder+'/SD2_rate_Go_mean_std'+netNr+'.npy')        
    STN_mean_Go,        STN_std_Go          = np.load('../data/'+loadFolder+'/STN_rate_Go_mean_std'+netNr+'.npy')
    SNr_mean_Go,        SNr_std_Go          = np.load('../data/'+loadFolder+'/SNr_rate_Go_mean_std'+netNr+'.npy')            
    Thal_mean_Go,       Thal_std_Go         = np.load('../data/'+loadFolder+'/Thal_rate_Go_mean_std'+netNr+'.npy')
    Cortex_G_mean_Go,   Cortex_G_std_Go     = np.load('../data/'+loadFolder+'/Cortex_G_rate_Go_mean_std'+netNr+'.npy')
    GPe_Proto2_mean_Go, GPe_Proto2_std_Go   = np.load('../data/'+loadFolder+'/GPeProto2_rate_Go_mean_std'+netNr+'.npy')
    PauseInput_mean_Go, PauseInput_std_Go   = np.load('../data/'+loadFolder+'/Stoppinput1_rate_Go_mean_std'+netNr+'.npy')
    Cortex_S_mean_Go,   Cortex_S_std_Go     = np.load('../data/'+loadFolder+'/Cortex_S_rate_Go_mean_std'+netNr+'.npy')
    STR_FSI_mean_Go,    STR_FSI_std_Go      = np.load('../data/'+loadFolder+'/FSI_rate_Go_mean_std'+netNr+'.npy')            

    # fast and slow Go
    GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo     = np.load('../data/'+loadFolder+'/GPe_Arky_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo    = np.load('../data/'+loadFolder+'/GPe_Proto_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    STR_D1_meanrate_FastGo,     STR_D1_std_FastGo,      STR_D1_meanrate_SlowGo,     STR_D1_std_SlowGo       = np.load('../data/'+loadFolder+'/SD1_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    STR_D2_meanrate_FastGo,     STR_D2_std_FastGo,      STR_D2_meanrate_SlowGo,     STR_D2_std_SlowGo       = np.load('../data/'+loadFolder+'/SD2_meanrate_std_Fast-Slow_Go'+netNr+'.npy')            
    STN_meanrate_FastGo,        STN_std_FastGo,         STN_meanrate_SlowGo,        STN_std_SlowGo          = np.load('../data/'+loadFolder+'/STN_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    SNr_meanrate_FastGo,        SNr_std_FastGo,         SNr_meanrate_SlowGo,        SNr_std_SlowGo          = np.load('../data/'+loadFolder+'/SNr_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    Thal_meanrate_FastGo,       Thal_std_FastGo,        Thal_meanrate_SlowGo,       Thal_std_SlowGo         = np.load('../data/'+loadFolder+'/Thal_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    Cortex_G_meanrate_FastGo,   Cortex_G_std_FastGo,    Cortex_G_meanrate_SlowGo,   Cortex_G_std_SlowGo     = np.load('../data/'+loadFolder+'/Cortex_G_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    GPe_Proto2_meanrate_FastGo, GPe_Proto2_std_FastGo,  GPe_Proto2_meanrate_SlowGo, GPe_Proto2_std_SlowGo   = np.load('../data/'+loadFolder+'/GPe_Proto2_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    PauseInput_meanrate_FastGo, PauseInput_std_FastGo,  PauseInput_meanrate_SlowGo, PauseInput_std_SlowGo   = np.load('../data/'+loadFolder+'/PauseInput_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    Cortex_S_meanrate_FastGo,   Cortex_S_std_FastGo,    Cortex_S_meanrate_SlowGo,   Cortex_S_std_SlowGo     = np.load('../data/'+loadFolder+'/Cortex_S_meanrate_std_Fast-Slow_Go'+netNr+'.npy')                        
    STR_FSI_meanrate_FastGo,    STR_FSI_std_FastGo,     STR_FSI_meanrate_SlowGo,    STR_FSI_std_SlowGo      = np.load('../data/'+loadFolder+'/FSI_meanrate_std_Fast-Slow_Go'+netNr+'.npy') 

    # load further things
    mInt_Go = np.load('../data/'+loadFolder+'/Integrator_ampa_Go_'+netNr+'_id'+str(params['general_id'])+'.npy')
    pvalue_list = {}
    nameList = ['failedStop_vs_allGo', 'failedStop_vs_fastGo', 'failedStop_vs_slowGo']
    for name in nameList:
        pvalue_list[name], pvalue_times = np.load('../data/'+loadFolder+'/p_value_list_'+name+'_times_'+str(params['general_id'])+netNr+'.npy', allow_pickle=True)

    plot_meanrate_All_FailedStopVsCorrectGo(resultsDir+'/', \
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
                                          params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], pvalue_list, pvalue_times, GO_Mode='fast')


    #rateperneuron_GPeArky_allStoptrials = np.load('../data/'+loadFolder+'/GPeArky_rateperneuron_allStoptrials_'+str(params['general_id'])+'.npy')
    STN_poprate_Stop_alltrials = np.load('../data/'+loadFolder+'/STN_rate_Stop_alltrials'+netNr+'.npy')
    SNr_poprate_Stop_alltrials = np.load('../data/'+loadFolder+'/SNr_rate_Stop_alltrials'+netNr+'.npy')
    GPe_Proto_poprate_Stop_alltrials = np.load('../data/'+loadFolder+'/GPeProto_rate_Stop_alltrials'+netNr+'.npy')
    GPe_Arky_poprate_Stop_alltrials = np.load('../data/'+loadFolder+'/GPeArky_rate_Stop_alltrials'+netNr+'.npy')        
    #get_peak_response_time(STN_poprate_Stop_alltrials, SNr_poprate_Stop_alltrials, GPe_Proto_poprate_Stop_alltrials, GPe_Arky_poprate_Stop_alltrials, params['t_init'] + params['t_SSD'], params['t_init'] + params['t_SSD'] + 200, params['general_dt'], params['general_id'], paramsS['trials'], params['general_id'], paramsS['trials'], paramname)    


##############################################################################################
if abfrage_loop=='RT':
    print("Calculating reaction times...")
    i_cycle=0

    #'''#
    spike_times, ranks = np.load('./data/Integrator_spike_Go'+netNr+'_cycle'+str(i_cycle)+'.npy')
    spike_times_stop, ranks_stop = np.load('./data/Integrator_spike_Stop'+netNr+'_cycle'+str(i_cycle)+'.npy')
    mInt_stop = np.load('../data/'+loadFolder+'/Integrator_ampa_Stop_'+netNr+'_id'+str(params['general_id'])+'.npy')
    mInt_go = np.load('../data/'+loadFolder+'/Integrator_ampa_Go_'+netNr+'_id'+str(params['general_id'])+'.npy')            
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(paramsS['trials'])), paramsS['trials']], order='F')
    rsp_mInt_go = np.reshape(mInt_go, [int(mInt_go.shape[0] / float(paramsS['trials'])), paramsS['trials']], order='F')
    mInt_maxpertrial = np.nanmax(rsp_mInt_go, 0)
    nz_FailedGo = np.nonzero(mInt_maxpertrial < Integrator.threshold)[0]
    nz_CorrectGo = np.nonzero(mInt_maxpertrial >= Integrator.threshold)[0]
    mInt_maxpertrial = np.nanmax(rsp_mInt_stop, 0)
    nz_FailedStop = np.nonzero(mInt_maxpertrial >= Integrator.threshold)[0]
    nz_CorrectStop = np.nonzero(mInt_maxpertrial < Integrator.threshold)[0]

    RT_Go = np.nan * np.ones(paramsS['trials'])
    RT_Stop = np.nan * np.ones(paramsS['trials'])
    for i_trial in range(paramsS['trials']):
        if np.nanmax(rsp_mInt_go[:, i_trial]) >= Integrator.threshold: 
            RT_Go[i_trial] = np.nonzero(rsp_mInt_go[:, i_trial] >= Integrator.threshold)[0][0]
            #RT_Go[i_trial] = np.nonzero(rsp_mInt_go[:, i_trial] >= Integrator.threshold)[0][0] * params['general_dt']            
        if np.nanmax(rsp_mInt_stop[:, i_trial]) >= Integrator.threshold: 
            RT_Stop[i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= Integrator.threshold)[0][0] # Correct
            #RT_Stop[i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= Integrator.threshold)[0][0] * params['general_dt'] # Correct                

    #plt.ion()
    plt.figure(figsize=(3.5,4), dpi=300)
    fsize = 12 # 15
    nz_Go = np.nonzero(np.isnan(RT_Go)==False)
    nz_Stop = np.nonzero(np.isnan(RT_Stop)==False)
    binSize=15
    counts_Go, bins_Go = np.histogram(RT_Go[nz_Go] * params['general_dt'], np.arange(params['t_init'],params['t_init']+501,binSize)) #
    counts_Stop, bins_Stop = np.histogram(RT_Stop[nz_Stop] * params['general_dt'], np.arange(params['t_init'],params['t_init']+501,binSize)) # 
    if counts_Go.max() > 0:
        #plt.bar(bins_Go[:-1], np.array(counts_Go) * (1.0/counts_Go.max()), width=np.diff(bins_Go)[0], alpha=0.5, color='b') #
        plt.bar(bins_Go[:-1], counts_Go/counts_Go.max(), width=binSize, alpha=1, color=np.array([107,139,164])/255.) #
    if counts_Stop.max() > 0: 
        #plt.bar(bins_Stop[:-1], np.array(counts_Stop) * (1.0/counts_Stop.max()), width=np.diff(bins_Stop)[0], alpha=0.5, color='g') #
        plt.bar(bins_Stop[:-1], counts_Stop/counts_Stop.max(), width=binSize, alpha=0.5, color='tomato') #
    mean_CorrectGo = np.round( (np.nanmean(RT_Go[nz_Go]) - params['t_init']/params['general_dt'])*params['general_dt'], 1)
    mean_FailedStop = np.round( (np.nanmean(RT_Stop[nz_Stop]) - params['t_init']/params['general_dt'])*params['general_dt'], 1)
    plt.legend(('correct Go, mean RT='+str(mean_CorrectGo),'failed Stop, mean RT='+str(mean_FailedStop)), fontsize=fsize-2)
    plt.xlabel('Reaction time [ms]', fontsize=fsize)    
    plt.ylabel('Normalized trial count', fontsize=fsize)
    ax = plt.gca()
    ax.set_yticks([0.0,0.5,1.0])
    #ax.axis([params['t_init'], 400 + params['t_init'], ax.axis()[2], ax.axis()[3]])
    ax.axis([params['t_init'], (500 + params['t_init']), ax.axis()[2], 1.22])                
    #ax.axis([params['t_init']/params['general_dt'], (500 + params['t_init'])/params['general_dt'], ax.axis()[2], ax.axis()[3]])
    xtks = ax.get_xticks()
    xlabels = []
    for i in range(len(xtks)):
        xlabels.append(str( int( xtks[i]-params['t_init'] ) ))
    ax.set_xticklabels(xlabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.tight_layout()
    #plt.ioff()
    plt.savefig(resultsDir+'/Reaction_times_paramsid'+str(params['general_id'])+'.png')
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

    #np.save('../data/'+loadFolder+'/resultsRT_id'+str(params['general_id'])+'.npy', results_RT)
    #np.save('../data/'+loadFolder+'/resultsRT_'+netNr+'id'+str(params['general_id'])+'.npy', results_RT) # 
    #np.save('../data/'+loadFolder+'/resultsRT_'+netNr+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(params['general_id'])+'.npy', results_RT) 
    params_loaded = np.load('../data/'+loadFolder+'/paramset_id'+str(params['general_id'])+netNr+'.npy', allow_pickle=True)
    print("params_loaded = ", params)


    thresh = Integrator.threshold
    #plt.ion()
    plt.figure(figsize=(3.5,4), dpi=300)
    plt.plot(rsp_mInt_go[:, :], color=np.array([107,139,164])/255., lw=1, label='correct Go')
    if len(nz_FailedStop) > 0:
        plt.plot(rsp_mInt_stop[:, nz_FailedStop], color='tomato', lw=1, label = 'failed Stop', linestyle='--')
    if len(nz_CorrectStop) > 0:
        plt.plot(rsp_mInt_stop[:, nz_CorrectStop], color='purple', lw=1, label = 'correct Stop')
    #gocue
    plt.axvline(params['t_init'] / params['general_dt'], color='green', linestyle=':')
    GoCueHandle = mlines.Line2D([], [], color='green', label='Go cue', linestyle=':')
    #stopcue
    plt.axvline((params['t_init'] + params['t_SSD']) / params['general_dt'], color='orange', linestyle=':')
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
    with open(resultsDir+'/stops_and_gos_'+str(paramsS['trials'])+'trials_paramsid'+str(params['general_id'])+'.txt', 'w') as f:
        mInt_Go = np.load('../data/'+loadFolder+'/Integrator_ampa_Go_'+netNr+'_id'+str(params['general_id'])+'.npy')
        mInt_Stop = np.load('../data/'+loadFolder+'/Integrator_ampa_Stop_'+netNr+'_id'+str(params['general_id'])+'.npy')
        nz_FastGo, nz_SlowGo = get_fast_slow_go_trials(mInt_Go, mInt_Stop, Integrator.threshold, paramsS['trials'])
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
    ax.axis([(params['t_init']-200)/params['general_dt'], (params['t_init']+600)/params['general_dt'], 0, ax.axis()[3]])        
    ax.set_xticks([params['t_init']/params['general_dt'], (params['t_init']+200)/params['general_dt'], (params['t_init']+400)/params['general_dt'], (params['t_init']+600)/params['general_dt']])
    xtks = ax.get_xticks()
    xlabels = []
    for i in range(len(xtks)):
        xlabels.append(str(int(params['general_dt']*xtks[i]-params['t_init'])))
    ax.set_xticklabels(xlabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.tight_layout()
    #plt.ioff()
    plt.savefig(resultsDir+'/Integrator_ampa_Stop_'+str(paramsS['trials'])+'trials_paramsid'+str(params['general_id'])+'.png')
    #plt.close()
    #plt.show()
