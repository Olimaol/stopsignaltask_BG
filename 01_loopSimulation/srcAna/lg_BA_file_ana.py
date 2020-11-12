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

from BGmodelSST.analysis import get_correct_failed_stop_trials, custom_zscore, custom_zscore_start0, \
    plot_zscore_stopVsGo, plot_zscore_stopVsGo_five, plot_zscore_stopVsGo_NewData, plot_zscore_stopVsGo_five_NewData, plot_correl_rates_Intmax, plot_meanrate_All_FailedVsCorrectStop, plot_meanrate_All_FailedStopVsCorrectGo, calc_KW_stats_all, rate_plot, \
    calc_meanrate_std_failed_correct, get_rates_failed_correct, get_poprate_aligned_onset, get_peak_response_time, custom_poprate, calc_meanrate_std_Fast_Slow, get_fast_slow_go_trials, get_rates_allGo_fastGo_slowGo
#from BGmodelSST.plotting import get_and_plot_syn_mon    
#from BGmodelSST.sim_params import params
#from BGmodelSST.init import init_neuronmodels
import matplotlib.lines as mlines
from analysisParams import paramsA


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

if len(sys.argv) <= 2:
    print('Arguments missing! first argument: network number, second argument: plotting mode')
    quit()
else:
    netNr = str(sys.argv[1])
    abfrage_loop = str(sys.argv[2])


params  = np.load('../data/'+paramsA['loadFolder']+'/paramset_id8007'+netNr+'.npy', allow_pickle=True)[0]
paramsS = np.load('../data/'+paramsA['loadFolder']+'/paramSset_id8007'+netNr+'.npy', allow_pickle=True)[0]

resultsDir='../results/'+paramsA['loadFolder']
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
    
    ### LOAD FAILED/CORRECT STOP RATES
    SD1_mean_FailedStop,        SD1_std_FailedStop,        SD1_mean_CorrectStop,        SD1_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SD1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    SD2_mean_FailedStop,        SD2_std_FailedStop,        SD2_mean_CorrectStop,        SD2_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SD2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    FSI_mean_FailedStop,        FSI_std_FailedStop,        FSI_mean_CorrectStop,        FSI_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/FSI_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    STN_mean_FailedStop,        STN_std_FailedStop,        STN_mean_CorrectStop,        STN_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/STN_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    GPe_Proto_mean_FailedStop,  GPe_Proto_std_FailedStop,  GPe_Proto_mean_CorrectStop,  GPe_Proto_std_CorrectStop  = np.load('../data/'+paramsA['loadFolder']+'/Proto_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    GPe_Proto2_mean_FailedStop, GPe_Proto2_std_FailedStop, GPe_Proto2_mean_CorrectStop, GPe_Proto2_std_CorrectStop = np.load('../data/'+paramsA['loadFolder']+'/Proto2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    GPe_Arky_mean_FailedStop,   GPe_Arky_std_FailedStop,   GPe_Arky_mean_CorrectStop,   GPe_Arky_std_CorrectStop   = np.load('../data/'+paramsA['loadFolder']+'/Arky_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    SNr_mean_FailedStop,        SNr_std_FailedStop,        SNr_mean_CorrectStop,        SNr_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SNr_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    Thal_mean_FailedStop,       Thal_std_FailedStop,       Thal_mean_CorrectStop,       Thal_std_CorrectStop       = np.load('../data/'+paramsA['loadFolder']+'/Thal_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')

    ### LOAD FAST/SLOW GO RATES
    StrD1_meanrate_FastGo,      StrD1_std_FastGo,       StrD1_meanrate_SlowGo,      StrD1_std_SlowGo        = np.load('../data/'+paramsA['loadFolder']+'/SD1_meanrate_std_Fast-Slow_Go'+netNr+'.npy')            
    StrD2_meanrate_FastGo,      StrD2_std_FastGo,       StrD2_meanrate_SlowGo,      StrD2_std_SlowGo        = np.load('../data/'+paramsA['loadFolder']+'/SD2_meanrate_std_Fast-Slow_Go'+netNr+'.npy')                        
    StrFSI_meanrate_FastGo,     StrFSI_std_FastGo,      StrFSI_meanrate_SlowGo,     StrFSI_std_SlowGo       = np.load('../data/'+paramsA['loadFolder']+'/FSI_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo     = np.load('../data/'+paramsA['loadFolder']+'/GPe_Arky_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo    = np.load('../data/'+paramsA['loadFolder']+'/GPe_Proto_meanrate_std_Fast-Slow_Go'+netNr+'.npy')


    ### FIGURE A: Z-SCORES, STOP CUE RESPONSE OF STRD2 AND STRFSI CORRECT_STOP/SLOW_GO
    tStart = (params['t_init'] + params['t_SSD'] + paramsA['STR_figAB_tmin'])/params['general_dt']
    tEnd   = (params['t_init'] + params['t_SSD'] + paramsA['STR_figAB_tmax'])/params['general_dt']
    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()    
    STRD2_all_mean = StrD2_meanrate_SlowGo
    STRFSI_all_mean = StrFSI_meanrate_SlowGo
    ## PLOT MEAN RATES
    plt.plot(custom_zscore_start0(SD2_mean_CorrectStop, tStart, STRD2_all_mean), color=params['toRGB']['purple'], lw=3)
    plt.plot(custom_zscore_start0(StrD2_meanrate_SlowGo, tStart, STRD2_all_mean), dash_capstyle='round', dashes=(0.05,2), color=params['toRGB']['purple']*0.7, lw=3) 
    plt.plot(custom_zscore_start0(FSI_mean_CorrectStop, tStart, STRFSI_all_mean), color=params['toRGB']['grey'], lw=3)     
    plt.plot(custom_zscore_start0(StrFSI_meanrate_SlowGo, tStart, STRFSI_all_mean), dash_capstyle='round', dashes=(0.05,2), color=params['toRGB']['grey']*0.7, lw=3) 
    ## STOP CUE MARKER
    plt.axvline((params['t_init'] + params['t_SSD'])/params['general_dt'], color='grey', lw=0.5)
    ## LABELS
    plt.xlabel('Time from Stop cue [s]', fontsize=paramsA['STR_fsize'])
    plt.ylabel('$\Delta$Firing rate (z-score)', fontsize=paramsA['STR_fsize'])
    ## LIMITS AND TICKS
    ax.axis([tStart, tEnd, paramsA['STR_figABC_ymin'], paramsA['STR_figABC_ymax']])
    ax.set_xticks([(params['t_init'] + params['t_SSD'] + i*1000)/params['general_dt'] for i in paramsA['STR_figAB_xticks']])
    ax.set_xticklabels(paramsA['STR_figAB_xticks'])
    ax.set_yticks(paramsA['STR_figABC_yticks'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(paramsA['STR_fsize'])
    ## LEGEND
    normalLine = mlines.Line2D([], [], color='k', label='Stop', lw=3)
    dashedLine = mlines.Line2D([], [], color='k', label='slow Go', dash_capstyle='round', dashes=(0.05,2), lw=3)
    purpleLine = mlines.Line2D([], [], color=params['toRGB']['purple'], label='StrD2', lw=3)
    greyLine = mlines.Line2D([], [], color=params['toRGB']['grey'], label='StrFSI', lw=3)
    plt.legend(handles=[dashedLine, normalLine, purpleLine, greyLine], bbox_to_anchor=(-0.02,-0.015), bbox_transform=plt.gca().transAxes, fontsize=10, loc='lower left')
    ## SAVE
    plt.tight_layout()
    plt.savefig(resultsDir+'/zscore_StopSTRD2FSI_paramsid'+str(params['general_id'])+'_'+str(paramsS['trials'])+'trials.png', dpi=300)
    

    ### FIGURE B: Z-SCORES, STOP CUE RESPONSE OF STRD1, GPE_Proto AND GPE_ARKY CORRECT_STOP/SLOW_GO
    tStart = (params['t_init'] + params['t_SSD'] + paramsA['STR_figAB_tmin'])/params['general_dt']
    tEnd   = (params['t_init'] + params['t_SSD'] + paramsA['STR_figAB_tmax'])/params['general_dt']
    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()
    STR_all_mean        = StrD1_meanrate_SlowGo
    GPe_Arky_all_mean   = GPe_Arky_meanrate_SlowGo
    GPe_Proto_all_mean  = GPe_Proto_meanrate_SlowGo
    ## PLOT MEAN RATES
    plt.plot(custom_zscore_start0(SD1_mean_CorrectStop, tStart, STR_all_mean), color=params['toRGB']['red'], lw=3)     
    plt.plot(custom_zscore_start0(StrD1_meanrate_SlowGo, tStart, STR_all_mean), dash_capstyle='round', dashes=(0.05,2), color=params['toRGB']['red']*0.7, lw=3)         
    plt.plot(custom_zscore_start0(GPe_Arky_mean_CorrectStop, tStart, GPe_Arky_all_mean), color=params['toRGB']['cyan'], lw=3)
    plt.plot(custom_zscore_start0(GPe_Arky_meanrate_SlowGo, tStart, GPe_Arky_all_mean), dash_capstyle='round', dashes=(0.05,2), color=params['toRGB']['cyan']*0.7, lw=3)
    plt.plot(custom_zscore_start0(GPe_Proto_mean_CorrectStop, tStart, GPe_Proto_all_mean), color=params['toRGB']['blue'], lw=3) 
    plt.plot(custom_zscore_start0(GPe_Proto_meanrate_SlowGo, tStart, GPe_Proto_all_mean), dash_capstyle='round', dashes=(0.05,2), color=params['toRGB']['blue']*0.7, lw=3)
    ## STOP CUE MARKER
    plt.axvline((params['t_init'] + params['t_SSD'])/params['general_dt'], color='grey', lw=0.5)
    ## LABELS
    plt.ylabel('$\Delta$Firing rate (z-score)', fontsize=paramsA['STR_fsize'])
    plt.xlabel('Time from Stop cue [s]', fontsize=paramsA['STR_fsize'])
    ## LIMITS AND TICKS
    ax.axis([tStart, tEnd, paramsA['STR_figABC_ymin'], paramsA['STR_figABC_ymax']])
    ax.set_xticks([(params['t_init'] + params['t_SSD'] + i*1000)/params['general_dt'] for i in paramsA['STR_figAB_xticks']])
    ax.set_xticklabels(paramsA['STR_figAB_xticks'])
    ax.set_yticks(paramsA['STR_figABC_yticks'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(paramsA['STR_fsize'])
    ## LEGEND
    normalLine = mlines.Line2D([], [], color='k', label='Stop', lw=3)
    dashedLine = mlines.Line2D([], [], color='k', label='slow Go', dash_capstyle='round', dashes=(0.05,2), lw=3)
    redLine = mlines.Line2D([], [], color=params['toRGB']['red'], label='StrD1', lw=3)
    cyanLine = mlines.Line2D([], [], color=params['toRGB']['cyan'], label='GPe-Arky', lw=3)
    blueLine = mlines.Line2D([], [], color=params['toRGB']['blue'], label='GPe-Proto', lw=3)
    plt.legend(handles=[dashedLine, normalLine, redLine, cyanLine, blueLine], bbox_to_anchor=(-0.02,1.015), bbox_transform=plt.gca().transAxes, fontsize=10, loc='upper left')
    ## SAVE
    plt.tight_layout()
    plt.savefig(resultsDir+'/zscore_StopSTR_paramsid'+str(params['general_id'])+'_'+str(paramsS['trials'])+'trials.png', dpi=300)
    
    
    ### FIGURE C: Z-SCORES, STOP CUE RESPONSE OF STN, SNR, GPE_PROTO AND GPE_ARKY CORRECT_STOP
    tStart = (params['t_init'] + params['t_SSD'] + paramsA['STR_figC_tmin'])/params['general_dt']
    tEnd   = (params['t_init'] + params['t_SSD'] + paramsA['STR_figC_tmax'])/params['general_dt']
    plt.figure(figsize=(3.5,4), dpi=300)
    ax=plt.gca()
    ## PLOT MEAN RATES
    plt.plot(custom_zscore_start0(STN_mean_CorrectStop, tStart, STN_mean_CorrectStop), color=params['toRGB']['gold'], lw=3) 
    plt.plot(custom_zscore_start0(SNr_mean_CorrectStop, tStart, SNr_mean_CorrectStop), color=params['toRGB']['orange'], lw=3) 
    plt.plot(custom_zscore_start0(GPe_Arky_mean_CorrectStop, tStart, GPe_Arky_mean_CorrectStop), color=params['toRGB']['cyan'], lw=3) 
    plt.plot(custom_zscore_start0(GPe_Proto_mean_CorrectStop, tStart, GPe_Proto_mean_CorrectStop), color=params['toRGB']['blue'], lw=3) 
    ## STOP CUE MARKER
    plt.axvline((params['t_init'] + params['t_SSD'])/params['general_dt'], color='grey', lw=0.5)
    ## LABELS
    plt.xlabel('Time from Stop cue [s]', fontsize=paramsA['STR_fsize'])
    plt.ylabel('$\Delta$Firing rate (z-score)', fontsize=paramsA['STR_fsize'])
    ## LIMITS AND TICKS
    ax.axis([tStart, tEnd, paramsA['STR_figABC_ymin'], paramsA['STR_figABC_ymax']])  
    ax.set_xticks([(params['t_init'] + params['t_SSD'] + i*1000)/params['general_dt'] for i in paramsA['STR_figC_xticks']])
    ax.set_xticklabels(paramsA['STR_figC_xticks'])
    ax.set_yticks(paramsA['STR_figABC_yticks'])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(paramsA['STR_fsize'])
    ## LEGEND
    plt.legend(('STN Stop', 'SNr Stop', 'GPe-Arky Stop', 'GPe-Proto Stop'), bbox_to_anchor=(-0.02,-0.015), bbox_transform=plt.gca().transAxes, fontsize=10, loc='lower left') # , 'Stop cue'
    ## SAVE
    plt.tight_layout()
    plt.savefig(resultsDir+'/zscore_StopDetail_paramsid'+str(params['general_id'])+'_'+str(paramsS['trials'])+'trials.png', dpi=300)
    
    
    ### FIGURE D: Z-SCORES, GO AND STOP CUE RESPONSES OF MULTIPLE POPS DURING CORRECT_STOP
    plot_zscore_stopVsGo_NewData(STN_mean_CorrectStop, STN_mean_CorrectStop, SNr_mean_CorrectStop, SNr_mean_CorrectStop, GPe_Arky_mean_CorrectStop, GPe_Arky_mean_CorrectStop, GPe_Proto_mean_CorrectStop, GPe_Proto_mean_CorrectStop, params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], resultsDir+'/', labels = ['STN', 'SNr', 'GPe-Arky', 'GPe-Proto'], linecol = [[params['toRGB']['gold'], params['toRGB']['gold']*0.7], [params['toRGB']['orange'], params['toRGB']['orange']*0.7], [params['toRGB']['cyan'],params['toRGB']['cyan']*0.7], [params['toRGB']['blue'],params['toRGB']['blue']*0.7]])
    plot_zscore_stopVsGo_five_NewData(SD1_mean_CorrectStop, SD1_mean_CorrectStop, SD2_mean_CorrectStop, SD2_mean_CorrectStop, FSI_mean_CorrectStop, FSI_mean_CorrectStop, GPe_Proto2_mean_CorrectStop, GPe_Proto2_mean_CorrectStop, Thal_mean_CorrectStop, Thal_mean_CorrectStop, params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], resultsDir+'/', labels=['StrD1', 'StrD2', 'StrFSI', 'GPe-Cp', 'Thalamus'], linecol=[[params['toRGB']['red'], params['toRGB']['red']*0.7], [params['toRGB']['purple'], params['toRGB']['purple']*0.7], [params['toRGB']['grey'], params['toRGB']['grey']*0.7], [params['toRGB']['light brown'], params['toRGB']['light brown']*0.7], [params['toRGB']['lime'], params['toRGB']['lime']*0.7]])                             


##############################################################################################
if abfrage_loop == 'extra':

    ### LOAD INTEGRATOR, CALCULATE FAILED/CORRECT STOP IDX
    mInt_Go = np.load('../data/'+paramsA['loadFolder']+'/Integrator_ampa_Go_'+netNr+'_id'+str(params['general_id'])+'.npy')
    mInt_stop = np.load('../data/'+paramsA['loadFolder']+'/Integrator_ampa_Stop_'+netNr+'_id'+str(params['general_id'])+'.npy')
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(paramsS['trials'])), paramsS['trials']], order='F')
    nz_FailedStop, nz_CorrectStop = get_correct_failed_stop_trials(mInt_stop, params['IntegratorNeuron_threshold'], paramsS['trials'])

    ### LOAD FAILED/CORRECT STOP RATES
    SD1_mean_FailedStop,        SD1_std_FailedStop,        SD1_mean_CorrectStop,        SD1_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SD1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    SD2_mean_FailedStop,        SD2_std_FailedStop,        SD2_mean_CorrectStop,        SD2_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SD2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    FSI_mean_FailedStop,        FSI_std_FailedStop,        FSI_mean_CorrectStop,        FSI_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/FSI_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    STN_mean_FailedStop,        STN_std_FailedStop,        STN_mean_CorrectStop,        STN_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/STN_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    GPe_Proto_mean_FailedStop,  GPe_Proto_std_FailedStop,  GPe_Proto_mean_CorrectStop,  GPe_Proto_std_CorrectStop  = np.load('../data/'+paramsA['loadFolder']+'/Proto_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    GPe_Proto2_mean_FailedStop, GPe_Proto2_std_FailedStop, GPe_Proto2_mean_CorrectStop, GPe_Proto2_std_CorrectStop = np.load('../data/'+paramsA['loadFolder']+'/Proto2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    GPe_Arky_mean_FailedStop,   GPe_Arky_std_FailedStop,   GPe_Arky_mean_CorrectStop,   GPe_Arky_std_CorrectStop   = np.load('../data/'+paramsA['loadFolder']+'/Arky_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    SNr_mean_FailedStop,        SNr_std_FailedStop,        SNr_mean_CorrectStop,        SNr_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SNr_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    Thal_mean_FailedStop,       Thal_std_FailedStop,       Thal_mean_CorrectStop,       Thal_std_CorrectStop       = np.load('../data/'+paramsA['loadFolder']+'/Thal_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    SNrE_mean_FailedStop,       SNrE_std_FailedStop,       SNrE_mean_CorrectStop,       SNrE_std_CorrectStop       = np.load('../data/'+paramsA['loadFolder']+'/SNrE_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    Cortex_G_mean_FailedStop,   Cortex_G_std_FailedStop,   Cortex_G_mean_CorrectStop,   Cortex_G_std_CorrectStop   = np.load('../data/'+paramsA['loadFolder']+'/Cortex_G_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    Cortex_S_mean_FailedStop,   Cortex_S_std_FailedStop,   Cortex_S_mean_CorrectStop,   Cortex_S_std_CorrectStop   = np.load('../data/'+paramsA['loadFolder']+'/Cortex_S_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    PauseInput_mean_FailedStop, PauseInput_std_FailedStop, PauseInput_mean_CorrectStop, PauseInput_std_CorrectStop = np.load('../data/'+paramsA['loadFolder']+'/Stoppinput1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    

    ### LOAD PVALUES
    pvalue_list, pvalue_times = np.load('../data/'+paramsA['loadFolder']+'/p_value_list_times_'+str(params['general_id'])+netNr+'.npy', allow_pickle=True)
    pvalue_list2 = {}
    nameList = ['failedStop_vs_allGo', 'failedStop_vs_fastGo', 'failedStop_vs_slowGo']
    for name in nameList:
        pvalue_list2[name], pvalue_times = np.load('../data/'+paramsA['loadFolder']+'/p_value_list_'+name+'_times_'+str(params['general_id'])+netNr+'.npy', allow_pickle=True)

    ### LOAD GO RATES
    GPe_Arky_mean_Go,   GPe_Arky_std_Go     = np.load('../data/'+paramsA['loadFolder']+'/GPeArky_rate_Go_mean_std'+netNr+'.npy')
    GPe_Proto_mean_Go,  GPe_Proto_std_Go    = np.load('../data/'+paramsA['loadFolder']+'/GPeProto_rate_Go_mean_std'+netNr+'.npy')  
    STR_D1_mean_Go,     STR_D1_std_Go       = np.load('../data/'+paramsA['loadFolder']+'/SD1_rate_Go_mean_std'+netNr+'.npy')     
    STR_D2_mean_Go,     STR_D2_std_Go       = np.load('../data/'+paramsA['loadFolder']+'/SD2_rate_Go_mean_std'+netNr+'.npy')        
    STN_mean_Go,        STN_std_Go          = np.load('../data/'+paramsA['loadFolder']+'/STN_rate_Go_mean_std'+netNr+'.npy')
    SNr_mean_Go,        SNr_std_Go          = np.load('../data/'+paramsA['loadFolder']+'/SNr_rate_Go_mean_std'+netNr+'.npy')            
    Thal_mean_Go,       Thal_std_Go         = np.load('../data/'+paramsA['loadFolder']+'/Thal_rate_Go_mean_std'+netNr+'.npy')
    Cortex_G_mean_Go,   Cortex_G_std_Go     = np.load('../data/'+paramsA['loadFolder']+'/Cortex_G_rate_Go_mean_std'+netNr+'.npy')
    GPe_Proto2_mean_Go, GPe_Proto2_std_Go   = np.load('../data/'+paramsA['loadFolder']+'/GPeProto2_rate_Go_mean_std'+netNr+'.npy')
    PauseInput_mean_Go, PauseInput_std_Go   = np.load('../data/'+paramsA['loadFolder']+'/Stoppinput1_rate_Go_mean_std'+netNr+'.npy')
    Cortex_S_mean_Go,   Cortex_S_std_Go     = np.load('../data/'+paramsA['loadFolder']+'/Cortex_S_rate_Go_mean_std'+netNr+'.npy')
    STR_FSI_mean_Go,    STR_FSI_std_Go      = np.load('../data/'+paramsA['loadFolder']+'/FSI_rate_Go_mean_std'+netNr+'.npy')            

    ### LOAD FAST/SLOW GO RATES
    GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo     = np.load('../data/'+paramsA['loadFolder']+'/GPe_Arky_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo    = np.load('../data/'+paramsA['loadFolder']+'/GPe_Proto_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    STR_D1_meanrate_FastGo,     STR_D1_std_FastGo,      STR_D1_meanrate_SlowGo,     STR_D1_std_SlowGo       = np.load('../data/'+paramsA['loadFolder']+'/SD1_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    STR_D2_meanrate_FastGo,     STR_D2_std_FastGo,      STR_D2_meanrate_SlowGo,     STR_D2_std_SlowGo       = np.load('../data/'+paramsA['loadFolder']+'/SD2_meanrate_std_Fast-Slow_Go'+netNr+'.npy')            
    STN_meanrate_FastGo,        STN_std_FastGo,         STN_meanrate_SlowGo,        STN_std_SlowGo          = np.load('../data/'+paramsA['loadFolder']+'/STN_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    SNr_meanrate_FastGo,        SNr_std_FastGo,         SNr_meanrate_SlowGo,        SNr_std_SlowGo          = np.load('../data/'+paramsA['loadFolder']+'/SNr_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    Thal_meanrate_FastGo,       Thal_std_FastGo,        Thal_meanrate_SlowGo,       Thal_std_SlowGo         = np.load('../data/'+paramsA['loadFolder']+'/Thal_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    Cortex_G_meanrate_FastGo,   Cortex_G_std_FastGo,    Cortex_G_meanrate_SlowGo,   Cortex_G_std_SlowGo     = np.load('../data/'+paramsA['loadFolder']+'/Cortex_G_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    GPe_Proto2_meanrate_FastGo, GPe_Proto2_std_FastGo,  GPe_Proto2_meanrate_SlowGo, GPe_Proto2_std_SlowGo   = np.load('../data/'+paramsA['loadFolder']+'/GPe_Proto2_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    PauseInput_meanrate_FastGo, PauseInput_std_FastGo,  PauseInput_meanrate_SlowGo, PauseInput_std_SlowGo   = np.load('../data/'+paramsA['loadFolder']+'/PauseInput_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    Cortex_S_meanrate_FastGo,   Cortex_S_std_FastGo,    Cortex_S_meanrate_SlowGo,   Cortex_S_std_SlowGo     = np.load('../data/'+paramsA['loadFolder']+'/Cortex_S_meanrate_std_Fast-Slow_Go'+netNr+'.npy')                        
    STR_FSI_meanrate_FastGo,    STR_FSI_std_FastGo,     STR_FSI_meanrate_SlowGo,    STR_FSI_std_SlowGo      = np.load('../data/'+paramsA['loadFolder']+'/FSI_meanrate_std_Fast-Slow_Go'+netNr+'.npy')

    ###TODO: PVALUES (generated in next functions) files is saved in srcAna
    ###TODO: CLEAN THESE FUNCTIONS
    ### FIGURE A: FAILED VS CORRECT STOPS
    plot_meanrate_All_FailedVsCorrectStop(resultsDir+'/', \
                                          GPe_Arky_mean_FailedStop, GPe_Arky_mean_CorrectStop, GPe_Arky_std_FailedStop, GPe_Arky_std_CorrectStop, \
                                          GPe_Proto_mean_FailedStop, GPe_Proto_mean_CorrectStop, \
                                          SD1_mean_FailedStop, SD1_mean_CorrectStop, \
                                          SD2_mean_FailedStop, SD2_mean_CorrectStop, \
                                          STN_mean_FailedStop, STN_mean_CorrectStop, \
                                          SNr_mean_FailedStop, SNr_mean_CorrectStop, \
                                          Thal_mean_FailedStop, Thal_mean_CorrectStop, \
                                          rsp_mInt_stop, nz_FailedStop, nz_CorrectStop, \
                                          params['IntegratorNeuron_threshold'], \
                                          Cortex_G_mean_FailedStop, Cortex_G_mean_CorrectStop, \
                                          GPe_Proto2_mean_FailedStop, GPe_Proto2_mean_CorrectStop, \
                                          PauseInput_mean_FailedStop, PauseInput_mean_CorrectStop, \
                                          Cortex_S_mean_FailedStop, Cortex_S_mean_CorrectStop, \
                                          SNrE_mean_FailedStop, SNrE_mean_CorrectStop, \
                                          FSI_mean_FailedStop, FSI_mean_CorrectStop, \
                                          params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], pvalue_list, pvalue_times)


    ### FIGURE B: FAILED STOP VS GO
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
                                          params['IntegratorNeuron_threshold'], \
                                          params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], pvalue_list2, pvalue_times, GO_Mode='fast')



##############################################################################################
if abfrage_loop=='RT':

    ### LOAD INTEGRATOR, CALCULATE IDX FAILED/CORRECT/FAST/SLOW GO/STOP
    mInt_stop = np.load('../data/'+paramsA['loadFolder']+'/Integrator_ampa_Stop_'+netNr+'_id'+str(params['general_id'])+'.npy')
    mInt_go = np.load('../data/'+paramsA['loadFolder']+'/Integrator_ampa_Go_'+netNr+'_id'+str(params['general_id'])+'.npy')            
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(paramsS['trials'])), paramsS['trials']], order='F')
    rsp_mInt_go = np.reshape(mInt_go, [int(mInt_go.shape[0] / float(paramsS['trials'])), paramsS['trials']], order='F')
    mInt_maxpertrial = np.nanmax(rsp_mInt_go, 0)
    nz_FailedGo = np.nonzero(mInt_maxpertrial < params['IntegratorNeuron_threshold'])[0]
    nz_CorrectGo = np.nonzero(mInt_maxpertrial >= params['IntegratorNeuron_threshold'])[0]
    mInt_maxpertrial = np.nanmax(rsp_mInt_stop, 0)
    nz_FailedStop = np.nonzero(mInt_maxpertrial >= params['IntegratorNeuron_threshold'])[0]
    nz_CorrectStop = np.nonzero(mInt_maxpertrial < params['IntegratorNeuron_threshold'])[0]
    nz_FastGo, nz_SlowGo = get_fast_slow_go_trials(mInt_go, mInt_stop, params['IntegratorNeuron_threshold'], paramsS['trials'])

    ### SAVE FAILED/CORRECT/FAST/SLOW ETC NUMBERS
    with open(resultsDir+'/stops_and_gos_'+str(paramsS['trials'])+'trials_paramsid'+str(params['general_id'])+'.txt', 'w') as f:
        print('correct Stops:',len(nz_CorrectStop),file=f)
        print('failed Stops:',len(nz_FailedStop),file=f)
        print('correct Gos:',len(nz_CorrectGo),file=f)
        print('failed Gos:',len(nz_FailedGo),file=f)
        print('slow Gos:',len(nz_SlowGo),file=f)
        print('fast Gos:',len(nz_FastGo),file=f)

    ### CALCULATE RT GO AND STOP
    RT_Go = np.nan * np.ones(paramsS['trials'])
    RT_Stop = np.nan * np.ones(paramsS['trials'])
    for i_trial in range(paramsS['trials']):
        if np.nanmax(rsp_mInt_go[:, i_trial]) >= params['IntegratorNeuron_threshold']: 
            RT_Go[i_trial] = np.nonzero(rsp_mInt_go[:, i_trial] >= params['IntegratorNeuron_threshold'])[0][0]           
        if np.nanmax(rsp_mInt_stop[:, i_trial]) >= params['IntegratorNeuron_threshold']: 
            RT_Stop[i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= params['IntegratorNeuron_threshold'])[0][0]              

    
    ### FIGURE A: REACTIONTIME DISTRIBUTION
    plt.figure(figsize=(3.5,4), dpi=300)
    fsize = 12
    nz_Go = np.nonzero(np.isnan(RT_Go)==False)
    nz_Stop = np.nonzero(np.isnan(RT_Stop)==False)
    binSize=15
    counts_Go, bins_Go = np.histogram(RT_Go[nz_Go] * params['general_dt'], np.arange(params['t_init'],params['t_init']+501,binSize)) #
    counts_Stop, bins_Stop = np.histogram(RT_Stop[nz_Stop] * params['general_dt'], np.arange(params['t_init'],params['t_init']+501,binSize)) # 
    if counts_Go.max() > 0:
        plt.bar(bins_Go[:-1], counts_Go/counts_Go.max(), width=binSize, alpha=1, color=np.array([107,139,164])/255.)
    if counts_Stop.max() > 0:
        plt.bar(bins_Stop[:-1], counts_Stop/counts_Stop.max(), width=binSize, alpha=0.5, color='tomato')
    mean_CorrectGo = np.round( (np.nanmean(RT_Go[nz_Go]) - params['t_init']/params['general_dt'])*params['general_dt'], 1)
    mean_FailedStop = np.round( (np.nanmean(RT_Stop[nz_Stop]) - params['t_init']/params['general_dt'])*params['general_dt'], 1)
    plt.legend(('correct Go, mean RT='+str(mean_CorrectGo),'failed Stop, mean RT='+str(mean_FailedStop)), fontsize=fsize-2)
    plt.xlabel('Reaction time [ms]', fontsize=fsize)    
    plt.ylabel('Normalized trial count', fontsize=fsize)
    ax = plt.gca()
    ax.set_yticks([0.0,0.5,1.0])
    ax.axis([params['t_init'], (500 + params['t_init']), ax.axis()[2], 1.22])
    xtks = ax.get_xticks()
    xlabels = []
    for i in range(len(xtks)):
        xlabels.append(str( int( xtks[i]-params['t_init'] ) ))
    ax.set_xticklabels(xlabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.tight_layout()
    plt.savefig(resultsDir+'/Reaction_times_paramsid'+str(params['general_id'])+'.png')


    ### FIGURE B: INTEGRATOR
    plt.figure(figsize=(3.5,4), dpi=300)
    ## INTEGRATOR LINES
    plt.plot(rsp_mInt_go[:, :], color=np.array([107,139,164])/255., lw=1, label='correct Go')
    if len(nz_FailedStop) > 0:
        plt.plot(rsp_mInt_stop[:, nz_FailedStop], color='tomato', lw=1, label = 'failed Stop', linestyle='--')
    if len(nz_CorrectStop) > 0:
        plt.plot(rsp_mInt_stop[:, nz_CorrectStop], color='purple', lw=1, label = 'correct Stop')
    ## GOCUE MARKER
    plt.axvline(params['t_init'] / params['general_dt'], color='green', linestyle=':')
    GoCueHandle = mlines.Line2D([], [], color='green', label='Go cue', linestyle=':')
    ## STOPCUE MARKER
    plt.axvline((params['t_init'] + params['t_SSD']) / params['general_dt'], color='orange', linestyle=':')
    StopCueHandle = mlines.Line2D([], [], color='orange', label='Stop cue', linestyle=':')
    ## THRESHOLD MARKER
    plt.axhline(params['IntegratorNeuron_threshold'], color='grey', linestyle=':')
    ThresholdHandle = mlines.Line2D([], [], color='grey', label='Threshold', linestyle=':')
    ## LINESTYLES FOR LEGEND TODO: SAVE THIS RGB COLOR AS GENERAL PARAMETER, MAYBE ALL COLORS
    CorrectGoHandle = mlines.Line2D([], [], color=np.array([107,139,164])/255., label='correct Go', linestyle='-')
    FailedStopHandle = mlines.Line2D([], [], color='tomato', label='failed Stop', linestyle='--')
    CorrectStopHandle = mlines.Line2D([], [], color='purple', label='correct Stop', linestyle='-')
    ## LEGEND
    plt.legend(handles = [ThresholdHandle, GoCueHandle, StopCueHandle, CorrectGoHandle, FailedStopHandle, CorrectStopHandle], loc='upper left', fontsize=fsize-2)
    ## TITLE
    plt.title(str(100 * len(nz_CorrectStop) / (len(nz_CorrectStop) + len(nz_FailedStop)) )+'% correct Stop trials' , fontsize=fsize) #fsize)
    ## LABEL
    plt.xlabel('Time from Go cue [ms]', fontsize=fsize)    
    plt.ylabel('Integrator value', fontsize=fsize)
    ## LIMITS AND TICKS
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
    ## SAVE
    plt.tight_layout()
    plt.savefig(resultsDir+'/Integrator_ampa_Stop_'+str(paramsS['trials'])+'trials_paramsid'+str(params['general_id'])+'.png')
