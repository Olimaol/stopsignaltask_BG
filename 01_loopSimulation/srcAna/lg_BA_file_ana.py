"""
Created on Fri Oct 16 2020

@authors: Oliver Maith, Lorenz Goenner, ilko
"""

import pylab as plt
import os
import sys
import numpy as np
import matplotlib.lines as mlines

from BGmodelSST.analysis import get_correct_failed_stop_trials, custom_zscore, custom_zscore_start0, plot_zscore_stopVsGo_NewData, plot_zscore_stopVsGo_five_NewData, plot_meanrate_All_FailedVsCorrectStop, plot_meanrate_All_FailedStopVsCorrectGo, get_fast_slow_go_trials

from analysisParams import paramsA


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
if abfrage_loop == 'STR':#FIGURE 5 AND 6
    
    ### LOAD FAILED/CORRECT STOP RATES
    meanRates_failedStop = {}
    meanRates_correctStop = {}
    meanRates_failedStop['StrD1'],        SD1_std_FailedStop,        meanRates_correctStop['StrD1'],        SD1_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SD1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    meanRates_failedStop['StrD2'],        SD2_std_FailedStop,        meanRates_correctStop['StrD2'],        SD2_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SD2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    meanRates_failedStop['StrFSI'],        FSI_std_FailedStop,        meanRates_correctStop['StrFSI'],        FSI_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/FSI_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    meanRates_failedStop['STN'],        STN_std_FailedStop,        meanRates_correctStop['STN'],        STN_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/STN_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    meanRates_failedStop['GPeProto'],  GPe_Proto_std_FailedStop,  meanRates_correctStop['GPeProto'],  GPe_Proto_std_CorrectStop  = np.load('../data/'+paramsA['loadFolder']+'/Proto_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    meanRates_failedStop['GPeCp'], GPe_Proto2_std_FailedStop, meanRates_correctStop['GPeCp'], GPe_Proto2_std_CorrectStop = np.load('../data/'+paramsA['loadFolder']+'/Proto2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    meanRates_failedStop['GPeArky'],   GPe_Arky_std_FailedStop,   meanRates_correctStop['GPeArky'],   GPe_Arky_std_CorrectStop   = np.load('../data/'+paramsA['loadFolder']+'/Arky_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    meanRates_failedStop['SNr'],        SNr_std_FailedStop,        meanRates_correctStop['SNr'],        SNr_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SNr_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    meanRates_failedStop['Thal'],       Thal_std_FailedStop,       meanRates_correctStop['Thal'],       Thal_std_CorrectStop       = np.load('../data/'+paramsA['loadFolder']+'/Thal_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')

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
    plt.plot(custom_zscore_start0(meanRates_correctStop['StrD2'], tStart, STRD2_all_mean), color=params['toRGB']['purple'], lw=3)
    plt.plot(custom_zscore_start0(StrD2_meanrate_SlowGo, tStart, STRD2_all_mean), dash_capstyle='round', dashes=(0.05,2), color=params['toRGB']['purple']*0.7, lw=3) 
    plt.plot(custom_zscore_start0(meanRates_correctStop['StrFSI'], tStart, STRFSI_all_mean), color=params['toRGB']['grey'], lw=3)     
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
    plt.plot(custom_zscore_start0(meanRates_correctStop['StrD1'], tStart, STR_all_mean), color=params['toRGB']['red'], lw=3)     
    plt.plot(custom_zscore_start0(StrD1_meanrate_SlowGo, tStart, STR_all_mean), dash_capstyle='round', dashes=(0.05,2), color=params['toRGB']['red']*0.7, lw=3)         
    plt.plot(custom_zscore_start0(meanRates_correctStop['GPeArky'], tStart, GPe_Arky_all_mean), color=params['toRGB']['cyan'], lw=3)
    plt.plot(custom_zscore_start0(GPe_Arky_meanrate_SlowGo, tStart, GPe_Arky_all_mean), dash_capstyle='round', dashes=(0.05,2), color=params['toRGB']['cyan']*0.7, lw=3)
    plt.plot(custom_zscore_start0(meanRates_correctStop['GPeProto'], tStart, GPe_Proto_all_mean), color=params['toRGB']['blue'], lw=3) 
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
    plt.plot(custom_zscore_start0(meanRates_correctStop['STN'], tStart, meanRates_correctStop['STN']), color=params['toRGB']['gold'], lw=3) 
    plt.plot(custom_zscore_start0(meanRates_correctStop['SNr'], tStart, meanRates_correctStop['SNr']), color=params['toRGB']['orange'], lw=3) 
    plt.plot(custom_zscore_start0(meanRates_correctStop['GPeArky'], tStart, meanRates_correctStop['GPeArky']), color=params['toRGB']['cyan'], lw=3) 
    plt.plot(custom_zscore_start0(meanRates_correctStop['GPeProto'], tStart, meanRates_correctStop['GPeProto']), color=params['toRGB']['blue'], lw=3) 
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
    plot_zscore_stopVsGo_NewData(meanRates_correctStop['STN'], meanRates_correctStop['STN'], meanRates_correctStop['SNr'], meanRates_correctStop['SNr'], meanRates_correctStop['GPeArky'], meanRates_correctStop['GPeArky'], meanRates_correctStop['GPeProto'], meanRates_correctStop['GPeProto'], params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], resultsDir+'/', labels = ['STN', 'SNr', 'GPe-Arky', 'GPe-Proto'], linecol = [[params['toRGB']['gold'], params['toRGB']['gold']*0.7], [params['toRGB']['orange'], params['toRGB']['orange']*0.7], [params['toRGB']['cyan'],params['toRGB']['cyan']*0.7], [params['toRGB']['blue'],params['toRGB']['blue']*0.7]])
    plot_zscore_stopVsGo_five_NewData(meanRates_correctStop['StrD1'], meanRates_correctStop['StrD1'], meanRates_correctStop['StrD2'], meanRates_correctStop['StrD2'], meanRates_correctStop['StrFSI'], meanRates_correctStop['StrFSI'], meanRates_correctStop['GPeCp'], meanRates_correctStop['GPeCp'], meanRates_correctStop['Thal'], meanRates_correctStop['Thal'], params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], resultsDir+'/', labels=['StrD1', 'StrD2', 'StrFSI', 'GPe-Cp', 'Thalamus'], linecol=[[params['toRGB']['red'], params['toRGB']['red']*0.7], [params['toRGB']['purple'], params['toRGB']['purple']*0.7], [params['toRGB']['grey'], params['toRGB']['grey']*0.7], [params['toRGB']['light brown'], params['toRGB']['light brown']*0.7], [params['toRGB']['lime'], params['toRGB']['lime']*0.7]])                             


##############################################################################################
if abfrage_loop == 'extra':#Figure 7 and 11

    ### LOAD INTEGRATOR, CALCULATE FAILED/CORRECT STOP IDX
    mInt_Go = np.load('../data/'+paramsA['loadFolder']+'/Integrator_ampa_Go_'+netNr+'_id'+str(params['general_id'])+'.npy')
    mInt_stop = np.load('../data/'+paramsA['loadFolder']+'/Integrator_ampa_Stop_'+netNr+'_id'+str(params['general_id'])+'.npy')
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(paramsS['trials'])), paramsS['trials']], order='F')
    nz_FailedStop, nz_CorrectStop = get_correct_failed_stop_trials(mInt_stop, params['IntegratorNeuron_threshold'], paramsS['trials'])

    ### LOAD FAILED/CORRECT STOP RATES
    meanRates_failedStop = {}
    meanRates_correctStop = {}
    meanRates_failedStop['StrD1'],        SD1_std_FailedStop,        meanRates_correctStop['StrD1'],        SD1_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SD1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    meanRates_failedStop['StrD2'],        SD2_std_FailedStop,        meanRates_correctStop['StrD2'],        SD2_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SD2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    meanRates_failedStop['StrFSI'],        FSI_std_FailedStop,        meanRates_correctStop['StrFSI'],        FSI_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/FSI_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    meanRates_failedStop['STN'],        STN_std_FailedStop,        meanRates_correctStop['STN'],        STN_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/STN_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    meanRates_failedStop['GPeProto'],  GPe_Proto_std_FailedStop,  meanRates_correctStop['GPeProto'],  GPe_Proto_std_CorrectStop  = np.load('../data/'+paramsA['loadFolder']+'/Proto_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    meanRates_failedStop['GPeCp'], GPe_Proto2_std_FailedStop, meanRates_correctStop['GPeCp'], GPe_Proto2_std_CorrectStop = np.load('../data/'+paramsA['loadFolder']+'/Proto2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    meanRates_failedStop['GPeArky'],   GPe_Arky_std_FailedStop,   meanRates_correctStop['GPeArky'],   GPe_Arky_std_CorrectStop   = np.load('../data/'+paramsA['loadFolder']+'/Arky_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')        
    meanRates_failedStop['SNr'],        SNr_std_FailedStop,        meanRates_correctStop['SNr'],        SNr_std_CorrectStop        = np.load('../data/'+paramsA['loadFolder']+'/SNr_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    meanRates_failedStop['Thal'],       Thal_std_FailedStop,       meanRates_correctStop['Thal'],       Thal_std_CorrectStop       = np.load('../data/'+paramsA['loadFolder']+'/Thal_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    meanRates_failedStop['SNrE'],       SNrE_std_FailedStop,       SNrE_mean_CorrectStop,       SNrE_std_CorrectStop       = np.load('../data/'+paramsA['loadFolder']+'/SNrE_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')
    meanRates_failedStop['cortexGo'],   Cortex_G_std_FailedStop,   meanRates_correctStop['cortexGo'],   Cortex_G_std_CorrectStop   = np.load('../data/'+paramsA['loadFolder']+'/Cortex_G_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    meanRates_failedStop['cortexStop'],   Cortex_S_std_FailedStop,   meanRates_correctStop['cortexStop'],   Cortex_S_std_CorrectStop   = np.load('../data/'+paramsA['loadFolder']+'/Cortex_S_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    
    meanRates_failedStop['cortexPause'], PauseInput_std_FailedStop, PauseInput_mean_CorrectStop, PauseInput_std_CorrectStop = np.load('../data/'+paramsA['loadFolder']+'/Stoppinput1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy')    

    ### LOAD PVALUES
    pvalue_list, pvalue_times = np.load('../data/'+paramsA['loadFolder']+'/p_value_list_times_'+str(params['general_id'])+netNr+'.npy', allow_pickle=True)
    pvalue_list2 = {}
    nameList = ['failedStop_vs_allGo', 'failedStop_vs_fastGo', 'failedStop_vs_slowGo']
    for name in nameList:
        pvalue_list2[name], pvalue_times = np.load('../data/'+paramsA['loadFolder']+'/p_value_list_'+name+'_times_'+str(params['general_id'])+netNr+'.npy', allow_pickle=True)

    ### LOAD GO RATES
    meanRates_Go = {}
    meanRates_Go['GPeArky'],   GPe_Arky_std_Go     = np.load('../data/'+paramsA['loadFolder']+'/GPeArky_rate_Go_mean_std'+netNr+'.npy')
    meanRates_Go['GPeProto'],  GPe_Proto_std_Go    = np.load('../data/'+paramsA['loadFolder']+'/GPeProto_rate_Go_mean_std'+netNr+'.npy')  
    meanRates_Go['StrD1'],     STR_D1_std_Go       = np.load('../data/'+paramsA['loadFolder']+'/SD1_rate_Go_mean_std'+netNr+'.npy')     
    meanRates_Go['StrD2'],     STR_D2_std_Go       = np.load('../data/'+paramsA['loadFolder']+'/SD2_rate_Go_mean_std'+netNr+'.npy')        
    meanRates_Go['STN'],        STN_std_Go          = np.load('../data/'+paramsA['loadFolder']+'/STN_rate_Go_mean_std'+netNr+'.npy')
    meanRates_Go['SNr'],        SNr_std_Go          = np.load('../data/'+paramsA['loadFolder']+'/SNr_rate_Go_mean_std'+netNr+'.npy')            
    meanRates_Go['Thal'],       Thal_std_Go         = np.load('../data/'+paramsA['loadFolder']+'/Thal_rate_Go_mean_std'+netNr+'.npy')
    meanRates_Go['cortexGo'],   Cortex_G_std_Go     = np.load('../data/'+paramsA['loadFolder']+'/Cortex_G_rate_Go_mean_std'+netNr+'.npy')
    meanRates_Go['GPeCp'], GPe_Proto2_std_Go   = np.load('../data/'+paramsA['loadFolder']+'/GPeProto2_rate_Go_mean_std'+netNr+'.npy')
    meanRates_Go['cortexPause'], PauseInput_std_Go   = np.load('../data/'+paramsA['loadFolder']+'/Stoppinput1_rate_Go_mean_std'+netNr+'.npy')
    meanRates_Go['cortexStop'],   Cortex_S_std_Go     = np.load('../data/'+paramsA['loadFolder']+'/Cortex_S_rate_Go_mean_std'+netNr+'.npy')
    meanRates_Go['StrFSI'],    STR_FSI_std_Go      = np.load('../data/'+paramsA['loadFolder']+'/FSI_rate_Go_mean_std'+netNr+'.npy')            

    ### LOAD FAST/SLOW GO RATES
    meanRates_fastGo = {}
    meanRates_slowGo = {}
    meanRates_fastGo['GPeArky'],   GPe_Arky_std_FastGo,    meanRates_slowGo['GPeArky'],   GPe_Arky_std_SlowGo     = np.load('../data/'+paramsA['loadFolder']+'/GPe_Arky_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    meanRates_fastGo['GPeProto'],  GPe_Proto_std_FastGo,   meanRates_slowGo['GPeProto'],  GPe_Proto_std_SlowGo    = np.load('../data/'+paramsA['loadFolder']+'/GPe_Proto_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    meanRates_fastGo['StrD1'],     STR_D1_std_FastGo,      meanRates_slowGo['StrD1'],     STR_D1_std_SlowGo       = np.load('../data/'+paramsA['loadFolder']+'/SD1_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    meanRates_fastGo['StrD2'],     STR_D2_std_FastGo,      meanRates_slowGo['StrD2'],     STR_D2_std_SlowGo       = np.load('../data/'+paramsA['loadFolder']+'/SD2_meanrate_std_Fast-Slow_Go'+netNr+'.npy')            
    meanRates_fastGo['STN'],        STN_std_FastGo,         meanRates_slowGo['STN'],        STN_std_SlowGo          = np.load('../data/'+paramsA['loadFolder']+'/STN_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    meanRates_fastGo['SNr'],        SNr_std_FastGo,         meanRates_slowGo['SNr'],        SNr_std_SlowGo          = np.load('../data/'+paramsA['loadFolder']+'/SNr_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    meanRates_fastGo['Thal'],       Thal_std_FastGo,        meanRates_slowGo['Thal'],       Thal_std_SlowGo         = np.load('../data/'+paramsA['loadFolder']+'/Thal_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    meanRates_fastGo['cortexGo'],   Cortex_G_std_FastGo,    meanRates_slowGo['cortexGo'],   Cortex_G_std_SlowGo     = np.load('../data/'+paramsA['loadFolder']+'/Cortex_G_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    meanRates_fastGo['GPeCp'], GPe_Proto2_std_FastGo,  meanRates_slowGo['GPeCp'], GPe_Proto2_std_SlowGo   = np.load('../data/'+paramsA['loadFolder']+'/GPe_Proto2_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    meanRates_fastGo['cortexPause'], PauseInput_std_FastGo,  meanRates_slowGo['cortexPause'], PauseInput_std_SlowGo   = np.load('../data/'+paramsA['loadFolder']+'/PauseInput_meanrate_std_Fast-Slow_Go'+netNr+'.npy')
    meanRates_fastGo['cortexStop'],   Cortex_S_std_FastGo,    meanRates_slowGo['cortexStop'],   Cortex_S_std_SlowGo     = np.load('../data/'+paramsA['loadFolder']+'/Cortex_S_meanrate_std_Fast-Slow_Go'+netNr+'.npy')                        
    meanRates_fastGo['StrFSI'],    STR_FSI_std_FastGo,     meanRates_slowGo['StrFSI'],    STR_FSI_std_SlowGo      = np.load('../data/'+paramsA['loadFolder']+'/FSI_meanrate_std_Fast-Slow_Go'+netNr+'.npy')

    dataFig7 = {}
    for popFig7 in params['Fig7_order']:
        dataFig7[popFig7] = [meanRates_failedStop[popFig7], meanRates_correctStop[popFig7]]
        
    dataFig11 = {}
    for popFig11 in params['Fig7_order']:
        dataFig11['all_'+popFig11]  = [meanRates_failedStop[popFig11], meanRates_Go[popFig11]]
        dataFig11['fast_'+popFig11] = [meanRates_failedStop[popFig11], meanRates_fastGo[popFig11]]
        dataFig11['slow_'+popFig11] = [meanRates_failedStop[popFig11], meanRates_slowGo[popFig11]]


    ### FIGURE A: FAILED VS CORRECT STOPS
    plot_meanrate_All_FailedVsCorrectStop(resultsDir, \
                                          dataFig7, \
                                          rsp_mInt_stop, nz_FailedStop, nz_CorrectStop, params['IntegratorNeuron_threshold'], \
                                          paramsA, \
                                          params['t_init'], params['t_SSD'], params['general_id'], paramsS['trials'], params['general_dt'], pvalue_list, pvalue_times)


    ### FIGURE B: FAILED STOP VS GO TODO: Adjust this like previous function
    plot_meanrate_All_FailedStopVsCorrectGo(resultsDir+'/', \
                                          dataFig11, \
                                          nz_FailedStop, nz_CorrectStop, \
                                          mInt_Go, mInt_stop, params['IntegratorNeuron_threshold'], \
                                          paramsA, \
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
