import os
import pylab as plt
import numpy as np
import scipy.stats as st
from scipy import interpolate
import csv
from analysis import plot_zscore_stopVsGo, custom_zscore
from sim_params import params
from lg_populations_BA_BG import Integrator


def plot_STR_stop(param_id):
    param_id = float(param_id)
    STR_D1_Stop_mean = np.load('plots/plotdata/STR_D1_Stop_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    STR_D1_Go_mean = np.load('plots/plotdata/STR_D1_Go_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    STR_D2_Stop_mean = np.load('plots/plotdata/STR_D2_Stop_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    STR_D2_Go_mean = np.load('plots/plotdata/STR_D2_Go_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    STR_FSI_Stop_mean = np.load('plots/plotdata/STR_FSI_Stop_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    STR_FSI_Go_mean = np.load('plots/plotdata/STR_FSI_Go_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    GPe_Arky_mean_Stop = np.load('plots/plotdata/Arky_Stop_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    GPe_Arky_mean_Go = np.load('plots/plotdata/Arky_Go_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    GPe_Proto_mean_Stop = np.load('plots/plotdata/Proto_Stop_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    GPe_Proto_mean_Go = np.load('plots/plotdata/Proto_Go_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    STN_mean_Stop = np.load('plots/plotdata/STN_Stop_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    STN_mean_Go = np.load('plots/plotdata/STN_Go_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    SNr_mean_Stop = np.load('plots/plotdata/SNr_Stop_mean_id'+str(param_id)+'.npy', allow_pickle=True)
    SNr_mean_Go = np.load('plots/plotdata/SNr_Go_mean_id'+str(param_id)+'.npy', allow_pickle=True)

    t_init = 300
    t_SSD = 250
    trials = 100

    #plot_zscore_stopVsGo(STN_mean_Stop, STN_mean_Go, SNr_mean_Stop, SNr_mean_Go, GPe_Arky_mean_Stop, GPe_Arky_mean_Go, GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, param_id, trials)

    plot_stop_rates(STN_mean_Stop, 'STN Stop', 'orange', \
                    SNr_mean_Stop, 'SNr Stop', 'tomato', \
                    GPe_Arky_mean_Stop, 'Arky Stop', 'cyan', \
                    GPe_Proto_mean_Stop, 'Proto', 'blue',
                    'Stop_detail', param_id, trials, t_init + t_SSD - 50, t_init + t_SSD + 150, -0.05, 0.15, 250) # 200)

    plot_stop_rates(STR_FSI_Stop_mean, 'FSI Stop', 'k', \
                    STR_FSI_Go_mean, 'FSI Go', '0.6', \
                    STR_D2_Stop_mean, 'Str D2 Stop', 'pink', \
                    STR_D2_Go_mean, 'Str D2 Go', 'purple',
                    'FSI_D2', param_id, trials, t_init + t_SSD - 200, t_init + t_SSD + 400, -0.2, 0.4, 150)

    plot_stop_rates(STR_D1_Stop_mean, 'Str D1 Stop', 'red', \
                    STR_D1_Go_mean, 'Str D1 Go', 'green', \
                    GPe_Arky_mean_Stop, 'Arky Stop', 'cyan', \
                    GPe_Proto_mean_Stop, 'Proto', 'blue',
                    'D1_Arky', param_id, trials, t_init + t_SSD - 200, t_init + t_SSD + 400, -0.2, 0.4, 150)

def plot_stop_rates(pop1_stop_mean, pop1name, pop1col, \
                    pop2_stop_mean, pop2name, pop2col, \
                    pop3_stop_mean, pop3name, pop3col, \
                    pop4_stop_mean, pop4name, pop4col,
                    filename, param_id, trials, tmin, tmax, tminlabel, tmaxlabel, ymax):
    plt.ion()
    plt.figure(figsize=(3.5,4), dpi=300)
    t_init = 300
    t_SSD = 250
    fsize = 12 # 15
    ax = plt.gca()
    plt.plot(pop1_stop_mean, lw=3, color = pop1col, label= pop1name)
    plt.plot(pop2_stop_mean, lw=3, color = pop2col, label= pop2name)
    plt.plot(pop3_stop_mean, lw=3, color = pop3col, label= pop3name)
    plt.plot(pop4_stop_mean, lw=3, color = pop4col, label= pop4name)
    #ax.axis([t_init + t_SSD - 200, t_init + t_SSD + 400, max(0, ax.axis()[2]), 150]) 
    ax.axis([tmin, tmax, max(0, ax.axis()[2]), ymax]) # 150
    plt.plot((t_init + t_SSD) * np.ones(2), [0, plt.axis()[3]], 'k--', lw=1.5, label='Stop cue')
    #ax.set_xticks([t_init + t_SSD - 200, t_init + t_SSD, t_init + t_SSD + 400]) 
    ax.set_xticks([tmin, t_init + t_SSD, tmax]) 
    #ax.set_xticklabels([-0.2, 0, 0.4]) 
    ax.set_xticklabels([tminlabel, 0, tmaxlabel])
    plt.xlabel('Time from Stop cue [sec]', fontsize=fsize)
    plt.ylabel('Firing rate [spk/s]', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.legend(fontsize=10) # 6
    plt.tight_layout()
    plt.savefig('plots/mean_rate_paramsid'+str(param_id)+'_'+str(trials)+'trials_'+filename+'.png', dpi=300) #         
    plt.ioff()
    plt.show()
    

def plot_percent_correctGo_vs_GoRate_cycles(param_id, network_array):
    paramname = 'CortexGo_rates'	
    loop_params, paramname = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True)
    loaded_data = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True)    
    print('loaded_data     = ', loaded_data)
    loop_params, paramname = loaded_data[0], loaded_data[1]   
    print("paramname, param_id, loop_params = ", paramname, param_id, loop_params)
    n_loop_cycles = len(loop_params)
    pct_corr_Go = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    pct_corr_Stop = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    Ctx_Go_rates = np.nan * np.ones(n_loop_cycles)    
    #for i_netw_rep in range(n_networks):
    for i_netw_rep in network_array:
        for i_cycle in range(n_loop_cycles):
            #results_RT = np.load('data/resultsRT_'+str(i_netw_rep+1)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)
            results_RT = np.load('data/resultsRT_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)            
            print('results_RT = ', results_RT)
            pct_corr_Go[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectGoTrials'] / float(results_RT.item()['nCorrectGoTrials'] + results_RT.item()['nFailedGoTrials'] )
            pct_corr_Stop[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectStopTrials'] / float( results_RT.item()['nCorrectStopTrials'] + results_RT.item()['nFailedStopTrials'] )
            params, wf = read_paramfile(param_id)
            Ctx_Go_rates[i_cycle] = loop_params[i_cycle] # params['CortexGo_rates']
    i_sorted = np.argsort(Ctx_Go_rates)
    #standard_plot(Ctx_Go_rates[i_sorted], np.nanmean(pct_corr_Go, 1)[i_sorted], 'Cortex Go input rate [spikes/s]', '% Correct Go trials', 'plots/summary_percent_correct_vs_GoRate_cycles_'+str(param_id)+'.png', 10)
    standard_plot_2lines_std(Ctx_Go_rates[i_sorted], np.nanmean(pct_corr_Go, 1)[i_sorted], np.nanstd(pct_corr_Go, 1)[i_sorted], 'Correct Go', \
                         Ctx_Go_rates[i_sorted], np.nanmean(pct_corr_Stop, 1)[i_sorted], np.nanstd(pct_corr_Stop, 1)[i_sorted], 'Correct Stop', \
                        'Cortex Go rates', '% correct trials', 'plots/CorrectGoStopTrials_vs_CortexGo_'+str(param_id)+'.png', 10, ymin=0, ymax=100) # sem # np.nanstd


def plot_percent_correctGo_vs_StopRate_cycles(param_id, network_array):
    paramname = 'CortexStop_rates'
    loop_params, paramname = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True)
    print("paramname, param_id, loop_params = ", paramname, param_id, loop_params)
    n_loop_cycles = len(loop_params)
    pct_corr_Go = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    pct_corr_Stop = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    Ctx_Stop_rates = np.nan * np.ones(n_loop_cycles)    
    #for i_netw_rep in range(n_networks):
    for i_netw_rep in network_array:
        for i_cycle in range(n_loop_cycles):
            #results_RT = np.load('data/resultsRT_'+str(i_netw_rep+1)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)
            results_RT = np.load('data/resultsRT_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)            
            #print('results_RT = ', results_RT)
            pct_corr_Go[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectGoTrials'] / float(results_RT.item()['nCorrectGoTrials'] + results_RT.item()['nFailedGoTrials'] )
            pct_corr_Stop[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectStopTrials'] / float( results_RT.item()['nCorrectStopTrials'] + results_RT.item()['nFailedStopTrials'] )
            params, wf = read_paramfile(param_id)
            Ctx_Stop_rates[i_cycle] = loop_params[i_cycle] # params['CortexGo_rates']
    i_sorted = np.argsort(Ctx_Stop_rates)
    standard_plot_2lines_std(Ctx_Stop_rates[i_sorted], np.nanmean(pct_corr_Go, 1)[i_sorted], np.nanstd(pct_corr_Go, 1)[i_sorted], 'Correct Go', \
                         Ctx_Stop_rates[i_sorted], np.nanmean(pct_corr_Stop, 1)[i_sorted], np.nanstd(pct_corr_Stop, 1)[i_sorted], 'Correct Stop', \
                        'Cortex Stop rates', '% correct trials', 'plots/CorrectGoStopTrials_vs_CortexStop_'+str(param_id)+'.png', 10, ymin=0, ymax=100) # sem # np.nanstd

def plot_percent_correctGo_vs_CortexPauseRate_cycles(param_id, network_array):
    paramname = 'Stop1rates'
    loop_params, paramname = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True)
    print("paramname, param_id, loop_params = ", paramname, param_id, loop_params)
    n_loop_cycles = len(loop_params)
    pct_corr_Go = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    pct_corr_Stop = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    Ctx_Stop_rates = np.nan * np.ones(n_loop_cycles)    
    #for i_netw_rep in range(n_networks):
    for i_netw_rep in network_array:
        for i_cycle in range(n_loop_cycles):
            #results_RT = np.load('data/resultsRT_'+str(i_netw_rep+1)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)
            results_RT = np.load('data/resultsRT_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)            
            pct_corr_Go[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectGoTrials'] / float(results_RT.item()['nCorrectGoTrials'] + results_RT.item()['nFailedGoTrials'] )
            pct_corr_Stop[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectStopTrials'] / float( results_RT.item()['nCorrectStopTrials'] + results_RT.item()['nFailedStopTrials'] )
            params, wf = read_paramfile(param_id)
            Ctx_Stop_rates[i_cycle] = loop_params[i_cycle] # params['CortexGo_rates']
    i_sorted = np.argsort(Ctx_Stop_rates)
    standard_plot_2lines_std(Ctx_Stop_rates[i_sorted], np.nanmean(pct_corr_Go, 1)[i_sorted], np.nanstd(pct_corr_Go, 1)[i_sorted], 'Correct Go', \
                         Ctx_Stop_rates[i_sorted], np.nanmean(pct_corr_Stop, 1)[i_sorted], np.nanstd(pct_corr_Stop, 1)[i_sorted], 'Correct Stop', \
                        'Cortex Pause rates', '% correct trials', 'plots/CorrectGoStopTrials_vs_CortexPause_'+str(param_id)+'.png', 10, ymin=0, ymax=100) # sem # np.nanstd

def plot_percent_correctGo_vs_weight_cycles(param_id, network_array, paramname):
    loop_params, paramname = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True) # new   
    print("paramname, param_id, loop_params = ", paramname, param_id, loop_params)
    n_loop_cycles = len(loop_params)
    pct_corr_Go = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    pct_corr_Stop = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    weights = np.nan * np.ones(n_loop_cycles)    
    for i_netw_rep in network_array:
        for i_cycle in range(n_loop_cycles):
            #results_RT = np.load('data/resultsRT_'+str(i_netw_rep+1)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True) # old
            results_RT = np.load('data/resultsRT_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)      
            #print('results_RT = ', results_RT)            
            pct_corr_Go[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectGoTrials'] / float(results_RT.item()['nCorrectGoTrials'] + results_RT.item()['nFailedGoTrials'] )
            pct_corr_Stop[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectStopTrials'] / float( results_RT.item()['nCorrectStopTrials'] + results_RT.item()['nFailedStopTrials'] )
            params, wf = read_paramfile(param_id)
            weights[i_cycle] = loop_params[i_cycle] # params['CortexGo_rates']
    i_sorted = np.argsort(weights)
    standard_plot_2lines_std(weights[i_sorted], np.nanmean(pct_corr_Go, 1)[i_sorted], np.nanstd(pct_corr_Go, 1)[i_sorted], 'Correct Go', \
                         weights[i_sorted], np.nanmean(pct_corr_Stop, 1)[i_sorted], np.nanstd(pct_corr_Stop, 1)[i_sorted], 'Correct Stop', \
                        paramname+' weights', '% correct trials', 'plots/CorrectGoStopTrials_vs_'+paramname+'_'+str(param_id)+'.png', 8, ymin=0, ymax=100) # 10

def plot_pctcorrect_timing_RT_vs_weight_cycles(param_id, network_array, paramname, loadFolder, loop_paramsSoll, saveFormat):
    n_loop_cycles = len(loop_paramsSoll)
    pct_corr_Go = np.nan * np.ones([n_loop_cycles, len(network_array)])
    pct_corr_Stop = np.nan * np.ones([n_loop_cycles, len(network_array)])
    STN_Timing = np.nan * np.ones([n_loop_cycles, len(network_array)])
    SNr_Timing = np.nan * np.ones([n_loop_cycles, len(network_array)])
    Proto_Timing = np.nan * np.ones([n_loop_cycles, len(network_array)])    
    Arky_Timing = np.nan * np.ones([n_loop_cycles, len(network_array)])
    RT_corrGo = np.nan * np.ones([n_loop_cycles, len(network_array)])
    RT_failedStop = np.nan * np.ones([n_loop_cycles, len(network_array)])
    weights = np.nan * np.ones(n_loop_cycles)    
    for i_netw_rep, netw_rep in enumerate(network_array):
        loop_params, paramname = np.load(loadFolder+'cycle_params_'+paramname+'_id'+str(int(param_id))+str(netw_rep)+'.npy', allow_pickle=True)
        loop_cycles = [np.where(loop_params==val)[0][0] for val in loop_paramsSoll] 
        if netw_rep==network_array[0]:
            print("paramname, param_id, loop_params[0], loop_paramsSoll = ", paramname, param_id, loop_params, loop_paramsSoll)
        for i_cycle, cycle in enumerate(loop_cycles):
            results_RT = np.load(loadFolder+'resultsRT_'+str(netw_rep)+'_param_'+paramname+'_cycle'+str(cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)      
            pct_corr_Go[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectGoTrials'] / float(results_RT.item()['nCorrectGoTrials'] + results_RT.item()['nFailedGoTrials'] )
            pct_corr_Stop[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectStopTrials'] / float( results_RT.item()['nCorrectStopTrials'] + results_RT.item()['nFailedStopTrials'] )
            #results_StopTiming = np.load('data/Stop_timing_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'_id'+str(int(param_id))+'.npy', allow_pickle=True)                  
            params, wf = read_paramfile(param_id)   
            t_StopCue = params['t_init'] + params['t_SSD']                       
            """STN_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[0] - t_StopCue           
            SNr_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[1] - t_StopCue                                   
            Proto_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[2] - t_StopCue           
            Arky_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[3] - t_StopCue"""           
            RT_corrGo[i_cycle, i_netw_rep] = results_RT.item()['meanRT_CorrectGo']
            RT_failedStop[i_cycle, i_netw_rep] = results_RT.item()['meanRT_FailedStop']                  
            weights[i_cycle] = loop_params[cycle] # params['CortexGo_rates']
    i_sorted = np.argsort(weights)
    plt.ion()    
    #plt.figure(figsize=(3.2,4), dpi=300)
    plt.figure(figsize=(2, 4), dpi=300)
 
    def changeParamName(paramname): 
        if paramname=='CortexGo_rates':
            return 'relative cortex-Go rate'
        else:
            return paramname

    print('paramname:',paramname)   
    standard_plot_2lines_std_subplot(211, weights[i_sorted], np.nanmean(pct_corr_Go, 1)[i_sorted], np.nanstd(pct_corr_Go, 1)[i_sorted], 'correct Go', \
                         weights[i_sorted], np.nanmean(pct_corr_Stop, 1)[i_sorted], np.nanstd(pct_corr_Stop, 1)[i_sorted], 'correct Stop', \
                        changeParamName(paramname)+[' weights',''][int(paramname[-5:]=='rates')], '% correct trials', 'plots/CorrectGoStopTrials_vs_'+paramname+'_'+str(param_id)+'.png', 6, ymin=-5, ymax=105, xmin=loop_paramsSoll[0], xmax=loop_paramsSoll[-1])

    """timing_plot_4lines_std_subplot(312, weights[i_sorted], np.nanmean(STN_Timing, 1)[i_sorted], np.nanstd(STN_Timing, 1)[i_sorted], 'STN', \
                           weights[i_sorted], np.nanmean(SNr_Timing, 1)[i_sorted], np.nanstd(SNr_Timing, 1)[i_sorted], 'SNr', \
                           weights[i_sorted], np.nanmean(Proto_Timing, 1)[i_sorted], np.nanstd(Proto_Timing, 1)[i_sorted], 'Proto', \
                           weights[i_sorted], np.nanmean(Arky_Timing, 1)[i_sorted], np.nanstd(Arky_Timing, 1)[i_sorted], 'Arky', \
                        t_StopCue, paramname+' weights', 'Timing', 'plots/StopTiming_vs_'+paramname+'_'+str(param_id)+'.png', 6, ymin=0, ymax=200)"""

    standard_plot_2lines_std_subplot(212, weights[i_sorted], np.nanmean(RT_corrGo, 1)[i_sorted], np.nanstd(RT_corrGo, 1)[i_sorted], 'correct Go', \
                         weights[i_sorted], np.nanmean(RT_failedStop, 1)[i_sorted], np.nanstd(RT_failedStop, 1)[i_sorted], 'failed Stop', \
                        changeParamName(paramname)+[' weights',''][int(paramname[-5:]=='rates')], 'Reaction time [ms]', 'plots/summary_RT_vs_w_'+paramname+'_'+str(param_id)+'.png', 6, ymin=0, ymax=600, xmin=loop_paramsSoll[0], xmax=loop_paramsSoll[-1]) # sem # np.nanstd
                        
    plt.tight_layout()                        

    saveFolder='plots/multiloops/'+str(param_id)+'/'
    try:
        os.mkdir(saveFolder)
    except:
        print(saveFolder+' not created')
    plt.savefig(saveFolder+'pctcorrect_timing_RT_vs_'+paramname+'_weight_paramsid'+str(param_id)+'.'+saveFormat, dpi=300)
         
    plt.ioff()                        
    plt.show()    



def plot_StopTiming_vs_weight_cycles(param_id, network_array, paramname):    
    #loop_params, paramname = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True) # old
    loop_params, paramname = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True) # new   
    print("paramname, param_id, loop_params = ", paramname, param_id, loop_params)
    n_loop_cycles = len(loop_params)
    STN_Timing = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    SNr_Timing = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    Proto_Timing = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])    
    Arky_Timing = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    weights = np.nan * np.ones(n_loop_cycles)    
    params, wf = read_paramfile(param_id)            
    t_StopCue = params['t_init'] + params['t_SSD']            
    #for i_netw_rep in range(n_networks):
    for i_netw_rep in network_array:        
        for i_cycle in range(n_loop_cycles):
            #print("i_cycle = ", i_cycle)
            results_StopTiming = np.load('data/Stop_timing_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'_id'+str(int(param_id))+'.npy', allow_pickle=True)      
            #results_StopTiming = np.load('data/Stop_timing_'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'_id'+str(int(param_id))+'.npy', allow_pickle=True)                  
            #print('results_StopTiming = ', results_StopTiming)            
            # [STN_median_peak, SNr_median_peak, Proto_median_peak, Arky_median_peak]
            STN_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[0] - t_StopCue           
            SNr_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[1] - t_StopCue                                   
            Proto_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[2] - t_StopCue           
            Arky_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[3] - t_StopCue     
            weights[i_cycle] = loop_params[i_cycle] # params['CortexGo_rates']
    i_sorted = np.argsort(weights)
    timing_plot_4lines_std(weights[i_sorted], np.nanmean(STN_Timing, 1)[i_sorted], np.nanstd(STN_Timing, 1)[i_sorted], 'STN', \
                           weights[i_sorted], np.nanmean(SNr_Timing, 1)[i_sorted], np.nanstd(SNr_Timing, 1)[i_sorted], 'SNr', \
                           weights[i_sorted], np.nanmean(Proto_Timing, 1)[i_sorted], np.nanstd(Proto_Timing, 1)[i_sorted], 'Proto', \
                           weights[i_sorted], np.nanmean(Arky_Timing, 1)[i_sorted], np.nanstd(Arky_Timing, 1)[i_sorted], 'Arky', \
                        t_StopCue, paramname+' weights', 'Timing', 'plots/StopTiming_vs_'+paramname+'_'+str(param_id)+'.png', 8, ymin=0, ymax=200) # 10


def plot_RT_vs_weight_cycles(param_id, network_array, paramname):    
    loop_params, paramname = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True)
    print("paramname, param_id, loop_params = ", paramname, param_id, loop_params)
    n_loop_cycles = len(loop_params)
    RT_corrGo = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    RT_failedStop = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    weights = np.nan * np.ones(n_loop_cycles)    
    #for i_netw_rep in range(n_networks):
    for i_netw_rep in network_array:            
        for i_cycle in range(n_loop_cycles):
            #results_RT = np.load('data/resultsRT_'+str(i_netw_rep+1)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)
            results_RT = np.load('data/resultsRT_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)            
            RT_corrGo[i_cycle, i_netw_rep] = results_RT.item()['meanRT_CorrectGo']
            RT_failedStop[i_cycle, i_netw_rep] = results_RT.item()['meanRT_FailedStop']
            params, wf = read_paramfile(param_id)
            weights[i_cycle] = loop_params[i_cycle] # params['CortexGo_rates']
    i_sorted = np.argsort(weights)
    standard_plot_2lines_std(weights[i_sorted], np.nanmean(RT_corrGo, 1)[i_sorted], np.nanstd(RT_corrGo, 1)[i_sorted], 'Correct Go', \
                         weights[i_sorted], np.nanmean(RT_failedStop, 1)[i_sorted], np.nanstd(RT_failedStop, 1)[i_sorted], 'Failed Stop', \
                        paramname+' weights', 'Reaction time [ms]', 'plots/summary_RT_vs_w_'+paramname+'_'+str(param_id)+'.png', 10, ymin=0, ymax=400) # sem # np.nanstd



def plot_percent_correctGo_vs_prob_cycles(param_id, n_networks, paramname):
    loop_params, paramname = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True)
    print("paramname, param_id, loop_params = ", paramname, param_id, loop_params)
    #print("paramname, param_id, loop_params = ", paramname, param_id, loop_params[0])
    n_loop_cycles = len(loop_params)
    #n_loop_cycles = len(loop_params[0])
    pct_corr_Go = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    pct_corr_Stop = np.nan * np.ones([n_loop_cycles, network_array[-1]+1])
    p_trans = np.nan * np.ones(n_loop_cycles)    
    for i_netw_rep in range(n_networks):
        for i_cycle in range(n_loop_cycles):
            results_RT = np.load('data/resultsRT_'+str(i_netw_rep+1)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)
            #pct_corr_Go[i_cycle, i_netw_rep] = results_RT.item()['nCorrectGoTrials']
            #pct_corr_Stop[i_cycle, i_netw_rep] = results_RT.item()['nCorrectStopTrials']
            pct_corr_Go[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectGoTrials'] / float(results_RT.item()['nCorrectGoTrials'] + results_RT.item()['nFailedGoTrials'] )
            pct_corr_Stop[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectStopTrials'] / float( results_RT.item()['nCorrectStopTrials'] + results_RT.item()['nFailedStopTrials'] )
            params, wf = read_paramfile(param_id)
            #print("loop_params[i_cycle] = ", loop_params[i_cycle][i_cycle])
            p_trans[i_cycle] = loop_params[i_cycle]#[i_cycle] # params['CortexGo_rates']
    i_sorted = np.argsort(p_trans)
    print("np.nanmean(pct_corr_Go, 1)[i_sorted] = ", np.nanmean(pct_corr_Go, 1)[i_sorted])
    print("p_trans[i_sorted] = ", p_trans[i_sorted])
    standard_plot_2lines_std(100*p_trans[i_sorted], np.nanmean(pct_corr_Go, 1)[i_sorted], np.nanstd(pct_corr_Go, 1)[i_sorted], 'Correct Go', \
                         100*p_trans[i_sorted], np.nanmean(pct_corr_Stop, 1)[i_sorted], np.nanstd(pct_corr_Stop, 1)[i_sorted], 'Correct Stop', \
                        'Residual Arkypallidal activity [%]', '% correct trials', 'plots/CorrectGoStopTrials_vs_'+paramname+'_'+str(param_id)+'.png', 8, ymin=0, ymax=100) # 10




def plot_percent_correct_vs_StopRate(param_range):
    pct_corr_stop = np.nan * np.ones(len(param_range))
    Ctx_Stop_rates = np.nan * np.ones(len(param_range))    
    for i_param in range(len(param_range)):
        results_RT = np.load('data/resultsRT_id'+str(float(param_range[i_param]))+'.npy', allow_pickle=True)
        #pct_corr_stop[i_param] = results_RT.item()['nCorrectStopTrials']
        pct_corr_stop[i_param] = 100 * results_RT.item()['nCorrectStopTrials'] / float( results_RT.item()['nCorrectStopTrials'] + results_RT.item()['nFailedStopTrials'] )
        params, wf = read_paramfile(param_range[i_param])
        Ctx_Stop_rates[i_param] = params['CortexStop_rates']
    i_sorted = np.argsort(Ctx_Stop_rates)
    standard_plot(Ctx_Stop_rates[i_sorted], pct_corr_stop[i_sorted], 'Cortex Stop input rate [spikes/s]', '% Correct Stop trials', 'plots/summary_percent_correct_vs_StopRate_'+str(param_range[0])+'-'+str(param_range[-1])+'.png', 10)


def plot_percent_correct_vs_weight(param_range, keystr, projname_str): 
    pct_corr_stop = np.nan * np.ones(len(param_range))
    weight_values = np.nan * np.ones(len(param_range))    
    for i_param in range(len(param_range)):
        results_RT = np.load('data/resultsRT_id'+str(float(param_range[i_param]))+'.npy', allow_pickle=True)
        #pct_corr_stop[i_param] = results_RT.item()['nCorrectStopTrials']
        pct_corr_stop[i_param] = 100 * results_RT.item()['nCorrectStopTrials'] / float( results_RT.item()['nCorrectStopTrials'] + results_RT.item()['nFailedStopTrials'] )        
        params, wf = read_paramfile(param_range[i_param])
        weight_values[i_param] = wf[keystr]
    i_sorted = np.argsort(weight_values)
    standard_plot(weight_values[i_sorted], pct_corr_stop[i_sorted], projname_str, '% Correct Stop trials', 'plots/summary_percent_correct_vs_w_'+keystr+'_'+str(param_range[0])+'-'+str(param_range[-1])+'.png', 10)


def plot_RT_vs_weight(param_range, keystr, projname_str): 
    weight_values = np.nan * np.ones(len(param_range))    
    RT_corrGo = np.nan * np.ones(len(param_range))    
    RT_failedStop = np.nan * np.ones(len(param_range))    
    for i_param in range(len(param_range)):
        results_RT = np.load('data/resultsRT_id'+str(float(param_range[i_param]))+'.npy', allow_pickle=True)
        RT_corrGo[i_param] = results_RT.item()['meanRT_CorrectGo']
        RT_failedStop[i_param] = results_RT.item()['meanRT_FailedStop']
        params, wf = read_paramfile(param_range[i_param])
        weight_values[i_param] = wf[keystr]
    i_sorted = np.argsort(weight_values)
    standard_plot_2lines(weight_values[i_sorted], RT_corrGo[i_sorted], 'Correct Go', weight_values[i_sorted], RT_failedStop[i_sorted], 'Failed Stop', projname_str, 'Reaction time [ms]', 'plots/summary_RT_vs_w_'+keystr+'_'+str(param_range[0])+'-'+str(param_range[-1])+'.png', 10, ymin=0, ymax=350)




def standard_plot(xdata, ydata, xlabel_text, ylabel_text, filename, fsize, ymin=0.0, ymax=100.0):
    plt.ion()
    plt.figure(figsize=(3,3), dpi=300)
    plt.plot(xdata, ydata, '.-')
    plt.xlabel(xlabel_text, fontsize=fsize)
    plt.ylabel(ylabel_text, fontsize=fsize)
    plt.axis([plt.axis()[0], plt.axis()[1], ymin, ymax])
    plt.tight_layout()
    plt.ioff()
    plt.savefig(filename, dpi=300)
    plt.show()

def standard_plot_2lines(xdata1, ydata1, label1, xdata2, ydata2, label2, xlabel_text, ylabel_text, filename, fsize, ymin=0.0, ymax=100.0):
    plt.ion()
    plt.figure(figsize=(3,3), dpi=300)
    plt.plot(xdata1, ydata1, '.-', label=label1)
    plt.plot(xdata2, ydata2, '.-', label=label2)
    plt.xlabel(xlabel_text, fontsize=fsize)
    plt.ylabel(ylabel_text, fontsize=fsize)
    plt.axis([plt.axis()[0], plt.axis()[1], ymin, ymax])
    plt.legend(fontsize=fsize)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(filename, dpi=300)
    plt.show()

def standard_plot_2lines_std(xdata1, ydata1, yerr1, label1, xdata2, ydata2, yerr2, label2, xlabel_text, ylabel_text, filename, fsize, ymin=0.0, ymax=100.0):
    plt.ion()
    plt.figure(figsize=(3,3), dpi=300)
    plt.errorbar(xdata1, ydata1, yerr=yerr1, label=label1)
    plt.errorbar(xdata2, ydata2, yerr=yerr2, label=label2)
    plt.xlabel(xlabel_text, fontsize=fsize)
    plt.ylabel(ylabel_text, fontsize=fsize)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.axis([plt.axis()[0], plt.axis()[1], ymin, ymax])
    plt.legend(fontsize=fsize)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_pctcorrect_timing_RT_vs_weight_cycles_FOURINAROW(param_id, network_array, paramname, loadFolder, loop_paramsSoll, paramIdx):
    n_loop_cycles = len(loop_paramsSoll)
    pct_corr_Go = np.nan * np.ones([n_loop_cycles, len(network_array)])
    pct_corr_Stop = np.nan * np.ones([n_loop_cycles, len(network_array)])
    STN_Timing = np.nan * np.ones([n_loop_cycles, len(network_array)])
    SNr_Timing = np.nan * np.ones([n_loop_cycles, len(network_array)])
    Proto_Timing = np.nan * np.ones([n_loop_cycles, len(network_array)])    
    Arky_Timing = np.nan * np.ones([n_loop_cycles, len(network_array)])
    RT_corrGo = np.nan * np.ones([n_loop_cycles, len(network_array)])
    RT_failedStop = np.nan * np.ones([n_loop_cycles, len(network_array)])
    weights = np.nan * np.ones(n_loop_cycles)    
    for i_netw_rep, netw_rep in enumerate(network_array):
        loop_params, paramname = np.load(loadFolder+'cycle_params_'+paramname+'_id'+str(int(param_id))+str(netw_rep)+'.npy', allow_pickle=True)
        loop_cycles = [np.where(loop_params==val)[0][0] for val in loop_paramsSoll] 
        if netw_rep==network_array[0]:
            print("paramname, param_id, loop_params[0], loop_paramsSoll = ", paramname, param_id, loop_params, loop_paramsSoll)
        for i_cycle, cycle in enumerate(loop_cycles):
            results_RT = np.load(loadFolder+'resultsRT_'+str(netw_rep)+'_param_'+paramname+'_cycle'+str(cycle)+'id'+str(int(param_id))+'.npy', allow_pickle=True)      
            pct_corr_Go[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectGoTrials'] / float(results_RT.item()['nCorrectGoTrials'] + results_RT.item()['nFailedGoTrials'] )
            pct_corr_Stop[i_cycle, i_netw_rep] = 100 * results_RT.item()['nCorrectStopTrials'] / float( results_RT.item()['nCorrectStopTrials'] + results_RT.item()['nFailedStopTrials'] )
            #results_StopTiming = np.load('data/Stop_timing_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'_id'+str(int(param_id))+'.npy', allow_pickle=True)                  
            params, wf = read_paramfile(param_id)   
            t_StopCue = params['t_init'] + params['t_SSD']                       
            """STN_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[0] - t_StopCue           
            SNr_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[1] - t_StopCue                                   
            Proto_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[2] - t_StopCue           
            Arky_Timing[i_cycle, i_netw_rep] = params['dt'] * results_StopTiming[3] - t_StopCue"""           
            RT_corrGo[i_cycle, i_netw_rep] = results_RT.item()['meanRT_CorrectGo']
            RT_failedStop[i_cycle, i_netw_rep] = results_RT.item()['meanRT_FailedStop']                  
            weights[i_cycle] = loop_params[cycle] # params['CortexGo_rates']
    i_sorted = np.argsort(weights)


    if paramIdx==0:
        plt.ion() 
        plt.figure(figsize=(160/25.4, 100/25.4), dpi=300)
 
    def changeParamName(paramname): 
        if paramname=='CortexGo_rates':
            return 'relative cortex-Go rate'
        else:
            return paramname

    print('paramname:',paramname)   

    ### significance of first and last value
    group1=pct_corr_Stop[0,:]
    group2=pct_corr_Stop[5,:]
    Hstat, pval = st.kruskal(group1, group2)

    standard_plot_2lines_std_subplot_FOURINAROW([Hstat, pval],[2,4,paramIdx+1], weights[i_sorted], np.nanmean(pct_corr_Go, 1)[i_sorted], np.nanstd(pct_corr_Go, 1)[i_sorted], 'correct Go', \
                         weights[i_sorted], np.nanmean(pct_corr_Stop, 1)[i_sorted], np.nanstd(pct_corr_Stop, 1)[i_sorted], 'correct Stop', \
                        changeParamName(paramname)+[' weights',''][int(paramname[-5:]=='rates')], '% correct trials', 'plots/CorrectGoStopTrials_vs_'+paramname+'_'+str(param_id)+'.png', 9, ymin=-5, ymax=105, xmin=loop_paramsSoll[0], xmax=loop_paramsSoll[-1])

    """timing_plot_4lines_std_subplot(312, weights[i_sorted], np.nanmean(STN_Timing, 1)[i_sorted], np.nanstd(STN_Timing, 1)[i_sorted], 'STN', \
                           weights[i_sorted], np.nanmean(SNr_Timing, 1)[i_sorted], np.nanstd(SNr_Timing, 1)[i_sorted], 'SNr', \
                           weights[i_sorted], np.nanmean(Proto_Timing, 1)[i_sorted], np.nanstd(Proto_Timing, 1)[i_sorted], 'Proto', \
                           weights[i_sorted], np.nanmean(Arky_Timing, 1)[i_sorted], np.nanstd(Arky_Timing, 1)[i_sorted], 'Arky', \
                        t_StopCue, paramname+' weights', 'Timing', 'plots/StopTiming_vs_'+paramname+'_'+str(param_id)+'.png', 6, ymin=0, ymax=200)"""

    ### significance of first and last value
    group1=RT_corrGo[0,:]
    group2=RT_corrGo[5,:]
    Hstat1, pval1 = st.kruskal(group1, group2)
    group1=RT_failedStop[0,:]
    group2=RT_failedStop[5,:]
    Hstat2, pval2 = st.kruskal(group1, group2)

    standard_plot_2lines_std_subplot_FOURINAROW([[Hstat1, pval1],[Hstat2, pval2]],[2,4,paramIdx+5], weights[i_sorted], np.nanmean(RT_corrGo, 1)[i_sorted], np.nanstd(RT_corrGo, 1)[i_sorted], 'correct Go', \
                         weights[i_sorted], np.nanmean(RT_failedStop, 1)[i_sorted], np.nanstd(RT_failedStop, 1)[i_sorted], 'failed Stop', \
                        changeParamName(paramname)+[' weights',''][int(paramname[-5:]=='rates')], 'Reaction time [ms]', 'plots/summary_RT_vs_w_'+paramname+'_'+str(param_id)+'.png', 9, ymin=0, ymax=450, xmin=loop_paramsSoll[0], xmax=loop_paramsSoll[-1]) # sem # np.nanstd
                        
                        
    if paramIdx==3:
        plt.subplots_adjust(left=0.09,bottom=0.13,top=0.92,right=0.98,hspace=0.05)
        left=plt.gcf().axes[0].get_position().x0
        right=plt.gcf().axes[7].get_position().x1
        plt.text(0.5*(left+right),0.03,'relative weight',fontsize=9, va='center', ha='center', transform=plt.gcf().transFigure)
        saveFolder='plots/multiloops/'
        try:
            os.mkdir(saveFolder)
        except:
            print(saveFolder+' not created')
        plt.savefig(saveFolder+'FourParameterVariations.svg', dpi=300)
        plt.ioff() 


def standard_plot_2lines_std_subplot_FOURINAROW(sig,subplotind, xdata1, ydata1, yerr1, label1, xdata2, ydata2, yerr2, label2, xlabel_text, ylabel_text, filename, fsize, ymin=0.0, ymax=100.0, xmin=0.0, xmax=1.0):
    #plt.ion()
    #plt.figure(figsize=(3,3), dpi=300)
    changeName = {'Cortex_SGPe_Arky weights':'cortex-Stop - GPe-Arky', 'ArkyD1Stop weights':'GPe-Arky - StrD1', 'ArkyD2Stop weights':'GPe-Arky - StrD2', 'ArkyFSIStop weights':'GPe-Arky - StrFSI'}
    nr=subplotind[2]
    plt.subplot(subplotind[0],subplotind[1],subplotind[2])  
    colorOfLabel = {'correct Stop':'purple', 'failed Stop':'tomato', 'correct Go':np.array([107,139,164])/255.}  
    plt.errorbar(xdata1, ydata1, yerr=yerr1, label=label1, color=colorOfLabel[label1])
    plt.errorbar(xdata2, ydata2, yerr=yerr2, label=label2, color=colorOfLabel[label2])
    
    if nr==1:
        fileopen='w'
    else:
        fileopen='a'
    with open('plots/multiloops/ArkyStop.txt', fileopen) as f:
        print(changeName[xlabel_text], file=f)
        if nr>4:
            print('correct Go RT: M(0) =',ydata1[0],'SD(0) =',yerr1[0],'M(1) =',ydata1[len(ydata1)-1],'SD(1) =',yerr1[len(yerr1)-1],'H = ',sig[0][0],'p = ',sig[0][1],'sig = ',sig[0][1]<(0.05/12.), file=f)
            print('failed Stop RT: M(0) =',ydata2[0],'SD(0) =',yerr2[0],'M(1) =',ydata2[len(ydata2)-1],'SD(1) =',yerr2[len(yerr2)-1],'H = ',sig[1][0],'p = ',sig[1][1],'sig = ',sig[1][1]<(0.05/12.), file=f)
        else:
            print('correct Go %: M(0) =',ydata1[0],'SD(0) =',yerr1[0],'M(1) =',ydata1[len(ydata1)-1],'SD(1) =',yerr1[len(yerr1)-1], file=f)
            print('correct Stop %: M(0) =',ydata2[0],'SD(0) =',yerr2[0],'M(1) =',ydata2[len(ydata2)-1],'SD(1) =',yerr2[len(yerr2)-1],'H = ',sig[0],'p = ',sig[1],'sig = ',sig[1]<(0.05/12.), file=f)
            

    #plt.xlabel(xlabel_text, fontsize=fsize)
    if paramIdx==0:
        plt.ylabel(ylabel_text, fontsize=fsize)
    ax = plt.gca()
    if nr>4:
        plt.xticks([0,0.2,0.4,0.6,0.8,1.0],[0,None,None,None,None,1.0])
    else:
        plt.xticks([0,0.2,0.4,0.6,0.8,1.0],[])
    if paramIdx==0 and nr==1:
        plt.yticks([0,25,50,75,100],[0,None,50,None,100])
    elif paramIdx==0 and nr==5:
        plt.yticks([0,100,200,300,400],[0,None,200,None,400])
    elif nr<=4:
        plt.yticks([0,25,50,75,100],[])
    else:
        plt.yticks([0,100,200,300,400],[])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.axis([xmin-(xmax-xmin)*0.02, xmax+(xmax-xmin)*0.02, ymin, ymax])
    if paramIdx==0 and nr==1:
        plt.legend(fontsize=fsize, loc='lower center')
    elif paramIdx==0 and nr==5:
        plt.legend(fontsize=fsize, loc='lower center')
    if nr<=4:
        plt.title(changeName[xlabel_text], fontsize=fsize)
        


def standard_plot_2lines_std_subplot(subplotind, xdata1, ydata1, yerr1, label1, xdata2, ydata2, yerr2, label2, xlabel_text, ylabel_text, filename, fsize, ymin=0.0, ymax=100.0, xmin=0.0, xmax=1.0):
    #plt.ion()
    #plt.figure(figsize=(3,3), dpi=300)
    plt.subplot(subplotind)  
    colorOfLabel = {'correct Stop':'purple', 'failed Stop':'tomato', 'correct Go':np.array([107,139,164])/255.}  
    plt.errorbar(xdata1, ydata1, yerr=yerr1, label=label1, color=colorOfLabel[label1])
    plt.errorbar(xdata2, ydata2, yerr=yerr2, label=label2, color=colorOfLabel[label2])
    plt.xlabel(xlabel_text, fontsize=fsize)
    plt.ylabel(ylabel_text, fontsize=fsize)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.axis([xmin-(xmax-xmin)*0.02, xmax+(xmax-xmin)*0.02, ymin, ymax])
    plt.legend(fontsize=fsize)
    #plt.tight_layout()
    #plt.ioff()
    #plt.savefig(filename, dpi=300)
    #plt.show()


def timing_plot_4lines_std(xdata1, ydata1, yerr1, label1, \
                           xdata2, ydata2, yerr2, label2, \
                           xdata3, ydata3, yerr3, label3, \
                           xdata4, ydata4, yerr4, label4, \
                           t_StopCue, xlabel_text, ylabel_text, filename, fsize, ymin=0.0, ymax=100.0):
    plt.ion()
    plt.figure(figsize=(3,3), dpi=300)
    plt.errorbar(xdata1, ydata1, yerr=yerr1, label=label1)
    plt.errorbar(xdata2, ydata2, yerr=yerr2, label=label2)
    plt.errorbar(xdata3, ydata3, yerr=yerr1, label=label3)
    plt.errorbar(xdata4, ydata4, yerr=yerr2, label=label4)    
    plt.xlabel(xlabel_text, fontsize=fsize)
    plt.ylabel(ylabel_text, fontsize=fsize)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.axis([plt.axis()[0], plt.axis()[1], ymin, ymax])
    plt.legend(fontsize=fsize)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(filename, dpi=300)
    plt.show()

def timing_plot_4lines_std_subplot(subplotind, xdata1, ydata1, yerr1, label1, \
                           xdata2, ydata2, yerr2, label2, \
                           xdata3, ydata3, yerr3, label3, \
                           xdata4, ydata4, yerr4, label4, \
                           t_StopCue, xlabel_text, ylabel_text, filename, fsize, ymin=0.0, ymax=100.0):
    #plt.ion()
    #plt.figure(figsize=(3,3), dpi=300)
    plt.subplot(subplotind)    
    plt.errorbar(xdata1, ydata1, yerr=yerr1, label=label1)
    plt.errorbar(xdata2, ydata2, yerr=yerr2, label=label2)
    plt.errorbar(xdata3, ydata3, yerr=yerr1, label=label3)
    plt.errorbar(xdata4, ydata4, yerr=yerr2, label=label4)    
    plt.xlabel(xlabel_text, fontsize=fsize)
    plt.ylabel(ylabel_text, fontsize=fsize)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plt.axis([plt.axis()[0], plt.axis()[1], ymin, ymax])
    plt.legend(fontsize=fsize)
    #plt.tight_layout()
    #plt.ioff()
    #plt.savefig(filename, dpi=300)
    #plt.show()

def sem(data, dim):
    N = len( np.nonzero(np.isnan(data)==False)[0] )
    return np.nanstd(data, dim) / np.sqrt(N)


def read_paramfile(curr_ID):
    file_rows = {}
    wf = {}
    params = {}
    #with open('sim_param_file.csv', 'rb') as csvfile: # Python 2
    with open('sim_param_file.csv', newline='') as csvfile:    # Python 3        
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            file_rows[str(int(reader.line_num))] = row
            if file_rows[str(int(reader.line_num))][0] == "params['id']":
                param_names = file_rows[str(int(reader.line_num))]
            if file_rows[str(int(reader.line_num))][0] == str(curr_ID):
                paramset = file_rows[str(int(reader.line_num))]
        for i_param in range(len(paramset)):
            strsplit_params = param_names[i_param].split('params')
            strsplit_wf = param_names[i_param].split('wf')
            strsplit_apostr = param_names[i_param].split("'")
            if len(strsplit_params) > 1: # name contains 'params'
                params[strsplit_apostr[1]] = float(paramset[i_param])
            elif len(strsplit_wf) > 1: # name contains 'wf'
                wf[strsplit_apostr[1]] = float(paramset[i_param])
    return params, wf


def plot_timing(param_id, network_array, paramname, loadFolder):
    #loop_params, paramname = np.load('data/cycle_params_'+paramname+'_id'+str(int(param_id))+'.npy', allow_pickle=True) # new   
    loop_params, paramname = np.load(loadFolder+'cycle_params_'+paramname+'_id'+str(int(param_id))+str(0)+'.npy', allow_pickle=True)

    print("paramname, param_id, loop_params = ", paramname, param_id, loop_params)
    n_loop_cycles = len(loop_params)

    STN_Stop_mean_0_0, STN_Stop_sem_0_0 = np.load('data/STN_rate_Stop_mean_std_'+ str(int(param_id))+'_'+str(network_array[0])+'_'+str(0)+'.npy')
    STN_mean_Stop = np.nan * np.ones([n_loop_cycles, len(network_array), len(STN_Stop_mean_0_0)])
    SNr_mean_Stop = np.nan * np.ones([n_loop_cycles, len(network_array), len(STN_Stop_mean_0_0)])    
    Arky_mean_Stop = np.nan * np.ones([n_loop_cycles, len(network_array), len(STN_Stop_mean_0_0)])        
    Proto_mean_Stop = np.nan * np.ones([n_loop_cycles, len(network_array), len(STN_Stop_mean_0_0)])    
    weights = np.nan * np.ones(n_loop_cycles)        

    plt.figure(figsize=(3.5,4), dpi=300)
    params, wf = read_paramfile(param_id)    
    t_init = params['t_init'] # 100
    t_SSD = params['t_SSD']    
    dt = float(params['dt'])
    itmin = int((t_init + t_SSD - 50)/dt)
    itmax = int((t_init + t_SSD + 150)/dt)
    fsize = 6
    ax = plt.gca()    

    for i_cycle in range(n_loop_cycles):
        for i_netw_rep in network_array:
            STN_mean_Stop[i_cycle, i_netw_rep, : ], STN_sem_stop = np.load(loadFolder+'STN_rate_Stop_mean_std_'+ str(int(param_id))+'_'+str(i_netw_rep)+'_'+str(i_cycle)+'.npy')
            SNr_mean_Stop[i_cycle, i_netw_rep, : ], SNr_sem_stop = np.load(loadFolder+'SNr_rate_Stop_mean_std_'+ str(int(param_id))+'_'+str(i_netw_rep)+'_'+str(i_cycle)+'.npy')            
            Arky_mean_Stop[i_cycle, i_netw_rep, : ], Arky_sem_stop = np.load(loadFolder+'GPeArky_rate_Stop_mean_std_'+ str(int(param_id))+'_'+str(i_netw_rep)+'_'+str(i_cycle)+'.npy')            
            Proto_mean_Stop[i_cycle, i_netw_rep, : ], Arky_sem_stop = np.load(loadFolder+'GPeProto_rate_Stop_mean_std_'+ str(int(param_id))+'_'+str(i_netw_rep)+'_'+str(i_cycle)+'.npy')                        
            weights[i_cycle] = loop_params[i_cycle] # params['CortexGo_rates']
        #norm_zsc_STN = custom_zscore(STN_mean_Stop[i_cycle, i_netw_rep, itmin : itmax], STN_mean_Stop[0, 0, itmin : itmax]) # <
        norm_zsc_STN = custom_zscore( np.nanmean(STN_mean_Stop[i_cycle, :, itmin : itmax], 0),  np.nanmean(STN_mean_Stop[i_cycle, :, itmin : itmax], 0)) # <        
        norm_zsc_STN -= np.nanmin(norm_zsc_STN)
        norm_zsc_STN /= np.nanmax(norm_zsc_STN) 
        norm_zsc_SNr = custom_zscore( np.nanmean(SNr_mean_Stop[i_cycle, :, itmin : itmax], 0), np.nanmean(SNr_mean_Stop[i_cycle, :, itmin : itmax], 0))
        norm_zsc_SNr -= np.nanmin(norm_zsc_SNr)
        norm_zsc_SNr /= np.nanmax(norm_zsc_SNr)    
        norm_zsc_Arky = custom_zscore( np.nanmean(Arky_mean_Stop[i_cycle, :, itmin : itmax], 0), np.nanmean(Arky_mean_Stop[i_cycle, :, itmin : itmax], 0))
        norm_zsc_Arky -= np.nanmin(norm_zsc_Arky)
        norm_zsc_Arky /= np.nanmax(norm_zsc_Arky)        
        norm_zsc_Proto = custom_zscore( np.nanmean(Proto_mean_Stop[i_cycle, :, itmin : itmax], 0), np.nanmean(Proto_mean_Stop[i_cycle, :, itmin : itmax], 0))
        norm_zsc_Proto -= np.nanmin(norm_zsc_Proto)               
        norm_zsc_Proto /= np.nanmax(norm_zsc_Proto) 
        yshift = i_cycle * 2                        
        plt.plot(range(len(norm_zsc_STN)), yshift + norm_zsc_STN, color='orange', lw=1)     
        plt.plot(range(len(norm_zsc_SNr)), yshift + norm_zsc_SNr, color='tomato', lw=1)              
        plt.plot(range(len(norm_zsc_Arky)), yshift + norm_zsc_Arky, color='cyan', lw=1)                  
        plt.plot(range(len(norm_zsc_Proto)), yshift + norm_zsc_Proto, color='blue', lw=1)   
    i_sorted = np.argsort(weights)

    ax=plt.gca()
    ax.set_xticks([0, 50/dt, 100/dt, 150/dt])    
    ax.set_xticklabels([-0.05, 0, 0.05, 0.1])
    plt.xlabel('Time from Stop cue [s]', fontsize=fsize)
    ax.set_yticks(2 * np.arange(n_loop_cycles))        
    ax.set_yticklabels(weights[i_sorted])        
    plt.ylabel('Parameter value', fontsize=fsize)
    plt.tight_layout()  

    saveFolder='plots/multiloops/'+str(param_id)+'/'
    try:
        os.mkdir(saveFolder)
    except:
        print(saveFolder+' not created')
    plt.savefig(saveFolder+'StopTiming_paramsid'+str(int(param_id))+'_'+str(paramname)+'.png')
    plt.show()    
    #'''        

def plot_RT_distributions_correctStop():
    param_id_list = ['8007', '8008', '8009', '8010', '8011', '8012','8013'] # 8008-8013
    network_IDs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    n_networks = len(network_IDs)
    trials = 200 
    t_init = params['t_init'] # 100
    dt = float(params['dt'])

    RT_Go = {
                    '8007' : np.nan * np.ones([n_networks, trials]),
                    '8008' : np.nan * np.ones([n_networks, trials]), 
                    '8009' : np.nan * np.ones([n_networks, trials]),                     
                    '8010' : np.nan * np.ones([n_networks, trials]),  
                    '8011' : np.nan * np.ones([n_networks, trials]),                                                                                
                    '8012' : np.nan * np.ones([n_networks, trials]),                                         
                    '8013' : np.nan * np.ones([n_networks, trials]),                                         
            }
    RT_Stop = {
                    '8007' : np.nan * np.ones([n_networks, trials]),
                    '8008' : np.nan * np.ones([n_networks, trials]), 
                    '8009' : np.nan * np.ones([n_networks, trials]),                     
                    '8010' : np.nan * np.ones([n_networks, trials]),                                         
                    '8011' : np.nan * np.ones([n_networks, trials]),                                                                                
                    '8012' : np.nan * np.ones([n_networks, trials]),                                         
                    '8013' : np.nan * np.ones([n_networks, trials]),                                         
            }

    pct_CorrectStop = {
                    '8007' : np.nan * np.ones(n_networks),
                    '8008' : np.nan * np.ones(n_networks), 
                    '8009' : np.nan * np.ones(n_networks),                     
                    '8010' : np.nan * np.ones(n_networks),                                         
                    '8011' : np.nan * np.ones(n_networks),                                                                                
                    '8012' : np.nan * np.ones(n_networks),                                         
                    '8013' : np.nan * np.ones(n_networks),                                         
            }

    pct_FailedStop = {
                    '8007' : np.nan * np.ones(n_networks),
                    '8008' : np.nan * np.ones(n_networks), 
                    '8009' : np.nan * np.ones(n_networks),                     
                    '8010' : np.nan * np.ones(n_networks),                                         
                    '8011' : np.nan * np.ones(n_networks),                                                                                
                    '8012' : np.nan * np.ones(n_networks),                                         
                    '8013' : np.nan * np.ones(n_networks),                                         
            }

    pct_CorrectGo = {
                    '8007' : np.nan * np.ones(n_networks),
                    '8008' : np.nan * np.ones(n_networks), 
                    '8009' : np.nan * np.ones(n_networks),                     
                    '8010' : np.nan * np.ones(n_networks),                                         
                    '8011' : np.nan * np.ones(n_networks),                                                                                
                    '8012' : np.nan * np.ones(n_networks),                                         
                    '8013' : np.nan * np.ones(n_networks),                                         
            }

    pct_FailedGo = {
                    '8007' : np.nan * np.ones(n_networks),
                    '8008' : np.nan * np.ones(n_networks), 
                    '8009' : np.nan * np.ones(n_networks),                     
                    '8010' : np.nan * np.ones(n_networks),                                         
                    '8011' : np.nan * np.ones(n_networks),                                                                                
                    '8012' : np.nan * np.ones(n_networks),                                         
                    '8013' : np.nan * np.ones(n_networks),                                         
            }


    rsp_RT_Go = {
                    '8007' : [],
                    '8008' : [], 
                    '8009' : [],                     
                    '8010' : [],                     
                    '8011' : [],                     
                    '8012' : [],                     
                    '8013' : [],                     
            }            
    rsp_RT_Stop = {
                    '8007' : [],
                    '8008' : [], 
                    '8009' : [],                     
                    '8010' : [],                     
                    '8011' : [],                     
                    '8012' : [],                     
                    '8013' : [],                     
            }            

    nz_Go = {
                    '8007' : [],
                    '8008' : [], 
                    '8009' : [],                     
                    '8010' : [],                     
                    '8011' : [],                     
                    '8012' : [],                     
                    '8013' : [],                   
            }            
    nz_Stop = {
                    '8007' : [],
                    '8008' : [], 
                    '8009' : [],   
                    '8010' : [],                     
                    '8011' : [],                     
                    '8012' : [],                     
                    '8013' : [],                     
            }            


    for param_id in param_id_list: 
        for i_netw, netw in enumerate(network_IDs):        
            datapath = 'data/'#'/media/lorenz/Volume/Lorenz_AFS/data_Oliver/'
            mInt_stop = np.load(datapath+'Integrator_ampa_Stop_'+str(netw)+'_id'+str(int(param_id))+'.npy')    
            mInt_go = np.load(datapath+'Integrator_ampa_Go_'+str(netw)+'_id'+str(int(param_id))+'.npy')     
            results_RT = np.load(datapath+'resultsRT_'+str(netw)+'_param_'+'CortexStop_rates'+'_cycle'+str(0)+'id'+str(int(param_id))+'.npy', allow_pickle=True) 
            n_StopTrials = results_RT.item()['nFailedStopTrials'] + results_RT.item()['nCorrectStopTrials']
            n_GoTrials = results_RT.item()['nFailedGoTrials'] + results_RT.item()['nCorrectGoTrials']
            pct_CorrectStop[param_id][i_netw] = 100 * results_RT.item()['nCorrectStopTrials'] / float(n_StopTrials)
            pct_FailedStop[param_id][i_netw] = 100 * results_RT.item()['nFailedStopTrials'] / float(n_StopTrials)
            pct_CorrectGo[param_id][i_netw] = 100 * results_RT.item()['nCorrectGoTrials'] / float(n_GoTrials)
            pct_FailedGo[param_id][i_netw] = 100 * results_RT.item()['nFailedGoTrials'] / float(n_GoTrials)
            #print('pct_CorrectStop[param_id][i_netw] = ', pct_CorrectStop[param_id][i_netw])
            rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(trials)), trials], order='F')
            rsp_mInt_go = np.reshape(mInt_go, [int(mInt_go.shape[0] / float(trials)), trials], order='F')
            mInt_maxpertrial = np.nanmax(rsp_mInt_go, 0)
            mInt_maxpertrial = np.nanmax(rsp_mInt_stop, 0)
            for i_trial in range(trials):
                if np.nanmax(rsp_mInt_go[:, i_trial]) >= Integrator.threshold: 
                    RT_Go[param_id][i_netw, i_trial] = np.nonzero(rsp_mInt_go[:, i_trial] >= Integrator.threshold)[0][0]            
                if np.nanmax(rsp_mInt_stop[:, i_trial]) >= Integrator.threshold: 
                    RT_Stop[param_id][i_netw, i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= Integrator.threshold)[0][0] # Correct

        rsp_RT_Go[param_id] = np.reshape(RT_Go[param_id], n_networks * trials)
        rsp_RT_Stop[param_id] = np.reshape(RT_Stop[param_id], n_networks * trials)    
        nz_Go[param_id] = np.nonzero(np.isnan(rsp_RT_Go[param_id])==False)    
        nz_Stop[param_id] = np.nonzero(np.isnan(rsp_RT_Stop[param_id])==False)
        counts_Go, bins_Go = np.histogram(rsp_RT_Go[param_id][nz_Go[param_id]] * dt, 10) #     
        counts_Stop, bins_Stop = np.histogram(rsp_RT_Stop[param_id][nz_Stop[param_id]] * dt, 10) # 

    mean_CorrectGo = np.round( (np.nanmean(rsp_RT_Go[param_id][nz_Go[param_id]]) - t_init/dt)*dt, 1)    
    mean_FailedStop = np.round( (np.nanmean(rsp_RT_Stop[param_id][nz_Stop[param_id]]) - t_init/dt)*dt, 1)

    #if counts_Go.max() > 0:
    #    plt.bar(bins_Go[:-1], np.array(counts_Go) * (1.0/counts_Go.max()), width=np.diff(bins_Go)[0], alpha=0.5, color='b') #
    #if counts_Stop.max() > 0: 
    #    plt.bar(bins_Stop[:-1], np.array(counts_Stop) * (1.0/counts_Stop.max()), width=np.diff(bins_Stop)[0], alpha=0.5, color='g') # 

    plt.ion()

    ### FIGURE PARAMETERS
    fig = {}
    fig['x']             = 87
    fig['y']             = 100
    fig['modeList']      = ['activate', 'deactivate']
    fig['paramIDList'] = [['8007','8008', '8009', '8010'], ['8007', '8011', '8012', '8013']]
    fig['conditionName'] = {'activate8007':'active Stop component:\nNone', 'activate8008':'active Stop component:\nGPe-Arky', 'activate8009':'active Stop component:\nSTN', 'activate8010':'active Stop component:\nGPe-Cp',
                            'deactivate8007':'deactive Stop component:\nNone','deactivate8011':'deactive Stop component:\nGPe-Arky','deactivate8012':'deactive Stop component:\nSTN','deactivate8013':'deactive Stop component:\nGPe-Cp'}
    fig['colors']        = {'8007':np.array([107,139,164])/255., '8008':np.array([0,255,255])/255., '8009':np.array([219,180,12])/255., '8010':np.array([173,129,80])/255.,
                                                           '8011':np.array([0,255,255])/255., '8012':np.array([219,180,12])/255., '8013':np.array([173,129,80])/255.}
    

    font = {}
    font["axTicks"] = {'labelsize': 9}
    font["axLabel"] = {'fontsize': 9, 'fontweight' : 'normal'}

    figRT = {}
    figRT['maxRT']       = 700#ms
    figRT['binSize']     = 15#ms
    figRT['xlim']        = [0,600]#ms
    figRT['ylim']        = [0,0.41]
    figRT['xtickNames']  = [0,200,400,600]
    figRT['ytickNames']  = [0,0.2,0.4]
    figRT['xlabel']      = 'Reaction time [ms]'
    figRT['ylabel']      = 'failed Stop trials PDF'
    figRT['bottom']      = 0.12
    figRT['top']         = 0.99
    figRT['left']        = 0.18
    figRT['right']       = 0.96
    figRT['hspace']      = 0.05
    figRT['linewidth']   = 2

    figBar = {}
    figBar['min_barH']      = 10
    figBar['conditionName'] = {'activate8007':'None', 'activate8008':'GPe-Arky', 'activate8009':'STN', 'activate8010':'GPe-Cp',
                               'deactivate8007':'None','deactivate8011':'GPe-Arky','deactivate8012':'STN','deactivate8013':'GPe-Cp'}
    figBar['xlabel']        = ['active Stop component', 'deactive Stop component']
    figBar['ylabel']        = ['% failed Stop trials', '% failed Stop trials']
    figBar['ylim']          = [0,105]
    figBar['alpha']         = 0.05
    figBar['ytickNames']    = [0,20,40,60,80,100]
    
    ### SIGNIFICANCE
    with open('plots/multiloops/stopComponents.txt', 'w') as f:
        sig = {}
        for modeIdx, mode in enumerate(fig['modeList']):
            paramIDList = fig['paramIDList'][modeIdx][1:]
            if mode=='activate':
                group1 = pct_CorrectGo['8007']
            else:
                group1 = pct_FailedStop['8007']
            print(figBar['xlabel'][modeIdx],file=f)
            print(figBar['conditionName'][mode+'8007'],'M =',np.nanmean(group1),'SD =',np.nanstd(group1), file=f)
            for paramID_Idx, paramID in enumerate(paramIDList):
                group2 = pct_FailedStop[paramID]
                print(figBar['conditionName'][mode+paramID],'M =',np.nanmean(group2),'SD =',np.nanstd(group2), file=f)
                if len(np.unique(group1))==1 and len(np.unique(group2))==1 and group1[0]==group2[0]:
                    sig[mode+'8007vs'+paramID] = [None, 1, False]
                else:
                    Hstat, pval = st.kruskal(group1, group2)
                    sig[mode+'8007vs'+paramID] = [Hstat, pval, pval<figBar['alpha']]
        correctedAlpha=figBar['alpha']/len(list(sig.keys()))
        for key in sig.keys():
            sig[key][2] = sig[key][1]<correctedAlpha



    ### FIGURE 1 and 2: RT-DISTRIBUTION FOR 1:ACTIVATING AND 2:Deactivating SINGLE COMPONENTS 
    for modeIdx, mode in enumerate(fig['modeList']):
        paramIDList = fig['paramIDList'][modeIdx]
        plt.figure(figsize=(fig['x']/25.4,fig['y']/25.4), dpi=300)

        for paramID_Idx, paramID in enumerate(paramIDList):
            if paramID == '8007' and mode == 'activate':
                RT_List = rsp_RT_Go[paramID][nz_Go[paramID]] * dt
            else:
                RT_List = rsp_RT_Stop[paramID][nz_Stop[paramID]] * dt
    
            counts, bins = np.histogram(RT_List, np.arange(t_init,t_init+figRT['maxRT']+1,figRT['binSize']))

            plt.subplot(len(paramIDList),1,paramID_Idx+1)
            interpolatedBins = np.arange(bins[0],bins[-2]+1,1).astype(int)
            interpolatedCounts = interpolate.interp1d(bins[:-1], counts/float(counts.sum()), kind='quadratic')
            plt.plot(interpolatedBins, interpolatedCounts(interpolatedBins), lw=figRT['linewidth'], color=fig['colors'][paramID])
            plt.xlim(figRT['xlim'][0]+t_init,figRT['xlim'][1]+t_init)
            plt.ylim(figRT['ylim'][0],figRT['ylim'][1])
            if paramID_Idx!=len(paramIDList)-1:
                plt.xticks(np.array(figRT['xtickNames'])+t_init,[])
                plt.yticks(figRT['ytickNames'],[])
            else:
                plt.xticks(np.array(figRT['xtickNames'])+t_init,figRT['xtickNames'])
                plt.yticks(figRT['ytickNames'],figRT['ytickNames'])
                plt.xlabel(figRT['xlabel'], **font["axLabel"])
            plt.gca().tick_params(axis='both', which='both', **font["axTicks"])
            plt.text(0.05,0.9,fig['conditionName'][mode+paramID],ha='left',va='top',transform=plt.gca().transAxes, **font["axLabel"])
        plt.subplots_adjust(bottom = figRT['bottom'], top = figRT['top'], right = figRT['right'], left = figRT['left'], hspace = figRT['hspace'])
        plotsBottom=plt.gcf().axes[len(plt.gcf().axes)-1].get_position().y0
        plotsTop=plt.gcf().axes[0].get_position().y1
        plt.text(0.05,0.5*(plotsTop+plotsBottom),figRT['ylabel'],ha='center',va='center',transform=plt.gcf().transFigure, rotation=90, **font["axLabel"])
        plt.savefig('plots/multiloops/RT_DISTRIBUTIONS_'+mode+'.png', dpi=300)
        plt.close()


    ### FIGURE 3 and 4: PERCENT CORRECT FOR 1:ACTIVATING AND 2:Deactivating SINGLE COMPONENTS 
    fullHeight = {}
    for modeIdx, mode in enumerate(fig['modeList']):
        yMaxList = []
        paramIDList = fig['paramIDList'][modeIdx]
        plt.figure(figsize=(fig['x']/25.4,fig['y']/25.4), dpi=300)

        for paramID_Idx, paramID in enumerate(paramIDList):
            if mode=='activate' and paramID=='8007':
                barH   = np.nanmean(pct_CorrectGo[paramID][:])
                barErr = np.nanstd(pct_CorrectGo[paramID][:])
            else:
                barH   = np.nanmean(pct_FailedStop[paramID][:])
                barErr = np.nanstd(pct_FailedStop[paramID][:])
            barH += figBar['min_barH']*int(barH==0)
            fullHeight[mode+paramID] = barH+barErr

            plt.bar([paramID_Idx], barH, yerr=barErr, width=1, label=fig['conditionName'][mode+paramID], color=fig['colors'][paramID], capsize=6)

            if paramID!='8007' and mode+'8007' in list(fullHeight.keys()) and sig[mode+'8007vs'+paramID][2]==True:
                x0 = 0
                x1 = paramID_Idx
                y0 = fullHeight[mode+'8007']+4
                y1 = fullHeight[mode+paramID]+4
                found_yMax=False
                change=0
                while found_yMax==False:
                    yMax = np.max([y0,y1])+4+change
                    mask1=np.array(yMaxList)>yMax-3
                    mask2=np.array(yMaxList)<yMax+3
                    mask=mask1*mask2
                    if True in mask:
                        change+=3.5
                    else:
                        found_yMax=True
                yMaxList.append(yMax)
                plt.plot([x0,x0],[y0,yMax],color='k',lw=1)
                plt.plot([x0,x1],[yMax,yMax],color='k',lw=1)
                plt.plot([x1,x1],[yMax,y1],color='k',lw=1)
                plt.text(0.5*(x0+x1),yMax-1.5,'*',va='bottom',ha='center', **font["axLabel"])

        #plt.xlim(figRT['xlim'][0]+t_init,figRT['xlim'][1]+t_init)
        if max(yMaxList)>figBar['ylim'][1]:
            ylimMax = max(yMaxList)+3.5
        else:
            ylimMax = figBar['ylim'][1]
        plt.ylim(figBar['ylim'][0],ylimMax)
        plt.xticks(np.arange(len(paramIDList)),[figBar['conditionName'][mode+paramID] for paramID in paramIDList])
        plt.yticks(figBar['ytickNames'],figBar['ytickNames'])
        plt.gca().tick_params(axis='both', which='both', **font["axTicks"])
        plt.xlabel(figBar['xlabel'][modeIdx], **font["axLabel"])
        plt.ylabel(figBar['ylabel'][modeIdx], **font["axLabel"])
        plt.subplots_adjust(bottom = figRT['bottom'], top = figRT['top'], right = figRT['right'], left = figRT['left'], hspace = figRT['hspace'])
        plt.savefig('plots/multiloops/PERFORMANCE_'+mode+'.png', dpi=300)
        plt.close()

       



if __name__ == '__main__':
    #plot_percent_correctGo_vs_GoRate_cycles(5111, 5) # 5111 # 5112
    #plot_percent_correctGo_vs_StopRate_cycles(5111, 5) # 5111 # 5112
    #plot_percent_correctGo_vs_weight_cycles(5111, 5, 'Proto-to-Arky')
    #plot_percent_correctGo_vs_weight_cycles(5111, 5, 'Proto-to-SNr')
    #plot_RT_vs_weight_cycles(5111, 5, 'Proto-to-SNr')
    #plot_percent_correctGo_vs_weight_cycles(5111, 5, 'STN-to-SNr')
    #plot_percent_correctGo_vs_weight_cycles(5111, 5, 'FSI-to-StrD1D2')
    #plot_percent_correctGo_vs_weight_cycles(5111, 5, 'GPe_ArkySTR_FSI')
    #plot_percent_correctGo_vs_weight_cycles(5111, 5, 'GPe_ArkySTR_D1')
    #plot_percent_correctGo_vs_weight_cycles(5111, 5, 'STR_D1SNr')
    #plot_percent_correctGo_vs_prob_cycles(5111, 5, 'GPe_Arky_spikeprob')
    #plot_STR_stop(5111)

    """
    FINAL RESULTS DATA:

    vary Arky-SD1 (id 8014, with tonic compensation): 
        - hinton:data/ArkyOutputs_gyrus_and_striatum/; network_array = [1,2,3,4,5,6,7,8,9,10]; X = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        - hinton:data/ArkyOutputs_gyrus_and_striatum/; network_array = [11,12,13,14,15,16,17,18,19,20]; X = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    vary Arky-SD2 (id 8014, with tonic compensation): 
        - hinton:data/ArkyOutputs_gyrus_and_striatum/; network_array = [1,2,3,4,5,6,7,8,9,10]; X = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        - hinton:data/ArkyOutputs_gyrus_and_striatum/; network_array = [11,12,13,14,15,16,17,18,19,20]; X = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    vary Arky-FSI (id 8014, with tonic compensation): 
        - hinton:data/ArkyOutputs_gyrus_and_striatum/; network_array = [1,2,3,4,5,6,7,8,9,10]; X = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        - hinton:data/ArkyOutputs_gyrus_and_striatum/; network_array = [11,12,13,14,15,16,17,18,19,20]; X = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    vary CorStop-Arky (id 8007): 
        - preliminar data: hinton:data/; network_array = [0,11,12,13,14,20]; X = [0.0, 0.5, 1.0, 1.5, 2.0]
        TODO: run network_array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]; X = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    vary Go rate (id 8007):
        - preliminar data: hinton:data/; network_array = [11,12,13,14]; X = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
        TODO: run network_array = [1,2,3,4,5,6,7,8,9,10,15,16,17,18,19,20]; X = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    vary Stop rate (id 8007):
        - preliminar data: hinton:data/; network_array = [11,12,13,14]; X = [0.0, 0.5, 1.0, 1.5, 2.0]
        TODO: run network_array = [1,2,3,4,5,6,7,8,9,10,15,16,17,18,19,20]; X = [0.0, 0.5, 1.0, 1.5, 2.0]

    first plot = [CorStop-Arky, Arky-SD1, Arky-SD2, Arky-FSI]
    second plot = [Go rate, Stop rate]
    """


    #n_networks = 20#10#6 # 2
    #network_array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]#[1,2,3,4,5,6,7,8,9,10]#[11,12,13,14] # [1] # range(n_networks) # [1,2,3,4] #  [0,2,3,4] #    [0]
    #params_id = 8014#8007 # 6097 # 6083
    #loadFolder = '/scratch/olmai/parameterVariationData/orig/'#'data/ArkyOutputs_gyrus_and_striatum/'#'data/'

    #plot_percent_correctGo_vs_GoRate_cycles(params_id, network_array) # Useful
    #plot_percent_correctGo_vs_StopRate_cycles(params_id, network_array) # Useful
    #plot_percent_correctGo_vs_CortexPauseRate_cycles(params_id, network_array) # Weird - counterintuitive?!  

    #paramNameList = ['ArkyD1Stop', 'ArkyD2Stop', 'ArkyFSIStop']#['Cortex_SGPe_Arky']#['CortexGo_rates','CortexStop_rates','Cortex_SGPe_Arky']#['CortexGo_rates', 'CortexStop_rates', 'Stop1rates', 'Cortex_SGPe_Arky', 'Stop_Proto2', 'STR_D2GPe_Arky', 'D2_Proto2', 'GPe_ArkySTR_D1', 'GPe_ArkySTR_D2', 'Proto2_Int', 'GPe_ProtoSTR_FSI', 'STN_SNr']

    paramNameList = ['Cortex_SGPe_Arky', 'ArkyD1Stop', 'ArkyD2Stop', 'ArkyFSIStop', 'CortexGo_rates']
    network_array = {
                    'Cortex_SGPe_Arky' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                    'ArkyD1Stop'       : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
                    'ArkyD2Stop'       : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
                    'ArkyFSIStop'      : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
                    'CortexGo_rates'   : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                    'CortexStop_rates' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                    }
    param_id =      {
                    'Cortex_SGPe_Arky' : 8007,
                    'ArkyD1Stop'       : 8014, 
                    'ArkyD2Stop'       : 8014, 
                    'ArkyFSIStop'      : 8014, 
                    'CortexGo_rates'   : 8007, 
                    'CortexStop_rates' : 8007
                    }
    loadFolder =    {
                    'Cortex_SGPe_Arky' : 'data/',
                    'ArkyD1Stop'       : 'data/ArkyOutputs_gyrus_and_striatum/', 
                    'ArkyD2Stop'       : 'data/ArkyOutputs_gyrus_and_striatum/', 
                    'ArkyFSIStop'      : 'data/ArkyOutputs_gyrus_and_striatum/', 
                    'CortexGo_rates'   : 'data/', 
                    'CortexStop_rates' : 'data/'
                    }
    variations =    {
                    'Cortex_SGPe_Arky' : [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    'ArkyD1Stop'       : [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                    'ArkyD2Stop'       : [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                    'ArkyFSIStop'      : [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                    'CortexGo_rates'   : [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5], 
                    'CortexStop_rates' : [0.0, 0.5, 1.0, 1.5, 2.0]
                    }
    saveFormat =    {
                    'Cortex_SGPe_Arky' : 'png',
                    'ArkyD1Stop'       : 'png', 
                    'ArkyD2Stop'       : 'png', 
                    'ArkyFSIStop'      : 'png', 
                    'CortexGo_rates'   : 'svg', 
                    'CortexStop_rates' : 'svg'
                    }
    
    # part of FIGURE 4 in manuscript
    for paramIdx, paramname in enumerate(['CortexGo_rates']):
        plot_pctcorrect_timing_RT_vs_weight_cycles(param_id[paramname], network_array[paramname], paramname, loadFolder[paramname], variations[paramname], saveFormat[paramname])

    # FIGURE 10 in manuscript
    for paramIdx, paramname in enumerate(['Cortex_SGPe_Arky', 'ArkyD1Stop', 'ArkyD2Stop', 'ArkyFSIStop']):
        plot_pctcorrect_timing_RT_vs_weight_cycles_FOURINAROW(param_id[paramname], network_array[paramname], paramname, loadFolder[paramname], variations[paramname], paramIdx)             
   
    # FIGURE 8 in manuscript
    plot_RT_distributions_correctStop()        

    '''#    
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'Cortex_SGPe_Arky') # TODO: Repeat at higher resolution   
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'Cortex_SGPe_Proto') # Unclear
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'STR_D1STR_D2')    
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'STR_D2STR_D1')        
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'STR_FSISTR_D1') #        
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'STR_FSISTR_D2') # 
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'STR_D2GPe_Arky')
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'STR_D2GPe_Proto')
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'GPe_ArkySTR_D1')
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'GPe_ArkySTR_D2')    
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'GPe_ArkySTR_FSI')  
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'GPe_ArkyGPe_Proto')        
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'GPe_ArkyCortex_G')       
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'GPe_ProtoSTR_FSI')    
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'GPe_ProtoGPe_Arky')            
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'GPe_ProtoSTN')  
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'STNGPe_Arky') 
    plot_percent_correctGo_vs_weight_cycles(params_id, network_array, 'STN_SNr')     
    '''
    '''#
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'Cortex_SGPe_Arky')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'Cortex_SGPe_Proto')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'STR_D1STR_D2')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'STR_D2STR_D1')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'STR_FSISTR_D1')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'STR_FSISTR_D2')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'STR_D2GPe_Arky')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'STR_D2GPe_Proto')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'GPe_ArkySTR_D1')
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'GPe_ArkySTR_D2')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'GPe_ArkySTR_FSI')   
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'GPe_ArkyGPe_Proto')        
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'GPe_ArkyCortex_G')
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'GPe_ProtoSTR_FSI')    
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'GPe_ProtoGPe_Arky')            
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'GPe_ProtoSTN')        
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'STNGPe_Arky')       
    plot_StopTiming_vs_weight_cycles(params_id, network_array, 'STN_SNr')     
    '''
    '''# Most results above somewhat unclear - increase param. range?
    plot_RT_vs_weight_cycles(params_id, network_array, 'Cortex_SGPe_Arky')       
    plot_RT_vs_weight_cycles(params_id, network_array, 'Cortex_SGPe_Proto')    
    plot_RT_vs_weight_cycles(params_id, network_array, 'STR_D1STR_D2')    
    plot_RT_vs_weight_cycles(params_id, network_array, 'STR_D2STR_D1')    
    plot_RT_vs_weight_cycles(params_id, network_array, 'STR_FSISTR_D1')    
    plot_RT_vs_weight_cycles(params_id, network_array, 'STR_FSISTR_D2')    
    plot_RT_vs_weight_cycles(params_id, network_array, 'STR_D2GPe_Arky')    
    plot_RT_vs_weight_cycles(params_id, network_array, 'STR_D2GPe_Proto')    
    plot_RT_vs_weight_cycles(params_id, network_array, 'GPe_ArkySTR_D1')
    plot_RT_vs_weight_cycles(params_id, network_array, 'GPe_ArkySTR_D2')    
    plot_RT_vs_weight_cycles(params_id, network_array, 'GPe_ArkySTR_FSI')   
    plot_RT_vs_weight_cycles(params_id, network_array, 'GPe_ArkyGPe_Proto')        
    plot_RT_vs_weight_cycles(params_id, network_array, 'GPe_ArkyCortex_G')
    plot_RT_vs_weight_cycles(params_id, network_array, 'GPe_ProtoSTR_FSI')    
    plot_RT_vs_weight_cycles(params_id, network_array, 'GPe_ProtoGPe_Arky')            
    plot_RT_vs_weight_cycles(params_id, network_array, 'GPe_ProtoSTN')        
    plot_RT_vs_weight_cycles(params_id, network_array, 'STNGPe_Arky')                
    plot_RT_vs_weight_cycles(params_id, network_array, 'STN_SNr')       
    '''    
    
