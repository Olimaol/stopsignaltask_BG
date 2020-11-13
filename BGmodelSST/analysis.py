import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin
import scipy.stats as st
from ANNarchy import population_rate
import matplotlib.lines as mlines
from BGmodelSST.sim_params import params


def calc_KW_stats_all(data, pvalue_times, name=''):
    # Input: Raw population rates across trials. Dims: [trials, timesteps]

    Hstat_all = []
    pval_all = []

    for pop, rates in data.items():
        H, p = calc_KruskalWallis(rates[0], rates[1], pvalue_times, 0, name)
        Hstat_all.append(H)
        pval_all.append(p)

    return Hstat_all, pval_all


def calc_KruskalWallis(rates_FailedStop, rates_CorrectStop, pvalue_times, number, name, printout = False):
    # Input:
    # rates_FailedStop, rates_CorrectStop:
    # Raw population rates across trials in ALL time bins
    n_bins = len(pvalue_times) - 1    
    Hstat = np.nan * np.ones(n_bins)
    pval = np.nan * np.ones(n_bins)      
    boxlen = int(1.0 / params['general_dt'])
    if printout: print(name, number)
    for i_bin in range(n_bins):
        meanrate_FailedStop_bin = np.nanmean(rates_FailedStop[:, pvalue_times[i_bin] : pvalue_times[i_bin+1]], 1) # Average across time steps
        meanrate_CorrectStop_bin = np.nanmean(rates_CorrectStop[:, pvalue_times[i_bin] : pvalue_times[i_bin+1]], 1) # Average across time steps
        meanrate_FailedStop_bin = meanrate_FailedStop_bin[np.nonzero(np.isnan(meanrate_FailedStop_bin)==False)[0]] # Exclude nan values        
        meanrate_CorrectStop_bin = meanrate_CorrectStop_bin[np.nonzero(np.isnan(meanrate_CorrectStop_bin)==False)[0]]         
        if np.sum(np.isnan(meanrate_CorrectStop_bin)==False) > 1:
            if (np.sum(meanrate_FailedStop_bin > 0) > 0 or np.sum(meanrate_CorrectStop_bin) > 0) and not(np.array_equal(meanrate_FailedStop_bin,meanrate_CorrectStop_bin)): # Avoid comparing two equal (e.g., zero) arrays                
                Hstat[i_bin], pval[i_bin] = st.kruskal(meanrate_FailedStop_bin, meanrate_CorrectStop_bin)
        if printout: 
            print('Bin %i: np.nanmean(meanrate_FailedStop_bin) = %.3f, np.nanmean(meanrate_CorrectStop_bin) = %.3f' %(i_bin, np.nanmean(meanrate_FailedStop_bin), np.nanmean(meanrate_CorrectStop_bin)))            
            print('Bin %i: p = %.3f' %(i_bin, pval[i_bin]))
    return Hstat, pval


def calc_meanrate_std_failed_correct(rate_data_Stop, mInt_stop, threshold, n_trials):
    # Input:     
    # rate_data_Stop: Raw population rate. Dims: [trials, timesteps]
    # Output:     
    # meanrate_FailedStop, meanrate_CorrectStop: 
    # Trial-averaged population rates. Dim: [timesteps]
    nz_FailedStop, nz_CorrectStop = get_correct_failed_stop_trials(mInt_stop, threshold, n_trials)
    meanrate_FailedStop = np.nanmean(rate_data_Stop[nz_FailedStop, :], 0)
    std_FailedStop = np.nanstd(rate_data_Stop[nz_FailedStop, :], 0)    
    meanrate_CorrectStop = np.nanmean(rate_data_Stop[nz_CorrectStop, :], 0)            
    std_CorrectStop = np.nanstd(rate_data_Stop[nz_CorrectStop, :], 0)                
    return meanrate_FailedStop, std_FailedStop, meanrate_CorrectStop, std_CorrectStop


def get_correct_failed_stop_trials(mInt_stop, threshold, n_trials):
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(n_trials)), n_trials], order='F')    
    mInt_maxpertrial = np.nanmax(rsp_mInt_stop, 0)
    nz_FailedStop = np.nonzero(mInt_maxpertrial >= threshold)[0]
    nz_CorrectStop = np.nonzero(mInt_maxpertrial < threshold)[0]
    return nz_FailedStop, nz_CorrectStop


def get_rates_failed_correct(rate_data_Stop, mInt_stop, threshold, n_trials):
    # Input:     
    # rate_data_Stop: Raw population rate. Dims: [trials, timesteps]    
    nz_FailedStop, nz_CorrectStop = get_correct_failed_stop_trials(mInt_stop, threshold, n_trials)
    rates_FailedStop = rate_data_Stop[nz_FailedStop, :]
    rates_CorrectStop = rate_data_Stop[nz_CorrectStop, :]
    # Output:     
    # rates_FailedStop, rates_CorrectStop: Raw population rate. Dims: [trials, timesteps]        
    return rates_FailedStop, rates_CorrectStop


def get_poprate_aligned_onset(mon, spikes, poprate, ratedata_currtrial, dt):
    te, ne = mon.raster_plot(spikes)
    if len(te) > 0:
        i_te_min = np.argmin(te)
        i_te_max = np.argmax(te)
        te_min = int( te[i_te_min] / dt )
        te_max = int( te[i_te_max] / dt )        
        len_ratedata = len(ratedata_currtrial[1:])               
        #poprate[te_min : te_min + len_ratedata] = ratedata_currtrial[1:] # Works, but often a large spike at onset                        
        shift = 0#int(2 / dt) # 2 works for Cortex_Stop and Go ## 100 is drastic - works, but errors with p-values?!
        #shift = int(20 / dt) # test for t_smooth_ms = 5.0 ms - Ok         
        #shift = int(50 / dt) # test for t_smooth_ms = 10.0 - still some errors for shift=40/dt, shift=50/dt is OK
        poprate[te_min + shift : te_min + len_ratedata] = ratedata_currtrial[1+shift: ] #        
        poprate[0 : te_min + shift] = np.zeros(te_min + shift)       
        poprate[te_min + len_ratedata : ] = 0

        
    return poprate


def custom_poprate(mon, pop_spikes, t_smooth_ms):
    t_mines = []
    for neuron in pop_spikes.keys():
        if len(pop_spikes[neuron]) == 0 : continue
        t_mines.append(np.min(pop_spikes[neuron]))
    t_min = np.min(t_mines)
    pop_spikes[list(pop_spikes.keys())[0]].insert(0,np.max([0,t_min-t_smooth_ms*100]).astype(int))# insert early fake spike to cause "spike" in population rate earlier

    te, ne = mon.raster_plot(pop_spikes)
    if len(te)>1: 
        rate_data_currtrial = population_rate( pop_spikes, smooth=t_smooth_ms)
        rate_data_currtrial = rate_data_currtrial[int(t_min-np.max([0,t_min-t_smooth_ms*100]).astype(int)):]
    else:
        rate_data_currtrial = []

    del pop_spikes[list(pop_spikes.keys())[0]][0]
    
    return rate_data_currtrial


def calc_meanrate_std_Fast_Slow(rate_data_Go, mInt_Go, mInt_Stop, threshold, n_trials):
    # Input:     
    # rate_data_Stop: Raw population rate. Dims: [trials, timesteps]
    # Output:     
    # meanrate_FastGo, std_FastGo, meanrate_SlowGo, std_SlowGo
    # Trial-averaged population rates. Dim: [timesteps]
    nz_FastGo, nz_SlowGo = get_fast_slow_go_trials(mInt_Go, mInt_Stop, threshold, n_trials)
    meanrate_FastGo = np.nanmean(rate_data_Go[nz_FastGo, :], 0)
    std_FastGo = np.nanstd(rate_data_Go[nz_FastGo, :], 0)    
    meanrate_SlowGo = np.nanmean(rate_data_Go[nz_SlowGo, :], 0)
    std_SlowGo = np.nanstd(rate_data_Go[nz_SlowGo, :], 0)    

    return meanrate_FastGo, std_FastGo, meanrate_SlowGo, std_SlowGo


def get_fast_slow_go_trials(mInt_go, mInt_stop, threshold, n_trials):
    rsp_mInt_go = np.reshape(mInt_go, [int(mInt_go.shape[0] / float(n_trials)), n_trials], order='F')    
    RT_Go = np.nan * np.ones(n_trials)
    for i_trial in range(n_trials):
        if np.nanmax(rsp_mInt_go[:, i_trial]) >= threshold: 
            RT_Go[i_trial] = np.nonzero(rsp_mInt_go[:, i_trial] >= threshold)[0][0]

    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(n_trials)), n_trials], order='F')    
    RT_Stop = np.nan * np.ones(n_trials)
    for i_trial in range(n_trials):
        if np.nanmax(rsp_mInt_stop[:, i_trial]) >= threshold: 
            RT_Stop[i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= threshold)[0][0]

    RT_Stop_95 = np.nanquantile(RT_Stop,0.95)
    #median_RT = np.nanmedian(RT_Go)
    #nz_FastGo = np.nonzero(RT_Go < median_RT)[0]
    #nz_SlowGo = np.nonzero(RT_Go >= median_RT)[0]
    nz_FastGo = np.nonzero(RT_Go < RT_Stop_95)[0]
    nz_SlowGo = np.nonzero(RT_Go >= RT_Stop_95)[0]
    return nz_FastGo, nz_SlowGo


def get_rates_allGo_fastGo_slowGo(rate_data_Go, mInt_go, mInt_stop, threshold, n_trials):
    # Input:     
    # rate_data_Go: Raw population rate. Dims: [trials, timesteps]
    nz_FastGo, nz_SlowGo = get_fast_slow_go_trials(mInt_go, mInt_stop, threshold, n_trials)
    nz_CorrectGo = np.concatenate([nz_FastGo, nz_SlowGo])

    rates_allGo = rate_data_Go[nz_CorrectGo, :]
    rates_fastGo = rate_data_Go[nz_FastGo, :]
    rates_slowGo = rate_data_Go[nz_SlowGo, :]
    # Output:     
    # rates_allGo, rates_fastGo, rates_slowGo: Raw population rate. Dims: [trials, timesteps]        
    return rates_allGo, rates_fastGo, rates_slowGo

def custom_zscore(data, mean_data=[]):
# normalizes the "data" array by subtracting the mean of "mean_data" and dividing by the std. of "mean_data"
    if len(mean_data) == 0:
        mean_data = data
    stdm = np.nanstd(mean_data)
    if stdm > 0:
        zsc = (data - np.nanmean(mean_data)) / stdm
    else:
        zsc = (data - np.nanmean(mean_data)) / np.nanstd(data)
    return zsc


def custom_zscore_start0(data, t_start, mean_data=[]):
# normalizes the "data" array by subtracting the mean of "mean_data" and dividing by the std. of "mean_data"
    if len(mean_data) == 0:
        mean_data = data
    stdm = np.nanstd(mean_data)
    if stdm > 0:
        zsc = (data - np.nanmean(mean_data)) / stdm
    else:
        zsc = (data - np.nanmean(mean_data)) / np.nanstd(data)
    zsc -= zsc[int(t_start)]    
    return zsc


def plot_zscore_stopVsGo_NewData(STN_mean_Stop, STN_mean_Go, SNr_mean_Stop, SNr_mean_Go, GPe_Arky_mean_Stop, GPe_Arky_mean_Go, GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, param_id, trials, dt, saveFolderPlots, \
                         labels = ['STN', 'SNr', 'Arky', 'Proto'], linecol = [['orange', 'yellow'], ['tomato', 'pink'], ['cyan','lightblue'], ['blue','navy']]):
    #plt.figure(figsize=(5,2), dpi=300)
    plt.figure(figsize=(6,4), dpi=300)    
    linew = 2.0 # 1.0
    fsize = 6    
    #plt.subplot(141)
    plt.subplot(251)    
    subplot_zscore(labels[0], STN_mean_Stop, STN_mean_Go, t_init, t_SSD, dt, linecol[0], linew, fsize, ymin=-1.3, ymax=3)
    plt.ylabel('Firing rate (z-score)', fontsize=6)    
    #plt.subplot(142)
    plt.subplot(252)        
    subplot_zscore(labels[1], SNr_mean_Stop, SNr_mean_Go, t_init, t_SSD, dt, linecol[1], linew, fsize, ymin=-1.3, ymax=3)
    ax=plt.gca(); ax.set_yticklabels([])
    #plt.subplot(143)
    plt.subplot(253)        
    subplot_zscore(labels[2], GPe_Arky_mean_Stop, GPe_Arky_mean_Go, t_init, t_SSD, dt, linecol[2], linew, fsize, ymin=-1.3, ymax=3)        
    ax=plt.gca(); ax.set_yticklabels([])    
    #plt.subplot(144)
    plt.subplot(254)        
    subplot_zscore(labels[3], GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, dt, linecol[3], linew, fsize, ymin=-1.3, ymax=3)      
    ax=plt.gca(); ax.set_yticklabels([])    
    fig=plt.gcf()
    fig.text(0.4, 0.02, 'Time from Go or Stop cue [sec]', fontsize=fsize)
    #plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4, hspace=0.3, left=0.08, right=0.95)        
    #plt.savefig(saveFolderPlots+'zscore_StopVsGo_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.png', dpi=300)
    #plt.savefig(saveFolderPlots+'zscore_StopVsGo_NewData_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_'+labels[0]+labels[1]+labels[2]+labels[3]+'.png', dpi=300)    
    #plt.ioff()
    #plt.show()


def plot_zscore_stopVsGo_five_NewData(STN_mean_Stop, STN_mean_Go, SNr_mean_Stop, SNr_mean_Go, GPe_Arky_mean_Stop, GPe_Arky_mean_Go, GPe_Proto_mean_Stop, GPe_Proto_mean_Go, Thal_mean_Stop, Thal_mean_Go, t_init, t_SSD, param_id, trials, dt, saveFolderPlots, \
                         labels = ['STN', 'SNr', 'Arky', 'Proto', 'Thal'], linecol = [['orange', 'yellow'], ['tomato', 'pink'], ['cyan','lightblue'], ['blue','navy'], ['black', 'grey']]):

    #plt.figure(figsize=(6,2), dpi=300) # (7,2) too wide
    linew = 2.0 # 1.0
    fsize = 6    
    #plt.subplot(151)
    plt.subplot(256)        
    subplot_zscore(labels[0], STN_mean_Stop, STN_mean_Go, t_init, t_SSD, dt, linecol[0], linew, fsize, ymin=-1.3, ymax=3)
    plt.ylabel('Firing rate (z-score)', fontsize=6)    
    #plt.subplot(152)
    plt.subplot(257)      
    subplot_zscore(labels[1], SNr_mean_Stop, SNr_mean_Go, t_init, t_SSD, dt, linecol[1], linew, fsize, ymin=-1.3, ymax=3)
    ax=plt.gca(); ax.set_yticklabels([])    
    #plt.subplot(153)
    plt.subplot(258)      
    subplot_zscore(labels[2], GPe_Arky_mean_Stop, GPe_Arky_mean_Go, t_init, t_SSD, dt, linecol[2], linew, fsize, ymin=-1.3, ymax=3)        
    ax=plt.gca(); ax.set_yticklabels([])    
    #plt.subplot(154)
    plt.subplot(259)      
    subplot_zscore(labels[3], GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, dt, linecol[3], linew, fsize, ymin=-1.3, ymax=3)      
    ax=plt.gca(); ax.set_yticklabels([])    
    #plt.subplot(155)
    plt.subplot(2,5,10)      
    subplot_zscore(labels[4], Thal_mean_Stop, Thal_mean_Go, t_init, t_SSD, dt, linecol[4], linew, fsize, ymin=-1.3, ymax=3)          
    axThal=ax=plt.gca(); ax.set_yticklabels([])    
    fig=plt.gcf()
    fig.text(0.4, 0.02, 'Time from Go or Stop cue [sec]', fontsize=fsize)
    #plt.tight_layout()
    #plt.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4, hspace=0.4, left=0.08, right=0.95)    
    plt.subplots_adjust(bottom=0.1, top=0.935, wspace=0.4, hspace=0.4, left=0.08, right=0.95) 

    normalLine = mlines.Line2D([], [], color='k', label='Stop cue')
    dashedLine = mlines.Line2D([], [], color='k', label='Go cue', dash_capstyle='round', dashes=(0.05,2))
    xLegend = axThal.get_position().x0
    yLegend = fig.axes[0].get_position().y1
    leg = plt.legend(handles=[normalLine,dashedLine],title='Alignment:', bbox_to_anchor=(xLegend,yLegend), bbox_transform=plt.gcf().transFigure, fontsize=fsize, title_fontsize=fsize, loc='upper left') 
    leg._legend_box.align = "left"  
    
    #plt.savefig(saveFolderPlots+'zscore_StopVsGo_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.png', dpi=300)
    plt.savefig(saveFolderPlots+'zscore_StopVsGo_NewData_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_'+labels[0]+labels[1]+labels[2]+labels[3]+labels[4]+'.svg', dpi=300)    
    plt.ioff()
    plt.show()


def plot_meanrate_All_FailedVsCorrectStop(resultsFolder, \
                                          data, \
                                          rsp_mInt_stop, nz_FailedStop, nz_CorrectStop, thresh, \
                                          t_init, t_SSD, param_id, trials, dt, pvalue_list=[], pvalue_times=[]):


    alpha = 0.01 # 0.05 
    pvalue_list = FDR_correction(pvalue_list, alpha) 

    '''#    
    # Bonferroni correction:
    n_pvals = np.sum(np.isnan(pvalue_list)==False) 
    print('n_pvals = ', n_pvals)    
    print('pvalues, before: max, min = ', np.nanmax(pvalue_list), np.nanmin(pvalue_list))
    if n_pvals > 0:
        for i in range(len(pvalue_list)):
            pvalue_list[i] *= n_pvals
    print('pvalues, after: max, min = ', np.nanmax(pvalue_list), np.nanmin(pvalue_list))    
    '''


    ### CONFIGURE FIGURE
    plt.figure(figsize=(6,4), dpi=300)
    linew = 1 # 0.5 # 1 # 2
    fsize = 6
    boxlen = int(10 / dt)
    smoothing = False#True # False # 
    nx = 2 # 3
    ny = 6 # 4 # 6
    tmin = (t_init + t_SSD - 300)/dt # 200    
    tn200 = (t_init + t_SSD - 200)/dt        
    tmax = (t_init + t_SSD + 300)/dt # 200
    t200 = (t_init + t_SSD + 200)/dt    
    t_stopCue = (t_init + t_SSD)/dt

    
    ### PLOT ALL RATES IN SUBPLOTS
    popIdx = 0
    for pop, rates in data.items():
        ## CREATE SUBPLOT
        ax_wrapper = []    
        ax_wrapper.append(plt.subplot(nx,ny,popIdx+1))
        ax=plt.gca()
        ## RATES SMOOTHED OR NOT SMOOTHED
        if smoothing:
            plt.plot(box_smooth(rates[0], boxlen), color='tomato', ls='--', lw=linew, label='failed Stop')
            plt.plot(box_smooth(rates[1], boxlen), color='purple', lw=linew, label='correct Stop')
        else:
            plt.plot(rates[0], color='tomato', ls='--', lw=linew, label='failed Stop')
            plt.plot(rates[1], color='purple', lw=linew, label='correct Stop')
        ## TITLE AND LABEL
        plt.title(params['titles_Code_to_Script'][pop], fontsize=fsize)
        if popIdx==0 or popIdx==ny: plt.ylabel('Firing rate [Hz]', fontsize=fsize)
        ## AXIS
        ax.axis([tmin, tmax, 0, params['Fig7_maxRates'][pop]])
        ax.set_xticks([tn200, t_stopCue, t200])
        ax.set_xticklabels([-0.2, 0, 0.2])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fsize)
        ## PVALUES
        plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, popIdx)
        popIdx += 1


    ### INTEGRATOR IN LAST SUBPLOT
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,2*ny))    
    ax=plt.gca()        
    ax.set_yticks([])    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)    
    ax_wrapper.append(ax_wrapper[0].twinx())
    ## CALCULATE AND PLOT INTEGRATOR LINES
    int_mean_failedStop  = np.nanmean(rsp_mInt_stop[:, nz_FailedStop], 1)
    int_mean_correctStop = np.nanmean(rsp_mInt_stop[:, nz_CorrectStop], 1)
    plt.plot(int_mean_correctStop, color='purple', lw=linew, label='Correct stop')     
    plt.plot(int_mean_failedStop, color='tomato', ls='--', lw=linew, label='Failed stop')
    ## THRESHOLD MARKER
    plt.plot([0, rsp_mInt_stop.shape[0]], thresh * np.ones(2), 'k--', lw=linew)
    ## TITLE, LABELS, AXIS
    plt.title('Integrator-Go', fontsize=fsize)
    ax=plt.gca()
    ax.axis([tmin, tmax, 0, params['Fig7_maxRates']['IntegratorGo']])
    ax.set_xticks([tn200, t_stopCue, t200])
    ax.set_xticklabels([-0.2, 0, 0.2])
    plt.ylabel('$g_{AMPA}$', fontsize=fsize)           
    ax_twin = plt.gca()
    ax_twin.set_ylabel('Integrator value', fontsize=fsize)
    for label in ax_twin.get_xticklabels() + ax_twin.get_yticklabels():
        label.set_fontsize(fsize)    
     

    ### GENERAL XLABEL
    fig=plt.gcf()
    fig.text(0.4, 0.12, 'Time from Stop cue [sec]', fontsize=fsize)


    ### LEGEND
    normalLine = mlines.Line2D([], [], color='purple', label='correct Stop')
    dashedLine = mlines.Line2D([], [], color='tomato', ls='--', label='failed Stop')
    xLegend = 0.5
    yLegend = 0.06
    leg = plt.legend(handles=[normalLine,dashedLine], bbox_to_anchor=(xLegend,yLegend), bbox_transform=plt.gcf().transFigure, fontsize=fsize, loc='center', ncol=2) 


    ### SAVE
    plt.subplots_adjust(bottom=0.2, top=0.95, wspace=0.5, hspace=0.4, left=0.08, right=0.9)
    if smoothing:
        plt.savefig(resultsFolder+'/meanrate_All_FailedVsCorrectStop_rate_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_smoothed.svg', dpi=300)
    else:
        plt.savefig(resultsFolder+'/meanrate_All_FailedVsCorrectStop_rate_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.svg', dpi=300)
    plt.ioff()
    plt.show()


def plot_meanrate_All_FailedStopVsCorrectGo(saveFolderPlots, \
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
                                          mInt_go, mInt_stop, \
                                          thresh, \
                                          t_init, t_SSD, param_id, trials, dt, pvalue_list=[], pvalue_times=[], GO_Mode=''):

    p_ind_arky, p_ind_SD1, p_ind_STN, p_ind_Proto, p_ind_Proto2, p_ind_SNr, p_ind_thal, p_ind_SD2, p_ind_CortexG, p_ind_CortexS, p_ind_FSI = range(11)#TODO: generalize pIdx

    nz_FastGo, nz_SlowGo = get_fast_slow_go_trials(mInt_go, mInt_stop, thresh, trials)
    nz_CorrectGo = np.concatenate([nz_FastGo, nz_SlowGo])

    rsp_mInt_go = np.reshape(mInt_go, [int(mInt_go.shape[0] / float(trials)), trials], order='F')
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(trials)), trials], order='F')

    if GO_Mode=='fast':
        pvalue_list = pvalue_list['failedStop_vs_fastGo']
        nz_Go = nz_FastGo
        GPe_Arky_mean_Go, GPe_Proto_mean_Go, STR_D1_mean_Go, STR_D2_mean_Go, STN_mean_Go, SNr_mean_Go, Thal_mean_Go, Cortex_G_mean_Go, GPe_Proto2_mean_Go, Cortex_S_mean_Go, STR_FSI_mean_Go = GPe_Arky_meanrate_FastGo, GPe_Proto_meanrate_FastGo, STR_D1_meanrate_FastGo, STR_D2_meanrate_FastGo, STN_meanrate_FastGo, SNr_meanrate_FastGo, Thal_meanrate_FastGo, Cortex_G_meanrate_FastGo, GPe_Proto2_meanrate_FastGo, Cortex_S_meanrate_FastGo, STR_FSI_meanrate_FastGo
    elif GO_Mode=='slow':
        pvalue_list = pvalue_list['failedStop_vs_slowGo']
        nz_Go = nz_SlowGo
        GPe_Arky_mean_Go, GPe_Proto_mean_Go, STR_D1_mean_Go, STR_D2_mean_Go, STN_mean_Go, SNr_mean_Go, Thal_mean_Go, Cortex_G_mean_Go, GPe_Proto2_mean_Go, Cortex_S_mean_Go, STR_FSI_mean_Go = GPe_Arky_meanrate_SlowGo, GPe_Proto_meanrate_SlowGo, STR_D1_meanrate_SlowGo, STR_D2_meanrate_SlowGo, STN_meanrate_SlowGo, SNr_meanrate_SlowGo, Thal_meanrate_SlowGo, Cortex_G_meanrate_SlowGo, GPe_Proto2_meanrate_SlowGo, Cortex_S_meanrate_SlowGo, STR_FSI_meanrate_SlowGo
    else:
        pvalue_list = pvalue_list['failedStop_vs_allGo']
        nz_Go = nz_CorrectGo

    alpha = 0.01 # 0.05
    pvalue_list = FDR_correction(pvalue_list, alpha)

    t_action0, t_action1 = [], []
    for mInt, nz in [[mInt_go, nz_Go], [mInt_stop, nz_FailedStop]]:
        a, b = get_fastest_slowest_action(mInt, nz, trials, thresh)
        t_action0.append(a)
        t_action1.append(b)
    t_action0 = min(t_action0)
    t_action1 = max(t_action1)

    '''#    
    # Bonferroni correction:
    n_pvals = np.sum(np.isnan(pvalue_list)==False) 
    print('n_pvals = ', n_pvals)    
    print('pvalues, before: max, min = ', np.nanmax(pvalue_list), np.nanmin(pvalue_list))
    if n_pvals > 0:
        for i in range(len(pvalue_list)):
            pvalue_list[i] *= n_pvals
    print('pvalues, after: max, min = ', np.nanmax(pvalue_list), np.nanmin(pvalue_list))    
    '''

    plt.figure(figsize=(6,4), dpi=300)
    linew = 1 # 0.5 # 1 # 2
    fsize = 6
    boxlen = int(10 / dt)
    smoothing = False#True # False # 
    nx = 2 # 3
    ny = 6 # 4 # 6
    tmin = (t_init - 50)/dt #(t_init + t_SSD - 300)/dt
    tmax = (t_init + 550)/dt #(t_init + t_SSD + 300)/dt
    t_goCue = t_init/dt
    t_stopCue = (t_init + t_SSD)/dt
    xTickValues = [t_init/dt, (t_init+250)/dt, (t_init+500)/dt]
    xTickLabels = [0, 0.25, 0.5]

    maxRate = {'GPe-Arky':60, 'StrD1':105, 'StrD2':70, 'STN':80, 'Cortex-Go':320, 'GPe-cp':75, 'GPe-Proto':50, 'SNr':130, 'Thalamus':50, 'Cortex-Stop':430, 'StrFSI':50, 'Integrator':0.32}

    labelList = ['failed Stop', GO_Mode+' Go']

    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,1))
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(GPe_Arky_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
        plt.plot(box_smooth(GPe_Arky_mean_Go, boxlen), color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
    else:
        plt.plot(GPe_Arky_mean_FailedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
        plt.plot(GPe_Arky_mean_Go, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
    plt.title('GPe-Arky', fontsize=fsize)
    plt.ylabel('Firing rate [Hz]', fontsize=fsize)
    ax.axis([tmin, tmax, 0, maxRate['GPe-Arky']])
    #ax.axis([tmin, tmax, 0, ax.axis()[3]])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #p_ind_arky = 0
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_arky)


    #'''#
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,ny+1))    
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(GPe_Proto_mean_Go, boxlen), color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(box_smooth(GPe_Proto_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    else:
        plt.plot(GPe_Proto_mean_Go, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(GPe_Proto_mean_FailedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('GPe-Proto', fontsize=fsize)
    plt.ylabel('Firing rate [Hz]', fontsize=fsize)  
    #ax.axis([tmin, tmax, 0, 50])
    ax.axis([tmin, tmax, 0, maxRate['GPe-Proto']])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #p_ind_proto = 3
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_Proto)                    
    #'''

    #'''#
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,2))
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(STR_D1_mean_Go, boxlen), color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(box_smooth(SD1_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    else:
        plt.plot(STR_D1_mean_Go, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(SD1_mean_FailedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('StrD1', fontsize=fsize)
    #ax.axis([tmin, tmax, 0, 50])
    ax.axis([tmin, tmax, 0, maxRate['StrD1']])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_SD1)            

    #'''

    #'''#
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,3))        
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(STR_D2_mean_Go, boxlen), color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(box_smooth(SD2_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    else:
       plt.plot(STR_D2_mean_Go, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
       plt.plot(SD2_mean_FailedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('StrD2', fontsize=fsize)
    #ax.axis([tmin, tmax, 0, 10])
    ax.axis([tmin, tmax, 0, maxRate['StrD2']])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_SD2)                
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #'''

    #'''#
    #plt.subplot(nx,ny,3)
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,4))
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(STN_mean_FailedStop, boxlen),  color='tomato', ls='--', lw=linew, label=labelList[0])
        plt.plot(box_smooth(STN_mean_Go, boxlen), color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
    else:
        plt.plot(STN_mean_Go, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(STN_mean_FailedStop, '--', color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('STN', fontsize=fsize)
    ax.axis([tmin, tmax, 0, maxRate['STN']])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #p_ind_STN = 2
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_STN)            

    
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,ny+2))    
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(SNr_mean_Go, boxlen), color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(box_smooth(SNr_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    else:
        plt.plot(SNr_mean_Go, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(SNr_mean_FailedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('SNr', fontsize=fsize)
    #ax.axis([tmin, tmax, 0, ax.axis()[3]])
    ax.axis([tmin, tmax, 0, maxRate['SNr']]) # 300
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #p_ind_SNr = 4
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_SNr)            

    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,2*ny-1))    
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(STR_FSI_mean_Go, boxlen), color=np.array([107,139,164])/255., ls='solid', lw=linew, label=labelList[1])
        plt.plot(box_smooth(FSI_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    else:
        plt.plot(box_smooth(STR_FSI_mean_Go, boxlen), color=np.array([107,139,164])/255., ls='solid', lw=linew, label=labelList[1])
        plt.plot(box_smooth(FSI_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('StrFSI', fontsize=fsize)
    ax.axis([tmin, tmax, 0, maxRate['StrFSI']])
    #ax.axis([tmin, tmax, 0, 150]) # 300
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_FSI)                        
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)

    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,ny+3))    
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(Thal_mean_Go, boxlen), color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(box_smooth(Thal_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    else:
        plt.plot(Thal_mean_Go, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(Thal_mean_FailedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('thalamus', fontsize=fsize)
    ax.axis([tmin, tmax, 0, maxRate['Thalamus']])
    #ax.axis([tmin, tmax, 0, ax.axis()[3]])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_thal)    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)

    #plt.subplot(nx,ny,2*ny)
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,2*ny))    
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)        
    ax.set_yticks([])    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)    
    ax_wrapper.append(ax_wrapper[0].twinx())            
    int_mean_failedStop = np.nanmean(rsp_mInt_stop[:, nz_FailedStop], 1)    
    int_std_failedStop = np.nanstd(rsp_mInt_stop[:, nz_FailedStop], 1)            
    int_mean_correctGo = np.nanmean(rsp_mInt_go[:, nz_Go], 1)
    int_std_correctGo = np.nanstd(rsp_mInt_go[:, nz_Go], 1)
    plt.plot(int_mean_correctGo, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])     
    plt.plot(int_mean_failedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    int_times = range(len(rsp_mInt_stop[:, 0]))
    #thresh = Integrator.threshold
    plt.plot([0, rsp_mInt_stop.shape[0]], thresh * np.ones(2), 'k--', lw=linew)
    plt.title('Integrator-Go', fontsize=fsize)
    ax=plt.gca()               
    ax.axis([tmin, tmax, 0, maxRate['Integrator']]) # 0.16
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    plt.ylabel('$g_{AMPA}$', fontsize=fsize)           
    ax_twin = plt.gca()
    ax_twin.set_ylabel('Integrator value', fontsize=fsize)
    for label in ax_twin.get_xticklabels() + ax_twin.get_yticklabels():
        label.set_fontsize(fsize)    
     

    #'''#
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,5))        
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(Cortex_G_mean_Go, boxlen), color=np.array([107,139,164])/255., ls='solid', lw=linew, label=labelList[1])
        plt.plot(box_smooth(Cortex_G_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    else:
        plt.plot(Cortex_G_mean_Go, color=np.array([107,139,164])/255., ls='solid', lw=linew, label=labelList[1])
        plt.plot(Cortex_G_mean_FailedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('cortex-Go', fontsize=fsize)
    #ax.axis([tmin, tmax, 0, 200])
    ax.axis([tmin, tmax, 0, maxRate['Cortex-Go']])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_CortexG)                    
    #'''


    #'''#
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,6))        
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(GPe_Proto2_mean_Go, boxlen), color=np.array([107,139,164])/255., ls='solid', lw=linew, label=labelList[1])
        plt.plot(box_smooth(GPe_Proto2_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    else:
        plt.plot(GPe_Proto2_mean_Go, color=np.array([107,139,164])/255., ls='solid', lw=linew, label=labelList[1])
        plt.plot(GPe_Proto2_mean_FailedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('GPe-Cp', fontsize=fsize)
    #ax.axis([tmin, tmax, 0, 200])
    ax.axis([tmin, tmax, 0, maxRate['GPe-cp']])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_Proto2)                    
    #'''


    #'''#
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,10))        
    ax=plt.gca()
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
    if smoothing:
        plt.plot(box_smooth(Cortex_S_mean_Go, boxlen), color=np.array([107,139,164])/255., ls='solid', lw=linew, label=labelList[1])
        plt.plot(box_smooth(Cortex_S_mean_FailedStop, boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
    else:
        plt.plot(Cortex_S_mean_Go, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        plt.plot(Cortex_S_mean_FailedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.title('cortex-Stop', fontsize=fsize)
    #ax.axis([tmin, tmax, 0, 200])
    ax.axis([tmin, tmax, 0, maxRate['Cortex-Stop']])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, p_ind_CortexS)            
    #'''



    fig=plt.gcf()
    fig.text(0.4, 0.12, 'Time from Go cue [sec]', fontsize=fsize)

    normalLine = mlines.Line2D([], [], color=np.array([107,139,164])/255., label=labelList[1])
    dashedLine = mlines.Line2D([], [], color='tomato', ls='--', label=labelList[0])
    xLegend = 0.5
    yLegend = 0.06
    leg = plt.legend(handles=[normalLine,dashedLine], bbox_to_anchor=(xLegend,yLegend), bbox_transform=plt.gcf().transFigure, fontsize=fsize, loc='center', ncol=2) 

    #plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.95, wspace=0.5, hspace=0.4, left=0.08, right=0.9) # wspace = 0.5, right=0.98

    if smoothing:
        plt.savefig(saveFolderPlots+'meanrate_All_FailedStopVsCorrectGo_rate_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_smoothed.svg', dpi=300)
    else:
        plt.savefig(saveFolderPlots+'meanrate_All_FailedStopVsCorrectGo_rate_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.svg', dpi=300)
    plt.ioff()
    plt.show()


def plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, pval_ind=0, show_axis=True):

    with open('../results/'+resultsFolder+'/PVALUES', 'a') as f:
        print('\n',file=f)
        NamesList= ['p_ind_arky', 'p_ind_SD1', 'p_ind_STN', 'p_ind_Proto', 'p_ind_Proto2', 'p_ind_SNr', 'p_ind_thal', 'p_ind_SD2', 'p_ind_CortexG', 'p_ind_CortexS', 'p_ind_FSI']
        print(NamesList[pval_ind],file=f)
        #print(np.array([i/10. for i in pvalue_times])[np.logical_not(np.isnan(pvalue_list[pval_ind])][0],file=f)
        #print(pvalue_list[pval_ind][np.logical_not(np.isnan(pvalue_list[pval_ind])][0]),file=f)

        IDX=np.logical_not(np.isnan(pvalue_list[pval_ind]))
        test=pvalue_list[pval_ind,IDX]
        test2=np.array([i/10. for i in pvalue_times[:-1]])[IDX]
        #print(pvalue_list[pval_ind],file=f)
        #print(np.array([i/10. for i in pvalue_times]),file=f)
        if len(test)>0:
            print(test[0],file=f)
            print(test2[0],file=f)


    fsize=6    
    if len(pvalue_list) > 0:            
        if len(pvalue_list[pval_ind]) > 0:
            if np.sum( pvalue_list[pval_ind] < alpha ) > 0:                
                for i_pv in range(len(pvalue_list[pval_ind])):
                    if pvalue_list[pval_ind][i_pv] < alpha:                    
                        plt.axvspan(pvalue_times[i_pv], pvalue_times[i_pv+1], ymax=0.05, facecolor='lightgray')


def FDR_correction(pvalue_list, alpha):
    len_pv = len(pvalue_list)
    len_pv_0 = len(pvalue_list[0])    
    print('len_pv, len_pv_0 = ', len_pv, len_pv_0)   
    rsp_pval_list = np.reshape(pvalue_list, (len_pv * len_pv_0))
    sortind = np.argsort(rsp_pval_list)
    rsp_pval_list_sorted = rsp_pval_list[sortind]        
    m = len(sortind)
    k_list = range(1, m+1)
    #thresh_pOverk = alpha / float(m)  # Benjamini-Hochberg
    thresh_pOverk = alpha / (float(m) *  Benjamini_Yekutieli_c(m))  # Benjamini-Yekutieli    
    accept_sorted =  (np.array(rsp_pval_list_sorted) / k_list) <= thresh_pOverk
    
    pvals_corrected_rsp = np.nan * np.ones(len(rsp_pval_list))
    for i in range(len(accept_sorted)):
        if accept_sorted[i]: 
            pvals_corrected_rsp[sortind[i]] = rsp_pval_list[sortind[i]]
        else:
            pvals_corrected_rsp[sortind[i]] = np.nan
    return np.reshape(pvals_corrected_rsp, (len_pv, len_pv_0))       


def Benjamini_Yekutieli_c(m):
    # Benjamini, Y., Yekutieli, D. (2001). 
    #"The control of the false discovery rate in multiple testing under dependency". 
    # Annals of Statistics. 29 (4): 1165-1188. doi:10.1214/aos/1013699998.
    c = np.zeros(m)
    c[0] = 1    
    for i in range(1,m):
        c[i] = c[i-1] + 1.0/i
    return c[-1]


#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
#####################################################################################################################################################
































































def get_peak_response_time(STN_poprate_Stop_alltrials, SNr_poprate_Stop_alltrials, Proto_poprate_Stop_alltrials, Arky_poprate_Stop_alltrials, t_min, t_max, dt, param_id, n_trials, i_cycle, i_netw_rep, paramname):
    STN_counts, STN_bins, STN_median_peak, STN_P25, STN_P75 = get_peak_response_time_hist(STN_poprate_Stop_alltrials, t_min, t_max, dt, 'STN')
    SNr_counts, SNr_bins, SNr_median_peak, SNr_P25, SNr_P75 = get_peak_response_time_hist(SNr_poprate_Stop_alltrials, t_min, t_max, dt, 'SNr')
    Proto_counts, Proto_bins, Proto_median_peak, Proto_P25, Proto_P75 = get_peak_response_time_hist(Proto_poprate_Stop_alltrials, t_min, t_max, dt, 'Proto')
    Arky_counts, Arky_bins, Arky_median_peak, Arky_P25, Arky_P75 = get_peak_response_time_hist(Arky_poprate_Stop_alltrials, t_min, t_max, dt, 'Arky')    

    linew = 3 # 2
    msize = 5
    plt.figure()    
    plot_custom_cdf(STN_bins, STN_counts, 'orange', linew)        
    plot_custom_cdf(SNr_bins, SNr_counts, 'tomato', linew)        
    plot_custom_cdf(Proto_bins, Proto_counts, 'blue', linew)        
    plot_custom_cdf(Arky_bins, Arky_counts, 'cyan', linew)            

    '''#
    plt.plot(STN_median_peak, 0.5, 'o', color='C0')
    plt.plot(SNr_median_peak, 0.5, 'o', color='C1')
    plt.plot(Proto_median_peak, 0.5, 'o', color='C2')
    plt.plot(Arky_median_peak, 0.5, 'o', color='C3')
    '''    
    '''#    
    plt.errorbar(STN_median_peak, 0.5 + 0.01*np.random.randn(), xerr = [[STN_median_peak - STN_P25], [STN_P75 - STN_median_peak]], fmt='o--', color='C0', alpha = 0.5, lw=linew, ms=msize)
    plt.errorbar(SNr_median_peak, 0.5 + 0.01*np.random.randn(), xerr = [[SNr_median_peak - SNr_P25], [SNr_P75 - SNr_median_peak]], fmt='o--', color='C1', alpha = 0.5, lw=linew, ms=msize)    
    plt.errorbar(Proto_median_peak, 0.5 + 0.01*np.random.randn(), xerr = [[Proto_median_peak - Proto_P25], [Proto_P75 - Proto_median_peak]], fmt='o--', color='C2', alpha = 0.5, lw=linew, ms=msize)    
    plt.errorbar(Arky_median_peak, 0.5 + 0.01*np.random.randn(), xerr = [[Arky_median_peak - Arky_P25], [Arky_P75 - Arky_median_peak]], fmt='o--', color='C3', alpha = 0.5, lw=linew, ms=msize)    
    '''        
    fsize=15
    #plt.legend(fontsize=fsize)
    ax=plt.gca()
    ax.axis([ax.axis()[0], (t_min + 200) / dt, ax.axis()[2], ax.axis()[3]])        
    #print('ax.axis() = ', ax.axis())
    xtks = ax.get_xticks()
    xlabels = []
    for i in range(len(xtks)):
        xlabels.append(str( int( xtks[i]*dt - t_min) )) # 
    ax.set_xticklabels(xlabels)        
    plt.xlabel('Time from Stop cue [ms]', fontsize=fsize)    
    plt.ylabel('Cumulative fraction of peak Stop response', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)       
    plt.savefig('plots/cdf_Stoptiming_paramsid'+str(int(param_id))+'_'+str(n_trials)+'trials'+'_cycle'+str(i_cycle)+'.png', dpi=300)   
    #plt.close()
    #np.save('data/Stop_timing'+'_'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'_id'+str(int(param_id))+'.npy', [STN_median_peak, SNr_median_peak, Proto_median_peak, Arky_median_peak])           
    np.save('data/Stop_timing'+'_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'_id'+str(int(param_id))+'.npy', [STN_median_peak, SNr_median_peak, Proto_median_peak, Arky_median_peak])
    return
    
def plot_custom_cdf(bins, counts, str_color, linew):
    y_factor = 1.0 / sum(counts)
    for i in range(len(counts)):
        plt.plot([bins[i], bins[i+1]], y_factor * sum(counts[0:i+1]) * np.ones(2), color=str_color, lw=linew)
        plt.plot(bins[i] * np.ones(2), [y_factor * sum(counts[0:i]), y_factor * sum(counts[0:i+1])], color=str_color, lw=linew)
    return

def get_peak_response_time_hist(poprate_Stop_alltrials, t_min, t_max, dt, label):
    n_trials = len(poprate_Stop_alltrials[:, 0])    
    n_timesteps = len(poprate_Stop_alltrials[0, :])    
    itmax = np.nan * np.ones(n_trials)
    #itmax = np.argmax(poprate_Stop_alltrials[:, int(t_min/dt) : ], axis=1)        
    for i_trial in range(n_trials):
        maxval = np.nanmax(poprate_Stop_alltrials[i_trial, int(t_min/dt) : int(t_max/dt)])
        itmax[i_trial] = np.nonzero(poprate_Stop_alltrials[i_trial, int(t_min/dt) : int(t_max/dt)] == maxval)[0][0] + int(t_min/dt)       
    counts, bins = np.histogram(itmax)
    median_peak = np.nanmedian(itmax)
    P25 = np.nanpercentile(itmax, 25)
    P75 = np.nanpercentile(itmax, 75)    
    #plt.hist(bins[:-1], bins, weights=counts, cumulative=True, histtype='step', normed=True)            
    #plt.savefig('plots/cdf_Stoptiming.png')            
    return counts, bins, median_peak, P25, P75





def get_syn_mon(mon, str_popname):
    gampa_data = mon.get('g_ampa')
    ggaba_data = mon.get('g_gaba')
    ampa_max = max(gampa_data)
    ampa_P99 = np.round(np.percentile(gampa_data,99), 2)    
    ampa_P95 = np.round(np.percentile(gampa_data,95), 2)    
    ampa_P90 = np.round(np.percentile(gampa_data,90), 2)       
    #ampa_P75 = np.round(np.percentile(gampa_data,75), 2)    
    #ampa_P50 = np.round(np.percentile(gampa_data,50), 2)    
    #ampa_P25 = np.round(np.percentile(gampa_data,25), 2)        
    #ampa_stats = [ampa_P75, ampa_P90, ampa_P95, ampa_P99, ampa_max] # ampa_P25, ampa_P50, 
    ampa_stats = [ampa_P90, ampa_P95, ampa_P99, ampa_max]      

    gaba_max = max(ggaba_data)    
    gaba_P99 = np.round(np.percentile(ggaba_data,99), 2)    
    gaba_P95 = np.round(np.percentile(ggaba_data,95), 2)    
    gaba_P90 = np.round(np.percentile(ggaba_data,90), 2)    
    #gaba_P75 = np.round(np.percentile(ggaba_data,75), 2)
    #gaba_P50 = np.round(np.percentile(gampa_data,50), 2)    
    #gaba_P25 = np.round(np.percentile(gampa_data,25), 2)            
    #gaba_stats = [gaba_P75, gaba_P90, gaba_P95, gaba_P99, gaba_max] # gaba_P25, gaba_P50,         
    gaba_stats = [gaba_P90, gaba_P95, gaba_P99, gaba_max]    

    #print(str_popname + ', Ampa: max, P99, P95, P90, P75 = ', ampa_max, ampa_P99, ampa_P95, ampa_P90, ampa_P75)
    #print(str_popname + ', Gaba: max, P99, P95, P90, P75 = ', gaba_max, gaba_P99, gaba_P95, gaba_P90, gaba_P75)    
    #print()
    return ampa_stats, gaba_stats

'''#
def get_rate_singletrial(mon, rate_data_array, trial_index):
    spikes = mon.get('spike')            
    times, ranks = mon.raster_plot(spikes)                        
    if len(times) > 0: 
        rate_data_currtrial = population_rate( spikes )
    else:
        rate_data_currtrial = np.nan * np.ones(len(rate_data_array[0,:]))
    return rate_data_currtrial
'''

def box_smooth(data, boxlen):
    smoothed = np.nan * np.ones(len(data) - boxlen + 1)
    smoothed[:] = data[ : -boxlen+1]
    for i in range(boxlen):
        if i==0: 
            smoothed[:] = data[ : -(boxlen-1)]
        elif i==boxlen-1: 
            smoothed[:] += data[i : ]
        else:
            smoothed[:] += data[i : -(boxlen-1)+i]
    smoothed /= float(boxlen)
    smoothed = np.append(np.nan * np.ones(boxlen-1), smoothed)
    return smoothed












def get_fastest_slowest_action(mInt, nz, n_trials, threshold):
    rsp_mInt = np.reshape(mInt, [int(mInt.shape[0] / float(n_trials)), n_trials], order='F')    
    RT_Go = np.nan * np.ones(n_trials)
    for i_trial in range(n_trials):
        if np.nanmax(rsp_mInt[:, i_trial]) >= threshold: 
            RT_Go[i_trial] = np.nonzero(rsp_mInt[:, i_trial] >= threshold)[0][0]
    
    if np.sum(np.logical_not(np.isnan(RT_Go[nz])))>0:
        fastest = RT_Go[nz].min()
        slowest = RT_Go[nz].max()
    else:
        fastest = None
        slowest = None

    return fastest, slowest



def rate_plot(mean_rate_Hz, std_rate, str_id, n_trials, str_color, plot_sem = True, pltlabel=''):
    plt.plot(mean_rate_Hz, str_color, lw=3, label=pltlabel)    
    if plot_sem:
        ax = plt.gca()
        ax.fill_between(range(len(mean_rate_Hz)), mean_rate_Hz - std_rate, mean_rate_Hz + std_rate, color=str_color, alpha=0.4, edgecolor='None')


def plot_meanrate(te, ne, str_id, n_trials, length_ms, str_color, plot_sem = True, pltlabel=''):
    print("Calculating "+str_id+" mean...")
    custom_rate_all_trialmean, custom_rate_all_trialsem = trial_averaged_firing_rate(te, ne, n_trials, length_ms)
    plt.plot(custom_rate_all_trialmean, str_color, lw=3, label=pltlabel) # lw=2
    if plot_sem:
        ax = plt.gca()
        ax.fill_between(range(len(custom_rate_all_trialmean)), custom_rate_all_trialmean - custom_rate_all_trialsem, custom_rate_all_trialmean + custom_rate_all_trialsem, color=str_color, alpha=0.4, edgecolor='None')
    return custom_rate_all_trialmean, custom_rate_all_trialsem

def plot_meanrate_selective(te, ne, str_id, n_trials, trial_list, length_ms, str_color, plot_sem = True):
    print("Calculating "+str_id+" mean...")
    #custom_rate_all_trialmean, custom_rate_all_trialsem = trial_averaged_firing_rate_selective(te, ne, n_trials, length_ms, trial_list)
    custom_rate_all_trialmean, custom_rate_all_trialsem, custom_rate_perneuron = trial_averaged_firing_rate_selective(te, ne, n_trials, length_ms, trial_list)    
    plt.plot(custom_rate_all_trialmean, str_color, lw=3) # lw=2
    if plot_sem:
        ax = plt.gca()
        ax.fill_between(range(len(custom_rate_all_trialmean)), custom_rate_all_trialmean - custom_rate_all_trialsem, custom_rate_all_trialmean + custom_rate_all_trialsem, color=str_color, alpha=0.4, edgecolor='None')
    #return custom_rate_all_trialmean, custom_rate_all_trialsem
    return custom_rate_all_trialmean, custom_rate_all_trialsem, custom_rate_perneuron    


def trial_averaged_firing_rate(te, ne, n_trials, simlength, calc_rate_per_trial=False, tmin=np.nan, tmax=np.nan):
    # tmin and tmax can be used to define a time window in which firing rates should be calculated
    reversal_index = np.nan * np.ones([100, n_trials])
    custom_rate_perneuron = np.zeros([100, n_trials, int(simlength)])
    meanrate_pertrial = np.nan * np.ones(n_trials)
    for j_neuron in range(100):
        nz_spike_j = np.nonzero(ne == j_neuron)
        ar_min = argrelmin(te[nz_spike_j]) [0]
        reversal_index[j_neuron, 0:len(ar_min)] = ar_min # 
        for k_trial in range(n_trials-1): # range(n_trials):
            min_index = reversal_index[j_neuron, k_trial-1]
            max_index = np.nanmax(reversal_index[j_neuron, k_trial])
            if np.isnan(min_index) == False and np.isnan(max_index) == False:
                if k_trial==0: 
                    i_sp_min = 0
                else:
                    i_sp_min = int(reversal_index[j_neuron, k_trial-1])
                i_sp_max = int(max_index)
                nz_it = np.nonzero(te[nz_spike_j][i_sp_min : i_sp_max])[0]
                for it in te[nz_spike_j][i_sp_min : i_sp_max][nz_it]:
                    if np.isnan(tmin) or np.isnan(tmax):
                        #custom_rate_perneuron[j_neuron, k_trial, it] += np.sum(te[nz_spike_j][i_sp_min : i_sp_max][nz_it] == it ) # for dt=1ms
                        custom_rate_perneuron[j_neuron, k_trial, int(it)] += np.sum(te[nz_spike_j][i_sp_min : i_sp_max][nz_it] == it )                        
                    else:
                        if it >= tmin and it <= tmax:
                            #custom_rate_perneuron[j_neuron, k_trial, it] += np.sum(te[nz_spike_j][i_sp_min : i_sp_max][nz_it] == it ) # for dt=1ms
                            custom_rate_perneuron[j_neuron, k_trial, int(it)] += np.sum(te[nz_spike_j][i_sp_min : i_sp_max][nz_it] == it )                            
            meanrate_pertrial[k_trial] = 1e3 * np.nanmean(custom_rate_perneuron[:, k_trial, :])
    custom_rate_meanAcrossNeurons = np.nanmean(custom_rate_perneuron, 1)     # units: spikes per msec
    custom_rate_all_trialmean = np.nanmean(custom_rate_meanAcrossNeurons, 0) # units: spikes per msec
    custom_rate_all_trialmean *= 1e3                                         # units: spikes per sec
    custom_rate_all_trialsem = 1.0 / np.sqrt(n_trials) * np.nanstd(1e3 * custom_rate_meanAcrossNeurons, 0) # units: spikes per sec
    if calc_rate_per_trial == False:
        return custom_rate_all_trialmean, custom_rate_all_trialsem
    else:
        return custom_rate_all_trialmean, custom_rate_all_trialsem, meanrate_pertrial

def trial_averaged_firing_rate_selective(te, ne, n_trials, simlength, trial_list):
    # Calculate the mean firing rate for the trials specified in trial_list
    # TODO: Can I vectorize some of the for loops below?!   
    reversal_index = np.nan * np.ones([100, n_trials])
    custom_rate_perneuron = np.zeros([100, n_trials, int(simlength)])
    for j_neuron in range(100):
        nz_spike_j = np.nonzero(ne == j_neuron) # Get (monitor) index values for spikes of neuron j
        ar_min = argrelmin(te[nz_spike_j]) [0]  # Get times at which a new trial starts (100 in a row, with time reset)
        reversal_index[j_neuron, 0:len(ar_min)] = ar_min # 
        for k_trial in range(n_trials-1): # range(n_trials):
            if k_trial in trial_list:
                min_index = reversal_index[j_neuron, k_trial-1]
                max_index = np.nanmax(reversal_index[j_neuron, k_trial])
                if np.isnan(min_index) == False and np.isnan(max_index) == False:
                    if k_trial==0: 
                        i_sp_min = 0
                    else:
                        i_sp_min = int(reversal_index[j_neuron, k_trial-1]) # Index of first spike in trial k
                    i_sp_max = int(max_index)                               # Index of last  spike in trial k
                    nz_it = np.nonzero(te[nz_spike_j][i_sp_min : i_sp_max])[0]
                    for it in te[nz_spike_j][i_sp_min : i_sp_max][nz_it]:   # Iterate over spike times of neuron j in trial k
                        #custom_rate_perneuron[j_neuron, k_trial, it] += np.sum(te[nz_spike_j][i_sp_min : i_sp_max][nz_it] == it ) # Why "+= " ?
                        custom_rate_perneuron[j_neuron, k_trial, it] = np.sum(te[nz_spike_j][i_sp_min : i_sp_max][nz_it] == it )                        
            else: #  k_trial not in trial_list:
                custom_rate_perneuron[:, k_trial, :] = np.nan
    custom_rate_meanAcrossNeurons = np.nanmean(custom_rate_perneuron, 1)     # units: spikes per msec
    custom_rate_all_trialmean = np.nanmean(custom_rate_meanAcrossNeurons, 0) # units: spikes per msec
    custom_rate_all_trialmean *= 1e3                                         # units: spikes per sec
    #custom_rate_all_trialstd = np.nanstd(1e3 * custom_rate_meanAcrossNeurons, 0) # units: spikes per sec
    custom_rate_all_trialsem = 1.0 / np.sqrt(n_trials) * np.nanstd(1e3 * custom_rate_meanAcrossNeurons, 0) # units: spikes per sec
    #return custom_rate_all_trialmean, custom_rate_all_trialstd
    #return custom_rate_all_trialmean, custom_rate_all_trialsem    
    return custom_rate_all_trialmean, custom_rate_all_trialsem, custom_rate_perneuron



def simple_mean_rate(te, n_trials):
    summed_spikes_perstep = np.bincount(te)
    mean_spikes_perstep = summed_spikes_perstep / float(n_trials)
    popmean_spikes_perstep = mean_spikes_perstep / 100.0 # population size = 100
    mean_poprate_Hz = popmean_spikes_perstep * 1e3 # dt = 1ms
    return mean_poprate_Hz




def plot_zscore_stopVsGo(STN_mean_Stop, STN_mean_Go, SNr_mean_Stop, SNr_mean_Go, GPe_Arky_mean_Stop, GPe_Arky_mean_Go, GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, param_id, trials, dt, saveFolderPlots, \
                         labels = ['STN', 'SNr', 'Arky', 'Proto'], linecol = [['orange', 'yellow'], ['tomato', 'pink'], ['cyan','lightblue'], ['blue','navy']]):
    #plt.figure(figsize=(5,2), dpi=300)
    plt.figure(figsize=(6,4), dpi=300)    
    linew = 2.0 # 1.0
    fsize = 6    
    #plt.subplot(141)
    plt.subplot(251)    
    subplot_zscore(labels[0], STN_mean_Stop, STN_mean_Go, t_init, t_SSD, dt, linecol[0], linew, fsize, ymin=-1, ymax=7)
    plt.ylabel('Firing rate (z-score)', fontsize=6)    
    #plt.subplot(142)
    plt.subplot(252)        
    subplot_zscore(labels[1], SNr_mean_Stop, SNr_mean_Go, t_init, t_SSD, dt, linecol[1], linew, fsize, ymin=-1, ymax=7)
    ax=plt.gca(); ax.set_yticklabels([])
    #plt.subplot(143)
    plt.subplot(253)        
    subplot_zscore(labels[2], GPe_Arky_mean_Stop, GPe_Arky_mean_Go, t_init, t_SSD, dt, linecol[2], linew, fsize, ymin=-1, ymax=7)        
    ax=plt.gca(); ax.set_yticklabels([])    
    #plt.subplot(144)
    plt.subplot(254)        
    subplot_zscore(labels[3], GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, dt, linecol[3], linew, fsize, ymin=-1, ymax=7)      
    ax=plt.gca(); ax.set_yticklabels([])    
    fig=plt.gcf()
    fig.text(0.4, 0.02, 'Time from Go or Stop cue [sec]', fontsize=fsize)
    #plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4, hspace=0.3, left=0.08, right=0.95)        
    #plt.savefig(saveFolderPlots+'zscore_StopVsGo_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.png', dpi=300)
    plt.savefig(saveFolderPlots+'zscore_StopVsGo_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_'+labels[0]+labels[1]+labels[2]+labels[3]+'.png', dpi=300)    
    #plt.ioff()
    #plt.show()

def plot_zscore_stopVsGo_five(STN_mean_Stop, STN_mean_Go, SNr_mean_Stop, SNr_mean_Go, GPe_Arky_mean_Stop, GPe_Arky_mean_Go, GPe_Proto_mean_Stop, GPe_Proto_mean_Go, Thal_mean_Stop, Thal_mean_Go, t_init, t_SSD, param_id, trials, dt, saveFolderPlots, \
                         labels = ['STN', 'SNr', 'Arky', 'Proto', 'Thal'], linecol = [['orange', 'yellow'], ['tomato', 'pink'], ['cyan','lightblue'], ['blue','navy'], ['black', 'grey']]):

    #plt.figure(figsize=(6,2), dpi=300) # (7,2) too wide
    linew = 2.0 # 1.0
    fsize = 6    
    #plt.subplot(151)
    plt.subplot(256)        
    subplot_zscore(labels[0], STN_mean_Stop, STN_mean_Go, t_init, t_SSD, dt, linecol[0], linew, fsize, ymin=-1, ymax=7)
    plt.ylabel('Firing rate (z-score)', fontsize=6)    
    #plt.subplot(152)
    plt.subplot(257)      
    subplot_zscore(labels[1], SNr_mean_Stop, SNr_mean_Go, t_init, t_SSD, dt, linecol[1], linew, fsize, ymin=-1, ymax=7)
    ax=plt.gca(); ax.set_yticklabels([])    
    #plt.subplot(153)
    plt.subplot(258)      
    subplot_zscore(labels[2], GPe_Arky_mean_Stop, GPe_Arky_mean_Go, t_init, t_SSD, dt, linecol[2], linew, fsize, ymin=-1, ymax=7)        
    ax=plt.gca(); ax.set_yticklabels([])    
    #plt.subplot(154)
    plt.subplot(259)      
    subplot_zscore(labels[3], GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, dt, linecol[3], linew, fsize, ymin=-1, ymax=7)      
    ax=plt.gca(); ax.set_yticklabels([])    
    #plt.subplot(155)
    plt.subplot(2,5,10)      
    subplot_zscore(labels[4], Thal_mean_Stop, Thal_mean_Go, t_init, t_SSD, dt, linecol[4], linew, fsize, ymin=-1, ymax=7)          
    ax=plt.gca(); ax.set_yticklabels([])    
    fig=plt.gcf()
    fig.text(0.4, 0.02, 'Time from Go or Stop cue [sec]', fontsize=fsize)
    #plt.tight_layout()
    #plt.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4, hspace=0.4, left=0.08, right=0.95)    
    plt.subplots_adjust(bottom=0.1, top=0.935, wspace=0.4, hspace=0.4, left=0.08, right=0.95)    
    
    #plt.savefig(saveFolderPlots+'zscore_StopVsGo_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.png', dpi=300)
    plt.savefig(saveFolderPlots+'zscore_StopVsGo_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_'+labels[0]+labels[1]+labels[2]+labels[3]+labels[4]+'.png', dpi=300)    
    plt.ioff()
    plt.show()












    
def subplot_zscore(label, meanrate_Stop, meanrate_Go, t_init, t_SSD, dt, linecol, linew, fsize, ymin=-1, ymax=7):
    if label=='StrD1' or label=='StrD2':
        da=(0.05,2.5)
    elif label=='GPe-Proto' or label=='GPe-Cp':
        da=(0.05,3)
    else:
        da=(0.05,2)
    plt.title(label, fontsize=6)    
    plt.plot(custom_zscore(meanrate_Stop, meanrate_Stop), color=linecol[0], lw=linew)  
    plt.plot(np.append(np.nan*np.ones(int(t_SSD / dt)), custom_zscore(meanrate_Go, meanrate_Go)), dash_capstyle='round', dashes=da, color=linecol[1], lw=linew) # Align to Go cue onset, shift by t_SSD towards the right / padding with NaN values from [-tSSD, 0]     
    #plt.plot((t_init + t_SSD)/dt * np.ones(2), [plt.axis()[2]+0.1, plt.axis()[3]], color='grey', lw=0.5) # 'k--', lw=1.5)
    ax = plt.gca()
    t_neg200 = (t_init + t_SSD - 200)/dt
    tmin = t_neg200
    t200 = (t_init + t_SSD + 200)/dt
    tmax = t200    
    t_stopCue = (t_init + t_SSD)/dt
    plt.axvline(t_stopCue, color='grey', lw=0.5)
    ax.axis([tmin, tmax, ymin, ymax])
    ax.set_xticks([t_neg200, t_stopCue, t200])
    ax.set_xticklabels([-0.2, 0, 0.2])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #plt.legend(('Stop', 'Go'), fontsize=fsize)    
                             

def plot_correl_rates_Intmax(GPe_Arky_ratepertrial_Stop, GPe_Proto_ratepertrial_Stop, STR_D1_ratepertrial_Stop,   STR_D2_ratepertrial_Stop,   STN_ratepertrial_Stop, \
                             SNr_ratepertrial_Stop,      Thal_ratepertrial_Stop,      Cortex_G_ratepertrial_Stop, Cortex_S_ratepertrial_Stop, mInt_maxpertrial, param_id, trials):
    plt.figure(figsize=(5,4), dpi=300)                             
    fsize = 6
    plt.subplot(251)
    nz_nonnan_trials = np.nonzero( np.isnan(GPe_Arky_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( GPe_Arky_ratepertrial_Stop )[0]
    nz_nonnan_nonzero = np.intersect1d(nz_nonnan_trials, nz_trials)
    plt.plot( GPe_Arky_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero], '.')
    regress = st.linregress(GPe_Arky_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero])
    print("rho = %.2f, p = %.2f (Arky vs. Int Stop)" %(regress.rvalue, regress.pvalue))
    xmin = np.nanmin(GPe_Arky_ratepertrial_Stop[nz_nonnan_nonzero])
    xmax = np.nanmax(GPe_Arky_ratepertrial_Stop[nz_nonnan_nonzero])
    plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
    ax=plt.gca()
    plt.xlabel('Arky rate', fontsize=fsize)
    plt.ylabel('Integrator max. value per trial', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ax.set_xticks([ax.axis()[0], ax.axis()[1]])
    if regress.pvalue < 0.05:
        plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize)    

    plt.subplot(252)
    nz_nonnan_trials = np.nonzero( np.isnan(GPe_Proto_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( GPe_Proto_ratepertrial_Stop )[0]
    nz_nonnan_nonzero = np.intersect1d(nz_nonnan_trials, nz_trials)
    plt.plot( GPe_Proto_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero], '.')
    regress = st.linregress(GPe_Proto_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero])
    print("rho, p (Proto vs. Int Stop) = ", regress.rvalue, regress.pvalue)
    xmin = np.nanmin(GPe_Proto_ratepertrial_Stop[nz_nonnan_nonzero])
    xmax = np.nanmax(GPe_Proto_ratepertrial_Stop[nz_nonnan_nonzero])
    plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
    ax=plt.gca()
    plt.xlabel('Proto rate', fontsize=fsize)
    #plt.ylabel('Integrator max. value per trial', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ax.set_yticklabels([])
    ax.set_xticks([ax.axis()[0], ax.axis()[1]])
    if regress.pvalue < 0.05:
        plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize)    

    plt.subplot(253)
    nz_nonnan_trials = np.nonzero( np.isnan(STR_D1_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( STR_D1_ratepertrial_Stop )[0]
    nz_nonnan_nonzero = np.intersect1d(nz_nonnan_trials, nz_trials)
    plt.plot( STR_D1_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero], '.')
    if len(nz_nonnan_nonzero) > 0:
        regress = st.linregress(STR_D1_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero])
        print("rho, p (Str D1 vs. Int Stop) = ", regress.rvalue, regress.pvalue)
        xmin = np.nanmin(STR_D1_ratepertrial_Stop[nz_nonnan_nonzero])
        xmax = np.nanmax(STR_D1_ratepertrial_Stop[nz_nonnan_nonzero])
        plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
    ax=plt.gca()
    plt.xlabel('Str D1 rate', fontsize=fsize)
    #plt.ylabel('Integrator max. value per trial', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ax.set_yticklabels([])
    ax.set_xticks([ax.axis()[0], ax.axis()[1]])
    if regress.pvalue < 0.05:
        plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize)    

    plt.subplot(254)
    nz_nonnan_trials = np.nonzero( np.isnan(STR_D2_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( STR_D2_ratepertrial_Stop )[0]
    nz_nonnan_nonzero = np.intersect1d(nz_nonnan_trials, nz_trials)
    plt.plot( STR_D2_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero], '.')
    if len(nz_nonnan_nonzero) > 0:
        regress = st.linregress(STR_D2_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero])
        print("rho, p (Str D2 vs. Int Stop) = ", regress.rvalue, regress.pvalue)
        xmin = np.nanmin(STR_D2_ratepertrial_Stop[nz_nonnan_nonzero])
        xmax = np.nanmax(STR_D2_ratepertrial_Stop[nz_nonnan_nonzero])
        plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
    ax=plt.gca()
    plt.xlabel('Str D2 rate', fontsize=fsize)
    #plt.ylabel('Integrator max. value per trial', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ax.set_yticklabels([])
    ax.set_xticks([ax.axis()[0], ax.axis()[1]])
    if regress.pvalue < 0.05:
        plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize)    

    plt.subplot(255)
    nz_nonnan_trials = np.nonzero( np.isnan(STN_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( STN_ratepertrial_Stop )[0]
    nz_nonnan_nonzero = np.intersect1d(nz_nonnan_trials, nz_trials)
    plt.plot( STN_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero], '.')
    if len(nz_nonnan_nonzero) > 0:
        regress = st.linregress(STN_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero])
        print("rho, p (STN vs. Int Stop) = ", regress.rvalue, regress.pvalue)
        xmin = np.nanmin(STN_ratepertrial_Stop[nz_nonnan_nonzero])
        xmax = np.nanmax(STN_ratepertrial_Stop[nz_nonnan_nonzero])
        plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
        if regress.pvalue < 0.05:
            plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize)
    ax=plt.gca()
    plt.xlabel('STN rate', fontsize=fsize)
    #plt.ylabel('Integrator max. value per trial', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ax.set_yticklabels([])


    plt.subplot(256)
    nz_nonnan_trials = np.nonzero( np.isnan(SNr_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( SNr_ratepertrial_Stop )[0]
    nz_nonnan_nonzero = np.intersect1d(nz_nonnan_trials, nz_trials)
    plt.plot( SNr_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero], '.')
    regress = st.linregress(SNr_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero])
    print("rho, p (SNr vs. Int Stop) = ", regress.rvalue, regress.pvalue)
    xmin = np.nanmin(SNr_ratepertrial_Stop[nz_nonnan_nonzero])
    xmax = np.nanmax(SNr_ratepertrial_Stop[nz_nonnan_nonzero])
    plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
    ax=plt.gca()
    plt.xlabel('SNr rate', fontsize=fsize)
    plt.ylabel('Integrator max. value per trial', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ax.set_xticks([ax.axis()[0], ax.axis()[1]])
    if regress.pvalue < 0.05:
        plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize)    

    plt.subplot(257)
    nz_nonnan_trials = np.nonzero( np.isnan(Thal_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( Thal_ratepertrial_Stop )[0]
    nz_nonnan_nonzero = np.intersect1d(nz_nonnan_trials, nz_trials)
    plt.plot( Thal_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero], '.')
    regress = st.linregress(Thal_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero])
    print("rho, p (Thal. vs. Int Stop) = ", regress.rvalue, regress.pvalue)
    xmin = np.nanmin(Thal_ratepertrial_Stop[nz_nonnan_nonzero])
    xmax = np.nanmax(Thal_ratepertrial_Stop[nz_nonnan_nonzero])
    plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
    ax=plt.gca()
    plt.xlabel('Thal. rate', fontsize=fsize)
    #plt.ylabel('Integrator max. value per trial', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ax.set_yticklabels([])
    ax.set_xticks([ax.axis()[0], ax.axis()[1]])
    if regress.pvalue < 0.05:
        print("Thal: p < 0.05")
        plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize)    

    plt.subplot(258)
    nz_nonnan_trials = np.nonzero( np.isnan(Cortex_G_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( Cortex_G_ratepertrial_Stop )[0]
    nz_nonnan_nonzero = np.intersect1d(nz_nonnan_trials, nz_trials)
    plt.plot( Cortex_G_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero], '.')
    if len(nz_nonnan_nonzero) > 0:
        regress = st.linregress(Cortex_G_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero])
        print("rho, p (Cortex_G vs. Int Stop) = ", regress.rvalue, regress.pvalue)
        xmin = np.nanmin(Cortex_G_ratepertrial_Stop[nz_nonnan_nonzero])
        xmax = np.nanmax(Cortex_G_ratepertrial_Stop[nz_nonnan_nonzero])
        plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
        ax=plt.gca()
        plt.xlabel('Cortex_G rate', fontsize=fsize)
        #plt.ylabel('Integrator max. value per trial', fontsize=fsize)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fsize)
        ax.set_yticklabels([])
        ax.set_xticks([ax.axis()[0], ax.axis()[1]])
        if regress.pvalue < 0.05:
            plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize)    

    plt.subplot(2,5,9)
    nz_nonnan_trials = np.nonzero( np.isnan(Cortex_S_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( Cortex_S_ratepertrial_Stop )[0]
    nz_nonnan_nonzero = np.intersect1d(nz_nonnan_trials, nz_trials)
    ax=plt.gca()    
    #if len(nz_nonnan_trials) > 0:
    if len(nz_nonnan_nonzero) > 0:        
        plt.plot( Cortex_S_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero], '.')        
        regress = st.linregress(Cortex_S_ratepertrial_Stop[nz_nonnan_nonzero], mInt_maxpertrial[nz_nonnan_nonzero]) # Correct order: st.linregress(x, y(x))
        print("rho, p (Cortex_S vs. Int Stop) = ", regress.rvalue, regress.pvalue)
        xmin = np.nanmin(Cortex_S_ratepertrial_Stop[nz_nonnan_nonzero])
        xmax = np.nanmax(Cortex_S_ratepertrial_Stop[nz_nonnan_nonzero])
        plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
        plt.xlabel('Cortex_S rate', fontsize=fsize)
        #plt.ylabel('Integrator max. value per trial', fontsize=fsize)
        if regress.pvalue < 0.05:
            plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize)    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    ax.set_yticklabels([])
    ax.set_xticks([ax.axis()[0], ax.axis()[1]])

    plt.subplot(2,5,10)
    nz_nonnan_trials = np.nonzero( np.isnan(GPe_Arky_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( GPe_Arky_ratepertrial_Stop )[0]
    nz_nonnan_nonzero_Arky = np.intersect1d(nz_nonnan_trials, nz_trials)
    nz_nonnan_trials = np.nonzero( np.isnan(STR_D1_ratepertrial_Stop)==False )[0]
    nz_trials = np.nonzero( STR_D1_ratepertrial_Stop )[0]
    nz_nonnan_nonzero_D1 = np.intersect1d(nz_nonnan_trials, nz_trials)
    plt.plot(GPe_Arky_ratepertrial_Stop[nz_nonnan_nonzero_D1], STR_D1_ratepertrial_Stop[nz_nonnan_nonzero_D1], '.')
    if len(nz_nonnan_nonzero_D1) > 0:
        regress = st.linregress(GPe_Arky_ratepertrial_Stop[nz_nonnan_nonzero_D1], STR_D1_ratepertrial_Stop[nz_nonnan_nonzero_D1])
        print("rho, p (Arky vs. Str D1) = ", regress.rvalue, regress.pvalue)
        xmin = np.nanmin(GPe_Arky_ratepertrial_Stop[nz_nonnan_nonzero_D1])
        xmax = np.nanmax(GPe_Arky_ratepertrial_Stop[nz_nonnan_nonzero_D1])
        plt.plot( [xmin, xmax], regress.slope * np.array([xmin, xmax]) + regress.intercept)
    ax=plt.gca()
    plt.xlabel('Arky rate', fontsize=fsize)
    plt.ylabel('Str D1 rate', fontsize=fsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)
    #ax.set_yticklabels([])
    ax.set_xticks([ax.axis()[0], ax.axis()[1]])
    if regress.pvalue < 0.05:
        plt.title('p < 0.05, $R^2$='+str(round(regress.rvalue**2,2)), fontsize=fsize) 
    plt.tight_layout()
    plt.savefig('plots/correl_rates_Intmax_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.png', dpi=300)
    plt.show()





   









def load_failed_vs_correct_stop(param_id, trials):
    param_id = float(param_id)
    GPe_Arky_mean_FailedStop = np.load('plots/plotdata/GPe_Arky_mean_FailedStop_id'+str(param_id)+'.npy')
    GPe_Arky_mean_CorrectStop = np.load('plots/plotdata/GPe_Arky_mean_CorrectStop_id'+str(param_id)+'.npy')
    GPe_Proto_mean_FailedStop = np.load('plots/plotdata/GPe_Proto_mean_FailedStop_id'+str(param_id)+'.npy')
    GPe_Proto_mean_CorrectStop = np.load('plots/plotdata/GPe_Proto_mean_CorrectStop_id'+str(param_id)+'.npy')
    SD1_mean_FailedStop = np.load('plots/plotdata/STR_D1_mean_FailedStop_id'+str(param_id)+'.npy')
    SD1_mean_CorrectStop = np.load('plots/plotdata/STR_D1_mean_CorrectStop_id'+str(param_id)+'.npy')
    SD2_mean_FailedStop = np.load('plots/plotdata/STR_D2_mean_FailedStop_id'+str(param_id)+'.npy')
    SD2_mean_CorrectStop = np.load('plots/plotdata/STR_D2_mean_CorrectStop_id'+str(param_id)+'.npy')
    FSI_mean_FailedStop = np.load('plots/plotdata/STR_FSI_mean_FailedStop_id'+str(param_id)+'.npy')
    FSI_mean_CorrectStop = np.load('plots/plotdata/STR_FSI_mean_CorrectStop_id'+str(param_id)+'.npy')
    STN_mean_FailedStop = np.load('plots/plotdata/STN_mean_FailedStop_id'+str(param_id)+'.npy')
    STN_mean_CorrectStop = np.load('plots/plotdata/STN_mean_CorrectStop_id'+str(param_id)+'.npy')
    SNr_mean_FailedStop = np.load('plots/plotdata/SNr_mean_FailedStop_id'+str(param_id)+'.npy')
    SNr_mean_CorrectStop = np.load('plots/plotdata/SNr_mean_CorrectStop_id'+str(param_id)+'.npy')
    SNrE_mean_FailedStop = np.load('plots/plotdata/SNrE_mean_FailedStop_id'+str(param_id)+'.npy')
    SNrE_mean_CorrectStop = np.load('plots/plotdata/SNrE_mean_CorrectStop_id'+str(param_id)+'.npy')
    Thal_mean_FailedStop = np.load('plots/plotdata/Thal_mean_FailedStop_id'+str(param_id)+'.npy')
    Thal_mean_CorrectStop = np.load('plots/plotdata/Thal_mean_CorrectStop_id'+str(param_id)+'.npy')
    Cortex_G_mean_FailedStop = np.load('plots/plotdata/CortexG_mean_FailedStop_id'+str(param_id)+'.npy')
    Cortex_G_mean_CorrectStop = np.load('plots/plotdata/CortexG_mean_CorrectStop_id'+str(param_id)+'.npy')
    Cortex_S_mean_FailedStop = np.load('plots/plotdata/CortexS_mean_FailedStop_id'+str(param_id)+'.npy')
    Cortex_S_mean_CorrectStop = np.load('plots/plotdata/CortexS_mean_CorrectStop_id'+str(param_id)+'.npy')
    PauseInput_mean_FailedStop = np.load('plots/plotdata/PauseInput_mean_FailedStop_id'+str(param_id)+'.npy')
    PauseInput_mean_CorrectStop = np.load('plots/plotdata/PauseInput_mean_CorrectStop_id'+str(param_id)+'.npy')

    mInt_stop = np.load('data/Integrator_ampa_Stop_id'+str(param_id)+'.npy')
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(trials)), trials], order='F') # order seems correct
    mInt_maxpertrial = np.nanmax(rsp_mInt_stop, 0)
    threshold = 0.13
    nz_FailedStop = np.nonzero(mInt_maxpertrial >= threshold)[0] # Integrator.threshold = 0.13
    nz_CorrectStop = np.nonzero(mInt_maxpertrial < threshold)[0]
    t_init = 300
    t_SSD = 250
    #results_RT = np.load('data/resultsRT_'+str('')+'_param_'+'STN-to-SNr'+'_cycle'+str(0)+'id'+str(param_id)+'.npy')

    
    plot_meanrate_All_FailedVsCorrectStop(GPe_Arky_mean_FailedStop, GPe_Arky_mean_CorrectStop, \
                                          GPe_Proto_mean_FailedStop, GPe_Proto_mean_CorrectStop, \
                                          SD1_mean_FailedStop, SD1_mean_CorrectStop, \
                                          SD2_mean_FailedStop, SD2_mean_CorrectStop, \
                                          STN_mean_FailedStop, STN_mean_CorrectStop, \
                                          SNr_mean_FailedStop, SNr_mean_CorrectStop, \
                                          Thal_mean_FailedStop, Thal_mean_CorrectStop, \
                                          rsp_mInt_stop, nz_FailedStop, nz_CorrectStop, \
                                          threshold, \
                                          Cortex_G_mean_FailedStop, Cortex_G_mean_CorrectStop, \
                                          PauseInput_mean_FailedStop, PauseInput_mean_CorrectStop, \
                                          Cortex_S_mean_FailedStop, Cortex_S_mean_CorrectStop, \
                                          SNrE_mean_FailedStop, SNrE_mean_CorrectStop, \
                                          t_init, t_SSD, param_id, trials, pvalue_list=[], pvalue_times=[]) #  # FSI_mean_FailedStop, FSI_mean_CorrectStop


    GPe_Arky_ratepertrial_Stop = np.load('plots/plotdata/Arky_ratepertrial_id'+str(param_id)+'.npy')
    GPe_Proto_ratepertrial_Stop = np.load('plots/plotdata/Proto_ratepertrial_id'+str(param_id)+'.npy')
    STR_D1_ratepertrial_Stop = np.load('plots/plotdata/STR_D1_ratepertrial_id'+str(param_id)+'.npy')
    STR_D2_ratepertrial_Stop = np.load('plots/plotdata/STR_D2_ratepertrial_id'+str(param_id)+'.npy')
    STN_ratepertrial_Stop = np.load('plots/plotdata/STN_ratepertrial_id'+str(param_id)+'.npy')
    SNr_ratepertrial_Stop = np.load('plots/plotdata/SNr_ratepertrial_id'+str(param_id)+'.npy')
    Thal_ratepertrial_Stop = np.load('plots/plotdata/Thal_ratepertrial_id'+str(param_id)+'.npy')
    Cortex_G_ratepertrial_Stop = np.load('plots/plotdata/CortexG_ratepertrial_id'+str(param_id)+'.npy')
    Cortex_S_ratepertrial_Stop = np.load('plots/plotdata/CortexS_ratepertrial_id'+str(param_id)+'.npy')

    #plot_correl_rates_Intmax(GPe_Arky_ratepertrial_Stop, GPe_Proto_ratepertrial_Stop, STR_D1_ratepertrial_Stop,   STR_D2_ratepertrial_Stop,   STN_ratepertrial_Stop, \
    #                         SNr_ratepertrial_Stop,      Thal_ratepertrial_Stop,      Cortex_G_ratepertrial_Stop, Cortex_S_ratepertrial_Stop, mInt_maxpertrial, param_id, trials)



if __name__ == '__main__':
    #load_failed_vs_correct_stop(5111.0, 100)
    # load_failed_vs_correct_stop(5113.0, 100) 
    load_failed_vs_correct_stop(6004.0, 100)    
    # Preliminary conclusion: If Arky-StrD1 weights are zero, GPi/SNr is extremely strongly driven by SNrE, and Str D1 has much weaker influence on SNr (if any)!!!


