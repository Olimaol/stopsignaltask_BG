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
        poprate[te_min : te_min + len_ratedata] = ratedata_currtrial[1: ]
        poprate[0 : te_min] = np.zeros(te_min)       
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
    plt.figure(figsize=(6,4), dpi=300)    
    linew = 2.0
    fsize = 6    
    plt.subplot(251)    
    subplot_zscore(labels[0], STN_mean_Stop, STN_mean_Go, t_init, t_SSD, dt, linecol[0], linew, fsize, ymin=-1.3, ymax=3)
    plt.ylabel('Firing rate (z-score)', fontsize=6)    
    plt.subplot(252)        
    subplot_zscore(labels[1], SNr_mean_Stop, SNr_mean_Go, t_init, t_SSD, dt, linecol[1], linew, fsize, ymin=-1.3, ymax=3)
    ax=plt.gca(); ax.set_yticklabels([])
    plt.subplot(253)        
    subplot_zscore(labels[2], GPe_Arky_mean_Stop, GPe_Arky_mean_Go, t_init, t_SSD, dt, linecol[2], linew, fsize, ymin=-1.3, ymax=3)        
    ax=plt.gca(); ax.set_yticklabels([])    
    plt.subplot(254)        
    subplot_zscore(labels[3], GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, dt, linecol[3], linew, fsize, ymin=-1.3, ymax=3)      
    ax=plt.gca(); ax.set_yticklabels([])    
    fig=plt.gcf()
    fig.text(0.4, 0.02, 'Time from Go or Stop cue [sec]', fontsize=fsize)
    plt.subplots_adjust(bottom=0.2, top=0.85, wspace=0.4, hspace=0.3, left=0.08, right=0.95)        



def plot_zscore_stopVsGo_five_NewData(STN_mean_Stop, STN_mean_Go, SNr_mean_Stop, SNr_mean_Go, GPe_Arky_mean_Stop, GPe_Arky_mean_Go, GPe_Proto_mean_Stop, GPe_Proto_mean_Go, Thal_mean_Stop, Thal_mean_Go, t_init, t_SSD, param_id, trials, dt, saveFolderPlots, \
                         labels = ['STN', 'SNr', 'Arky', 'Proto', 'Thal'], linecol = [['orange', 'yellow'], ['tomato', 'pink'], ['cyan','lightblue'], ['blue','navy'], ['black', 'grey']]):

    linew = 2.0
    fsize = 6    
    plt.subplot(256)        
    subplot_zscore(labels[0], STN_mean_Stop, STN_mean_Go, t_init, t_SSD, dt, linecol[0], linew, fsize, ymin=-1.3, ymax=3)
    plt.ylabel('Firing rate (z-score)', fontsize=6)    
    plt.subplot(257)      
    subplot_zscore(labels[1], SNr_mean_Stop, SNr_mean_Go, t_init, t_SSD, dt, linecol[1], linew, fsize, ymin=-1.3, ymax=3)
    ax=plt.gca(); ax.set_yticklabels([])    
    plt.subplot(258)      
    subplot_zscore(labels[2], GPe_Arky_mean_Stop, GPe_Arky_mean_Go, t_init, t_SSD, dt, linecol[2], linew, fsize, ymin=-1.3, ymax=3)        
    ax=plt.gca(); ax.set_yticklabels([])    
    plt.subplot(259)      
    subplot_zscore(labels[3], GPe_Proto_mean_Stop, GPe_Proto_mean_Go, t_init, t_SSD, dt, linecol[3], linew, fsize, ymin=-1.3, ymax=3)      
    ax=plt.gca(); ax.set_yticklabels([])    
    plt.subplot(2,5,10)      
    subplot_zscore(labels[4], Thal_mean_Stop, Thal_mean_Go, t_init, t_SSD, dt, linecol[4], linew, fsize, ymin=-1.3, ymax=3)          
    axThal=ax=plt.gca(); ax.set_yticklabels([])    
    fig=plt.gcf()
    fig.text(0.4, 0.02, 'Time from Go or Stop cue [sec]', fontsize=fsize)

    plt.subplots_adjust(bottom=0.1, top=0.935, wspace=0.4, hspace=0.4, left=0.08, right=0.95) 

    normalLine = mlines.Line2D([], [], color='k', label='Stop cue')
    dashedLine = mlines.Line2D([], [], color='k', label='Go cue', dash_capstyle='round', dashes=(0.05,2))
    xLegend = axThal.get_position().x0
    yLegend = fig.axes[0].get_position().y1
    leg = plt.legend(handles=[normalLine,dashedLine],title='Alignment:', bbox_to_anchor=(xLegend,yLegend), bbox_transform=plt.gcf().transFigure, fontsize=fsize, title_fontsize=fsize, loc='upper left') 
    leg._legend_box.align = "left"  
    
    plt.savefig(saveFolderPlots+'zscore_StopVsGo_NewData_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_'+labels[0]+labels[1]+labels[2]+labels[3]+labels[4]+'.svg', dpi=300)    
    plt.ioff()
    plt.show()


def plot_meanrate_All_FailedVsCorrectStop(resultsFolder, \
                                          data, \
                                          rsp_mInt_stop, nz_FailedStop, nz_CorrectStop, thresh, \
                                          paramsA, \
                                          t_init, t_SSD, param_id, trials, dt, pvalue_list=[], pvalue_times=[], alpha=0.01):


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
    linew = 1
    fsize = 6
    boxlen = int(10 / dt)
    smoothing = False
    nx = 2
    ny = 6
    tmin = (t_init + t_SSD - 300)/dt
    tn200 = (t_init + t_SSD - 200)/dt        
    tmax = (t_init + t_SSD + 300)/dt
    t200 = (t_init + t_SSD + 200)/dt    
    t_stopCue = (t_init + t_SSD)/dt

    
    ### PLOT ALL RATES IN SUBPLOTS
    popIdx = 0
    for pop, rates in data.items():
        ## CREATE SUBPLOT
        ax_wrapper = []    
        ax_wrapper.append(plt.subplot(nx,ny,popIdx+1))
        ax=plt.gca()
        ## PLOT RATES SMOOTHED OR NOT SMOOTHED
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
        ax.axis([tmin, tmax, 0, paramsA['Fig7_maxRates'][pop]])
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
    ax.axis([tmin, tmax, 0, paramsA['Fig7_maxRates']['IntegratorGo']])
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


def plot_meanrate_All_FailedStopVsCorrectGo(resultsFolder, \
                                          data, \
                                          nz_FailedStop, nz_CorrectStop, \
                                          mInt_go, mInt_stop, thresh, \
                                          paramsA, \
                                          t_init, t_SSD, param_id, trials, dt, pvalue_list=[], pvalue_times=[], GO_Mode='all', alpha=0.01):
    
    
    nz_FastGo, nz_SlowGo = get_fast_slow_go_trials(mInt_go, mInt_stop, thresh, trials)
    nz_CorrectGo = np.concatenate([nz_FastGo, nz_SlowGo])

    rsp_mInt_go = np.reshape(mInt_go, [int(mInt_go.shape[0] / float(trials)), trials], order='F')
    rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(trials)), trials], order='F')

    if GO_Mode=='fast':
        pvalue_list = pvalue_list['failedStop_vs_fastGo']
        nz_Go = nz_FastGo
    elif GO_Mode=='slow':
        pvalue_list = pvalue_list['failedStop_vs_slowGo']
        nz_Go = nz_SlowGo
    elif GO_Mode=='all':
        pvalue_list = pvalue_list['failedStop_vs_allGo']
        nz_Go = nz_CorrectGo

    pvalue_list = FDR_correction(pvalue_list, alpha)

    ### EARLIEST AND LATEST RESPONSE
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

    ### CONFIGURE FIGURE
    plt.figure(figsize=(6,4), dpi=300)
    linew = 1
    fsize = 6
    boxlen = int(10 / dt)
    smoothing = False
    nx = 2
    ny = 6
    tmin = (t_init - 50)/dt
    tmax = (t_init + 550)/dt
    t_goCue = t_init/dt
    t_stopCue = (t_init + t_SSD)/dt
    xTickValues = [t_init/dt, (t_init+250)/dt, (t_init+500)/dt]
    xTickLabels = [0, 0.25, 0.5]
    labelList = ['failed Stop', GO_Mode+' Go']


    ### PLOT ALL RATES IN SUBPLOTS
    popIdx = 0
    for pop in params['Fig7_order']:
        ### SELECT RATES
        rates = data[GO_Mode+'_'+pop]
        ### CREATE SUBPLOT
        ax_wrapper = []    
        ax_wrapper.append(plt.subplot(nx,ny,popIdx+1))
        ax=plt.gca()
        ### MARK RESPONSE REGION
        plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
        ### MARK GOU CUE AND STOP CUE
        plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
        plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)
        ### PLOT RATES
        if smoothing:
            plt.plot(box_smooth(rates[0], boxlen), color='tomato', ls='--', lw=linew, label=labelList[0])
            plt.plot(box_smooth(rates[1], boxlen), color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        else:
            plt.plot(rates[0], color='tomato', ls='--', lw=linew, label=labelList[0])
            plt.plot(rates[1], color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
        ## TITLE AND LABEL
        plt.title(params['titles_Code_to_Script'][pop], fontsize=fsize)
        if popIdx==0 or popIdx==ny: plt.ylabel('Firing rate [Hz]', fontsize=fsize)
        ## AXIS
        ax.axis([tmin, tmax, 0, paramsA['Fig7_maxRates'][pop]])
        ax.set_xticks(xTickValues)
        ax.set_xticklabels(xTickLabels)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(fsize)
        ## PVALUES
        plot_pvalues(resultsFolder, alpha, dt, ax_wrapper, pvalue_list, pvalue_times, popIdx)
        popIdx += 1


    ### INTEGRATOR IN LAST SUBPLOT
    ax_wrapper = []    
    ax_wrapper.append(plt.subplot(nx,ny,2*ny))    
    ax=plt.gca() 
    ### MARKER FOR RESPONSES TIME AND GO CUE AND STOP CUE
    plt.axvspan(t_action0, t_action1, ymin=0, facecolor='g', alpha=0.1)
    plt.axvline(t_goCue, color='g', linestyle='dotted', linewidth=linew)
    plt.axvline(t_stopCue, color='r', linestyle='dotted', linewidth=linew)       
    ax.set_yticks([])    
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(fsize)    
    ax_wrapper.append(ax_wrapper[0].twinx())
    ## CALCULATE AND PLOT INTEGRATOR LINES
    int_mean_failedStop = np.nanmean(rsp_mInt_stop[:, nz_FailedStop], 1)
    int_mean_correctGo = np.nanmean(rsp_mInt_go[:, nz_Go], 1)
    plt.plot(int_mean_failedStop, color='tomato', ls='--', lw=linew, label=labelList[0])
    plt.plot(int_mean_correctGo, color=np.array([107,139,164])/255., lw=linew, label=labelList[1])
    ## THRESHOLD MARKER
    plt.plot([0, rsp_mInt_stop.shape[0]], thresh * np.ones(2), 'k--', lw=linew)
    ## TITLE, LABELS, AXIS
    plt.title('Integrator-Go', fontsize=fsize)
    ax=plt.gca()
    ax.axis([tmin, tmax, 0, paramsA['Fig7_maxRates']['IntegratorGo']])
    ax.set_xticks(xTickValues)
    ax.set_xticklabels(xTickLabels)
    plt.ylabel('$g_{AMPA}$', fontsize=fsize)           
    ax_twin = plt.gca()
    ax_twin.set_ylabel('Integrator value', fontsize=fsize)
    for label in ax_twin.get_xticklabels() + ax_twin.get_yticklabels():
        label.set_fontsize(fsize)   
     
        
    ### GENERAL XLABEL
    fig=plt.gcf()
    fig.text(0.4, 0.12, 'Time from Stop cue [sec]', fontsize=fsize)


    ### LEGEND
    normalLine = mlines.Line2D([], [], color=np.array([107,139,164])/255., label=labelList[1])
    dashedLine = mlines.Line2D([], [], color='tomato', ls='--', label=labelList[0])
    xLegend = 0.5
    yLegend = 0.06
    leg = plt.legend(handles=[normalLine,dashedLine], bbox_to_anchor=(xLegend,yLegend), bbox_transform=plt.gcf().transFigure, fontsize=fsize, loc='center', ncol=2)
    

    ### SAVE
    plt.subplots_adjust(bottom=0.2, top=0.95, wspace=0.5, hspace=0.4, left=0.08, right=0.9)
    if smoothing:
        plt.savefig(resultsFolder+'meanrate_All_FailedStopVsCorrectGo_rate_paramsid'+str(int(param_id))+'_'+str(trials)+'trials_smoothed.svg', dpi=300)
    else:
        plt.savefig(resultsFolder+'meanrate_All_FailedStopVsCorrectGo_rate_paramsid'+str(int(param_id))+'_'+str(trials)+'trials.svg', dpi=300)
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
