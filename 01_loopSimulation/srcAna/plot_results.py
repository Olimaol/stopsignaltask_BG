import os
import pylab as plt
import numpy as np
import scipy.stats as st
from scipy import interpolate
from BGmodelSST.sim_params import params


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
            t_StopCue = params['t_init'] + params['t_SSD']
            RT_corrGo[i_cycle, i_netw_rep] = results_RT.item()['meanRT_CorrectGo']
            RT_failedStop[i_cycle, i_netw_rep] = results_RT.item()['meanRT_FailedStop']                  
            weights[i_cycle] = loop_params[cycle]
    i_sorted = np.argsort(weights)
    plt.ion()
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

    standard_plot_2lines_std_subplot(212, weights[i_sorted], np.nanmean(RT_corrGo, 1)[i_sorted], np.nanstd(RT_corrGo, 1)[i_sorted], 'correct Go', \
                         weights[i_sorted], np.nanmean(RT_failedStop, 1)[i_sorted], np.nanstd(RT_failedStop, 1)[i_sorted], 'failed Stop', \
                        changeParamName(paramname)+[' weights',''][int(paramname[-5:]=='rates')], 'Reaction time [ms]', 'plots/summary_RT_vs_w_'+paramname+'_'+str(param_id)+'.png', 6, ymin=0, ymax=600, xmin=loop_paramsSoll[0], xmax=loop_paramsSoll[-1]) # sem # np.nanstd
                        
    plt.tight_layout()

    saveFolder='../results/parametervariations/'+str(param_id)
    try:
        os.makedirs(saveFolder)
    except:
        if os.path.isdir(saveFolder):
            print(saveFolder+' already exists')
        else:
            print('could not create '+saveFolder+' folder')
            quit()

    plt.savefig(saveFolder+'/pctcorrect_timing_RT_vs_'+paramname+'_weight_paramsid'+str(param_id)+'.'+saveFormat, dpi=300)
         
    plt.ioff()                        
    plt.show()


def standard_plot_2lines_std_subplot(subplotind, xdata1, ydata1, yerr1, label1, xdata2, ydata2, yerr2, label2, xlabel_text, ylabel_text, filename, fsize, ymin=0.0, ymax=100.0, xmin=0.0, xmax=1.0):
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
            t_StopCue = params['t_init'] + params['t_SSD']                       
            RT_corrGo[i_cycle, i_netw_rep] = results_RT.item()['meanRT_CorrectGo']
            RT_failedStop[i_cycle, i_netw_rep] = results_RT.item()['meanRT_FailedStop']                  
            weights[i_cycle] = loop_params[cycle]
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
        saveFolder='../results/parametervariations'
        try:
            os.makedirs(saveFolder)
        except:
            if os.path.isdir(saveFolder):
                print(saveFolder+' already exists')
            else:
                print('could not create '+saveFolder+' folder')
                quit()
        plt.savefig(saveFolder+'/FourParameterVariations.svg', dpi=300)
        plt.ioff()


def standard_plot_2lines_std_subplot_FOURINAROW(sig,subplotind, xdata1, ydata1, yerr1, label1, xdata2, ydata2, yerr2, label2, xlabel_text, ylabel_text, filename, fsize, ymin=0.0, ymax=100.0, xmin=0.0, xmax=1.0):
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
    with open('../results/parametervariations/ArkyStop.txt', fileopen) as f:
        print(changeName[xlabel_text], file=f)
        if nr>4:
            print('correct Go RT: M(0) =',ydata1[0],'SD(0) =',yerr1[0],'M(1) =',ydata1[len(ydata1)-1],'SD(1) =',yerr1[len(yerr1)-1],'H = ',sig[0][0],'p = ',sig[0][1],'sig = ',sig[0][1]<(0.05/12.), file=f)
            print('failed Stop RT: M(0) =',ydata2[0],'SD(0) =',yerr2[0],'M(1) =',ydata2[len(ydata2)-1],'SD(1) =',yerr2[len(yerr2)-1],'H = ',sig[1][0],'p = ',sig[1][1],'sig = ',sig[1][1]<(0.05/12.), file=f)
        else:
            print('correct Go %: M(0) =',ydata1[0],'SD(0) =',yerr1[0],'M(1) =',ydata1[len(ydata1)-1],'SD(1) =',yerr1[len(yerr1)-1], file=f)
            print('correct Stop %: M(0) =',ydata2[0],'SD(0) =',yerr2[0],'M(1) =',ydata2[len(ydata2)-1],'SD(1) =',yerr2[len(yerr2)-1],'H = ',sig[0],'p = ',sig[1],'sig = ',sig[1]<(0.05/12.), file=f)
            

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


def plot_RT_distributions_correctStop():
    param_id_list = ['8007', '8008', '8009', '8010', '8011', '8012','8013'] # 8008-8013
    network_IDs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    n_networks = len(network_IDs)
    trials = 200 
    t_init = params['t_init'] # 100
    dt = float(params['general_dt'])

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
            datapath = '../data/parametervariations/'
            mInt_stop = np.load(datapath+'Integrator_ampa_Stop_'+str(netw)+'_id'+str(int(param_id))+'.npy')    
            mInt_go = np.load(datapath+'Integrator_ampa_Go_'+str(netw)+'_id'+str(int(param_id))+'.npy')     
            results_RT = np.load(datapath+'resultsRT_'+str(netw)+'_param_'+'CortexStop_rates'+'_cycle'+str(0)+'id'+str(int(param_id))+'.npy', allow_pickle=True) 
            n_StopTrials = results_RT.item()['nFailedStopTrials'] + results_RT.item()['nCorrectStopTrials']
            n_GoTrials = results_RT.item()['nFailedGoTrials'] + results_RT.item()['nCorrectGoTrials']
            pct_CorrectStop[param_id][i_netw] = 100 * results_RT.item()['nCorrectStopTrials'] / float(n_StopTrials)
            pct_FailedStop[param_id][i_netw] = 100 * results_RT.item()['nFailedStopTrials'] / float(n_StopTrials)
            pct_CorrectGo[param_id][i_netw] = 100 * results_RT.item()['nCorrectGoTrials'] / float(n_GoTrials)
            pct_FailedGo[param_id][i_netw] = 100 * results_RT.item()['nFailedGoTrials'] / float(n_GoTrials)
            rsp_mInt_stop = np.reshape(mInt_stop, [int(mInt_stop.shape[0] / float(trials)), trials], order='F')
            rsp_mInt_go = np.reshape(mInt_go, [int(mInt_go.shape[0] / float(trials)), trials], order='F')
            mInt_maxpertrial = np.nanmax(rsp_mInt_go, 0)
            mInt_maxpertrial = np.nanmax(rsp_mInt_stop, 0)
            for i_trial in range(trials):
                if np.nanmax(rsp_mInt_go[:, i_trial]) >= params['IntegratorNeuron_threshold']: 
                    RT_Go[param_id][i_netw, i_trial] = np.nonzero(rsp_mInt_go[:, i_trial] >= params['IntegratorNeuron_threshold'])[0][0]            
                if np.nanmax(rsp_mInt_stop[:, i_trial]) >= params['IntegratorNeuron_threshold']: 
                    RT_Stop[param_id][i_netw, i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= params['IntegratorNeuron_threshold'])[0][0] # Correct

        rsp_RT_Go[param_id] = np.reshape(RT_Go[param_id], n_networks * trials)
        rsp_RT_Stop[param_id] = np.reshape(RT_Stop[param_id], n_networks * trials)    
        nz_Go[param_id] = np.nonzero(np.isnan(rsp_RT_Go[param_id])==False)    
        nz_Stop[param_id] = np.nonzero(np.isnan(rsp_RT_Stop[param_id])==False)
        counts_Go, bins_Go = np.histogram(rsp_RT_Go[param_id][nz_Go[param_id]] * dt, 10) #     
        counts_Stop, bins_Stop = np.histogram(rsp_RT_Stop[param_id][nz_Stop[param_id]] * dt, 10) # 

    mean_CorrectGo = np.round( (np.nanmean(rsp_RT_Go[param_id][nz_Go[param_id]]) - t_init/dt)*dt, 1)    
    mean_FailedStop = np.round( (np.nanmean(rsp_RT_Stop[param_id][nz_Stop[param_id]]) - t_init/dt)*dt, 1)

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
    with open('../results/parametervariations/stopComponents.txt', 'w') as f:
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
        plt.savefig('../results/parametervariations/RT_DISTRIBUTIONS_'+mode+'.png', dpi=300)
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
        plt.savefig('../results/parametervariations/PERFORMANCE_'+mode+'.png', dpi=300)
        plt.close()


if __name__ == '__main__':

    ### PARAMETERS FOR PARAMETERVARIATION PLOTS
    ## NETWORKNUMBERS OF PARAMETERVARIATION SIMULATIONS
    network_array = {
                    'Cortex_SGPe_Arky' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                    'ArkyD1Stop'       : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
                    'ArkyD2Stop'       : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
                    'ArkyFSIStop'      : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], 
                    'CortexGo_rates'   : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                    }
    ## GENERAL_ID OF PARAMETERVARIATION SIMULATIONS
    param_id =      {
                    'Cortex_SGPe_Arky' : 8007,
                    'ArkyD1Stop'       : 8014, 
                    'ArkyD2Stop'       : 8014, 
                    'ArkyFSIStop'      : 8014, 
                    'CortexGo_rates'   : 8007
                    }
    ## DATAFOLDER OF PARAMETERVARIATION SIMULATIONS
    loadFolder =    {
                    'Cortex_SGPe_Arky' : '../data/parametervariations/',
                    'ArkyD1Stop'       : '../data/parametervariations/', 
                    'ArkyD2Stop'       : '../data/parametervariations/', 
                    'ArkyFSIStop'      : '../data/parametervariations/', 
                    'CortexGo_rates'   : '../data/parametervariations/'
                    }
    ## PARAMETERVARIATIONS OF PARAMETERVARIATION SIMULATIONS
    variations =    {
                    'Cortex_SGPe_Arky' : [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    'ArkyD1Stop'       : [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                    'ArkyD2Stop'       : [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                    'ArkyFSIStop'      : [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                    'CortexGo_rates'   : [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
                    }
    ## FORMAT FOR RESULT FIGURE
    saveFormat =    {
                    'Cortex_SGPe_Arky' : 'png',
                    'ArkyD1Stop'       : 'png', 
                    'ArkyD2Stop'       : 'png', 
                    'ArkyFSIStop'      : 'png', 
                    'CortexGo_rates'   : 'svg'
                    }

    
    ### FIGURE 4C,D IN MANUSCRIPT
    for paramIdx, paramname in enumerate(['CortexGo_rates']):
        plot_pctcorrect_timing_RT_vs_weight_cycles(param_id[paramname], network_array[paramname], paramname, loadFolder[paramname], variations[paramname], saveFormat[paramname])

    # FIGURE 10 IN MANUSCRIPT
    for paramIdx, paramname in enumerate(['Cortex_SGPe_Arky', 'ArkyD1Stop', 'ArkyD2Stop', 'ArkyFSIStop']):
        plot_pctcorrect_timing_RT_vs_weight_cycles_FOURINAROW(param_id[paramname], network_array[paramname], paramname, loadFolder[paramname], variations[paramname], paramIdx)             
   
    # FIGURE 8 IN MANUSCRIPT
    plot_RT_distributions_correctStop()        

    
