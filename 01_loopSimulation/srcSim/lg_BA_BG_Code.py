"""
Created on Fri Oct 16 2020

@authors: Oliver Maith, Lorenz Goenner, ilko
"""

import sys
#sys.path.insert(1, '../..')# to import mainScripts
from ANNarchy import*
import pylab 
import random
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patch # fuer legende
import scipy.stats as st
from scipy import stats #fuer z score -> mean firing rate
from scipy.signal import argrelmin

from BGmodelSST.analysis import custom_zscore, custom_zscore_start0, plot_zscore_stopVsGo, plot_zscore_stopVsGo_five, plot_zscore_stopVsGo_NewData, plot_zscore_stopVsGo_five_NewData, plot_correl_rates_Intmax, plot_meanrate_All_FailedVsCorrectStop, plot_meanrate_All_FailedStopVsCorrectGo, calc_KW_stats_all, rate_plot, calc_meanrate_std_failed_correct, get_rates_failed_correct, get_poprate_aligned_onset, get_peak_response_time, custom_poprate, calc_meanrate_std_Fast_Slow, get_fast_slow_go_trials, get_rates_allGo_fastGo_slowGo
from BGmodelSST.plotting import get_and_plot_syn_mon    
from BGmodelSST.sim_params import params
from BGmodelSST.init import init_neuronmodels
import matplotlib.lines as mlines


setup(dt=0.1)
setup(num_threads=1)

from BGmodelSST.neuronmodels import Izhikevich_neuron, Izhikevich_STR_neuron, STR_FSI_neuron, Integrator_neuron
from BGmodelSST.populations import Stoppinput1, Cortex_S, Cortex_G, STR_D1, STR_D2, STN, SNr, GPe_Proto, Thal, Integrator, IntegratorStop, GPeE, SNrE, STNE, STR_FSI, STRE, GPe_Arky, TestThalnoise, GPe_Proto2
from BGmodelSST.projections import Stoppinput1STN, Cortex_GSTR_D1, Cortex_GSTR_D2, Cortex_GSTR_FSI, Cortex_GThal, Cortex_SGPe_Arky, STR_D1SNr, STR_D2GPe_Proto, STNSNr, STNGPe_Proto, GPe_ProtoSTN, GPe_ProtoSNr, SNrThal, ThalIntegrator, GPeEGPe_Proto, GPEGPe_Arky, SNrESNr, STNESTN, STR_FSISTR_D1, STR_FSISTR_D2, STRESTR_D1, STRESTR_D2, GPe_ArkySTR_D1, GPe_ArkySTR_D2, TestThalnoiseThal,STRESTR_FSI, STR_FSISTR_FSI, GPe_ArkySTR_FSI, GPe_ArkyGPe_Proto, GPe_ProtoGPe_Arky, STR_D2GPe_Arky, GPe_ProtoSTR_FSI, STR_D1STR_D1, STR_D1STR_D2, STR_D2STR_D1, STR_D2STR_D2, Cortex_GSTR_D2, Cortex_SGPe_Proto, STNGPe_Arky, ThalSD1, ThalSD2, ThalFSI, GPe_ProtoGPe_Proto2, GPe_Proto2GPe_Proto, STR_D2GPe_Proto2, STR_D1GPe_Proto2, STNGPe_Proto2, Cortex_SGPe_Proto2, GPe_ArkyGPe_Proto2, GPe_Proto2STR_D1, GPe_Proto2STR_D2, GPe_Proto2STR_FSI, GPe_Proto2GPe_Arky, GPe_Proto2IntegratorStop, EProto1GPe_Proto, EProto2GPe_Proto2, EArkyGPe_Arky, Cortex_SGPe_Arky2, STR_D2GPe_Arky2, GPe_ProtoGPe_Arky2, STNGPe_Arky2, GPe_Proto2GPe_Arky2, EArkyGPe_Arky2, GPe_Arky2STR_D1, GPe_Arky2STR_D2, GPe_Arky2STR_FSI

population_size=params['general_populationSize']
print(population_size)
quit()

def Kernel(x,smooth):
    s=smooth
    K=(1./(np.sqrt(2*np.pi*s**2)))*np.exp(-x**2/(2*s**2))
    
    return K

def get_smoothed_rate(spikes,dt,MonitoringTime):
    MonitoringTime=int(MonitoringTime+1)
    numNeurons=len(spikes)
    #timesteps = 10ms bins
    timesteps=np.arange(0,MonitoringTime,10)
    smoothedRate=np.zeros((numNeurons,timesteps.size))
    smoothAlpha=0.25    

    for neuron in range(numNeurons):
        #smooth = 0.5/meanFiringrate in ms
        meanFr=len(spikes[neuron])/float(MonitoringTime)
        if meanFr>0:
            smooth=smoothAlpha/meanFr
            if key=='Cortex_STN' or key=='STN':
                smooth=smoothAlpha/(30/1000.)
            if key=='Cortex_GO' or key=='Thal':
                smooth=smoothAlpha/(20/1000.)
            for idx,t in enumerate(timesteps):
                #spiketimes * dt --> spiketimes in ms
                spikeTimes=np.array(spikes[neuron])*dt
                kernelInput=t-spikeTimes
                kernelOutput=Kernel(kernelInput,smooth)
                smoothedRate[neuron,idx]=np.sum(kernelOutput)*1000#rate calculated in ms --> for Hz multiply by 1000
        else:
            smoothedRate[neuron,:]=0
    return [timesteps,smoothedRate]

def changeStopMechanism(fArky, fSTN, fProto2):
    Cortex_SGPe_Arky.mod_factor              = params['weights_cortexStop_GPeArky']*fArky
    params['cortexPause_ratesSecondRespMod'] = params['cortexPause_ratesSecondRespMod']*fSTN
    Cortex_SGPe_Proto2.mod_factor            = params['weights_cortexStop_GPeCp']*fProto2



# Apply weight factors here, to avoid re-compiling. 
# CAUTION: Will not be considered in the automatic report!
# CAUTION: appears to slow down spike transmission!

GPe_ArkySTR_D1.mod_factor = params['weights_GPeArky_StrD1']
GPe_ArkySTR_D2.mod_factor = params['weights_GPeArky_StrD2']
GPe_ArkyGPe_Proto.mod_factor = params['weights_GPeArky_GPeProto']
GPe_ArkySTR_FSI.mod_factor =  params['weights_GPeArky_StrFSI']
Cortex_GSTR_D1.mod_factor = params['weights_cortexGo_StrD1']
Cortex_GSTR_D2.mod_factor = params['weights_cortexGo_StrD2']
Cortex_GThal.mod_factor = params['weights_cortexGo_Thal']
Cortex_GSTR_FSI.mod_factor = params['weights_cortexGo_StrFSI']
Stoppinput1STN.mod_factor = params['weights_cortexPause_STN']
Cortex_SGPe_Arky.mod_factor = params['weights_cortexStop_GPeArky']
Cortex_SGPe_Proto.mod_factor = 0
STR_FSISTR_D1.mod_factor = params['weights_StrFSI_StrD1']
STR_FSISTR_D2.mod_factor = params['weights_StrFSI_StrD2']
STR_FSISTR_FSI.mod_factor = params['weights_StrFSI_StrFSI']
GPeEGPe_Proto.mod_factor = 0
GPEGPe_Arky.mod_factor = 0
GPe_ProtoSTN.mod_factor = params['weights_GPeProto_STN']
GPe_ProtoSNr.mod_factor = params['weights_GPeProto_Snr']
GPe_ProtoGPe_Arky.mod_factor = params['weights_GPeProto_GPeArky']
GPe_ProtoSTR_FSI.mod_factor = params['weights_GPeProto_StrFSI'] 
STR_D1SNr.mod_factor = params['weights_StrD1_SNr']
STR_D1STR_D1.mod_factor = params['weights_StrD1_StrD1']
STR_D1STR_D2.mod_factor = params['weights_StrD1_StrD2']
STR_D2GPe_Proto.mod_factor = params['weights_StrD2_GPeProto']
STR_D2GPe_Arky.mod_factor = params['weights_StrD2_GPeArky']
STR_D2STR_D1.mod_factor = params['weights_StrD2_StrD1']
STR_D2STR_D2.mod_factor = params['weights_StrD2_StrD2']
SNrThal.mod_factor = params['weights_SNr_Thal']
SNrESNr.mod_factor = params['weights_SNrE_SNr']
STNSNr.mod_factor = params['weights_STN_SNr']
STNGPe_Proto.mod_factor = params['weights_STN_GPeProto']
STNGPe_Arky.mod_factor = params['weights_STN_GPeArky']
STNESTN.mod_factor = params['weights_STNE_STN']
STRESTR_D1.mod_factor = params['weights_StrD1E_StrD1'] 
STRESTR_D2.mod_factor = params['weights_StrD2E_StrD2']
STRESTR_FSI.mod_factor = params['weights_StrFSIE_StrFSI']
ThalIntegrator.mod_factor = params['weights_Thal_IntGo'] 
TestThalnoiseThal.mod_factor = params['weights_ThalE_Thal']
ThalSD1.mod_factor = params['weights_Thal_StrSD1']
ThalSD2.mod_factor = params['weights_Thal_StrSD2']
ThalFSI.mod_factor = params['weights_Thal_StrFSI']
GPe_ProtoGPe_Proto2.mod_factor      = params['weights_GPeProto_GPeCp']
GPe_Proto2GPe_Proto.mod_factor      = params['weights_GPeCp_GPeProto']
STR_D2GPe_Proto2.mod_factor         = params['weights_StrD2_GPeCp']
STR_D1GPe_Proto2.mod_factor         = params['weights_StrD1_GPeCp']
STNGPe_Proto2.mod_factor            = params['weights_STN_GPeCp']
Cortex_SGPe_Proto2.mod_factor       = params['weights_cortexStop_GPeCp']
GPe_ArkyGPe_Proto2.mod_factor       = params['weights_GPeArky_GPeCp']
GPe_Proto2STR_D1.mod_factor         = params['weights_GPeCp_StrD1']
GPe_Proto2STR_D2.mod_factor         = params['weights_GPeCp_StrD2']
GPe_Proto2STR_FSI.mod_factor        = params['weights_GPeCp_StrFSI']
GPe_Proto2GPe_Arky.mod_factor       = params['weights_GPeCp_GPeArky']
GPe_Proto2IntegratorStop.mod_factor = params['weights_GPeCp_IntStop']
EProto1GPe_Proto.mod_factor         = params['weights_GPeProtoE_GPeProto']
EProto2GPe_Proto2.mod_factor        = params['weights_GPeCpE_GPeCp']
EArkyGPe_Arky.mod_factor            = params['weights_GPeArkyE_GPeArky']

Cortex_SGPe_Arky2.mod_factor   = 0
STR_D2GPe_Arky2.mod_factor     = params['weights_StrD2_GPeArky']*params['GPeArkyCopy_On']
GPe_ProtoGPe_Arky2.mod_factor  = params['weights_GPeProto_GPeArky']*params['GPeArkyCopy_On']
STNGPe_Arky2.mod_factor        = params['weights_STN_GPeArky']*params['GPeArkyCopy_On']
GPe_Proto2GPe_Arky2.mod_factor = params['weights_GPeCp_GPeArky']*params['GPeArkyCopy_On']
EArkyGPe_Arky2.mod_factor      = params['weights_GPeArkyE_GPeArky']*params['GPeArkyCopy_On']
GPe_Arky2STR_D1.mod_factor     = params['weights_GPeArky_StrD1']*params['GPeArkyCopy_On']
GPe_Arky2STR_D2.mod_factor     = params['weights_GPeArky_StrD2']*params['GPeArkyCopy_On']
GPe_Arky2STR_FSI.mod_factor    = params['weights_GPeArky_StrFSI']*params['GPeArkyCopy_On']

def changeArkyStopOutput(fD1,fD2,fFSI):
    """
    change Arky-Str weights but compansate it with second Arky population, thus only stop input effect is changed (second Arky doesn't get stop input)
    """
    GPe_ArkySTR_D1.mod_factor = params['weights_GPeArky_StrD1']*fD1
    GPe_ArkySTR_D2.mod_factor = params['weights_GPeArky_StrD2']*fD2
    GPe_ArkySTR_FSI.mod_factor =  params['weights_GPeArky_StrFSI']*fFSI
    GPe_Arky2STR_D1.mod_factor     = params['weights_GPeArky_StrD1']*params['GPeArkyCopy_On']*(1-fD1)#np.clip((1-fD1),0,None)
    GPe_Arky2STR_D2.mod_factor     = params['weights_GPeArky_StrD2']*params['GPeArkyCopy_On']*(1-fD2)#np.clip((1-fD2),0,None)
    GPe_Arky2STR_FSI.mod_factor    = params['weights_GPeArky_StrFSI']*params['GPeArkyCopy_On']*(1-fFSI)#np.clip((1-fFSI),0,None)







t = 0
#general parameters
trials = 10 # 25 # 300 # 100 # Caution - memory consumption gets critical! 300 is OK.
t_smooth_ms = 20.0 # 5.0 # 1.0 # Errors for 5.0-20.0 ?!
t_decay = 300 # 500 #


#main loop

if len(sys.argv) <= 1:
    compile()
    setup(seed=0)
    np.random.seed(0)
    i_netw_rep = ''
elif len(sys.argv) > 1:
    compile('Annarchy'+str(sys.argv[1]))
    print('Argument List:', str(sys.argv[1:]))
    i_netw_rep = str(sys.argv[1])
    print("i_netw_rep = ", i_netw_rep)

    try:
        os.mkdir('plots/'+str(i_netw_rep))
    except:
        print(str(i_netw_rep)+' not created')



if len(i_netw_rep)==0:
    saveFolderPlots='plots/'
else:
    saveFolderPlots='plots/'+i_netw_rep+'/'


simulate(50)              # for stable point
reset(populations=True, projections=False, synapses=False, net_id=0) # Does the reset cause the "transients" in firing rates?

#STR_D1.compute_firing_rate(5.0) # Errors?!
simlen = t_decay - 250 + 950 + (params['t_GoAndStop'] - 100) +params['t_init']-300+params['t_SSD']-250#changed t_init and t_SSD
###
t_init = params['t_init'] # 100
t_SSD = params['t_SSD'] # 170 # = SSD (stop signal delay), Defines the onset of the stop cue / pause signal , vorher 170
t_delaySTOP = params['t_delaySTOP']
t_CorSTOPDuration = params['t_GoAndStop']
simlen = t_SSD + t_delaySTOP*int(params['id']>=7000) + t_CorSTOPDuration + t_decay + t_init
###
rate_data_SD1_Go = np.nan*np.ones([trials, int(simlen / dt())]) # 1040 # 950 = t_init + t_SSD + t_PauseDuration + t_delay + 30 + 70 + 250 = 300 + 250 + 10 + 40 + 350 
rate_data_SD1_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rate_data_SD2_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_SD2_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rate_data_FSI_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_FSI_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rate_data_STN_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_STN_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rate_data_GPeProto_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_GPeProto2_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_GPeProto_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rate_data_GPeProto2_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rate_data_GPeArky_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_GPeArky_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rateperneuron_GPeArky_allStoptrials = np.nan*np.ones([trials, population_size, int(simlen / dt())])
rate_data_SNr_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_SNr_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rate_data_SNrE_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_SNrE_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rate_data_Thal_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_Thal_Stop = np.nan*np.ones([trials, int(simlen / dt())])
rate_data_Cortex_G_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_Cortex_G_Stop = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_Cortex_S_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_Cortex_S_Stop = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_Stoppinput1_Go = np.nan*np.ones([trials, int(simlen / dt())]) 
rate_data_Stoppinput1_Stop = np.nan*np.ones([trials, int(simlen / dt())]) 

m_STR_D1   = Monitor(STR_D1,['spike']) 
m_STR_D2   = Monitor(STR_D2,['spike']) 
m_STR_FSI   = Monitor(STR_FSI,['spike'])
m_STN   = Monitor(STN,['spike']) 
m_GPe_Proto  = Monitor(GPe_Proto,['spike']) 
m_GPe_Arky  = Monitor(GPe_Arky,['spike']) 
m_SNr = Monitor(SNr,['spike'])
m_Thal = Monitor(Thal,['spike'])
m_Int_ampa = Monitor(Integrator, ['g_ampa', 'spike'])
m_StopInt_ampa = Monitor(IntegratorStop, ['g_ampa', 'spike'])
m_SNrE = Monitor(SNrE,['spike'])
m_Cortex_G  = Monitor(Cortex_G,'spike') 
m_Cortex_S = Monitor(Cortex_S,'spike') 
m_Stoppinput1 = Monitor(Stoppinput1,'spike')

m_STR_D1_new   = Monitor(STR_D1, ['spike'])
m_STR_D2_new   = Monitor(STR_D2, ['spike'])
m_STR_FSI_new   = Monitor(STR_FSI, ['spike'])
m_STN_new = Monitor(STN,['spike'])
m_GPe_Proto_new  = Monitor(GPe_Proto,['spike']) 
m_GPe_Proto2_new  = Monitor(GPe_Proto2,['spike']) 
m_GPe_Arky_new  = Monitor(GPe_Arky,['spike']) 
m_SNr_new = Monitor(SNr,['spike'])
m_SNrE_new = Monitor(SNrE,['spike'])
m_Thal_new = Monitor(Thal,['spike'])
m_Cortex_G_new  = Monitor(Cortex_G,'spike') 
m_Cortex_S_new = Monitor(Cortex_S,'spike') 
m_Stoppinput1_new = Monitor(Stoppinput1,'spike')

m_syn_D1 = Monitor(STR_D1[0], ['g_ampa', 'g_gaba'])
m_syn_D2 = Monitor(STR_D2[0], ['g_ampa', 'g_gaba'])
m_syn_FSI = Monitor(STR_FSI[0], ['g_ampa', 'g_gaba'])
m_syn_STN = Monitor(STN[0], ['g_ampa', 'g_gaba'])
m_syn_Proto = Monitor(GPe_Proto[0], ['g_ampa', 'g_gaba'])
m_syn_Arky = Monitor(GPe_Arky[0], ['g_ampa', 'g_gaba'])
m_syn_SNr = Monitor(SNr[0], ['g_ampa', 'g_gaba'])
m_syn_Thal = Monitor(Thal[0], ['g_ampa', 'g_gaba'])

selection = np.zeros(trials)
timeovertrial = np.zeros(trials)
zaehler_go = 0


#'''#
params['wmean_Stoppinput1STN'] = np.array(Stoppinput1STN.w).mean()
params['wmean_Cortex_GSTR_D1'] = np.array(Cortex_GSTR_D1.w).mean()
params['wmean_Cortex_GThal'] = np.array(Cortex_GThal.w).mean()
params['wmean_Cortex_SGPe_Arky'] = np.array(Cortex_SGPe_Arky.w).mean()
params['wmean_STR_D1SNr'] = np.array(STR_D1SNr.w).mean()
params['wmean_STNSNr'] = np.array(STNSNr.w).mean()
params['wmean_STNGPe_Proto'] = np.array(STNGPe_Proto.w).mean()
params['wmean_GPe_ProtoSTN'] = np.array(GPe_ProtoSTN.w).mean()
params['wmean_GPe_ProtoSNr'] = np.array(GPe_ProtoSNr.w).mean()
params['wmean_SNrThal'] = np.array(SNrThal.w).mean()
params['wmean_ThalIntegrator'] = np.array(ThalIntegrator.w).mean()
#params['wmean_GPe_ArkySTR_D1'] = np.array(GPe_ArkySTR_D1.w).mean()
params['wmean_GPeEGPe_Proto'] = np.array(GPeEGPe_Proto.w).mean()
params['wmean_GPEGPe_Arky'] = np.array(GPEGPe_Arky.w).mean()
params['wmean_SNrESNr'] = np.array(SNrESNr.w).mean()
params['wmean_STNESTN'] = np.array(STNESTN.w).mean()
np.save('data/paramset_id'+str(int(params['id']))+str(i_netw_rep)+'.npy', params)
#'''

sigma_Go = params['sigma_Go'] # 0.0 # 5.0 # 10.0 #
sigma_Stop = params['sigma_Stop'] #
sigma_Pause = params['sigma_Pause'] #

print('starte versuche')

# loop fuer 100 		Versuche -> Eingabe "loop" notwendig ! 
#abfrage_loop = raw_input("Eingabe 'GO' fuer '+str(trials)' Go trials oder 'STOP' fuer '+str(trials)' Stop trials, 'STR' fuer STR und GPe_Proto Plot, 'RT' fuer Reaktionszeiten, 'loop' fuer GO-STOP-RT, 'extra' fuer neue Vergleichsplots, ' ' (space) fuer Plots fuer einen run: ")
if len(sys.argv) <= 1:
    abfrage_loop = input("Eingabe 'STR' fuer STR und GPe_Proto Plot oder 'RT' fuer Reaktionszeiten, 'loop' fuer GO-STOP-RT, 'extra' fuer neue Vergleichsplots, ' ' (space) fuer Plots fuer einen run: ")
else:	
    abfrage_loop = str(sys.argv[2])






if abfrage_loop == 'loop':

    # Start loop for parameter variations here:
    makeParameterVariations=0 #2
    i_CortexGo_rates = i_CortexStop_rates = i_Stop1rates = i_weights_cortexStop_GPeArky = i_weights_cortexStop_GPeCp = i_weights_StrD2_GPeArky = i_weights_StrD2_GPeCp = i_weights_GPeArky_StrD1 = i_GPe_ArkySTR_D2 = i_weights_GPeCp_IntStop = i_GPe_ProtoSTR_FSI = i_STN_SNr = i_ArkyD1Stop = i_ArkyD2Stop = i_ArkyFSIStop = i_activateStopArky = i_activateStopSTN = i_activateStopProto2 = i_deactivateStopArky = i_deactivateStopSTN = i_deactivateStopProto2 = 99
    if makeParameterVariations==0:
        ###only one param wihtout variation
        paramsFactorList = np.array([1.0])
        paramnames = ['CortexStop_rates']
        i_CortexStop_rates = 0
    elif makeParameterVariations==1:
        ###one param with variation
        paramsFactorList = np.array([0.0, 0.5, 1.0, 1.5, 2])
        paramnames = ['weights_cortexStop_GPeArky']
        i_weights_cortexStop_GPeArky = 0
    elif makeParameterVariations==2:
        ###multiple params with variation
        paramsFactorList = np.array([0.0, 0.5, 1.0, 1.5, 2])
        #paramnames = ['CortexGo_rates', 'CortexStop_rates', 'Stop1rates', 'weights_cortexStop_GPeArky', 'weights_cortexStop_GPeCp', 'weights_StrD2_GPeArky', 'weights_StrD2_GPeCp', 'weights_GPeArky_StrD1', 'weights_GPeArky_StrD2', 'weights_GPeCp_IntStop', 'GPe_ProtoSTR_FSI', 'STN_SNr', 'i_ArkyD1Stop', 'i_ArkyD2Stop', 'i_ArkyFSIStop']
        #i_CortexGo_rates, i_CortexStop_rates, i_Stop1rates, i_weights_cortexStop_GPeArky, i_weights_cortexStop_GPeCp, i_weights_StrD2_GPeArky, i_weights_StrD2_GPeCp, i_weights_GPeArky_StrD1, i_weights_GPeArky_StrD2, i_weights_GPeCp_IntStop, i_GPe_ProtoSTR_FSI, i_STN_SNr, i_ArkyD1Stop, i_ArkyD2Stop, i_ArkyFSIStop = range(len(paramnames))
        #paramnames = ['i_ArkyD1Stop', 'i_ArkyD2Stop', 'i_ArkyFSIStop']
        #i_ArkyD1Stop, i_ArkyD2Stop, i_ArkyFSIStop = range(len(paramnames))
        #paramnames = ['activateStopArky', 'activateStopSTN', 'activateStopProto2', 'deactivateStopArky', 'deactivateStopSTN', 'deactivateStopProto2']
        #i_activateStopArky, i_activateStopSTN, i_activateStopProto2, i_deactivateStopArky, i_deactivateStopSTN, i_deactivateStopProto2 = range(len(paramnames))
        paramnames = ['CortexGo_rates', 'CortexStop_rates', 'weights_cortexStop_GPeArky']
        i_CortexGo_rates, i_CortexStop_rates, i_weights_cortexStop_GPeArky = range(len(paramnames))
    
    

    n_param_vars = len(paramnames) # 1

    params_orig_CortexGo = params['CortexGo_rates']
    params_orig_CortexStop = params['CortexStop_rates']
    params_orig_CortexPause = params['Stop1rates']
    params_orig_CortexPauseMod = params['cortexPause_ratesSecondRespMod']

    for i_paramname in range(n_param_vars):       
        paramname = paramnames[i_paramname]
        print('paramname = ', paramname)        

        if i_paramname == i_CortexGo_rates:
            params_orig =  params['CortexGo_rates']
            param_factors = np.array([0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5])
        elif i_paramname == i_CortexStop_rates:
            params_orig =  params['CortexStop_rates']
            param_factors = paramsFactorList                         
        elif i_paramname == i_Stop1rates:
            params_orig =  params['Stop1rates']
            param_factors = paramsFactorList 
        elif i_paramname == i_weights_cortexStop_GPeArky:
            params_orig, paramname = Cortex_SGPe_Arky.w, 'Cortex_SGPe_Arky'
            param_factors = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        elif i_paramname == i_weights_cortexStop_GPeCp:
            params_orig, paramname = Cortex_SGPe_Proto2.w, 'weights_cortexStop_GPeCp' 
            param_factors = paramsFactorList                                              
        elif i_paramname == i_weights_StrD2_GPeArky:
            params_orig, paramname = STR_D2GPe_Arky.w, 'STR_D2GPe_Arky'    
            param_factors = paramsFactorList                                             
        elif i_paramname == i_weights_StrD2_GPeCp:
            params_orig, paramname = STR_D2GPe_Proto2.w, 'weights_StrD2_GPeCp'  
            param_factors = paramsFactorList             
        elif i_paramname == i_weights_GPeArky_StrD1:
            params_orig, paramname = GPe_ArkySTR_D1.w, 'weights_GPeArky_StrD1'  
            param_factors = paramsFactorList                               
        elif i_paramname == i_weights_GPeArky_StrD2:
            params_orig, paramname = GPe_ArkySTR_D2.w, 'weights_GPeArky_StrD2'
            param_factors = paramsFactorList                             
        elif i_paramname == i_weights_GPeCp_IntStop:
            params_orig, paramname = GPe_Proto2IntegratorStop.w, 'weights_GPeCp_IntStop'
            param_factors = np.array([0.5, 0.6, 0.7, 0.8, 1.0])            
        elif i_paramname == i_GPe_ProtoSTR_FSI:
            params_orig, paramname = GPe_ProtoSTR_FSI.w, 'GPe_ProtoSTR_FSI'  
            param_factors = paramsFactorList                                       
        elif i_paramname == i_STN_SNr:
            params_orig, paramname = STNSNr.w, 'STN_SNr' 
            param_factors = paramsFactorList                              
        elif i_paramname == i_ArkyD1Stop:
            params_orig, paramname = 1, 'ArkyD1Stop'
            param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])                             
        elif i_paramname == i_ArkyD2Stop:
            params_orig, paramname = 1, 'ArkyD2Stop'
            param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])                           
        elif i_paramname == i_ArkyFSIStop:
            params_orig, paramname = 1, 'ArkyFSIStop'
            param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        elif i_paramname == i_activateStopArky:
            params_orig, paramname = 1, 'activateStopArky'
            param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])                         
        elif i_paramname == i_activateStopSTN:
            params_orig, paramname = 1, 'activateStopSTN'
            param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])                         
        elif i_paramname == i_activateStopProto2:
            params_orig, paramname = 1, 'activateStopProto2'
            param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        elif i_paramname == i_deactivateStopArky:
            params_orig, paramname = 1, 'deactivateStopArky'
            param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])                         
        elif i_paramname == i_deactivateStopSTN:
            params_orig, paramname = 1, 'deactivateStopSTN'
            param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])                         
        elif i_paramname == i_deactivateStopProto2:
            params_orig, paramname = 1, 'deactivateStopProto2'
            param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

                               
   

        n_loop_cycles = len(param_factors) # 1
        loop_params = param_factors# * params_orig
        

        np.save('data/cycle_params_'+paramname+'_id'+str(int(params['id']))+str(i_netw_rep)+'.npy', [loop_params, paramname])
        print('loop_params = ', loop_params)  

        for i_cycle in range(n_loop_cycles):
            plt.close('all')
            plt.ioff()
            print('i_cycle = ', i_cycle)


            t0 = 50 # 50ms simulation for initialization is not recorded, see above
            t_init = params['t_init'] # 100
            t_SSD = params['t_SSD'] # 170 # = SSD (stop signal delay), Defines the onset of the stop cue / pause signal , vorher 170
            t_PauseDuration = params['t_PauseDuration']
            t_delayGO = params['t_delayGO'] # 40 # 0 # Delay between Go/Stop cue presentation and onset of cortical Go/Stop activity
            t_delay = t_delayGO
            t_delaySTOP = params['t_delaySTOP']
            t_Pausedelay = 0 # 10 # 0 # Delay between Stop cue presentation and onset of Cortical "Pause" activity

            t_CorGODuration = params['t_CorGODuration']
            t_CorSTOPDuration = params['t_GoAndStop']
            t_motorSTOPDuration = params['t_motorStop']
            t_delaymotorSTOP = params['t_delaymotorSTOP']
            t_delayGOSD = params['t_delayGOSD']

            print("params = ", params)
            mode='GO'
            for i in range(0,trials):
                reset(populations=True, projections=True, synapses=False, net_id=0)
                if i_paramname == i_CortexGo_rates:             
                    params['CortexGo_rates'] = params_orig * param_factors[i_cycle] # Caution - must be reset after loop over this parameter !!!
                elif i_paramname == i_CortexStop_rates:
                    params['CortexStop_rates'] = params_orig * param_factors[i_cycle] 
                elif i_paramname == i_Stop1rates:
                    params['Stop1rates'] = params_orig * param_factors[i_cycle] 
                elif i_paramname == i_weights_cortexStop_GPeArky:            
                    Cortex_SGPe_Arky.mod_factor = param_factors[i_cycle] * params['weights_cortexStop_GPeArky']                
                elif i_paramname == i_weights_cortexStop_GPeCp:            
                    Cortex_SGPe_Proto2.mod_factor = param_factors[i_cycle] * params['weights_cortexStop_GPeCp']                
                elif i_paramname == i_weights_StrD2_GPeArky:  
                    STR_D2GPe_Arky.mod_factor = param_factors[i_cycle] * params['weights_StrD2_GPeArky']              
                elif i_paramname == i_weights_StrD2_GPeCp: 
                    STR_D2GPe_Proto2.mod_factor = param_factors[i_cycle] * params['weights_StrD2_GPeCp']   
                elif i_paramname == i_weights_GPeArky_StrD1:            
                    GPe_ArkySTR_D1.mod_factor = param_factors[i_cycle] * params['weights_GPeArky_StrD1']
                elif i_paramname == i_weights_GPeArky_StrD2:            
                    GPe_ArkySTR_D2.mod_factor = param_factors[i_cycle] * params['weights_GPeArky_StrD2'] 
                elif i_paramname == i_weights_GPeCp_IntStop:
                    GPe_Proto2IntegratorStop.mod_factor = param_factors[i_cycle] * params['weights_GPeCp_IntStop']   
                elif i_paramname == i_GPe_ProtoSTR_FSI:
                    GPe_ProtoSTR_FSI.mod_factor = param_factors[i_cycle] * params['weights_GPeProto_StrFSI']                 
                elif i_paramname == i_STN_SNr:                    
                    STNSNr.mod_factor = param_factors[i_cycle] * params['weights_STN_SNr']                
                elif i_paramname == i_ArkyD1Stop:                    
                    changeArkyStopOutput(param_factors[i_cycle],1,1)               
                elif i_paramname == i_ArkyD2Stop:                    
                    changeArkyStopOutput(1,param_factors[i_cycle],1)              
                elif i_paramname == i_ArkyFSIStop:                    
                    changeArkyStopOutput(1,1,param_factors[i_cycle])
                elif i_paramname == i_activateStopArky:                    
                    changeStopMechanism(param_factors[i_cycle],0,0)              
                elif i_paramname == i_activateStopSTN:                    
                    changeStopMechanism(0,param_factors[i_cycle],0)              
                elif i_paramname == i_activateStopProto2:                    
                    changeStopMechanism(0,0,param_factors[i_cycle])
                elif i_paramname == i_deactivateStopArky:                    
                    changeStopMechanism(param_factors[i_cycle],1,1)              
                elif i_paramname == i_deactivateStopSTN:                    
                    changeStopMechanism(1,param_factors[i_cycle],1)              
                elif i_paramname == i_deactivateStopProto2:                    
                    changeStopMechanism(1,1,param_factors[i_cycle])

                

                ### TRIAL START
                ### trial INITIALIZATION simulation to get stable state
                Cortex_G.rates = 0
                Cortex_S.rates = 0
                Stoppinput1.rates = 0

                simulate(t_init)# t_init rest

                ### Integrator Reset
                Integrator.decision = 0
                Integrator.g_ampa = 0  
                IntegratorStop.decision = 0 

                ### calculate all eventTIMES
                GOCUE     = np.round(get_time(),1)
                STOPCUE   = GOCUE + t_SSD
                STN1ON    = GOCUE
                STN1OFF   = STN1ON + t_PauseDuration
                GOON      = GOCUE + t_delayGO + int(np.clip(t_delayGOSD * np.random.randn(),-t_delayGO,None))
                STN2ON    = STOPCUE
                STN2OFF   = STN2ON + t_PauseDuration
                STOPON    = STOPCUE + t_delaySTOP
                STOPOFF   = STOPON + t_CorSTOPDuration
                ENDTIME   = STOPOFF + t_decay
                GOOFF     = ENDTIME
                motorSTOP = ENDTIME
                motorSTOPOFF = ENDTIME
                eventTimes = np.array([GOCUE,STOPCUE,STN1ON,STN1OFF,GOON,GOOFF,STN2ON,STN2OFF,STOPON,STOPOFF,ENDTIME,motorSTOP,motorSTOPOFF])
                randn_trial_Pause = np.random.randn()
                randn_trial = np.random.randn()
                randn_trial_S = np.random.randn()

                ### simulate all EVENTS
                motorResponse=0
                GOOFFResponse=0
                responseThal=0
                responseProto2=0
                t=np.round(get_time(),1)
                end=False
                tempMode=mode
                while not(end):
                    if t == STN1ON:
                        Stoppinput1.rates = params['Stop1rates'] + sigma_Pause * randn_trial_Pause
                        #print('STN1ON',STN1ON)
                    if t == STN1OFF:
                        Stoppinput1.rates = 0
                        #print('STN1OFF',STN1OFF)
                    if t == GOON:
                        Cortex_G.rates = params['CortexGo_rates'] + sigma_Go * randn_trial
                        #print('GOON',GOON)
                    if t == GOOFF:
                        Cortex_G.rates = 0
                        #print('GOOFF',GOOFF)                    
                    if t == STN2ON and tempMode=='STOP':
                        Stoppinput1.rates = params['cortexPause_ratesSecondRespMod']*params['Stop1rates'] + sigma_Pause * randn_trial_Pause
                        #print('STN2ON',STN2ON)
                    if t == STN2OFF and tempMode=='STOP':
                        Stoppinput1.rates = 0
                        #print('STN2OFF',STN2OFF)
                    if t == STOPON and tempMode=='STOP':
                        Cortex_S.rates = params['CortexStop_rates'] + sigma_Stop * randn_trial_S * (params['CortexStop_rates'] > 0)
                        #print('STOPON',STOPON)
                    if t == motorSTOP:
                        Cortex_S.rates = params['CortexmotorStop_rates'] + sigma_Stop * randn_trial_S * (params['CortexmotorStop_rates'] > 0)
                        motorSTOPOFF = motorSTOP + t_motorSTOPDuration
                        #print('motorSTOP',motorSTOP)
                    if t == STOPOFF and tempMode=='STOP' and ((motorSTOPOFF==ENDTIME) or (motorSTOPOFF!=ENDTIME and t>motorSTOPOFF)):
                        Cortex_S.rates = 0
                        #print('STOPOFF',STOPOFF)
                    if t == motorSTOPOFF:
                        Cortex_S.rates = 0
                        #print('motorSTOPOFF',motorSTOPOFF)
                    if t == ENDTIME:
                        end=True
                        #print('ENDTIME',ENDTIME)
                    else:
                        nowTIME = np.round(get_time(),1)
                        #print('nowTIME',nowTIME)
                        eventTimes = np.array([GOCUE,STOPCUE,STN1ON,STN1OFF,GOON,GOOFF,STN2ON,STN2OFF,STOPON,STOPOFF,ENDTIME,motorSTOP,motorSTOPOFF])
                        nextEvent = np.max([np.min(eventTimes[eventTimes>=nowTIME]-nowTIME),1])
                        #print('nextEvent',nextEvent)
                        if responseThal==0 and responseProto2==0:
                            simulate_until(max_duration=nextEvent, population=[Integrator,IntegratorStop], operator='or')
                        elif responseProto2==0:
                            simulate_until(max_duration=nextEvent, population=IntegratorStop)
                        elif responseThal==0:
                            simulate_until(max_duration=nextEvent, population=Integrator)
                        else:
                            simulate(nextEvent)
                        responseThal = int(Integrator.decision)
                        responseProto2 = int(IntegratorStop.decision)
                        t = np.round(get_time(),1)
                        #print('time:',t,'restsimulation:',np.ceil(t)-t)
                        simulate(np.round(np.ceil(t)-t,1))
                        t = np.round(get_time(),1)
                        if responseThal == -1 and motorResponse == 0:
                            motorResponse=1
                            motorSTOP = t + t_delaymotorSTOP
                            if t<STOPCUE and tempMode=='STOP':
                                tempMode='GO'
                        if responseProto2 == -1 and GOOFFResponse == 0 and ((t>STOPON and tempMode=='STOP') or motorResponse==1):
                            GOOFFResponse=1
                            GOOFF = t
                ### TRIAL END

                

                SD1_spikes = m_STR_D1_new.get('spike')            
                rate_data_SD1_currtrial = custom_poprate(m_STR_D1_new, SD1_spikes, t_smooth_ms )
                SD2_spikes = m_STR_D2_new.get('spike')
                te, ne = m_STR_D2_new.raster_plot(SD2_spikes)            
                rate_data_SD2_currtrial = custom_poprate(m_STR_D2_new, SD2_spikes, t_smooth_ms)
                FSI_spikes = m_STR_FSI_new.get('spike')
                rate_data_FSI_currtrial = custom_poprate(m_STR_FSI_new, FSI_spikes, t_smooth_ms)                        
                STN_spikes = m_STN_new.get('spike')
                rate_data_STN_currtrial = custom_poprate(m_STN_new, STN_spikes, t_smooth_ms)                        
                Proto_spikes = m_GPe_Proto_new.get('spike')
                Proto2_spikes = m_GPe_Proto2_new.get('spike')
                rate_data_GPeProto2_currtrial = custom_poprate(m_GPe_Proto2_new, Proto2_spikes, t_smooth_ms)
                rate_data_GPeProto_currtrial = custom_poprate(m_GPe_Proto_new, Proto_spikes, t_smooth_ms)                                     
                Arky_spikes = m_GPe_Arky_new.get('spike')
                rate_data_GPeArky_currtrial = custom_poprate(m_GPe_Arky_new, Arky_spikes, t_smooth_ms)
                SNr_spikes = m_SNr_new.get('spike')
                rate_data_SNr_currtrial = custom_poprate(m_SNr_new, SNr_spikes, t_smooth_ms)
                SNrE_spikes = m_SNrE_new.get('spike')
                rate_data_SNrE_currtrial = custom_poprate(m_SNrE_new, SNrE_spikes, t_smooth_ms)
                Thal_spikes = m_Thal_new.get('spike')
                rate_data_Thal_currtrial = custom_poprate(m_Thal_new, Thal_spikes, t_smooth_ms)
                Cortex_G_spikes = m_Cortex_G_new.get('spike')
                rate_data_Cortex_G_currtrial = custom_poprate(m_Cortex_G_new, Cortex_G_spikes, t_smooth_ms)            
                Cortex_S_spikes = m_Cortex_S_new.get('spike')
                rate_data_Cortex_S_currtrial = custom_poprate(m_Cortex_S_new, Cortex_S_spikes, t_smooth_ms)
                Stoppinput1_spikes = m_Stoppinput1_new.get('spike')
                rate_data_Stoppinput1_currtrial = custom_poprate(m_Stoppinput1_new, Stoppinput1_spikes, t_smooth_ms)               

                rate_data_SD1_Go[i, : ] = get_poprate_aligned_onset(m_STR_D1_new, SD1_spikes, rate_data_SD1_Go[i, : ], rate_data_SD1_currtrial, dt())            
                rate_data_SD2_Go[i, : ] = get_poprate_aligned_onset(m_STR_D2_new, SD2_spikes, rate_data_SD2_Go[i, : ], rate_data_SD2_currtrial, dt())                        
                rate_data_FSI_Go[i, : ] = get_poprate_aligned_onset(m_STR_FSI_new, FSI_spikes, rate_data_FSI_Go[i, : ], rate_data_FSI_currtrial, dt())                        
                rate_data_STN_Go[i, : ] = get_poprate_aligned_onset(m_STN_new, STN_spikes, rate_data_STN_Go[i, : ], rate_data_STN_currtrial, dt())                                    
                rate_data_GPeProto_Go[i, : ] = get_poprate_aligned_onset(m_GPe_Proto_new, Proto_spikes, rate_data_GPeProto_Go[i, : ], rate_data_GPeProto_currtrial, dt())             
                rate_data_GPeProto2_Go[i, : ] = get_poprate_aligned_onset(m_GPe_Proto2_new, Proto2_spikes, rate_data_GPeProto2_Go[i, : ], rate_data_GPeProto2_currtrial, dt())
                rate_data_GPeArky_Go[i, : ] = get_poprate_aligned_onset(m_GPe_Arky_new, Arky_spikes, rate_data_GPeArky_Go[i, : ], rate_data_GPeArky_currtrial, dt())            
                rate_data_SNr_Go[i, : ] = get_poprate_aligned_onset(m_SNr_new, SNr_spikes, rate_data_SNr_Go[i, : ], rate_data_SNr_currtrial, dt())                                                
                rate_data_SNrE_Go[i, : ] = get_poprate_aligned_onset(m_SNrE_new, SNrE_spikes, rate_data_SNrE_Go[i, : ], rate_data_SNrE_currtrial, dt())                                                            
                rate_data_Thal_Go[i, : ] = get_poprate_aligned_onset(m_Thal_new, Thal_spikes, rate_data_Thal_Go[i, : ], rate_data_Thal_currtrial, dt())                         
                rate_data_Cortex_G_Go[i, : ] = get_poprate_aligned_onset(m_Cortex_G_new, Cortex_G_spikes, rate_data_Cortex_G_Go[i, : ], rate_data_Cortex_G_currtrial, dt())
                rate_data_Cortex_S_Go[i, : ] = get_poprate_aligned_onset(m_Cortex_S_new, Cortex_S_spikes, rate_data_Cortex_S_Go[i, : ], rate_data_Cortex_S_currtrial, dt())
                rate_data_Stoppinput1_Go[i, : ] = get_poprate_aligned_onset(m_Stoppinput1_new, Stoppinput1_spikes, rate_data_Stoppinput1_Go[i, : ], rate_data_Stoppinput1_currtrial, dt())            

                t = get_time()

                if Integrator.decision == -1 :
                    print('Spike Time Relative to Start:')
                    t= get_current_step()
                    print(t)
                    print('Trialnumber:')
                    print(i)
                    print(Integrator.decision)
                    print(Integrator.g_ampa)
                    selection[i] = Integrator.decision
                    timeovertrial[i]=t
                    zaehler_go = zaehler_go + 1
                    print('resetting after fire')
                    Integrator.g_ampa = 0                             
                else:
                    selection[i] = Integrator.decision
                    timeovertrial[i]=t
                    print(Integrator.decision)
                    print('timeout')
                    print(Integrator.g_ampa)
                reset(populations=True, projections=True, synapses=False, net_id=0)
            
            print('zaehler_go:')
            print(zaehler_go)
            
            np.save('data/selection_Go'+str(i_netw_rep)+'.npy', selection)
            np.save('data/Timeovertrials_Go'+str(i_netw_rep)+'.npy', timeovertrial)
            np.save('data/Integrator_ampa_Go_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy', m_Int_ampa.get('g_ampa'))
            spike_times, ranks = m_Int_ampa.raster_plot(m_Int_ampa.get('spike'))
            np.save('data/Integrator_spike_Go'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'.npy', [spike_times, ranks])
            STR_D1_spikedata = m_STR_D1.get('spike')
            STR_D2_spikedata = m_STR_D2.get('spike')
            STR_FSI_spikedata = m_STR_FSI.get('spike')
            STN_spikedata = m_STN.get('spike')
            GPe_Proto_spikedata = m_GPe_Proto.get('spike')
            GPe_Arky_spikedata = m_GPe_Arky.get('spike')
            SNr_spikedata = m_SNr.get('spike')
            SNrE_spikedata = m_SNrE.get('spike')
            Thal_spikedata = m_Thal.get('spike')
            CortexG_spikedata = m_Cortex_G.get('spike')
            CortexS_spikedata = m_Cortex_S.get('spike')
            Stoppinput1_spikedata = m_Stoppinput1.get('spike')
           
            np.save('data/SD1_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_SD1_Go, 0), np.nanstd(rate_data_SD1_Go, 0)])
            np.save('data/SD2_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_SD2_Go, 0), np.nanstd(rate_data_SD2_Go, 0)])
            np.save('data/FSI_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_FSI_Go, 0), np.nanstd(rate_data_FSI_Go, 0)])
            np.save('data/STN_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_STN_Go, 0), np.nanstd(rate_data_STN_Go, 0)])
            np.save('data/GPeProto_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_GPeProto_Go, 0), np.nanstd(rate_data_GPeProto_Go, 0)])
            np.save('data/GPeProto2_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_GPeProto2_Go, 0), np.nanstd(rate_data_GPeProto2_Go, 0)])            
            np.save('data/GPeArky_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_GPeArky_Go, 0), np.nanstd(rate_data_GPeArky_Go, 0)])        
            np.save('data/SNr_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_SNr_Go, 0), np.nanstd(rate_data_SNr_Go, 0)])
            np.save('data/SNrE_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_SNrE_Go, 0), np.nanstd(rate_data_SNrE_Go, 0)])        
            np.save('data/Thal_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_Thal_Go, 0), np.nanstd(rate_data_Thal_Go, 0)])        
            np.save('data/Cortex_G_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_Cortex_G_Go, 0), np.nanstd(rate_data_Cortex_G_Go, 0)])        
            np.save('data/Cortex_S_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_Cortex_S_Go, 0), np.nanstd(rate_data_Cortex_S_Go, 0)])                
            np.save('data/Stoppinput1_rate_Go_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_Stoppinput1_Go, 0), np.nanstd(rate_data_Stoppinput1_Go, 0)])                        

                        

            print('fertig')





            t0 = 50 # 50ms simulation for initialization is not recorded, see above
            t_init = params['t_init'] # 100
            t_SSD = params['t_SSD'] # 170 # = SSD (stop signal delay), Defines the onset of the stop cue / pause signal , vorher 170
            t_PauseDuration = params['t_PauseDuration']
            t_delayGO = params['t_delayGO'] # 40 # 0 # Delay between Go/Stop cue presentation and onset of cortical Go/Stop activity
            t_delay = t_delayGO
            t_delaySTOP = params['t_delaySTOP']
            t_Pausedelay = 0 # 10 # 0 # Delay between Stop cue presentation and onset of Cortical "Pause" activity

            t_CorGODuration = params['t_CorGODuration']
            t_CorSTOPDuration = params['t_GoAndStop']
            t_motorSTOPDuration = params['t_motorStop'] 
            t_delaymotorSTOP = params['t_delaymotorSTOP']
            t_delayGOSD = params['t_delayGOSD']
            t_dauerinput = 100 # Dauer der Inputpraesentation für Gosignal & Stoppsignal
            t_dauerinput_go = 180

            print("params = ", params)
            mode='STOP'
            for i in range (0,trials):
                reset(populations=True, projections=True, synapses=False, net_id=0)
                if i_paramname == i_CortexGo_rates:             
                    params['CortexGo_rates'] = params_orig * param_factors[i_cycle] # Caution - must be reset after loop over this parameter !!!
                elif i_paramname == i_CortexStop_rates:
                    params['CortexStop_rates'] = params_orig * param_factors[i_cycle] 
                elif i_paramname == i_Stop1rates:
                    params['Stop1rates'] = params_orig * param_factors[i_cycle] 
                elif i_paramname == i_weights_cortexStop_GPeArky:            
                    Cortex_SGPe_Arky.mod_factor = param_factors[i_cycle] * params['weights_cortexStop_GPeArky']                
                elif i_paramname == i_weights_cortexStop_GPeCp:            
                    Cortex_SGPe_Proto2.mod_factor = param_factors[i_cycle] * params['weights_cortexStop_GPeCp']                
                elif i_paramname == i_weights_StrD2_GPeArky:  
                    STR_D2GPe_Arky.mod_factor = param_factors[i_cycle] * params['weights_StrD2_GPeArky']              
                elif i_paramname == i_weights_StrD2_GPeCp: 
                    STR_D2GPe_Proto2.mod_factor = param_factors[i_cycle] * params['weights_StrD2_GPeCp']   
                elif i_paramname == i_weights_GPeArky_StrD1:            
                    GPe_ArkySTR_D1.mod_factor = param_factors[i_cycle] * params['weights_GPeArky_StrD1']
                elif i_paramname == i_weights_GPeArky_StrD2:            
                    GPe_ArkySTR_D2.mod_factor = param_factors[i_cycle] * params['weights_GPeArky_StrD2'] 
                elif i_paramname == i_weights_GPeCp_IntStop:
                    GPe_Proto2IntegratorStop.mod_factor = param_factors[i_cycle] * params['weights_GPeCp_IntStop']   
                elif i_paramname == i_GPe_ProtoSTR_FSI:
                    GPe_ProtoSTR_FSI.mod_factor = param_factors[i_cycle] * params['weights_GPeProto_StrFSI']                 
                elif i_paramname == i_STN_SNr:                    
                    STNSNr.mod_factor = param_factors[i_cycle] * params['weights_STN_SNr']               
                elif i_paramname == i_ArkyD1Stop:                    
                    changeArkyStopOutput(param_factors[i_cycle],1,1)               
                elif i_paramname == i_ArkyD2Stop:                    
                    changeArkyStopOutput(1,param_factors[i_cycle],1)              
                elif i_paramname == i_ArkyFSIStop:                    
                    changeArkyStopOutput(1,1,param_factors[i_cycle])


                

                ### TRIAL START
                ### trial INITIALIZATION simulation to get stable state
                Cortex_G.rates = 0
                Cortex_S.rates = 0
                Stoppinput1.rates = 0

                simulate(t_init)# t_init rest

                ### Integrator Reset
                Integrator.decision = 0
                Integrator.g_ampa = 0  
                IntegratorStop.decision = 0 

                ### calculate all eventTIMES
                GOCUE     = np.round(get_time(),1)
                STOPCUE   = GOCUE + t_SSD
                STN1ON    = GOCUE
                STN1OFF   = STN1ON + t_PauseDuration
                GOON      = GOCUE + t_delayGO + int(np.clip(t_delayGOSD * np.random.randn(),-t_delayGO,None))
                STN2ON    = STOPCUE
                STN2OFF   = STN2ON + t_PauseDuration
                STOPON    = STOPCUE + t_delaySTOP
                STOPOFF   = STOPON + t_CorSTOPDuration
                ENDTIME   = STOPOFF + t_decay
                GOOFF     = ENDTIME
                motorSTOP = ENDTIME
                motorSTOPOFF = ENDTIME
                eventTimes = np.array([GOCUE,STOPCUE,STN1ON,STN1OFF,GOON,GOOFF,STN2ON,STN2OFF,STOPON,STOPOFF,ENDTIME,motorSTOP,motorSTOPOFF])
                randn_trial_Pause = np.random.randn()
                randn_trial = np.random.randn()
                randn_trial_S = np.random.randn()

                ### simulate all EVENTS
                motorResponse=0
                GOOFFResponse=0
                responseThal=0
                responseProto2=0
                t=np.round(get_time(),1)
                end=False
                tempMode=mode
                while not(end):
                    if t == STN1ON:
                        Stoppinput1.rates = params['Stop1rates'] + sigma_Pause * randn_trial_Pause
                        #print('STN1ON',STN1ON)
                    if t == STN1OFF:
                        Stoppinput1.rates = 0
                        #print('STN1OFF',STN1OFF)
                    if t == GOON:
                        Cortex_G.rates = params['CortexGo_rates'] + sigma_Go * randn_trial
                        #print('GOON',GOON)
                    if t == GOOFF:
                        Cortex_G.rates = 0
                        #print('GOOFF',GOOFF)                    
                    if t == STN2ON and mode=='STOP':
                        Stoppinput1.rates = params['cortexPause_ratesSecondRespMod']*params['Stop1rates'] + sigma_Pause * randn_trial_Pause
                        #print('STN2ON',STN2ON)
                    if t == STN2OFF and mode=='STOP':
                        Stoppinput1.rates = 0
                        #print('STN2OFF',STN2OFF)
                    if t == STOPON and mode=='STOP':
                        Cortex_S.rates = params['CortexStop_rates'] + sigma_Stop * randn_trial_S * (params['CortexStop_rates'] > 0)
                        #print('STOPON',STOPON)
                    if t == motorSTOP:
                        Cortex_S.rates = params['CortexmotorStop_rates'] + sigma_Stop * randn_trial_S * (params['CortexmotorStop_rates'] > 0)
                        motorSTOPOFF = motorSTOP + t_motorSTOPDuration
                        #print('motorSTOP',motorSTOP)
                    if t == STOPOFF and mode=='STOP' and ((motorSTOPOFF==ENDTIME) or (motorSTOPOFF!=ENDTIME and t>motorSTOPOFF)):
                        Cortex_S.rates = 0
                        #print('STOPOFF',STOPOFF)
                    if t == motorSTOPOFF:
                        Cortex_S.rates = 0
                        #print('motorSTOPOFF',motorSTOPOFF)
                    if t == ENDTIME:
                        end=True
                        #print('ENDTIME',ENDTIME)
                    else:
                        nowTIME = np.round(get_time(),1)
                        #print('nowTIME',nowTIME)
                        eventTimes = np.array([GOCUE,STOPCUE,STN1ON,STN1OFF,GOON,GOOFF,STN2ON,STN2OFF,STOPON,STOPOFF,ENDTIME,motorSTOP,motorSTOPOFF])
                        nextEvent = np.max([np.min(eventTimes[eventTimes>=nowTIME]-nowTIME),1])
                        #print('nextEvent',nextEvent)
                        if responseThal==0 and responseProto2==0:
                            simulate_until(max_duration=nextEvent, population=[Integrator,IntegratorStop], operator='or')
                        elif responseProto2==0:
                            simulate_until(max_duration=nextEvent, population=IntegratorStop)
                        elif responseThal==0:
                            simulate_until(max_duration=nextEvent, population=Integrator)
                        else:
                            simulate(nextEvent)
                        responseThal = int(Integrator.decision)
                        responseProto2 = int(IntegratorStop.decision)
                        t = np.round(get_time(),1)
                        #print('time:',t,'restsimulation:',np.ceil(t)-t)
                        simulate(np.round(np.ceil(t)-t,1))
                        t = np.round(get_time(),1)
                        if responseThal == -1 and motorResponse == 0:
                            motorResponse=1
                            motorSTOP = t + t_delaymotorSTOP
                            if t<STOPCUE and tempMode=='STOP':
                                tempMode='GO'
                        if responseProto2 == -1 and GOOFFResponse == 0 and ((t>STOPON and tempMode=='STOP') or motorResponse==1):
                            GOOFFResponse=1
                            GOOFF = t
                ### TRIAL END

                


                SD1_spikes = m_STR_D1_new.get('spike')
                rate_data_SD1_currtrial = custom_poprate(m_STR_D1_new, SD1_spikes, t_smooth_ms)
                SD2_spikes = m_STR_D2_new.get('spike')
                rate_data_SD2_currtrial = custom_poprate(m_STR_D2_new, SD2_spikes, t_smooth_ms)
                FSI_spikes = m_STR_FSI_new.get('spike')
                rate_data_FSI_currtrial = custom_poprate(m_STR_FSI_new, FSI_spikes, t_smooth_ms)                        
                STN_spikes = m_STN_new.get('spike')
                rate_data_STN_currtrial = custom_poprate(m_STN_new, STN_spikes, t_smooth_ms)                        
                Proto_spikes = m_GPe_Proto_new.get('spike')                  
                Proto2_spikes = m_GPe_Proto2_new.get('spike')
                rate_data_GPeProto_currtrial = custom_poprate(m_GPe_Proto_new, Proto_spikes, t_smooth_ms)
                rate_data_GPeProto2_currtrial = custom_poprate(m_GPe_Proto2_new, Proto2_spikes, t_smooth_ms)
                Arky_spikes = m_GPe_Arky_new.get('spike')
                rate_data_GPeArky_currtrial = custom_poprate(m_GPe_Arky_new, Arky_spikes, t_smooth_ms)
                
                rateperneuron_GPeArky_currtrial = m_GPe_Arky_new.smoothed_rate( Arky_spikes )
                rateperneuron_GPeArky_allStoptrials[i, : , 0:len(rateperneuron_GPeArky_currtrial[0,:])] = rateperneuron_GPeArky_currtrial
                
                SNr_spikes = m_SNr_new.get('spike')
                rate_data_SNr_currtrial = custom_poprate(m_SNr_new, SNr_spikes, t_smooth_ms)                                                
                SNrE_spikes = m_SNrE_new.get('spike')
                rate_data_SNrE_currtrial = custom_poprate(m_SNrE_new, SNrE_spikes, t_smooth_ms)                                                            
                Thal_spikes = m_Thal_new.get('spike')
                rate_data_Thal_currtrial = custom_poprate(m_Thal_new, Thal_spikes, t_smooth_ms)                        
                te, ne = m_Thal_new.raster_plot(Thal_spikes)
                Cortex_G_spikes = m_Cortex_G_new.get('spike')
                rate_data_Cortex_G_currtrial = custom_poprate(m_Cortex_G_new, Cortex_G_spikes, t_smooth_ms)                        
                Cortex_S_spikes = m_Cortex_S_new.get('spike')
                te, ne = m_Cortex_S_new.raster_plot(Cortex_S_spikes) 
                rate_data_Cortex_S_currtrial = custom_poprate(m_Cortex_S_new, Cortex_S_spikes, t_smooth_ms)
                Stoppinput1_spikes = m_Stoppinput1_new.get('spike')
                rate_data_Stoppinput1_currtrial = custom_poprate(m_Stoppinput1_new, Stoppinput1_spikes, t_smooth_ms)            

                rate_data_SD1_Stop[i, : ] = get_poprate_aligned_onset(m_STR_D1_new, SD1_spikes, rate_data_SD1_Stop[i, : ], rate_data_SD1_currtrial, dt())            
                rate_data_SD2_Stop[i, : ] = get_poprate_aligned_onset(m_STR_D2_new, SD2_spikes, rate_data_SD2_Stop[i, : ], rate_data_SD2_currtrial, dt())                        
                rate_data_FSI_Stop[i, : ] = get_poprate_aligned_onset(m_STR_FSI_new, FSI_spikes, rate_data_FSI_Stop[i, : ], rate_data_FSI_currtrial, dt())                        
                rate_data_STN_Stop[i, : ] = get_poprate_aligned_onset(m_STN_new, STN_spikes, rate_data_STN_Stop[i, : ], rate_data_STN_currtrial, dt())                                    
                rate_data_GPeProto_Stop[i, : ] = get_poprate_aligned_onset(m_GPe_Proto_new, Proto_spikes, rate_data_GPeProto_Stop[i, : ], rate_data_GPeProto_currtrial, dt())             
                rate_data_GPeProto2_Stop[i, : ] = get_poprate_aligned_onset(m_GPe_Proto2_new, Proto2_spikes, rate_data_GPeProto2_Stop[i, : ], rate_data_GPeProto2_currtrial, dt()) 
                rate_data_GPeArky_Stop[i, : ] = get_poprate_aligned_onset(m_GPe_Arky_new, Arky_spikes, rate_data_GPeArky_Stop[i, : ], rate_data_GPeArky_currtrial, dt())            
                rate_data_SNr_Stop[i, : ] = get_poprate_aligned_onset(m_SNr_new, SNr_spikes, rate_data_SNr_Stop[i, : ], rate_data_SNr_currtrial, dt())                                                
                rate_data_SNrE_Stop[i, : ] = get_poprate_aligned_onset(m_SNrE_new, SNrE_spikes, rate_data_SNrE_Stop[i, : ], rate_data_SNrE_currtrial, dt())                                                            
                rate_data_Thal_Stop[i, : ] = get_poprate_aligned_onset(m_Thal_new, Thal_spikes, rate_data_Thal_Stop[i, : ], rate_data_Thal_currtrial, dt())                         
                rate_data_Cortex_G_Stop[i, : ] = get_poprate_aligned_onset(m_Cortex_G_new, Cortex_G_spikes, rate_data_Cortex_G_Stop[i, : ], rate_data_Cortex_G_currtrial, dt())
                rate_data_Cortex_S_Stop[i, : ] = get_poprate_aligned_onset(m_Cortex_S_new, Cortex_S_spikes, rate_data_Cortex_S_Stop[i, : ], rate_data_Cortex_S_currtrial, dt())
                rate_data_Stoppinput1_Stop[i, : ] = get_poprate_aligned_onset(m_Stoppinput1_new, Stoppinput1_spikes, rate_data_Stoppinput1_Stop[i, : ], rate_data_Stoppinput1_currtrial, dt())



                if i==trials-1: 
                    get_and_plot_syn_mon(m_syn_D1, m_syn_D2, m_syn_FSI, m_syn_STN, m_syn_Proto, m_syn_Arky, m_syn_SNr, m_syn_Thal, params['id'])
                t = get_time()

                if Integrator.decision == -1 :
                    print('Spike Time Relative to Start:')
                    t= get_current_step()
                    print(t)
                    print('Trialnumber:')
                    print(i)
                    print(Integrator.decision)
                    print(Integrator.g_ampa)
                    selection[i] = Integrator.decision
                    timeovertrial[i]=t
                    zaehler_go = zaehler_go + 1
                    print('resetting after fire')
                    Integrator.g_ampa = 0                                          
                else:
                    selection[i] = Integrator.decision
                    timeovertrial[i]=t
                    print(Integrator.decision)
                    print('timeout')
                    print(Integrator.g_ampa)
                reset(populations=True, projections=True, synapses=False, net_id=0)
            
            print('zaehler_go:')
            print(zaehler_go)
            
            np.save('data/selection_Stop'+str(i_netw_rep)+'.npy', selection)
            np.save('data/Timeovertrials_Stop'+str(i_netw_rep)+'.npy', timeovertrial)
            np.save('data/Integrator_ampa_Stop_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy', m_Int_ampa.get('g_ampa'))            
            spike_times, ranks = m_Int_ampa.raster_plot(m_Int_ampa.get('spike'))
            np.save('data/Integrator_spike_Stop'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'.npy', [spike_times, ranks]) 
            STR_D1_spikedata = m_STR_D1.get('spike')
            STR_D2_spikedata = m_STR_D2.get('spike')
            STN_spikedata = m_STN.get('spike')
            GPe_Proto_spikedata = m_GPe_Proto.get('spike')
            GPe_Arky_spikedata = m_GPe_Arky.get('spike')
            SNr_spikedata = m_SNr.get('spike')
            SNrE_spikedata = m_SNrE.get('spike')
            STR_FSI_spikedata = m_STR_FSI.get('spike')
            Thal_spikedata = m_Thal.get('spike')
            CortexG_spikedata = m_Cortex_G.get('spike')
            CortexS_spikedata = m_Cortex_S.get('spike')
            Stoppinput1_spikedata = m_Stoppinput1.get('spike')


            mInt_stop = np.load('data/Integrator_ampa_Stop_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')
            np.save('data/SD1_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_SD1_Stop, 0), np.nanstd(rate_data_SD1_Stop, 0)])
            np.save('data/SD2_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_SD2_Stop, 0), np.nanstd(rate_data_SD2_Stop, 0)])
            np.save('data/FSI_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_FSI_Stop, 0), np.nanstd(rate_data_FSI_Stop, 0)])             
            np.save('data/STN_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_STN_Stop, 0), np.nanstd(rate_data_STN_Stop, 0)])
            np.save('data/STN_rate_Stop_mean_std_'+ str(int(params['id']))+'_'+str(i_netw_rep)+'_'+str(i_cycle)+'.npy', [np.nanmean(rate_data_STN_Stop, 0), np.nanstd(rate_data_STN_Stop, 0)])
            np.save('data/GPeProto_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_GPeProto_Stop, 0), np.nanstd(rate_data_GPeProto_Stop, 0)])
            np.save('data/GPeProto2_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_GPeProto2_Stop, 0), np.nanstd(rate_data_GPeProto2_Stop, 0)])            
            np.save('data/GPeProto_rate_Stop_mean_std_'+ str(int(params['id']))+'_'+str(i_netw_rep)+'_'+str(i_cycle)+'.npy', [np.nanmean(rate_data_GPeProto_Stop, 0), np.nanstd(rate_data_GPeProto_Stop, 0)])            
            np.save('data/GPeProto2_rate_Stop_mean_std_'+ str(int(params['id']))+'_'+str(i_netw_rep)+'_'+str(i_cycle)+'.npy', [np.nanmean(rate_data_GPeProto2_Stop, 0), np.nanstd(rate_data_GPeProto2_Stop, 0)])                        
            np.save('data/GPeArky_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_GPeArky_Stop, 0), np.nanstd(rate_data_GPeArky_Stop, 0)])   
            np.save('data/GPeArky_rate_Stop_mean_std_'+ str(int(params['id']))+'_'+str(i_netw_rep)+'_'+str(i_cycle)+'.npy', [np.nanmean(rate_data_GPeArky_Stop, 0), np.nanstd(rate_data_GPeArky_Stop, 0)])               
            np.save('data/SNr_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_SNr_Stop, 0), np.nanstd(rate_data_SNr_Stop, 0)])
            np.save('data/SNr_rate_Stop_mean_std_'+ str(int(params['id']))+'_'+str(i_netw_rep)+'_'+str(i_cycle)+'.npy', [np.nanmean(rate_data_SNr_Stop, 0), np.nanstd(rate_data_SNr_Stop, 0)])            
            np.save('data/SNrE_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_SNrE_Stop, 0), np.nanstd(rate_data_SNrE_Stop, 0)])        
            np.save('data/Thal_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_Thal_Stop, 0), np.nanstd(rate_data_Thal_Stop, 0)])        
            np.save('data/Cortex_G_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_Cortex_G_Stop, 0), np.nanstd(rate_data_Cortex_G_Stop, 0)])
            np.save('data/Cortex_S_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_Cortex_S_Stop, 0), np.nanstd(rate_data_Cortex_S_Stop, 0)])
            np.save('data/Stoppinput1_rate_Stop_mean_std'+str(i_netw_rep)+'.npy', [np.nanmean(rate_data_Stoppinput1_Stop, 0), np.nanstd(rate_data_Stoppinput1_Stop, 0)])        

            
            np.save('data/STN_rate_Stop_alltrials'+str(i_netw_rep)+'.npy', rate_data_STN_Stop)         
            np.save('data/SNr_rate_Stop_alltrials'+str(i_netw_rep)+'.npy', rate_data_SNr_Stop) 
            np.save('data/GPeProto_rate_Stop_alltrials'+str(i_netw_rep)+'.npy', rate_data_GPeProto_Stop) 
            np.save('data/GPeArky_rate_Stop_alltrials'+str(i_netw_rep)+'.npy', rate_data_GPeArky_Stop)         


            StrD1_meanrate_FailedStop, StrD1_std_FailedStop, StrD1_meanrate_CorrectStop, StrD1_std_CorrectStop              = calc_meanrate_std_failed_correct(rate_data_SD1_Stop, mInt_stop, Integrator.threshold, trials) 
            StrD2_meanrate_FailedStop, StrD2_std_FailedStop, StrD2_meanrate_CorrectStop, StrD2_std_CorrectStop              = calc_meanrate_std_failed_correct(rate_data_SD2_Stop, mInt_stop, Integrator.threshold, trials)        
            StrFSI_meanrate_FailedStop, StrFSI_std_FailedStop, StrFSI_meanrate_CorrectStop, StrFSI_std_CorrectStop          = calc_meanrate_std_failed_correct(rate_data_FSI_Stop, mInt_stop, Integrator.threshold, trials)        
            STN_meanrate_FailedStop, STN_std_FailedStop, STN_meanrate_CorrectStop, STN_std_CorrectStop                      = calc_meanrate_std_failed_correct(rate_data_STN_Stop, mInt_stop, Integrator.threshold, trials)         
            Proto_meanrate_FailedStop, Proto_std_FailedStop, Proto_meanrate_CorrectStop, Proto_std_CorrectStop              = calc_meanrate_std_failed_correct(rate_data_GPeProto_Stop, mInt_stop, Integrator.threshold, trials)                 
            Proto2_meanrate_FailedStop, Proto2_std_FailedStop, Proto2_meanrate_CorrectStop, Proto2_std_CorrectStop          = calc_meanrate_std_failed_correct(rate_data_GPeProto2_Stop, mInt_stop, Integrator.threshold, trials)            
            Arky_meanrate_FailedStop, Arky_std_FailedStop, Arky_meanrate_CorrectStop, Arky_std_CorrectStop                  = calc_meanrate_std_failed_correct(rate_data_GPeArky_Stop, mInt_stop, Integrator.threshold, trials)
            SNr_meanrate_FailedStop, SNr_std_FailedStop, SNr_meanrate_CorrectStop, SNr_std_CorrectStop                      = calc_meanrate_std_failed_correct(rate_data_SNr_Stop, mInt_stop, Integrator.threshold, trials)
            SNrE_meanrate_FailedStop, SNrE_std_FailedStop, SNrE_meanrate_CorrectStop, SNrE_std_CorrectStop                  = calc_meanrate_std_failed_correct(rate_data_SNrE_Stop, mInt_stop, Integrator.threshold, trials)
            Thal_meanrate_FailedStop, Thal_std_FailedStop, Thal_meanrate_CorrectStop, Thal_std_CorrectStop                  = calc_meanrate_std_failed_correct(rate_data_Thal_Stop, mInt_stop, Integrator.threshold, trials)
            Cortex_G_meanrate_FailedStop, Cortex_G_std_FailedStop, Cortex_G_meanrate_CorrectStop, Cortex_G_std_CorrectStop  = calc_meanrate_std_failed_correct(rate_data_Cortex_G_Stop, mInt_stop, Integrator.threshold, trials)
            Cortex_S_meanrate_FailedStop, Cortex_S_std_FailedStop, Cortex_S_meanrate_CorrectStop, Cortex_S_std_CorrectStop  = calc_meanrate_std_failed_correct(rate_data_Cortex_S_Stop, mInt_stop, Integrator.threshold, trials)
            Stoppinput1_meanrate_FailedStop, Stoppinput1_std_FailedStop, Stoppinput1_meanrate_CorrectStop, Stoppinput1_std_CorrectStop = calc_meanrate_std_failed_correct(rate_data_Stoppinput1_Stop, mInt_stop, Integrator.threshold, trials)        

            np.save('data/SD1_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [StrD1_meanrate_FailedStop, StrD1_std_FailedStop, StrD1_meanrate_CorrectStop, StrD1_std_CorrectStop])
            np.save('data/SD2_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [StrD2_meanrate_FailedStop, StrD2_std_FailedStop, StrD2_meanrate_CorrectStop, StrD2_std_CorrectStop])
            np.save('data/FSI_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [StrFSI_meanrate_FailedStop, StrFSI_std_FailedStop, StrFSI_meanrate_CorrectStop, StrFSI_std_CorrectStop])
            np.save('data/STN_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [STN_meanrate_FailedStop, STN_std_FailedStop, STN_meanrate_CorrectStop, STN_std_CorrectStop])
            np.save('data/Proto_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [Proto_meanrate_FailedStop, Proto_std_FailedStop, Proto_meanrate_CorrectStop, Proto_std_CorrectStop])
            np.save('data/Proto2_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [Proto2_meanrate_FailedStop, Proto2_std_FailedStop, Proto2_meanrate_CorrectStop, Proto2_std_CorrectStop])            
            np.save('data/Arky_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [Arky_meanrate_FailedStop, Arky_std_FailedStop, Arky_meanrate_CorrectStop, Arky_std_CorrectStop])
            np.save('data/SNr_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [SNr_meanrate_FailedStop, SNr_std_FailedStop, SNr_meanrate_CorrectStop, SNr_std_CorrectStop])
            np.save('data/SNrE_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [SNrE_meanrate_FailedStop, SNrE_std_FailedStop, SNrE_meanrate_CorrectStop, SNrE_std_CorrectStop])        
            np.save('data/Thal_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [Thal_meanrate_FailedStop, Thal_std_FailedStop, Thal_meanrate_CorrectStop, Thal_std_CorrectStop])        
            np.save('data/Cortex_G_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [Cortex_G_meanrate_FailedStop, Cortex_G_std_FailedStop, Cortex_G_meanrate_CorrectStop, Cortex_G_std_CorrectStop])
            np.save('data/Cortex_S_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [Cortex_S_meanrate_FailedStop, Cortex_S_std_FailedStop, Cortex_S_meanrate_CorrectStop, Cortex_S_std_CorrectStop])        
            np.save('data/Stoppinput1_meanrate_std_Failed_Correct_Stop'+str(i_netw_rep)+'.npy', [Stoppinput1_meanrate_FailedStop, Stoppinput1_std_FailedStop, Stoppinput1_meanrate_CorrectStop, Stoppinput1_std_CorrectStop])                
            

            mInt_Go = np.load('data/Integrator_ampa_Go_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')
            GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo     = calc_meanrate_std_Fast_Slow(rate_data_GPeArky_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo    = calc_meanrate_std_Fast_Slow(rate_data_GPeProto_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            StrD1_meanrate_FastGo,      StrD1_std_FastGo,       StrD1_meanrate_SlowGo,      StrD1_std_SlowGo        = calc_meanrate_std_Fast_Slow(rate_data_SD1_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)                 
            StrD2_meanrate_FastGo,      StrD2_std_FastGo,       StrD2_meanrate_SlowGo,      StrD2_std_SlowGo        = calc_meanrate_std_Fast_Slow(rate_data_SD2_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            STN_meanrate_FastGo,        STN_std_FastGo,         STN_meanrate_SlowGo,        STN_std_SlowGo          = calc_meanrate_std_Fast_Slow(rate_data_STN_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            SNr_meanrate_FastGo,        SNr_std_FastGo,         SNr_meanrate_SlowGo,        SNr_std_SlowGo          = calc_meanrate_std_Fast_Slow(rate_data_SNr_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            Thal_meanrate_FastGo,       Thal_std_FastGo,        Thal_meanrate_SlowGo,       Thal_std_SlowGo         = calc_meanrate_std_Fast_Slow(rate_data_Thal_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            Cortex_G_meanrate_FastGo,   Cortex_G_std_FastGo,    Cortex_G_meanrate_SlowGo,   Cortex_G_std_SlowGo     = calc_meanrate_std_Fast_Slow(rate_data_Cortex_G_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            GPe_Proto2_meanrate_FastGo, GPe_Proto2_std_FastGo,  GPe_Proto2_meanrate_SlowGo, GPe_Proto2_std_SlowGo   = calc_meanrate_std_Fast_Slow(rate_data_GPeProto2_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            PauseInput_meanrate_FastGo, PauseInput_std_FastGo,  PauseInput_meanrate_SlowGo, PauseInput_std_SlowGo   = calc_meanrate_std_Fast_Slow(rate_data_Stoppinput1_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            Cortex_S_meanrate_FastGo,   Cortex_S_std_FastGo,    Cortex_S_meanrate_SlowGo,   Cortex_S_std_SlowGo     = calc_meanrate_std_Fast_Slow(rate_data_Cortex_S_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)                
            StrFSI_meanrate_FastGo,     StrFSI_std_FastGo,      StrFSI_meanrate_SlowGo,     StrFSI_std_SlowGo       = calc_meanrate_std_Fast_Slow(rate_data_FSI_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)                 

            np.save('data/GPe_Arky_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',   [GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo])
            np.save('data/GPe_Proto_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',  [GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo])
            np.save('data/SD1_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',        [StrD1_meanrate_FastGo,      StrD1_std_FastGo,       StrD1_meanrate_SlowGo,      StrD1_std_SlowGo])
            np.save('data/SD2_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',        [StrD2_meanrate_FastGo,      StrD2_std_FastGo,       StrD2_meanrate_SlowGo,      StrD2_std_SlowGo])            
            np.save('data/STN_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',        [STN_meanrate_FastGo,        STN_std_FastGo,         STN_meanrate_SlowGo,        STN_std_SlowGo])
            np.save('data/SNr_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',        [SNr_meanrate_FastGo,        SNr_std_FastGo,         SNr_meanrate_SlowGo,        SNr_std_SlowGo])
            np.save('data/Thal_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',       [Thal_meanrate_FastGo,       Thal_std_FastGo,        Thal_meanrate_SlowGo,       Thal_std_SlowGo])
            np.save('data/Cortex_G_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',   [Cortex_G_meanrate_FastGo,   Cortex_G_std_FastGo,    Cortex_G_meanrate_SlowGo,   Cortex_G_std_SlowGo])
            np.save('data/GPe_Proto2_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy', [GPe_Proto2_meanrate_FastGo, GPe_Proto2_std_FastGo,  GPe_Proto2_meanrate_SlowGo, GPe_Proto2_std_SlowGo])
            np.save('data/PauseInput_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy', [PauseInput_meanrate_FastGo, PauseInput_std_FastGo,  PauseInput_meanrate_SlowGo, PauseInput_std_SlowGo])
            np.save('data/Cortex_S_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',   [Cortex_S_meanrate_FastGo,   Cortex_S_std_FastGo,    Cortex_S_meanrate_SlowGo,   Cortex_S_std_SlowGo])                        
            np.save('data/FSI_meanrate_std_Fast-Slow_Go'+str(i_netw_rep)+'.npy',        [StrFSI_meanrate_FastGo,     StrFSI_std_FastGo,      StrFSI_meanrate_SlowGo,     StrFSI_std_SlowGo])



            t_stopCue = int(t_init + t_SSD)
            t_min = int((t_stopCue - 300)/dt())   
            t_max = int((t_stopCue + 300)/dt()) # 100        
            binsize_ms = int(20 / dt()) # 25 
            pvalue_times = range(t_min, t_max + 1, binsize_ms)
            SD1_rates_FailedStop, SD1_rates_CorrectStop = get_rates_failed_correct(rate_data_SD1_Stop, mInt_stop, Integrator.threshold, trials)
            SD2_rates_FailedStop, SD2_rates_CorrectStop = get_rates_failed_correct(rate_data_SD2_Stop, mInt_stop, Integrator.threshold, trials)        
            STN_rates_FailedStop, STN_rates_CorrectStop = get_rates_failed_correct(rate_data_STN_Stop, mInt_stop, Integrator.threshold, trials)
            GPe_Proto_rates_FailedStop, GPe_Proto_rates_CorrectStop = get_rates_failed_correct(rate_data_GPeProto_Stop, mInt_stop, Integrator.threshold, trials)
            GPe_Proto2_rates_FailedStop, GPe_Proto2_rates_CorrectStop = get_rates_failed_correct(rate_data_GPeProto2_Stop, mInt_stop, Integrator.threshold, trials)        
            GPe_Arky_rates_FailedStop, GPe_Arky_rates_CorrectStop = get_rates_failed_correct(rate_data_GPeArky_Stop, mInt_stop, Integrator.threshold, trials)
            SNr_rates_FailedStop, SNr_rates_CorrectStop = get_rates_failed_correct(rate_data_SNr_Stop, mInt_stop, Integrator.threshold, trials)        
            Thal_rates_FailedStop, Thal_rates_CorrectStop = get_rates_failed_correct(rate_data_Thal_Stop, mInt_stop, Integrator.threshold, trials)        
            CortexG_rates_FailedStop, CortexG_rates_CorrectStop = get_rates_failed_correct(rate_data_Cortex_G_Stop, mInt_stop, Integrator.threshold, trials)                
            CortexS_rates_FailedStop, CortexS_rates_CorrectStop = get_rates_failed_correct(rate_data_Cortex_S_Stop, mInt_stop, Integrator.threshold, trials)                        
            FSI_rates_FailedStop, FSI_rates_CorrectStop = get_rates_failed_correct(rate_data_FSI_Stop, mInt_stop, Integrator.threshold, trials)        

            print('t_stopCue =', t_stopCue)
            Hstat_all, pval_all = calc_KW_stats_all(GPe_Arky_rates_FailedStop, GPe_Arky_rates_CorrectStop, \
                                                SD1_rates_FailedStop, SD1_rates_CorrectStop, \
                                                SD2_rates_FailedStop, SD2_rates_CorrectStop, \
                                                STN_rates_FailedStop, STN_rates_CorrectStop, \
                                                GPe_Proto_rates_FailedStop, GPe_Proto_rates_CorrectStop, \
                                                GPe_Proto2_rates_FailedStop, GPe_Proto2_rates_CorrectStop, \
                                                SNr_rates_FailedStop, SNr_rates_CorrectStop, \
                                                Thal_rates_FailedStop, Thal_rates_CorrectStop, \
                                                CortexG_rates_FailedStop, CortexG_rates_CorrectStop, \
                                                CortexS_rates_FailedStop, CortexS_rates_CorrectStop, \
                                                FSI_rates_FailedStop, FSI_rates_CorrectStop, \
                                                pvalue_times, dt(), 'test')
      
            p_ind_arky, p_ind_SD1, p_ind_STN, p_ind_Proto, p_ind_Proto2, p_ind_SNr, p_ind_thal, p_ind_SD2, p_ind_CortexG, p_ind_CortexS, p_ind_FSI = range(11)
            print('pval_all = ', pval_all)
      
            pvalue_list = pval_all    
            np.save('data/p_value_list_times_'+str(int(params['id']))+str(i_netw_rep)+'.npy', [pvalue_list, pvalue_times], allow_pickle=True)




            ### same pvalue calculation for new plot (similar to failed stops vs correct stops)
            t_stopCue = int(t_init + t_SSD)
            t_min = int((t_init - 50)/dt())   
            t_max = int((t_init + 550)/dt()) # 100        
            binsize_ms = int(20 / dt()) # 25 
            pvalue_times = range(t_min, t_max + 1, binsize_ms) 
            SD1_rates_allGo,        SD1_rates_fastGo,           SD1_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_SD1_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            SD2_rates_allGo,        SD2_rates_fastGo,           SD2_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_SD2_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            STN_rates_allGo,        STN_rates_fastGo,           STN_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_STN_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            GPe_Proto_rates_allGo,  GPe_Proto_rates_fastGo,     GPe_Proto_rates_slowGo  = get_rates_allGo_fastGo_slowGo(rate_data_GPeProto_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            GPe_Proto2_rates_allGo, GPe_Proto2_rates_fastGo,    GPe_Proto2_rates_slowGo = get_rates_allGo_fastGo_slowGo(rate_data_GPeProto2_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            GPe_Arky_rates_allGo,   GPe_Arky_rates_fastGo,      GPe_Arky_rates_slowGo   = get_rates_allGo_fastGo_slowGo(rate_data_GPeArky_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            SNr_rates_allGo,        SNr_rates_fastGo,           SNr_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_SNr_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            Thal_rates_allGo,       Thal_rates_fastGo,          Thal_rates_slowGo       = get_rates_allGo_fastGo_slowGo(rate_data_Thal_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            CortexG_rates_allGo,    CortexG_rates_fastGo,       CortexG_rates_slowGo    = get_rates_allGo_fastGo_slowGo(rate_data_Cortex_G_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            CortexS_rates_allGo,    CortexS_rates_fastGo,       CortexS_rates_slowGo    = get_rates_allGo_fastGo_slowGo(rate_data_Cortex_S_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)
            FSI_rates_allGo,        FSI_rates_fastGo,           FSI_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_FSI_Go, mInt_Go, mInt_stop, Integrator.threshold, trials)        

            failedStopList  = [GPe_Arky_rates_FailedStop, SD1_rates_FailedStop, SD2_rates_FailedStop, STN_rates_FailedStop, GPe_Proto_rates_FailedStop, GPe_Proto2_rates_FailedStop, SNr_rates_FailedStop, Thal_rates_FailedStop, CortexG_rates_FailedStop, CortexS_rates_FailedStop, FSI_rates_FailedStop]
            allGoList       = [GPe_Arky_rates_allGo, SD1_rates_allGo, SD2_rates_allGo, STN_rates_allGo, GPe_Proto_rates_allGo, GPe_Proto2_rates_allGo, SNr_rates_allGo, Thal_rates_allGo, CortexG_rates_allGo, CortexS_rates_allGo, FSI_rates_allGo]
            fastGoList      = [GPe_Arky_rates_fastGo, SD1_rates_fastGo, SD2_rates_fastGo, STN_rates_fastGo, GPe_Proto_rates_fastGo, GPe_Proto2_rates_fastGo, SNr_rates_fastGo, Thal_rates_fastGo, CortexG_rates_fastGo, CortexS_rates_fastGo, FSI_rates_fastGo]
            slowGoList      = [GPe_Arky_rates_slowGo, SD1_rates_slowGo, SD2_rates_slowGo, STN_rates_slowGo, GPe_Proto_rates_slowGo, GPe_Proto2_rates_slowGo, SNr_rates_slowGo, Thal_rates_slowGo, CortexG_rates_slowGo, CortexS_rates_slowGo, FSI_rates_slowGo]
            nameList = ['failedStop_vs_allGo', 'failedStop_vs_fastGo', 'failedStop_vs_slowGo']
            for Groups_Idx, Groups_Val in enumerate([[failedStopList, allGoList], [failedStopList, fastGoList], [failedStopList, slowGoList]]):
                GroupA = Groups_Val[0]
                GroupB = Groups_Val[1]
                Hstat_all, pval_all = calc_KW_stats_all(GroupA[0], GroupB[0], \
                                                        GroupA[1], GroupB[1], \
                                                        GroupA[2], GroupB[2], \
                                                        GroupA[3], GroupB[3], \
                                                        GroupA[4], GroupB[4], \
                                                        GroupA[5], GroupB[5], \
                                                        GroupA[6], GroupB[6], \
                                                        GroupA[7], GroupB[7], \
                                                        GroupA[8], GroupB[8], \
                                                        GroupA[9], GroupB[9], \
                                                        GroupA[10], GroupB[10], \
                                                        pvalue_times, dt(), nameList[Groups_Idx])

                pvalue_list = pval_all    
                np.save('data/p_value_list_'+nameList[Groups_Idx]+'_times_'+str(int(params['id']))+str(i_netw_rep)+'.npy', [pvalue_list, pvalue_times], allow_pickle=True)



            # TODO: Save temporal average of mean firing rate per trial here
            np.save('data/SD1_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy', np.nanmean(rate_data_SD1_Stop[:, t_stopCue : t_stopCue + 200], 1)) # temporal mean across trials
            np.save('data/SD2_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy', np.nanmean(rate_data_SD2_Stop[:, t_stopCue : t_stopCue + 200], 1))
            np.save('data/STN_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy', np.nanmean(rate_data_STN_Stop[:, t_stopCue : t_stopCue + 200], 1))
            np.save('data/GPeProto_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy', np.nanmean(rate_data_GPeProto_Stop[:, t_stopCue : t_stopCue + 200], 1))        
            np.save('data/GPeArky_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy', np.nanmean(rate_data_GPeArky_Stop[:, t_stopCue : t_stopCue + 200], 1))        
            np.save('data/SNr_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy', np.nanmean(rate_data_SNr_Stop[:, t_stopCue : t_stopCue + 200], 1))
            np.save('data/Thal_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy', np.nanmean(rate_data_Thal_Stop[:, t_stopCue : t_stopCue + 200], 1))        
            np.save('data/Cortex_G_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy', np.nanmean(rate_data_Cortex_G_Stop[:, t_stopCue : t_stopCue + 200], 1))        
            np.save('data/Cortex_S_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy', np.nanmean(rate_data_Cortex_S_Stop[:, t_stopCue : t_stopCue + 200], 1))                

            print('fertig')




            print("Calculating reaction times...")

            Integrator.g_ampa = 0                    

            param_id = params['id']
            t_init = params['t_init']


            spike_times, ranks = np.load('./data/Integrator_spike_Go'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'.npy')
            spike_times_stop, ranks_stop = np.load('./data/Integrator_spike_Stop'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'.npy')
            mInt_stop = np.load('data/Integrator_ampa_Stop_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')
            mInt_go = np.load('data/Integrator_ampa_Go_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')            
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
                if np.nanmax(rsp_mInt_stop[:, i_trial]) >= Integrator.threshold: 
                    RT_Stop[i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= Integrator.threshold)[0][0]


            plt.figure(figsize=(3.5,4), dpi=300)
            fsize = 10
            nz_Go = np.nonzero(np.isnan(RT_Go)==False)
            nz_Stop = np.nonzero(np.isnan(RT_Stop)==False)
            counts_Go, bins_Go = np.histogram(RT_Go[nz_Go] * dt(), 10) # 
            counts_Stop, bins_Stop = np.histogram(RT_Stop[nz_Stop] * dt(), 10) # 
            if counts_Go.max() > 0:
                plt.bar(bins_Go[:-1], np.array(counts_Go) * (1.0/counts_Go.max()), width=np.diff(bins_Go)[0], alpha=0.5, color='b') #
            if counts_Stop.max() > 0: 
                plt.bar(bins_Stop[:-1], np.array(counts_Stop) * (1.0/counts_Stop.max()), width=np.diff(bins_Stop)[0], alpha=0.5, color='g') #
            mean_CorrectGo = np.round( (np.nanmean(RT_Go[nz_Go]) - t_init/dt())*dt(), 1)
            mean_FailedStop = np.round( (np.nanmean(RT_Stop[nz_Stop]) - t_init/dt())*dt(), 1)
            plt.legend(('Correct Go, mean RT='+str(mean_CorrectGo),'Failed stop, mean RT='+str(mean_FailedStop)), fontsize=fsize)
            plt.xlabel('Reaction time [ms]', fontsize=fsize)    
            plt.ylabel('Normalized trial count', fontsize=fsize)
            ax = plt.gca()
            ax.axis([t_init, (500 + t_init), ax.axis()[2], ax.axis()[3]])
            xtks = ax.get_xticks()
            xlabels = []
            for i in range(len(xtks)):
                xlabels.append(str( int( xtks[i]-t_init ) ))
            ax.set_xticklabels(xlabels)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(fsize)
            plt.tight_layout()
            plt.savefig(saveFolderPlots+'Reaction_times_paramsid'+str(int(param_id))+'.png')

            results_RT = {}
            results_RT['nFailedGoTrials'] = len(nz_FailedGo)
            results_RT['nCorrectGoTrials'] = len(nz_CorrectGo)
            results_RT['nFailedStopTrials'] = len(nz_FailedStop)
            results_RT['nCorrectStopTrials'] = len(nz_CorrectStop)
            results_RT['meanRT_CorrectGo'] = mean_CorrectGo
            results_RT['meanRT_FailedStop'] = mean_FailedStop

            np.save('data/resultsRT_'+str(i_netw_rep)+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(int(param_id))+'.npy', results_RT) 
            params_loaded = np.load('data/paramset_id'+str(int(param_id))+str(i_netw_rep)+'.npy', allow_pickle=True)
            print("params_loaded = ", params)


            t_init = params['t_init']
            t_SSD = params['t_SSD']
            thresh = Integrator.threshold
            plt.figure(figsize=(3.5,4), dpi=300)
            plt.plot([0, rsp_mInt_stop.shape[0]], thresh * np.ones(2), '--', lw=1, color='grey', label='Threshold')
            Thresh_grey_patch = patch.Patch(color = 'grey', label = 'Threshold' )
            plt.plot(rsp_mInt_go[:, :], 'b', lw=1, label='correct Go')
            CorrectGo_blue_patch = patch.Patch(color = 'blue', label = 'Correct Go trial')
            plt.plot((t_init / dt()) * np.ones(2), [0, 0.16], 'g--', lw=1.5, label = 'Go cue')
            GoCue_green_patch = patch.Patch(color = 'green', label = 'Go cue' )
            plt.plot((t_init + t_SSD) / dt() * np.ones(2), [0, 0.16], '--', color='orange', lw=1.5, label = 'Stop cue')
            StopCue_blue_patch = patch.Patch(color = 'orange', label = 'Stop cue' ) # color = 'blue'
            if len(nz_FailedStop) > 0:
                plt.plot(rsp_mInt_stop[:, nz_FailedStop], 'r', lw=1, label = 'Failed stop')
            if len(nz_CorrectStop) > 0:
                plt.plot(rsp_mInt_stop[:, nz_CorrectStop], 'k', lw=1, label = 'Correct stop')

            FailedStop_red_patch = patch.Patch(color = 'red', label = 'Failed stop trial')
            CorrectStop_black_patch = patch.Patch(color = 'black', label = 'Correct stop trial')
            plt.legend(handles = [Thresh_grey_patch, CorrectGo_blue_patch, GoCue_green_patch, StopCue_blue_patch, FailedStop_red_patch, CorrectStop_black_patch], loc='lower right', fontsize=6)
            plt.title(str(100 * len(nz_CorrectStop) / (len(nz_CorrectStop) + len(nz_FailedStop)) )+'% correct stop trials, ' + str(100 * len(nz_CorrectGo) / (len(nz_CorrectGo) + len(nz_FailedGo)) )+'% correct Go trials', fontsize=9) #fsize)
            plt.xlabel('Time [ms]', fontsize=fsize)    
            plt.ylabel('Integrator value', fontsize=fsize)
            ax = plt.gca()
            ax.set_yticks(np.arange(0, 0.2, 0.04))
            ax.axis([(t_init-200)/dt(), ax.axis()[1], 0, ax.axis()[3]])        
            ax.set_xticks([t_init/dt(), (t_init+200)/dt(), (t_init+400)/dt(), (t_init+600)/dt()])
            xtks = ax.get_xticks()
            xlabels = []
            for i in range(len(xtks)):
                xlabels.append(str(int(dt()*xtks[i]-t_init)))
            ax.set_xticklabels(xlabels)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(fsize)
            plt.tight_layout()
            plt.savefig(saveFolderPlots+'Integrator_ampa_Stop_'+str(trials)+'trials_paramsid'+str(int(param_id))+'.png')
        
            # RESET of all modified parameters: 
            params['CortexGo_rates'] = params_orig_CortexGo
            params['CortexStop_rates'] = params_orig_CortexStop
            params['Stop1rates'] = params_orig_CortexPause 
            params['cortexPause_ratesSecondRespMod'] = params_orig_CortexPauseMod
            Cortex_SGPe_Arky.mod_factor = params['weights_cortexStop_GPeArky']
            Cortex_SGPe_Proto2.mod_factor = params['weights_cortexStop_GPeCp']                     
            STR_D2GPe_Arky.mod_factor = params['weights_StrD2_GPeArky']                   
            STR_D2GPe_Proto2.mod_factor = params['weights_StrD2_GPeCp']  
            GPe_ArkySTR_D1.mod_factor = params['weights_GPeArky_StrD1']
            GPe_ArkySTR_D2.mod_factor = params['weights_GPeArky_StrD2'] 
            GPe_Proto2IntegratorStop.mod_factor = params['weights_GPeCp_IntStop']   
            GPe_ProtoSTR_FSI.mod_factor = params['weights_GPeProto_StrFSI'] 
            STNSNr.mod_factor = params['weights_STN_SNr']  
            changeArkyStopOutput(1,1,1)
            changeStopMechanism(1,1,1)





##############################################################################################
if abfrage_loop == 'STR' or abfrage_loop == 'loop': 
    #plt.ion()

    param_id = params['id']
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
if abfrage_loop == 'extra' or abfrage_loop == 'loop': 
    plt.ion()
    param_id = params['id']
    t_init = params['t_init'] # 100
    t_SSD = params['t_SSD'] # 170
    t_stopCue = int(t_init + t_SSD)    
    fsize = 6    

    #'''#
    plt.figure(figsize=(3.5,4), dpi=300)
    #mInt_stop = np.load('data/Integrator_ampa_Stop_id'+str(int(param_id))+'.npy')
    mInt_stop = np.load('data/Integrator_ampa_Stop_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')
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
    pvalue_list, pvalue_times = np.load('data/p_value_list_times_'+str(int(params['id']))+str(i_netw_rep)+'.npy', allow_pickle=True)

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
    STR_D1_ratepertrial_Stop = np.load('data/SD1_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy')
    STR_D2_ratepertrial_Stop = np.load('data/SD2_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy')
    STN_ratepertrial_Stop = np.load('data/STN_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy')
    GPe_Proto_ratepertrial_Stop = np.load('data/GPeProto_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy')        
    GPe_Arky_ratepertrial_Stop = np.load('data/GPeArky_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy')        
    SNr_ratepertrial_Stop = np.load('data/SNr_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy')
    Thal_ratepertrial_Stop = np.load('data/Thal_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy')
    Cortex_G_ratepertrial_Stop = np.load('data/Cortex_G_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy')        
    Cortex_S_ratepertrial_Stop = np.load('data/Cortex_S_rate_Stop_tempmean_'+str(int(params['id']))+str(i_netw_rep)+'.npy')                

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
    mInt_Go = np.load('data/Integrator_ampa_Go_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')
    pvalue_list = {}
    nameList = ['failedStop_vs_allGo', 'failedStop_vs_fastGo', 'failedStop_vs_slowGo']
    for name in nameList:
        pvalue_list[name], pvalue_times = np.load('data/p_value_list_'+name+'_times_'+str(int(params['id']))+str(i_netw_rep)+'.npy', allow_pickle=True)

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


    #rateperneuron_GPeArky_allStoptrials = np.load('data/GPeArky_rateperneuron_allStoptrials_'+str(int(params['id']))+'.npy')
    STN_poprate_Stop_alltrials = np.load('data/STN_rate_Stop_alltrials'+str(i_netw_rep)+'.npy')
    SNr_poprate_Stop_alltrials = np.load('data/SNr_rate_Stop_alltrials'+str(i_netw_rep)+'.npy')
    GPe_Proto_poprate_Stop_alltrials = np.load('data/GPeProto_rate_Stop_alltrials'+str(i_netw_rep)+'.npy')
    GPe_Arky_poprate_Stop_alltrials = np.load('data/GPeArky_rate_Stop_alltrials'+str(i_netw_rep)+'.npy')        
    #get_peak_response_time(STN_poprate_Stop_alltrials, SNr_poprate_Stop_alltrials, GPe_Proto_poprate_Stop_alltrials, GPe_Arky_poprate_Stop_alltrials, t_init + t_SSD, t_init + t_SSD + 200, dt(), param_id, trials, param_id, trials, paramname)    


##############################################################################################
if abfrage_loop=='RT':
    print("Calculating reaction times...")                  

    param_id = params['id']
    t_init = params['t_init'] # 100
    i_cycle=0

    #'''#
    spike_times, ranks = np.load('./data/Integrator_spike_Go'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'.npy')
    spike_times_stop, ranks_stop = np.load('./data/Integrator_spike_Stop'+str(i_netw_rep)+'_cycle'+str(i_cycle)+'.npy')
    mInt_stop = np.load('data/Integrator_ampa_Stop_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')
    mInt_go = np.load('data/Integrator_ampa_Go_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')            
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
        mInt_Go = np.load('data/Integrator_ampa_Go_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')
        mInt_Stop = np.load('data/Integrator_ampa_Stop_'+str(i_netw_rep)+'_id'+str(int(params['id']))+'.npy')
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
