#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:00 2020

@authors: Oliver Maith, Lorenz Goenner
"""

from ANNarchy import*
import pylab 
import random
#import timeit import default_timer as timer
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patch # fuer legende
import scipy.stats as st
#from scipy.interpolate import spline # fuer smooth 
from scipy import stats #fuer z score -> mean firing rate
from scipy.signal import argrelmin

from analysis import custom_zscore, \
    plot_zscore_stopVsGo, plot_correl_rates_Intmax, plot_meanrate_All_FailedVsCorrectStop, calc_KW_stats_all, rate_plot, \
    calc_meanrate_std_failed_correct, get_rates_failed_correct, get_poprate_aligned_onset, get_peak_response_time, custom_poprate # get_rate_singletrial,
    # trial_averaged_firing_rate, tria_averaged_firing_rate_selective, simple_mean_rate, plot_meanrate, plot_meanrate_selective, box_smooth, calc_KruskalWallis, get_correct_failed_stop_trials, get_syn_mon, plot_syn_stats,
from plotting import get_and_plot_syn_mon    
from sim_params import params, wf
from init import init_neuronmodels



#setup(dt=1)
setup(dt=0.1) # Extremely different results!!!
setup(num_threads=1)
# TODO: Adapt all plots to dt=0.1 !!!

#setup(seed=0)
#np.random.seed(0)

from lg_neuronmodels_BA_BG import Izhikevich_neuron, Izhikevich_STR_neuron,STR_FSI_neuron, Integrator_neuron, Poisson_neuron, FixedSynapse
from lg_populations_BA_BG import Stoppinput1, Cortex_S, Cortex_G, STR_D1, STR_D2, STN, SNr, GPe_Proto, Thal, Integrator, IntegratorStop, GPeE, SNrE, STNE, STR_FSI, STRE, GPe_Arky, TestThalnoise, population_size, GPe_Proto2
from lg_projections_BA_BG import Stoppinput1STN, Cortex_GSTR_D1, Cortex_GSTR_D2, Cortex_GSTR_FSI, Cortex_GThal, Cortex_SGPe_Arky, STR_D1SNr, STR_D2GPe_Proto, STNSNr, \
                                    STNGPe_Proto, GPe_ProtoSTN, GPe_ProtoSNr, SNrThal, ThalIntegrator, GPeEGPe_Proto, GPEGPe_Arky, SNrESNr, STNESTN, \
                                    STR_FSISTR_D1, STR_FSISTR_D2, STRESTR_D1, STRESTR_D2, GPe_ArkySTR_D1, GPe_ArkySTR_D2, TestThalnoiseThal,STRESTR_FSI, \
                                    STR_FSISTR_FSI, GPe_ArkySTR_FSI, GPe_ArkyGPe_Proto, GPe_ProtoGPe_Arky, STR_D2GPe_Arky, \
                                    GPe_ProtoSTR_FSI, STR_D1STR_D1, STR_D1STR_D2, STR_D2STR_D1, STR_D2STR_D2, Cortex_GSTR_D2, Cortex_SGPe_Proto, STNGPe_Arky, GPe_ArkyCortex_G, \
                                    ThalSD1, ThalSD2, ThalFSI, \
                                    GPe_ProtoGPe_Proto2, GPe_Proto2GPe_Proto, STR_D2GPe_Proto2, STR_D1GPe_Proto2, STNGPe_Proto2, Cortex_SGPe_Proto2, GPe_ArkyGPe_Proto2, GPe_Proto2STR_D1, GPe_Proto2STR_D2, GPe_Proto2STR_FSI, GPe_Proto2GPe_Arky, GPe_Proto2IntegratorStop, EProto1GPe_Proto, EProto2GPe_Proto2, EArkyGPe_Arky, \
                                    Cortex_SGPe_Arky2, STR_D2GPe_Arky2, GPe_ProtoGPe_Arky2, STNGPe_Arky2, GPe_Proto2GPe_Arky2, EArkyGPe_Arky2, GPe_Arky2STR_D1, GPe_Arky2STR_D2, GPe_Arky2STR_FSI



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



# New: Apply weight factors here, to avoid re-compiling. 
# CAUTION: Will not be considered in the automatic report!
# CAUTION - appears to slow down spike transmission! Apply only to one projection at a time!
# Changing Poisson rates doesn't trigger recompiling.

GPe_ArkySTR_D1.mod_factor = wf['Arky_D1']
GPe_ArkySTR_D2.mod_factor = wf['Arky_D2']
GPe_ArkyGPe_Proto.mod_factor = wf['Arky_Proto']
GPe_ArkySTR_FSI.mod_factor =  wf['Arky_FSI']
GPe_ArkyCortex_G.mod_factor =  wf['Arky_CortexGo']
Cortex_GSTR_D1.mod_factor = wf['Go_D1']
Cortex_GSTR_D2.mod_factor = wf['Go_D2']
Cortex_GThal.mod_factor = wf['CortexG_Thal']
Cortex_GSTR_FSI.mod_factor = wf['CtxGo_FSI']
Stoppinput1STN.mod_factor = wf['Pause_STN']
Cortex_SGPe_Arky.mod_factor = wf['Stop_Arky']
Cortex_SGPe_Proto.mod_factor = wf['Stop_Proto']
STR_FSISTR_D1.mod_factor = wf['FSI_D1']
STR_FSISTR_D2.mod_factor = wf['FSI_D2']
STR_FSISTR_FSI.mod_factor = wf['FSI_FSI']
GPeEGPe_Proto.mod_factor = wf['E_Proto']*int(params['id']<8000)
GPEGPe_Arky.mod_factor = wf['E_Arky']*int(params['id']<8000)
GPe_ProtoSTN.mod_factor = wf['Proto_STN']
GPe_ProtoSNr.mod_factor = wf['Proto_Snr']
GPe_ProtoGPe_Arky.mod_factor = wf['Proto_Arky']
GPe_ProtoSTR_FSI.mod_factor = wf['Proto_FSI'] 
STR_D1SNr.mod_factor = wf['D1_Snr']
STR_D1STR_D1.mod_factor = wf['D1_D1']
STR_D1STR_D2.mod_factor = wf['D1_D2']
STR_D2GPe_Proto.mod_factor = wf['D2_Gpe']
STR_D2GPe_Arky.mod_factor = wf['D2_Arky']
STR_D2STR_D1.mod_factor = wf['D2_D1']
STR_D2STR_D2.mod_factor = wf['D2_D2']
SNrThal.mod_factor = wf['Snr_Thal']
SNrESNr.mod_factor = wf['E_Snr']
STNSNr.mod_factor = wf['STN_Snr']
STNGPe_Proto.mod_factor = wf['STN_Proto']
STNGPe_Arky.mod_factor = wf['STN_Arky']
STNESTN.mod_factor = wf['E_Stn']
STRESTR_D1.mod_factor = wf['E_D1'] 
STRESTR_D2.mod_factor = wf['E_D2']
STRESTR_FSI.mod_factor = wf['E_FSI']
ThalIntegrator.mod_factor = wf['Thal_Int'] 
TestThalnoiseThal.mod_factor = wf['ThalNoise']
ThalSD1.mod_factor = wf['ThalSD1']
ThalSD2.mod_factor = wf['ThalSD2']
ThalFSI.mod_factor = wf['ThalFSI']

if params['id'] >= 8000:
    GPe_ProtoGPe_Proto2.mod_factor      = wf['Proto1_Proto2']
    GPe_Proto2GPe_Proto.mod_factor      = wf['Proto2_Proto1']
    STR_D2GPe_Proto2.mod_factor         = wf['D2_Proto2']
    STR_D1GPe_Proto2.mod_factor         = wf['D1_Proto2']
    STNGPe_Proto2.mod_factor            = wf['STN_Proto2']
    Cortex_SGPe_Proto2.mod_factor       = wf['Stop_Proto2']
    GPe_ArkyGPe_Proto2.mod_factor       = wf['Arky_Proto2']
    GPe_Proto2STR_D1.mod_factor         = wf['Proto2_D1']
    GPe_Proto2STR_D2.mod_factor         = wf['Proto2_D2']
    GPe_Proto2STR_FSI.mod_factor        = wf['Proto2_FSI']
    GPe_Proto2GPe_Arky.mod_factor       = wf['Proto2_Arky']
    GPe_Proto2IntegratorStop.mod_factor = wf['Proto2_Int']
    EProto1GPe_Proto.mod_factor         = wf['E_Proto']
    EProto2GPe_Proto2.mod_factor        = wf['E_Proto2']
    EArkyGPe_Arky.mod_factor            = wf['E_Arky']

Cortex_SGPe_Arky2.mod_factor   = 0
STR_D2GPe_Arky2.mod_factor     = wf['D2_Arky']*int(params['Arky2'])
GPe_ProtoGPe_Arky2.mod_factor  = wf['Proto_Arky']*int(params['Arky2'])
STNGPe_Arky2.mod_factor        = wf['STN_Arky']*int(params['Arky2'])
GPe_Proto2GPe_Arky2.mod_factor = wf['Proto2_Arky']*int(params['Arky2'])
EArkyGPe_Arky2.mod_factor      = wf['E_Arky']*int(params['Arky2'])
GPe_Arky2STR_D1.mod_factor     = wf['Arky_D1']*int(params['Arky2'])
GPe_Arky2STR_D2.mod_factor     = wf['Arky_D2']*int(params['Arky2'])
GPe_Arky2STR_FSI.mod_factor    = wf['Arky_FSI']*int(params['Arky2'])

Integrator.tau = params['tau_Int']
IntegratorStop.tau = 30.0






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






simulate(50)              # for stable point
reset(populations=True, projections=False, synapses=False, net_id=0) # Does the reset cause the "transients" in firing rates?



############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################





### general parameters
trials = 200 
mode = 'STOP'
TrialType = params['TrialType']
t_decay = 300
t_init = params['t_init'] 
t_PauseDuration = params['t_PauseDuration']
t_StopDuration = params['t_StopDuration']
t_delayGO = params['t_delayGO']
t_delaySTOP = params['t_delaySTOP']
t_Pausedelay = 0
t_CorGODuration = params['t_CorGODuration']
t_CorSTOPDuration = params['t_GoAndStop']
t_motorSTOPDuration = params['t_motorStop']
t_delaymotorSTOP = params['t_delaymotorSTOP']
t_delayGOSD = params['t_delayGOSD']
lessionList = ['None','Arky', 'STN', 'Proto2']
t_SSD_List = np.arange(5,500,5).tolist()
saveFolder = 'data/'
sigma_Go = params['sigma_Go']
sigma_Stop = params['sigma_Stop']
sigma_Pause = params['sigma_Pause']

### results
stoppingPerformance = np.zeros((len(lessionList),len(t_SSD_List)))

### simulation loop
for lession_Idx, lession in enumerate(lessionList):
    for t_SSD_Idx, t_SSD in enumerate(t_SSD_List):
        successfulStops=0
        for i in range(trials):
            ### trial INITIALIZATION simulation to get stable state
            reset(populations=True, projections=True, synapses=False, net_id=0)
            Cortex_G.rates = 0
            Cortex_S.rates = 0
            Stoppinput1.rates = 0
            TrialType = params['TrialType']

            ###activate lession
            if lession=='Arky':
                GPe_ArkySTR_D1.mod_factor = 0
                GPe_ArkySTR_D2.mod_factor = 0
                GPe_ArkyGPe_Proto.mod_factor = 0
                GPe_ArkySTR_FSI.mod_factor = 0
                GPe_ArkyCortex_G.mod_factor = 0
                GPe_ArkyGPe_Proto2.mod_factor = 0
            elif lession=='Proto2':
                GPe_Proto2GPe_Proto.mod_factor = 0
                GPe_Proto2STR_D1.mod_factor = 0
                GPe_Proto2STR_D2.mod_factor = 0
                GPe_Proto2STR_FSI.mod_factor = 0
                GPe_Proto2GPe_Arky.mod_factor = 0
                GPe_Proto2IntegratorStop.mod_factor = 0
            elif lession=='STN':
                STNSNr.mod_factor = 0
                STNGPe_Proto.mod_factor = 0
                STNGPe_Arky.mod_factor = 0
                STNGPe_Proto2.mod_factor = 0
            elif lession=='None':
                nothing=0
            else:
                print('wrong lession argument'+lession)
                break

            simulate(t_init)# t_init rest

            ### Integrator Reset
            Integrator.decision = 0
            Integrator.g_ampa = 0  
            IntegratorStop.decision = 0                                                         
            
            if TrialType == 2:
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
                        Stoppinput1.rates = params['Stop2ratesMod']*params['Stop1rates'] + sigma_Pause * randn_trial_Pause
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
            ### end of trial
            #print('trial:',i,'response:',responseThal)
            if responseThal==0:
                successfulStops+=1
        ### end of all trials
        print(lession,t_SSD,'finished')
        stoppingPerformance[lession_Idx,t_SSD_Idx] = successfulStops/float(trials)
        np.save(saveFolder+'SSD_variation_stoppingPerformance'+str(lession)+'_'+str(t_SSD)+'_paramId'+str(int(params['id']))+'_netID'+str(i_netw_rep)+'.npz',stoppingPerformance[lession_Idx,t_SSD_Idx])

### Saves
np.savez(saveFolder+'SSD_variation_stoppingPerformance_paramId'+str(int(params['id']))+'_netID'+str(i_netw_rep)+'.npz', stoppingPerformance=stoppingPerformance, lessionList=lessionList, t_SSD_List=t_SSD_List)




























































        
