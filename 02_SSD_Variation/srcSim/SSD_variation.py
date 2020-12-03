#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:00 2020

@authors: Oliver Maith, Lorenz Goenner
"""

from ANNarchy import*
from BGmodelSST.sim_params import params
setup(dt=params['general_dt'])
setup(num_threads=params['general_threads'])

import sys
import os
import numpy as np

#from BGmodelSST.populations import Stoppinput1, Cortex_S, Cortex_G, Integrator, IntegratorStop
#from BGmodelSST.projections import STNSNr, STNGPe_Proto, STNESTN, GPe_ArkySTR_D1, GPe_ArkySTR_D2, GPe_ArkySTR_FSI, GPe_ArkyGPe_Proto, STNGPe_Arky, GPe_Proto2GPe_Proto, STNGPe_Proto2, GPe_ArkyGPe_Proto2, GPe_Proto2STR_D1, GPe_Proto2STR_D2, GPe_Proto2STR_FSI, GPe_Proto2GPe_Arky, GPe_Proto2IntegratorStop, STNGPe_Arky2, GPe_Proto2GPe_Arky2

from BGmodelSST.analysis import calc_KW_stats_all, calc_meanrate_std_failed_correct, get_rates_failed_correct, get_poprate_aligned_onset, custom_poprate, calc_meanrate_std_Fast_Slow, get_rates_allGo_fastGo_slowGo
from BGmodelSST.neuronmodels import Izhikevich_neuron, Izhikevich_STR_neuron, STR_FSI_neuron, Integrator_neuron
from BGmodelSST.populations import Stoppinput1, Cortex_S, Cortex_G, STR_D1, STR_D2, STN, SNr, GPe_Proto, Thal, Integrator, IntegratorStop, GPeE, SNrE, STNE, STR_FSI, STRE, GPe_Arky, TestThalnoise, GPe_Proto2
from BGmodelSST.projections import Stoppinput1STN, Cortex_GSTR_D1, Cortex_GSTR_D2, Cortex_GSTR_FSI, Cortex_GThal, Cortex_SGPe_Arky, STR_D1SNr, STR_D2GPe_Proto, STNSNr, STNGPe_Proto, GPe_ProtoSTN, GPe_ProtoSNr, SNrThal, ThalIntegrator, GPeEGPe_Proto, GPEGPe_Arky, SNrESNr, STNESTN, STR_FSISTR_D1, STR_FSISTR_D2, STRESTR_D1, STRESTR_D2, GPe_ArkySTR_D1, GPe_ArkySTR_D2, TestThalnoiseThal,STRESTR_FSI, STR_FSISTR_FSI, GPe_ArkySTR_FSI, GPe_ArkyGPe_Proto, GPe_ProtoGPe_Arky, STR_D2GPe_Arky, GPe_ProtoSTR_FSI, STR_D1STR_D1, STR_D1STR_D2, STR_D2STR_D1, STR_D2STR_D2, Cortex_GSTR_D2, Cortex_SGPe_Proto, STNGPe_Arky, ThalSD1, ThalSD2, ThalFSI, GPe_ProtoGPe_Proto2, GPe_Proto2GPe_Proto, STR_D2GPe_Proto2, STR_D1GPe_Proto2, STNGPe_Proto2, Cortex_SGPe_Proto2, GPe_ArkyGPe_Proto2, GPe_Proto2STR_D1, GPe_Proto2STR_D2, GPe_Proto2STR_FSI, GPe_Proto2GPe_Arky, GPe_Proto2IntegratorStop, EProto1GPe_Proto, EProto2GPe_Proto2, EArkyGPe_Arky, Cortex_SGPe_Arky2, STR_D2GPe_Arky2, GPe_ProtoGPe_Arky2, STNGPe_Arky2, GPe_Proto2GPe_Arky2, EArkyGPe_Arky2, GPe_Arky2STR_D1, GPe_Arky2STR_D2, GPe_Arky2STR_FSI


from simulationParams import paramsS


### COMPILATION AND DATA DIRECTORY CREATION
if len(sys.argv) <= 1:
    print('Network number missing')
    quit()
else:
    netNr = str(sys.argv[1])
    compile('Annarchy'+netNr)
    print("\nnetNr = ", netNr)

    dataDir='../data/'+paramsS['saveFolder']
    try:
        os.makedirs(dataDir)
    except:
        if os.path.isdir(dataDir):
            print(dataDir+' already exists')
        else:
            print('could not create '+dataDir+' folder')
            quit()
            
            
### SAVE PARAMETERS
np.save('../data/'+paramsS['saveFolder']+'/paramset_id'+str(params['general_id'])+netNr+'.npy', np.array([params]))#TODO: remove general_id
np.save('../data/'+paramsS['saveFolder']+'/paramSset_id'+str(params['general_id'])+netNr+'.npy', np.array([paramsS]))


### STOPPINGPERFORMANCE FOR EACH LESION AND SSD
stoppingPerformance = np.zeros((len(paramsS['lesionList']),len(paramsS['SSDList'])))

### SIMULATION LOOP
for lession_Idx, lession in enumerate(paramsS['lesionList']):
    for t_SSD_Idx, t_SSD in enumerate(paramsS['SSDList']):
        successfulStops=0
        for i in range(paramsS['trials']):
            ### trial INITIALIZATION simulation to get stable state
            reset(populations=True, projections=True, synapses=False, net_id=0)
            Cortex_G.rates = 0
            Cortex_S.rates = 0
            Stoppinput1.rates = 0

            ###activate lession
            if lession=='Arky':
                GPe_ArkySTR_D1.mod_factor = 0
                GPe_ArkySTR_D2.mod_factor = 0
                GPe_ArkyGPe_Proto.mod_factor = 0
                GPe_ArkySTR_FSI.mod_factor = 0
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

            simulate(params['t_init'])# t_init rest

            ### Integrator Reset
            Integrator.decision = 0
            Integrator.g_ampa = 0  
            IntegratorStop.decision = 0 

            ### calculate all eventTIMES
            GOCUE     = np.round(get_time(),1)
            STOPCUE   = GOCUE + t_SSD
            STN1ON    = GOCUE
            STN1OFF   = STN1ON + params['t_cortexPauseDuration']
            GOON      = GOCUE + params['t_delayGo'] + int(np.clip(params['t_delayGoSD'] * np.random.randn(),-params['t_delayGo'],None))
            STN2ON    = STOPCUE
            STN2OFF   = STN2ON + params['t_cortexPauseDuration']
            STOPON    = STOPCUE + params['t_delayStopAfterCue']
            STOPOFF   = STOPON + params['t_cortexStopDurationAfterCue']
            ENDTIME   = STOPOFF + params['t_decay']
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
            tempMode=paramsS['trialMode']
            while not(end):
                if t == STN1ON:
                    Stoppinput1.rates = params['cortexPause_rates'] + params['cortexPause_ratesSD'] * randn_trial_Pause
                    #print('STN1ON',STN1ON)
                if t == STN1OFF:
                    Stoppinput1.rates = 0
                    #print('STN1OFF',STN1OFF)
                if t == GOON:
                    Cortex_G.rates = params['cortexGo_rates'] + params['cortexGo_ratesSD'] * randn_trial
                    #print('GOON',GOON)
                if t == GOOFF:
                    Cortex_G.rates = 0
                    #print('GOOFF',GOOFF)                    
                if t == STN2ON and tempMode=='STOP':
                    Stoppinput1.rates = params['cortexPause_ratesSecondRespMod']*params['cortexPause_rates'] + params['cortexPause_ratesSD'] * randn_trial_Pause
                    #print('STN2ON',STN2ON)
                if t == STN2OFF and tempMode=='STOP':
                    Stoppinput1.rates = 0
                    #print('STN2OFF',STN2OFF)
                if t == STOPON and tempMode=='STOP':
                    Cortex_S.rates = params['cortexStop_ratesAfterCue'] + params['cortexStop_ratesSD'] * randn_trial_S * (params['cortexStop_ratesAfterCue'] > 0)
                    #print('STOPON',STOPON)
                if t == motorSTOP:
                    Cortex_S.rates = params['cortexStop_ratesAfterAction'] + params['cortexStop_ratesSD'] * randn_trial_S * (params['cortexStop_ratesAfterAction'] > 0)
                    motorSTOPOFF = motorSTOP + params['t_cortexStopDurationAfterAction']
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
                        motorSTOP = t + params['t_delayStopAfterAction']
                        if t<STOPCUE and tempMode=='STOP':
                            tempMode='GO'
                    if responseProto2 == -1 and GOOFFResponse == 0 and ((t>STOPON and tempMode=='STOP') or motorResponse==1):
                        GOOFFResponse=1
                        GOOFF = t
            ### TRIAL END
            
            ### COUNT SUCCESSFUL STOPS
            if responseThal==0:
                successfulStops+=1
                
                
        ### end of all trials
        print(lession,t_SSD,'finished')
        stoppingPerformance[lession_Idx,t_SSD_Idx] = successfulStops/float(paramsS['trials'])
        np.save('../data/'+paramsS['saveFolder']+'/SSD_variation_stoppingPerformance'+str(lession)+'_'+str(t_SSD)+'_paramId'+str(int(params['general_id']))+'_netID'+netNr+'.npz',stoppingPerformance[lession_Idx,t_SSD_Idx])

### Saves
np.savez('../data/'+paramsS['saveFolder']+'/SSD_variation_stoppingPerformance_paramId'+str(int(params['general_id']))+'_netID'+netNr+'.npz', stoppingPerformance=stoppingPerformance, lesionList=paramsS['lesionList'], t_SSD_List=paramsS['SSDList'])




























































        
