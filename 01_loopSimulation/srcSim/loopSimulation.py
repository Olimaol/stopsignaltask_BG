"""
Created on Fri Oct 16 2020

@authors: Oliver Maith, Lorenz Goenner, ilko
"""

from ANNarchy import*
from BGmodelSST.sim_params import params
setup(dt=params['general_dt'])
setup(num_threads=params['general_threads'])

import sys
import os
import numpy as np

from BGmodelSST.analysis import calc_KW_stats_all, calc_meanrate_std_failed_correct, get_rates_failed_correct, get_poprate_aligned_onset, custom_poprate, calc_meanrate_std_Fast_Slow, get_rates_allGo_fastGo_slowGo
from BGmodelSST.neuronmodels import Izhikevich_neuron, Izhikevich_STR_neuron, STR_FSI_neuron, Integrator_neuron
from BGmodelSST.populations import Stoppinput1, Cortex_S, Cortex_G, STR_D1, STR_D2, STN, SNr, GPe_Proto, Thal, Integrator, IntegratorStop, GPeE, SNrE, STNE, STR_FSI, STRE, GPe_Arky, TestThalnoise, GPe_Proto2
from BGmodelSST.projections import Stoppinput1STN, Cortex_GSTR_D1, Cortex_GSTR_D2, Cortex_GSTR_FSI, Cortex_GThal, Cortex_SGPe_Arky, STR_D1SNr, STR_D2GPe_Proto, STNSNr, STNGPe_Proto, GPe_ProtoSTN, GPe_ProtoSNr, SNrThal, ThalIntegrator, GPeEGPe_Proto, GPEGPe_Arky, SNrESNr, STNESTN, STR_FSISTR_D1, STR_FSISTR_D2, STRESTR_D1, STRESTR_D2, GPe_ArkySTR_D1, GPe_ArkySTR_D2, TestThalnoiseThal,STRESTR_FSI, STR_FSISTR_FSI, GPe_ArkySTR_FSI, GPe_ArkyGPe_Proto, GPe_ProtoGPe_Arky, STR_D2GPe_Arky, GPe_ProtoSTR_FSI, STR_D1STR_D1, STR_D1STR_D2, STR_D2STR_D1, STR_D2STR_D2, Cortex_GSTR_D2, Cortex_SGPe_Proto, STNGPe_Arky, ThalSD1, ThalSD2, ThalFSI, GPe_ProtoGPe_Proto2, GPe_Proto2GPe_Proto, STR_D2GPe_Proto2, STR_D1GPe_Proto2, STNGPe_Proto2, Cortex_SGPe_Proto2, GPe_ArkyGPe_Proto2, GPe_Proto2STR_D1, GPe_Proto2STR_D2, GPe_Proto2STR_FSI, GPe_Proto2GPe_Arky, GPe_Proto2IntegratorStop, EProto1GPe_Proto, EProto2GPe_Proto2, EArkyGPe_Arky, Cortex_SGPe_Arky2, STR_D2GPe_Arky2, GPe_ProtoGPe_Arky2, STNGPe_Arky2, GPe_Proto2GPe_Arky2, EArkyGPe_Arky2, GPe_Arky2STR_D1, GPe_Arky2STR_D2, GPe_Arky2STR_FSI

from simulationParams import paramsS


### PARAMETER VARIATION FUNCTIONS
def changeStopMechanism(fArky, fSTN, fProto2):
    Cortex_SGPe_Arky.mod_factor              = params['weights_cortexStop_GPeArky']*fArky
    params['cortexPause_ratesSecondRespMod'] = params['cortexPause_ratesSecondRespMod']*fSTN
    Cortex_SGPe_Proto2.mod_factor            = params['weights_cortexStop_GPeCp']*fProto2

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


### COMPILATION AND DIRECTORY CREATION
if len(sys.argv) <= 1:
    print('Network number missing')
else:
    netNr = str(sys.argv[1])
    compile('Annarchy'+netNr)
    print("\nnetNr = ", netNr)
    saveFolderPlots='plots/'+netNr+'/'

    try:
        os.mkdir(saveFolderPlots[:-1])
    except:
        if os.path.isdir(saveFolderPlots[:-1]):
            print(saveFolderPlots[:-1]+' already exists')
        else:
            print('could not create '+saveFolderPlots[:-1]+' folder')
            quit()

    try:
        os.makedirs('data')
    except:
        if os.path.isdir('data'):
            print('data already exists')
        else:
            print('could not create data folder')
            quit()


### POPULATIONS RATE DATA FOR EACH TRIAL OF GO AND STOP CUE ###
simlen = params['t_SSD'] + params['t_delayStopAfterCue'] + params['t_cortexStopDurationAfterCue'] + params['t_decay'] + params['t_init']
rate_data_SD1_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_SD1_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_SD2_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_SD2_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_FSI_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_FSI_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_STN_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_STN_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_GPeProto_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_GPeProto2_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_GPeProto_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_GPeProto2_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_GPeArky_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_GPeArky_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_SNr_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_SNr_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_SNrE_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_SNrE_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_Thal_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_Thal_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())])
rate_data_Cortex_G_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_Cortex_G_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_Cortex_S_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_Cortex_S_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_Stoppinput1_Go = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 
rate_data_Stoppinput1_Stop = np.nan*np.ones([paramsS['trials'], int(simlen / dt())]) 


### MONITORS ###
m_Int_ampa    = Monitor(Integrator, ['g_ampa', 'spike'])
m_STR_D1      = Monitor(STR_D1, ['spike'])
m_STR_D2      = Monitor(STR_D2, ['spike'])
m_STR_FSI     = Monitor(STR_FSI, ['spike'])
m_STN         = Monitor(STN,['spike'])
m_GPe_Proto   = Monitor(GPe_Proto,['spike']) 
m_GPe_Proto2  = Monitor(GPe_Proto2,['spike']) 
m_GPe_Arky    = Monitor(GPe_Arky,['spike']) 
m_SNr         = Monitor(SNr,['spike'])
m_SNrE        = Monitor(SNrE,['spike'])
m_Thal        = Monitor(Thal,['spike'])
m_Cortex_G    = Monitor(Cortex_G,'spike') 
m_Cortex_S    = Monitor(Cortex_S,'spike') 
m_Stoppinput1 = Monitor(Stoppinput1,'spike')

selection = np.zeros(paramsS['trials'])
timeovertrial = np.zeros(paramsS['trials'])

np.save('data/paramset_id'+str(params['general_id'])+netNr+'.npy', params)
np.save('data/paramSset_id'+str(params['general_id'])+netNr+'.npy', paramsS)


### WHICH PARAMETER VARIATIONS ###
i_cortexGo_rates = i_cortexStop_rates = i_cortexPause_rates = i_weights_cortexStop_GPeArky = i_weights_cortexStop_GPeCp = i_weights_StrD2_GPeArky = i_weights_StrD2_GPeCp = i_weights_GPeArky_StrD1 = i_weights_GPeArky_StrD2 = i_weights_GPeCp_IntStop = i_weights_GPeProto_StrFSI = i_weights_STN_SNr = i_ArkyD1Stop = i_ArkyD2Stop = i_ArkyFSIStop = i_activateStopArky = i_activateStopSTN = i_activateStopCp = i_deactivateStopArky = i_deactivateStopSTN = i_deactivateStopCp = 99
if paramsS['makeParameterVariations']==0:
    ###only one param wihtout variation
    paramsFactorList = np.array([1.0])
    paramnames = ['cortexStop_rates']
    i_cortexStop_rates = 0
elif paramsS['makeParameterVariations']==1:
    ###one param with variation
    paramsFactorList = np.array([0.0, 0.5, 1.0, 1.5, 2])
    paramnames = ['weights_cortexStop_GPeArky']
    i_weights_cortexStop_GPeArky = 0
elif paramsS['makeParameterVariations']==2:
    ###multiple params with variation
    paramsFactorList = np.array([0.0, 0.5, 1.0, 1.5, 2])
    #paramnames = ['cortexGo_rates', 'cortexStop_rates', 'cortexPause_rates', 'weights_cortexStop_GPeArky', 'weights_cortexStop_GPeCp', 'weights_StrD2_GPeArky', 'weights_StrD2_GPeCp', 'weights_GPeArky_StrD1', 'weights_GPeArky_StrD2', 'weights_GPeCp_IntStop', 'weights_GPeProto_StrFSI', 'weights_STN_SNr', 'i_ArkyD1Stop', 'i_ArkyD2Stop', 'i_ArkyFSIStop']
    #i_cortexGo_rates, i_cortexStop_rates, i_cortexPause_rates, i_weights_cortexStop_GPeArky, i_weights_cortexStop_GPeCp, i_weights_StrD2_GPeArky, i_weights_StrD2_GPeCp, i_weights_GPeArky_StrD1, i_weights_GPeArky_StrD2, i_weights_GPeCp_IntStop, i_weights_GPeProto_StrFSI, i_weights_STN_SNr, i_ArkyD1Stop, i_ArkyD2Stop, i_ArkyFSIStop = range(len(paramnames))
    #paramnames = ['i_ArkyD1Stop', 'i_ArkyD2Stop', 'i_ArkyFSIStop']
    #i_ArkyD1Stop, i_ArkyD2Stop, i_ArkyFSIStop = range(len(paramnames))
    #paramnames = ['activateStopArky', 'activateStopSTN', 'activateStopCp', 'deactivateStopArky', 'deactivateStopSTN', 'deactivateStopCp']
    #i_activateStopArky, i_activateStopSTN, i_activateStopCp, i_deactivateStopArky, i_deactivateStopSTN, i_deactivateStopCp = range(len(paramnames))
    paramnames = ['cortexGo_rates', 'cortexStop_rates', 'weights_cortexStop_GPeArky']
    i_cortexGo_rates, i_cortexStop_rates, i_weights_cortexStop_GPeArky = range(len(paramnames))

n_param_vars = len(paramnames)
params_orig_CortexGo = params['cortexGo_rates']
params_orig_CortexStop = params['cortexStop_ratesAfterCue']
params_orig_CortexPause = params['cortexPause_rates']
params_orig_CortexPauseMod = params['cortexPause_ratesSecondRespMod']

### LOOP OVER PARAMETERS (for variations)
for i_paramname in range(n_param_vars):       
    paramname = paramnames[i_paramname]
            
    ### SELECT PARAMETER AND PARAMETERVARIATIONS
    if i_paramname == i_cortexGo_rates:
        params_orig =  params['cortexGo_rates']
        param_factors = np.array([0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5])
    elif i_paramname == i_cortexStop_rates:
        params_orig =  params['cortexStop_ratesAfterCue']
        param_factors = paramsFactorList                         
    elif i_paramname == i_cortexPause_rates:
        params_orig =  params['cortexPause_rates']
        param_factors = paramsFactorList 
    elif i_paramname == i_weights_cortexStop_GPeArky:
        params_orig, paramname = Cortex_SGPe_Arky.w, 'weights_cortexStop_GPeArky'
        param_factors = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    elif i_paramname == i_weights_cortexStop_GPeCp:
        params_orig, paramname = Cortex_SGPe_Proto2.w, 'weights_cortexStop_GPeCp' 
        param_factors = paramsFactorList                                              
    elif i_paramname == i_weights_StrD2_GPeArky:
        params_orig, paramname = STR_D2GPe_Arky.w, 'weights_StrD2_GPeArky'    
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
    elif i_paramname == i_weights_GPeProto_StrFSI:
        params_orig, paramname = GPe_ProtoSTR_FSI.w, 'weights_GPeProto_StrFSI'  
        param_factors = paramsFactorList                                       
    elif i_paramname == i_weights_STN_SNr:
        params_orig, paramname = STNSNr.w, 'weights_STN_SNr' 
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
    elif i_paramname == i_activateStopCp:
        params_orig, paramname = 1, 'activateStopCp'
        param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    elif i_paramname == i_deactivateStopArky:
        params_orig, paramname = 1, 'deactivateStopArky'
        param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])                         
    elif i_paramname == i_deactivateStopSTN:
        params_orig, paramname = 1, 'deactivateStopSTN'
        param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])                         
    elif i_paramname == i_deactivateStopCp:
        params_orig, paramname = 1, 'deactivateStopCp'
        param_factors = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])

    n_loop_cycles = len(param_factors)
    loop_params = param_factors# * params_orig
    
    np.save('data/cycle_params_'+paramname+'_id'+str(params['general_id'])+netNr+'.npy', [loop_params, paramname])
    print('\n\nSTART PARAMETER VARIATION')
    print('paramname = '+paramname+', variations:',loop_params,'\n\n') 

    ### LOOP OVER PARAMETERVARIATIONS ###
    for i_cycle in range(n_loop_cycles):
        print('\n\nSTART CYCLE '+paramname+' '+str(i_cycle)+'/'+str(n_loop_cycles),'\n\n')

        ### CHANGE PARAMETER
        if i_paramname == i_cortexGo_rates:             
            params['cortexGo_rates'] = params_orig * param_factors[i_cycle] # Caution - params[''] must be reset after loop over the parameter !!!
        elif i_paramname == i_cortexStop_rates:
            params['cortexStop_ratesAfterCue'] = params_orig * param_factors[i_cycle] 
        elif i_paramname == i_cortexPause_rates:
            params['cortexPause_rates'] = params_orig * param_factors[i_cycle] 
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
        elif i_paramname == i_weights_GPeProto_StrFSI:
            GPe_ProtoSTR_FSI.mod_factor = param_factors[i_cycle] * params['weights_GPeProto_StrFSI']                 
        elif i_paramname == i_weights_STN_SNr:                    
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
        elif i_paramname == i_activateStopCp:                    
            changeStopMechanism(0,0,param_factors[i_cycle])
        elif i_paramname == i_deactivateStopArky:                    
            changeStopMechanism(param_factors[i_cycle],1,1)              
        elif i_paramname == i_deactivateStopSTN:                    
            changeStopMechanism(1,param_factors[i_cycle],1)              
        elif i_paramname == i_deactivateStopCp:                    
            changeStopMechanism(1,1,param_factors[i_cycle])
        

        ### GO TRIALS ###
        print('\n\nSTART GO TRIALS')
        mode='GO'
        zaehler_go = 0
        for i in range(0,paramsS['trials']):
            ### RESET MODEL ###
            print('\nreset before trial...')
            reset(populations=True, projections=True, synapses=False, net_id=0)

            ### TRIAL START
            print('TRIAL START, cycle: '+str(i_cycle+1)+'/'+str(n_loop_cycles)+', trial: '+str(i))

            ### trial INITIALIZATION simulation to get stable state
            Cortex_G.rates = 0
            Cortex_S.rates = 0
            Stoppinput1.rates = 0

            print('simulate t_init and events...')
            simulate(params['t_init'])# t_init rest

            ### Integrator Reset
            Integrator.decision = 0
            Integrator.g_ampa = 0  
            IntegratorStop.decision = 0 

            ### calculate all eventTIMES
            GOCUE     = np.round(get_time(),1)
            STOPCUE   = GOCUE + params['t_SSD']
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
            tempMode=mode
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
            print('Integrator decision:',Integrator.decision[0])
            print('TRIAL END, store trial data...')

            ### GET SPIKE DATA OF TRIAL
            SD1_spikes = m_STR_D1.get('spike')
            SD2_spikes = m_STR_D2.get('spike')
            FSI_spikes = m_STR_FSI.get('spike')
            STN_spikes = m_STN.get('spike')
            Proto_spikes = m_GPe_Proto.get('spike')
            Proto2_spikes = m_GPe_Proto2.get('spike')
            Arky_spikes = m_GPe_Arky.get('spike')
            SNr_spikes = m_SNr.get('spike')
            SNrE_spikes = m_SNrE.get('spike')
            Thal_spikes = m_Thal.get('spike')
            Cortex_G_spikes = m_Cortex_G.get('spike')
            Cortex_S_spikes = m_Cortex_S.get('spike')
            Stoppinput1_spikes = m_Stoppinput1.get('spike')

            ### GET POPULATIONS RATE DATA OF TRIAL
            rate_data_SD1_currtrial = custom_poprate(m_STR_D1, SD1_spikes, params['general_msSmooth'] )
            rate_data_SD2_currtrial = custom_poprate(m_STR_D2, SD2_spikes, params['general_msSmooth'])
            rate_data_FSI_currtrial = custom_poprate(m_STR_FSI, FSI_spikes, params['general_msSmooth'])
            rate_data_STN_currtrial = custom_poprate(m_STN, STN_spikes, params['general_msSmooth'])
            rate_data_GPeProto2_currtrial = custom_poprate(m_GPe_Proto2, Proto2_spikes, params['general_msSmooth'])
            rate_data_GPeProto_currtrial = custom_poprate(m_GPe_Proto, Proto_spikes, params['general_msSmooth'])
            rate_data_GPeArky_currtrial = custom_poprate(m_GPe_Arky, Arky_spikes, params['general_msSmooth'])
            rate_data_SNr_currtrial = custom_poprate(m_SNr, SNr_spikes, params['general_msSmooth'])
            rate_data_SNrE_currtrial = custom_poprate(m_SNrE, SNrE_spikes, params['general_msSmooth'])
            rate_data_Thal_currtrial = custom_poprate(m_Thal, Thal_spikes, params['general_msSmooth'])
            rate_data_Cortex_G_currtrial = custom_poprate(m_Cortex_G, Cortex_G_spikes, params['general_msSmooth'])
            rate_data_Cortex_S_currtrial = custom_poprate(m_Cortex_S, Cortex_S_spikes, params['general_msSmooth'])
            rate_data_Stoppinput1_currtrial = custom_poprate(m_Stoppinput1, Stoppinput1_spikes, params['general_msSmooth'])

            ### ADD RATE DATA OF TRIAL TO RATE DATA OF ALL GO TRIALS            
            rate_data_SD1_Go[i, : ] = get_poprate_aligned_onset(m_STR_D1, SD1_spikes, rate_data_SD1_Go[i, : ], rate_data_SD1_currtrial, dt())            
            rate_data_SD2_Go[i, : ] = get_poprate_aligned_onset(m_STR_D2, SD2_spikes, rate_data_SD2_Go[i, : ], rate_data_SD2_currtrial, dt())                        
            rate_data_FSI_Go[i, : ] = get_poprate_aligned_onset(m_STR_FSI, FSI_spikes, rate_data_FSI_Go[i, : ], rate_data_FSI_currtrial, dt())                        
            rate_data_STN_Go[i, : ] = get_poprate_aligned_onset(m_STN, STN_spikes, rate_data_STN_Go[i, : ], rate_data_STN_currtrial, dt())                                    
            rate_data_GPeProto_Go[i, : ] = get_poprate_aligned_onset(m_GPe_Proto, Proto_spikes, rate_data_GPeProto_Go[i, : ], rate_data_GPeProto_currtrial, dt())             
            rate_data_GPeProto2_Go[i, : ] = get_poprate_aligned_onset(m_GPe_Proto2, Proto2_spikes, rate_data_GPeProto2_Go[i, : ], rate_data_GPeProto2_currtrial, dt())
            rate_data_GPeArky_Go[i, : ] = get_poprate_aligned_onset(m_GPe_Arky, Arky_spikes, rate_data_GPeArky_Go[i, : ], rate_data_GPeArky_currtrial, dt())            
            rate_data_SNr_Go[i, : ] = get_poprate_aligned_onset(m_SNr, SNr_spikes, rate_data_SNr_Go[i, : ], rate_data_SNr_currtrial, dt())                                                
            rate_data_SNrE_Go[i, : ] = get_poprate_aligned_onset(m_SNrE, SNrE_spikes, rate_data_SNrE_Go[i, : ], rate_data_SNrE_currtrial, dt())                                                            
            rate_data_Thal_Go[i, : ] = get_poprate_aligned_onset(m_Thal, Thal_spikes, rate_data_Thal_Go[i, : ], rate_data_Thal_currtrial, dt())                         
            rate_data_Cortex_G_Go[i, : ] = get_poprate_aligned_onset(m_Cortex_G, Cortex_G_spikes, rate_data_Cortex_G_Go[i, : ], rate_data_Cortex_G_currtrial, dt())
            rate_data_Cortex_S_Go[i, : ] = get_poprate_aligned_onset(m_Cortex_S, Cortex_S_spikes, rate_data_Cortex_S_Go[i, : ], rate_data_Cortex_S_currtrial, dt())
            rate_data_Stoppinput1_Go[i, : ] = get_poprate_aligned_onset(m_Stoppinput1, Stoppinput1_spikes, rate_data_Stoppinput1_Go[i, : ], rate_data_Stoppinput1_currtrial, dt())            


            if Integrator.decision == -1 :
                t= get_current_step()
                selection[i] = Integrator.decision
                timeovertrial[i]=t
                zaehler_go = zaehler_go + 1              
            else:
                selection[i] = Integrator.decision
                timeovertrial[i]=t
            
        
        ### END GO TRIALS ###
        print('\nGO TRIALS FINISHED\nzaehler_go:',zaehler_go)
        
        ### SAVE GO TRIALS DATA
        print('save Go trials data...\n')
        np.save('data/selection_Go'+netNr+'.npy', selection)
        np.save('data/Timeovertrials_Go'+netNr+'.npy', timeovertrial)
        mInt_go = m_Int_ampa.get('g_ampa')
        np.save('data/Integrator_ampa_Go_'+netNr+'_id'+str(params['general_id'])+'.npy', mInt_go)
        spike_times, ranks = m_Int_ampa.raster_plot(m_Int_ampa.get('spike'))
        np.save('data/Integrator_spike_Go'+netNr+'_cycle'+str(i_cycle)+'.npy', [spike_times, ranks])

        np.save('data/SD1_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_SD1_Go, 0), np.nanstd(rate_data_SD1_Go, 0)])
        np.save('data/SD2_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_SD2_Go, 0), np.nanstd(rate_data_SD2_Go, 0)])
        np.save('data/FSI_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_FSI_Go, 0), np.nanstd(rate_data_FSI_Go, 0)])
        np.save('data/STN_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_STN_Go, 0), np.nanstd(rate_data_STN_Go, 0)])
        np.save('data/GPeProto_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_GPeProto_Go, 0), np.nanstd(rate_data_GPeProto_Go, 0)])
        np.save('data/GPeProto2_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_GPeProto2_Go, 0), np.nanstd(rate_data_GPeProto2_Go, 0)])            
        np.save('data/GPeArky_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_GPeArky_Go, 0), np.nanstd(rate_data_GPeArky_Go, 0)])        
        np.save('data/SNr_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_SNr_Go, 0), np.nanstd(rate_data_SNr_Go, 0)])
        np.save('data/SNrE_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_SNrE_Go, 0), np.nanstd(rate_data_SNrE_Go, 0)])        
        np.save('data/Thal_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_Thal_Go, 0), np.nanstd(rate_data_Thal_Go, 0)])        
        np.save('data/Cortex_G_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_Cortex_G_Go, 0), np.nanstd(rate_data_Cortex_G_Go, 0)])        
        np.save('data/Cortex_S_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_Cortex_S_Go, 0), np.nanstd(rate_data_Cortex_S_Go, 0)])                
        np.save('data/Stoppinput1_rate_Go_mean_std'+netNr+'.npy', [np.nanmean(rate_data_Stoppinput1_Go, 0), np.nanstd(rate_data_Stoppinput1_Go, 0)])                        

                    


        ### STOP TRIALS ###
        print('\nSTART STOP TRIALS')
        mode='STOP'
        for i in range (0,paramsS['trials']):
            ### RESET MODEL ###
            print('\nreset before trial...')
            reset(populations=True, projections=True, synapses=False, net_id=0)

            ### TRIAL START
            print('TRIAL START, cycle: '+str(i_cycle+1)+'/'+str(n_loop_cycles)+', trial: '+str(i))

            ### trial INITIALIZATION simulation to get stable state
            Cortex_G.rates = 0
            Cortex_S.rates = 0
            Stoppinput1.rates = 0

            print('simulate t_init and events...')
            simulate(params['t_init'])# t_init rest

            ### Integrator Reset
            Integrator.decision = 0
            Integrator.g_ampa = 0  
            IntegratorStop.decision = 0 

            ### calculate all eventTIMES
            GOCUE     = np.round(get_time(),1)
            STOPCUE   = GOCUE + params['t_SSD']
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
            tempMode=mode
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
                if t == STN2ON and mode=='STOP':
                    Stoppinput1.rates = params['cortexPause_ratesSecondRespMod']*params['cortexPause_rates'] + params['cortexPause_ratesSD'] * randn_trial_Pause
                    #print('STN2ON',STN2ON)
                if t == STN2OFF and mode=='STOP':
                    Stoppinput1.rates = 0
                    #print('STN2OFF',STN2OFF)
                if t == STOPON and mode=='STOP':
                    Cortex_S.rates = params['cortexStop_ratesAfterCue'] + params['cortexStop_ratesSD'] * randn_trial_S * (params['cortexStop_ratesAfterCue'] > 0)
                    #print('STOPON',STOPON)
                if t == motorSTOP:
                    Cortex_S.rates = params['cortexStop_ratesAfterAction'] + params['cortexStop_ratesSD'] * randn_trial_S * (params['cortexStop_ratesAfterAction'] > 0)
                    motorSTOPOFF = motorSTOP + params['t_cortexStopDurationAfterAction']
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
                        motorSTOP = t + params['t_delayStopAfterAction']
                        if t<STOPCUE and tempMode=='STOP':
                            tempMode='GO'
                    if responseProto2 == -1 and GOOFFResponse == 0 and ((t>STOPON and tempMode=='STOP') or motorResponse==1):
                        GOOFFResponse=1
                        GOOFF = t
            ### TRIAL END
            print('Integrator decision:',Integrator.decision[0])
            print('TRIAL END, store trial data...')
            
            ### GET SPIKE DATA OF TRIAL
            SD1_spikes = m_STR_D1.get('spike')
            SD2_spikes = m_STR_D2.get('spike')
            FSI_spikes = m_STR_FSI.get('spike')
            STN_spikes = m_STN.get('spike')
            Proto_spikes = m_GPe_Proto.get('spike')
            Proto2_spikes = m_GPe_Proto2.get('spike')
            Arky_spikes = m_GPe_Arky.get('spike')
            SNr_spikes = m_SNr.get('spike')
            SNrE_spikes = m_SNrE.get('spike')
            Thal_spikes = m_Thal.get('spike')
            Cortex_G_spikes = m_Cortex_G.get('spike')
            Cortex_S_spikes = m_Cortex_S.get('spike')
            Stoppinput1_spikes = m_Stoppinput1.get('spike')

            ### GET POPULATIONS RATE DATA OF TRIAL
            rate_data_SD1_currtrial = custom_poprate(m_STR_D1, SD1_spikes, params['general_msSmooth'])
            rate_data_SD2_currtrial = custom_poprate(m_STR_D2, SD2_spikes, params['general_msSmooth'])
            rate_data_FSI_currtrial = custom_poprate(m_STR_FSI, FSI_spikes, params['general_msSmooth'])
            rate_data_STN_currtrial = custom_poprate(m_STN, STN_spikes, params['general_msSmooth'])
            rate_data_GPeProto_currtrial = custom_poprate(m_GPe_Proto, Proto_spikes, params['general_msSmooth'])
            rate_data_GPeProto2_currtrial = custom_poprate(m_GPe_Proto2, Proto2_spikes, params['general_msSmooth'])
            rate_data_GPeArky_currtrial = custom_poprate(m_GPe_Arky, Arky_spikes, params['general_msSmooth'])
            rate_data_SNr_currtrial = custom_poprate(m_SNr, SNr_spikes, params['general_msSmooth'])
            rate_data_SNrE_currtrial = custom_poprate(m_SNrE, SNrE_spikes, params['general_msSmooth'])
            rate_data_Thal_currtrial = custom_poprate(m_Thal, Thal_spikes, params['general_msSmooth'])
            rate_data_Cortex_G_currtrial = custom_poprate(m_Cortex_G, Cortex_G_spikes, params['general_msSmooth'])
            rate_data_Cortex_S_currtrial = custom_poprate(m_Cortex_S, Cortex_S_spikes, params['general_msSmooth'])
            rate_data_Stoppinput1_currtrial = custom_poprate(m_Stoppinput1, Stoppinput1_spikes, params['general_msSmooth'])            

            ### ADD RATE DATA OF TRIAL TO RATE DATA OF ALL STOP TRIALS
            rate_data_SD1_Stop[i, : ] = get_poprate_aligned_onset(m_STR_D1, SD1_spikes, rate_data_SD1_Stop[i, : ], rate_data_SD1_currtrial, dt())            
            rate_data_SD2_Stop[i, : ] = get_poprate_aligned_onset(m_STR_D2, SD2_spikes, rate_data_SD2_Stop[i, : ], rate_data_SD2_currtrial, dt())                        
            rate_data_FSI_Stop[i, : ] = get_poprate_aligned_onset(m_STR_FSI, FSI_spikes, rate_data_FSI_Stop[i, : ], rate_data_FSI_currtrial, dt())                        
            rate_data_STN_Stop[i, : ] = get_poprate_aligned_onset(m_STN, STN_spikes, rate_data_STN_Stop[i, : ], rate_data_STN_currtrial, dt())                                    
            rate_data_GPeProto_Stop[i, : ] = get_poprate_aligned_onset(m_GPe_Proto, Proto_spikes, rate_data_GPeProto_Stop[i, : ], rate_data_GPeProto_currtrial, dt())             
            rate_data_GPeProto2_Stop[i, : ] = get_poprate_aligned_onset(m_GPe_Proto2, Proto2_spikes, rate_data_GPeProto2_Stop[i, : ], rate_data_GPeProto2_currtrial, dt()) 
            rate_data_GPeArky_Stop[i, : ] = get_poprate_aligned_onset(m_GPe_Arky, Arky_spikes, rate_data_GPeArky_Stop[i, : ], rate_data_GPeArky_currtrial, dt())            
            rate_data_SNr_Stop[i, : ] = get_poprate_aligned_onset(m_SNr, SNr_spikes, rate_data_SNr_Stop[i, : ], rate_data_SNr_currtrial, dt())                                                
            rate_data_SNrE_Stop[i, : ] = get_poprate_aligned_onset(m_SNrE, SNrE_spikes, rate_data_SNrE_Stop[i, : ], rate_data_SNrE_currtrial, dt())                                                            
            rate_data_Thal_Stop[i, : ] = get_poprate_aligned_onset(m_Thal, Thal_spikes, rate_data_Thal_Stop[i, : ], rate_data_Thal_currtrial, dt())                         
            rate_data_Cortex_G_Stop[i, : ] = get_poprate_aligned_onset(m_Cortex_G, Cortex_G_spikes, rate_data_Cortex_G_Stop[i, : ], rate_data_Cortex_G_currtrial, dt())
            rate_data_Cortex_S_Stop[i, : ] = get_poprate_aligned_onset(m_Cortex_S, Cortex_S_spikes, rate_data_Cortex_S_Stop[i, : ], rate_data_Cortex_S_currtrial, dt())
            rate_data_Stoppinput1_Stop[i, : ] = get_poprate_aligned_onset(m_Stoppinput1, Stoppinput1_spikes, rate_data_Stoppinput1_Stop[i, : ], rate_data_Stoppinput1_currtrial, dt())


            if Integrator.decision == -1 :
                t= get_current_step()
                selection[i] = Integrator.decision
                timeovertrial[i]=t
                zaehler_go = zaehler_go + 1                                         
            else:
                selection[i] = Integrator.decision
                timeovertrial[i]=t

        ### END OF STOP TRIALS ###
        print('zaehler_go:',zaehler_go)

        
        ### SAVE DATA OF STOP TRIALS
        np.save('data/selection_Stop'+netNr+'.npy', selection)
        np.save('data/Timeovertrials_Stop'+netNr+'.npy', timeovertrial)
        mInt_stop = m_Int_ampa.get('g_ampa')
        np.save('data/Integrator_ampa_Stop_'+netNr+'_id'+str(params['general_id'])+'.npy', mInt_stop)            
        spike_times, ranks = m_Int_ampa.raster_plot(m_Int_ampa.get('spike'))
        np.save('data/Integrator_spike_Stop'+netNr+'_cycle'+str(i_cycle)+'.npy', [spike_times, ranks]) 

        np.save('data/SD1_rate_Stop_mean_std'                                           + netNr                      + '.npy', [np.nanmean(rate_data_SD1_Stop, 0), np.nanstd(rate_data_SD1_Stop, 0)])
        np.save('data/SD2_rate_Stop_mean_std'                                           + netNr                      + '.npy', [np.nanmean(rate_data_SD2_Stop, 0), np.nanstd(rate_data_SD2_Stop, 0)])
        np.save('data/FSI_rate_Stop_mean_std'                                           + netNr                      + '.npy', [np.nanmean(rate_data_FSI_Stop, 0), np.nanstd(rate_data_FSI_Stop, 0)])             
        np.save('data/STN_rate_Stop_mean_std'                                           + netNr                      + '.npy', [np.nanmean(rate_data_STN_Stop, 0), np.nanstd(rate_data_STN_Stop, 0)])
        np.save('data/GPeProto_rate_Stop_mean_std'                                      + netNr                      + '.npy', [np.nanmean(rate_data_GPeProto_Stop, 0), np.nanstd(rate_data_GPeProto_Stop, 0)])
        np.save('data/GPeProto2_rate_Stop_mean_std'                                     + netNr                      + '.npy', [np.nanmean(rate_data_GPeProto2_Stop, 0), np.nanstd(rate_data_GPeProto2_Stop, 0)])            
        np.save('data/GPeArky_rate_Stop_mean_std'                                       + netNr                      + '.npy', [np.nanmean(rate_data_GPeArky_Stop, 0), np.nanstd(rate_data_GPeArky_Stop, 0)])   
        np.save('data/SNr_rate_Stop_mean_std'                                           + netNr                      + '.npy', [np.nanmean(rate_data_SNr_Stop, 0), np.nanstd(rate_data_SNr_Stop, 0)])
        np.save('data/SNrE_rate_Stop_mean_std'                                          + netNr                      + '.npy', [np.nanmean(rate_data_SNrE_Stop, 0), np.nanstd(rate_data_SNrE_Stop, 0)])        
        np.save('data/Thal_rate_Stop_mean_std'                                          + netNr                      + '.npy', [np.nanmean(rate_data_Thal_Stop, 0), np.nanstd(rate_data_Thal_Stop, 0)])        
        np.save('data/Cortex_G_rate_Stop_mean_std'                                      + netNr                      + '.npy', [np.nanmean(rate_data_Cortex_G_Stop, 0), np.nanstd(rate_data_Cortex_G_Stop, 0)])
        np.save('data/Cortex_S_rate_Stop_mean_std'                                      + netNr                      + '.npy', [np.nanmean(rate_data_Cortex_S_Stop, 0), np.nanstd(rate_data_Cortex_S_Stop, 0)])
        np.save('data/Stoppinput1_rate_Stop_mean_std'                                   + netNr                      + '.npy', [np.nanmean(rate_data_Stoppinput1_Stop, 0), np.nanstd(rate_data_Stoppinput1_Stop, 0)])        

        np.save('data/STN_rate_Stop_mean_std_'        + str(params['general_id']) + '_' + netNr + '_' + str(i_cycle) + '.npy', [np.nanmean(rate_data_STN_Stop, 0), np.nanstd(rate_data_STN_Stop, 0)])
        np.save('data/GPeProto_rate_Stop_mean_std_'   + str(params['general_id']) + '_' + netNr + '_' + str(i_cycle) + '.npy', [np.nanmean(rate_data_GPeProto_Stop, 0), np.nanstd(rate_data_GPeProto_Stop, 0)])            
        np.save('data/GPeProto2_rate_Stop_mean_std_'  + str(params['general_id']) + '_' + netNr + '_' + str(i_cycle) + '.npy', [np.nanmean(rate_data_GPeProto2_Stop, 0), np.nanstd(rate_data_GPeProto2_Stop, 0)])                        
        np.save('data/GPeArky_rate_Stop_mean_std_'    + str(params['general_id']) + '_' + netNr + '_' + str(i_cycle) + '.npy', [np.nanmean(rate_data_GPeArky_Stop, 0), np.nanstd(rate_data_GPeArky_Stop, 0)])               
        np.save('data/SNr_rate_Stop_mean_std_'        + str(params['general_id']) + '_' + netNr + '_' + str(i_cycle) + '.npy', [np.nanmean(rate_data_SNr_Stop, 0), np.nanstd(rate_data_SNr_Stop, 0)])            
        
        
        np.save('data/STN_rate_Stop_alltrials'+netNr+'.npy', rate_data_STN_Stop)         
        np.save('data/SNr_rate_Stop_alltrials'+netNr+'.npy', rate_data_SNr_Stop) 
        np.save('data/GPeProto_rate_Stop_alltrials'+netNr+'.npy', rate_data_GPeProto_Stop) 
        np.save('data/GPeArky_rate_Stop_alltrials'+netNr+'.npy', rate_data_GPeArky_Stop)         

        ### GET AND SAVE CORRECT AND FAILED STOP DATA
        StrD1_meanrate_FailedStop, StrD1_std_FailedStop, StrD1_meanrate_CorrectStop, StrD1_std_CorrectStop              = calc_meanrate_std_failed_correct(rate_data_SD1_Stop, mInt_stop, Integrator.threshold, paramsS['trials']) 
        StrD2_meanrate_FailedStop, StrD2_std_FailedStop, StrD2_meanrate_CorrectStop, StrD2_std_CorrectStop              = calc_meanrate_std_failed_correct(rate_data_SD2_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])        
        StrFSI_meanrate_FailedStop, StrFSI_std_FailedStop, StrFSI_meanrate_CorrectStop, StrFSI_std_CorrectStop          = calc_meanrate_std_failed_correct(rate_data_FSI_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])        
        STN_meanrate_FailedStop, STN_std_FailedStop, STN_meanrate_CorrectStop, STN_std_CorrectStop                      = calc_meanrate_std_failed_correct(rate_data_STN_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])         
        Proto_meanrate_FailedStop, Proto_std_FailedStop, Proto_meanrate_CorrectStop, Proto_std_CorrectStop              = calc_meanrate_std_failed_correct(rate_data_GPeProto_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])                 
        Proto2_meanrate_FailedStop, Proto2_std_FailedStop, Proto2_meanrate_CorrectStop, Proto2_std_CorrectStop          = calc_meanrate_std_failed_correct(rate_data_GPeProto2_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])            
        Arky_meanrate_FailedStop, Arky_std_FailedStop, Arky_meanrate_CorrectStop, Arky_std_CorrectStop                  = calc_meanrate_std_failed_correct(rate_data_GPeArky_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        SNr_meanrate_FailedStop, SNr_std_FailedStop, SNr_meanrate_CorrectStop, SNr_std_CorrectStop                      = calc_meanrate_std_failed_correct(rate_data_SNr_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        SNrE_meanrate_FailedStop, SNrE_std_FailedStop, SNrE_meanrate_CorrectStop, SNrE_std_CorrectStop                  = calc_meanrate_std_failed_correct(rate_data_SNrE_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        Thal_meanrate_FailedStop, Thal_std_FailedStop, Thal_meanrate_CorrectStop, Thal_std_CorrectStop                  = calc_meanrate_std_failed_correct(rate_data_Thal_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        Cortex_G_meanrate_FailedStop, Cortex_G_std_FailedStop, Cortex_G_meanrate_CorrectStop, Cortex_G_std_CorrectStop  = calc_meanrate_std_failed_correct(rate_data_Cortex_G_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        Cortex_S_meanrate_FailedStop, Cortex_S_std_FailedStop, Cortex_S_meanrate_CorrectStop, Cortex_S_std_CorrectStop  = calc_meanrate_std_failed_correct(rate_data_Cortex_S_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        Stoppinput1_meanrate_FailedStop, Stoppinput1_std_FailedStop, Stoppinput1_meanrate_CorrectStop, Stoppinput1_std_CorrectStop = calc_meanrate_std_failed_correct(rate_data_Stoppinput1_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])        

        np.save('data/SD1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [StrD1_meanrate_FailedStop, StrD1_std_FailedStop, StrD1_meanrate_CorrectStop, StrD1_std_CorrectStop])
        np.save('data/SD2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [StrD2_meanrate_FailedStop, StrD2_std_FailedStop, StrD2_meanrate_CorrectStop, StrD2_std_CorrectStop])
        np.save('data/FSI_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [StrFSI_meanrate_FailedStop, StrFSI_std_FailedStop, StrFSI_meanrate_CorrectStop, StrFSI_std_CorrectStop])
        np.save('data/STN_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [STN_meanrate_FailedStop, STN_std_FailedStop, STN_meanrate_CorrectStop, STN_std_CorrectStop])
        np.save('data/Proto_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [Proto_meanrate_FailedStop, Proto_std_FailedStop, Proto_meanrate_CorrectStop, Proto_std_CorrectStop])
        np.save('data/Proto2_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [Proto2_meanrate_FailedStop, Proto2_std_FailedStop, Proto2_meanrate_CorrectStop, Proto2_std_CorrectStop])            
        np.save('data/Arky_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [Arky_meanrate_FailedStop, Arky_std_FailedStop, Arky_meanrate_CorrectStop, Arky_std_CorrectStop])
        np.save('data/SNr_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [SNr_meanrate_FailedStop, SNr_std_FailedStop, SNr_meanrate_CorrectStop, SNr_std_CorrectStop])
        np.save('data/SNrE_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [SNrE_meanrate_FailedStop, SNrE_std_FailedStop, SNrE_meanrate_CorrectStop, SNrE_std_CorrectStop])        
        np.save('data/Thal_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [Thal_meanrate_FailedStop, Thal_std_FailedStop, Thal_meanrate_CorrectStop, Thal_std_CorrectStop])        
        np.save('data/Cortex_G_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [Cortex_G_meanrate_FailedStop, Cortex_G_std_FailedStop, Cortex_G_meanrate_CorrectStop, Cortex_G_std_CorrectStop])
        np.save('data/Cortex_S_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [Cortex_S_meanrate_FailedStop, Cortex_S_std_FailedStop, Cortex_S_meanrate_CorrectStop, Cortex_S_std_CorrectStop])        
        np.save('data/Stoppinput1_meanrate_std_Failed_Correct_Stop'+netNr+'.npy', [Stoppinput1_meanrate_FailedStop, Stoppinput1_std_FailedStop, Stoppinput1_meanrate_CorrectStop, Stoppinput1_std_CorrectStop])                
        
        ### GET AND SAVE FAST AND SLOW GO DATA
        GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo     = calc_meanrate_std_Fast_Slow(rate_data_GPeArky_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo    = calc_meanrate_std_Fast_Slow(rate_data_GPeProto_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        StrD1_meanrate_FastGo,      StrD1_std_FastGo,       StrD1_meanrate_SlowGo,      StrD1_std_SlowGo        = calc_meanrate_std_Fast_Slow(rate_data_SD1_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])                 
        StrD2_meanrate_FastGo,      StrD2_std_FastGo,       StrD2_meanrate_SlowGo,      StrD2_std_SlowGo        = calc_meanrate_std_Fast_Slow(rate_data_SD2_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        STN_meanrate_FastGo,        STN_std_FastGo,         STN_meanrate_SlowGo,        STN_std_SlowGo          = calc_meanrate_std_Fast_Slow(rate_data_STN_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        SNr_meanrate_FastGo,        SNr_std_FastGo,         SNr_meanrate_SlowGo,        SNr_std_SlowGo          = calc_meanrate_std_Fast_Slow(rate_data_SNr_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        Thal_meanrate_FastGo,       Thal_std_FastGo,        Thal_meanrate_SlowGo,       Thal_std_SlowGo         = calc_meanrate_std_Fast_Slow(rate_data_Thal_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        Cortex_G_meanrate_FastGo,   Cortex_G_std_FastGo,    Cortex_G_meanrate_SlowGo,   Cortex_G_std_SlowGo     = calc_meanrate_std_Fast_Slow(rate_data_Cortex_G_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        GPe_Proto2_meanrate_FastGo, GPe_Proto2_std_FastGo,  GPe_Proto2_meanrate_SlowGo, GPe_Proto2_std_SlowGo   = calc_meanrate_std_Fast_Slow(rate_data_GPeProto2_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        PauseInput_meanrate_FastGo, PauseInput_std_FastGo,  PauseInput_meanrate_SlowGo, PauseInput_std_SlowGo   = calc_meanrate_std_Fast_Slow(rate_data_Stoppinput1_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        Cortex_S_meanrate_FastGo,   Cortex_S_std_FastGo,    Cortex_S_meanrate_SlowGo,   Cortex_S_std_SlowGo     = calc_meanrate_std_Fast_Slow(rate_data_Cortex_S_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])                
        StrFSI_meanrate_FastGo,     StrFSI_std_FastGo,      StrFSI_meanrate_SlowGo,     StrFSI_std_SlowGo       = calc_meanrate_std_Fast_Slow(rate_data_FSI_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])                 

        np.save('data/GPe_Arky_meanrate_std_Fast-Slow_Go'+netNr+'.npy',   [GPe_Arky_meanrate_FastGo,   GPe_Arky_std_FastGo,    GPe_Arky_meanrate_SlowGo,   GPe_Arky_std_SlowGo])
        np.save('data/GPe_Proto_meanrate_std_Fast-Slow_Go'+netNr+'.npy',  [GPe_Proto_meanrate_FastGo,  GPe_Proto_std_FastGo,   GPe_Proto_meanrate_SlowGo,  GPe_Proto_std_SlowGo])
        np.save('data/SD1_meanrate_std_Fast-Slow_Go'+netNr+'.npy',        [StrD1_meanrate_FastGo,      StrD1_std_FastGo,       StrD1_meanrate_SlowGo,      StrD1_std_SlowGo])
        np.save('data/SD2_meanrate_std_Fast-Slow_Go'+netNr+'.npy',        [StrD2_meanrate_FastGo,      StrD2_std_FastGo,       StrD2_meanrate_SlowGo,      StrD2_std_SlowGo])            
        np.save('data/STN_meanrate_std_Fast-Slow_Go'+netNr+'.npy',        [STN_meanrate_FastGo,        STN_std_FastGo,         STN_meanrate_SlowGo,        STN_std_SlowGo])
        np.save('data/SNr_meanrate_std_Fast-Slow_Go'+netNr+'.npy',        [SNr_meanrate_FastGo,        SNr_std_FastGo,         SNr_meanrate_SlowGo,        SNr_std_SlowGo])
        np.save('data/Thal_meanrate_std_Fast-Slow_Go'+netNr+'.npy',       [Thal_meanrate_FastGo,       Thal_std_FastGo,        Thal_meanrate_SlowGo,       Thal_std_SlowGo])
        np.save('data/Cortex_G_meanrate_std_Fast-Slow_Go'+netNr+'.npy',   [Cortex_G_meanrate_FastGo,   Cortex_G_std_FastGo,    Cortex_G_meanrate_SlowGo,   Cortex_G_std_SlowGo])
        np.save('data/GPe_Proto2_meanrate_std_Fast-Slow_Go'+netNr+'.npy', [GPe_Proto2_meanrate_FastGo, GPe_Proto2_std_FastGo,  GPe_Proto2_meanrate_SlowGo, GPe_Proto2_std_SlowGo])
        np.save('data/PauseInput_meanrate_std_Fast-Slow_Go'+netNr+'.npy', [PauseInput_meanrate_FastGo, PauseInput_std_FastGo,  PauseInput_meanrate_SlowGo, PauseInput_std_SlowGo])
        np.save('data/Cortex_S_meanrate_std_Fast-Slow_Go'+netNr+'.npy',   [Cortex_S_meanrate_FastGo,   Cortex_S_std_FastGo,    Cortex_S_meanrate_SlowGo,   Cortex_S_std_SlowGo])                        
        np.save('data/FSI_meanrate_std_Fast-Slow_Go'+netNr+'.npy',        [StrFSI_meanrate_FastGo,     StrFSI_std_FastGo,      StrFSI_meanrate_SlowGo,     StrFSI_std_SlowGo])

        ### GET RATE DATA AND PVALUES FOR FAILED VS CORRECT STOP
        t_stopCue = int(params['t_init'] + params['t_SSD'])
        t_min = int((t_stopCue - params['sigTestRateComparisonFailedCorrect_tMin'])/dt())   
        t_max = int((t_stopCue + params['sigTestRateComparisonFailedCorrect_tMax'])/dt())      
        binsize_ms = int(params['sigTestRateComparisonFailedCorrect_binSize'] / dt())
        pvalue_times = range(t_min, t_max + 1, binsize_ms)
        SD1_rates_FailedStop, SD1_rates_CorrectStop = get_rates_failed_correct(rate_data_SD1_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        SD2_rates_FailedStop, SD2_rates_CorrectStop = get_rates_failed_correct(rate_data_SD2_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])        
        STN_rates_FailedStop, STN_rates_CorrectStop = get_rates_failed_correct(rate_data_STN_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        GPe_Proto_rates_FailedStop, GPe_Proto_rates_CorrectStop = get_rates_failed_correct(rate_data_GPeProto_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        GPe_Proto2_rates_FailedStop, GPe_Proto2_rates_CorrectStop = get_rates_failed_correct(rate_data_GPeProto2_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])        
        GPe_Arky_rates_FailedStop, GPe_Arky_rates_CorrectStop = get_rates_failed_correct(rate_data_GPeArky_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])
        SNr_rates_FailedStop, SNr_rates_CorrectStop = get_rates_failed_correct(rate_data_SNr_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])        
        Thal_rates_FailedStop, Thal_rates_CorrectStop = get_rates_failed_correct(rate_data_Thal_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])        
        CortexG_rates_FailedStop, CortexG_rates_CorrectStop = get_rates_failed_correct(rate_data_Cortex_G_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])                
        CortexS_rates_FailedStop, CortexS_rates_CorrectStop = get_rates_failed_correct(rate_data_Cortex_S_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])                        
        FSI_rates_FailedStop, FSI_rates_CorrectStop = get_rates_failed_correct(rate_data_FSI_Stop, mInt_stop, Integrator.threshold, paramsS['trials'])        

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
  
        #p_ind_arky, p_ind_SD1, p_ind_STN, p_ind_Proto, p_ind_Proto2, p_ind_SNr, p_ind_thal, p_ind_SD2, p_ind_CortexG, p_ind_CortexS, p_ind_FSI = range(11)  
        np.save('data/p_value_list_times_'+str(params['general_id'])+netNr+'.npy', [pval_all, pvalue_times], allow_pickle=True)

        ### GET RATE DATA AND PVALUES FOR FAILED STOP VS FAST GO
        t_stopCue = int(params['t_init'] + params['t_SSD'])
        t_min = int((params['t_init'] - params['sigTestRateComparisonFailedFast_tMin'])/dt())   
        t_max = int((params['t_init'] + params['sigTestRateComparisonFailedFast_tMax'])/dt()) 
        binsize_ms = int(params['sigTestRateComparisonFailedFast_binSize'] / dt())
        pvalue_times = range(t_min, t_max + 1, binsize_ms) 
        SD1_rates_allGo,        SD1_rates_fastGo,           SD1_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_SD1_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        SD2_rates_allGo,        SD2_rates_fastGo,           SD2_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_SD2_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        STN_rates_allGo,        STN_rates_fastGo,           STN_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_STN_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        GPe_Proto_rates_allGo,  GPe_Proto_rates_fastGo,     GPe_Proto_rates_slowGo  = get_rates_allGo_fastGo_slowGo(rate_data_GPeProto_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        GPe_Proto2_rates_allGo, GPe_Proto2_rates_fastGo,    GPe_Proto2_rates_slowGo = get_rates_allGo_fastGo_slowGo(rate_data_GPeProto2_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        GPe_Arky_rates_allGo,   GPe_Arky_rates_fastGo,      GPe_Arky_rates_slowGo   = get_rates_allGo_fastGo_slowGo(rate_data_GPeArky_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        SNr_rates_allGo,        SNr_rates_fastGo,           SNr_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_SNr_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        Thal_rates_allGo,       Thal_rates_fastGo,          Thal_rates_slowGo       = get_rates_allGo_fastGo_slowGo(rate_data_Thal_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        CortexG_rates_allGo,    CortexG_rates_fastGo,       CortexG_rates_slowGo    = get_rates_allGo_fastGo_slowGo(rate_data_Cortex_G_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        CortexS_rates_allGo,    CortexS_rates_fastGo,       CortexS_rates_slowGo    = get_rates_allGo_fastGo_slowGo(rate_data_Cortex_S_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])
        FSI_rates_allGo,        FSI_rates_fastGo,           FSI_rates_slowGo        = get_rates_allGo_fastGo_slowGo(rate_data_FSI_Go, mInt_go, mInt_stop, Integrator.threshold, paramsS['trials'])        

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

            np.save('data/p_value_list_'+nameList[Groups_Idx]+'_times_'+str(params['general_id'])+netNr+'.npy', [pval_all, pvalue_times], allow_pickle=True)

        ### SAVE MEAN RATES 200ms AFTER STOP CUE
        np.save('data/SD1_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy', np.nanmean(rate_data_SD1_Stop[:, t_stopCue : t_stopCue + 200], 1))
        np.save('data/SD2_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy', np.nanmean(rate_data_SD2_Stop[:, t_stopCue : t_stopCue + 200], 1))
        np.save('data/STN_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy', np.nanmean(rate_data_STN_Stop[:, t_stopCue : t_stopCue + 200], 1))
        np.save('data/GPeProto_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy', np.nanmean(rate_data_GPeProto_Stop[:, t_stopCue : t_stopCue + 200], 1))        
        np.save('data/GPeArky_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy', np.nanmean(rate_data_GPeArky_Stop[:, t_stopCue : t_stopCue + 200], 1))        
        np.save('data/SNr_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy', np.nanmean(rate_data_SNr_Stop[:, t_stopCue : t_stopCue + 200], 1))
        np.save('data/Thal_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy', np.nanmean(rate_data_Thal_Stop[:, t_stopCue : t_stopCue + 200], 1))        
        np.save('data/Cortex_G_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy', np.nanmean(rate_data_Cortex_G_Stop[:, t_stopCue : t_stopCue + 200], 1))        
        np.save('data/Cortex_S_rate_Stop_tempmean_'+str(params['general_id'])+netNr+'.npy', np.nanmean(rate_data_Cortex_S_Stop[:, t_stopCue : t_stopCue + 200], 1))                

        ### GET FAILED AND CORRECT GO AND STOP TRIALS, SAVE REAKTION TIME DATA
        print("Calculating reaction times...")
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
            if np.nanmax(rsp_mInt_stop[:, i_trial]) >= Integrator.threshold: 
                RT_Stop[i_trial] = np.nonzero(rsp_mInt_stop[:, i_trial] >= Integrator.threshold)[0][0]


        nz_Go = np.nonzero(np.isnan(RT_Go)==False)
        nz_Stop = np.nonzero(np.isnan(RT_Stop)==False)
        counts_Go, bins_Go = np.histogram(RT_Go[nz_Go] * dt(), 10) # 
        counts_Stop, bins_Stop = np.histogram(RT_Stop[nz_Stop] * dt(), 10) # 
        
        mean_CorrectGo = np.round( (np.nanmean(RT_Go[nz_Go]) - params['t_init']/dt())*dt(), 1)
        mean_FailedStop = np.round( (np.nanmean(RT_Stop[nz_Stop]) - params['t_init']/dt())*dt(), 1)

        results_RT = {}
        results_RT['nFailedGoTrials'] = len(nz_FailedGo)
        results_RT['nCorrectGoTrials'] = len(nz_CorrectGo)
        results_RT['nFailedStopTrials'] = len(nz_FailedStop)
        results_RT['nCorrectStopTrials'] = len(nz_CorrectStop)
        results_RT['meanRT_CorrectGo'] = mean_CorrectGo
        results_RT['meanRT_FailedStop'] = mean_FailedStop

        np.save('data/resultsRT_'+netNr+'_param_'+paramname+'_cycle'+str(i_cycle)+'id'+str(params['general_id'])+'.npy', results_RT) 

        ### RESET ALL MODIFIED PARAMETERS BEFOR NEW CYCLE
        params['cortexGo_rates'] = params_orig_CortexGo
        params['cortexStop_ratesAfterCue'] = params_orig_CortexStop
        params['cortexPause_rates'] = params_orig_CortexPause 
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

        print('\n')
        print(Cortex_SGPe_Arky.mod_factor[0], params['cortexPause_ratesSecondRespMod'], Cortex_SGPe_Proto2.mod_factor[0])
        changeStopMechanism(0.5,0.5,0.5)
        print(Cortex_SGPe_Arky.mod_factor[0], params['cortexPause_ratesSecondRespMod'], Cortex_SGPe_Proto2.mod_factor[0])
        


print('\nFINISHED\n')






