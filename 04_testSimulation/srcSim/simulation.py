import sys
sys.path.insert(1, '../..')# to import mainScripts

from ANNarchy import*
setup(dt=0.1)
setup(num_threads=1)

from model_OM_LG_mainScripts.sim_params import params
from model_OM_LG_mainScripts.neuronmodels import Izhikevich_neuron, Izhikevich_STR_neuron,STR_FSI_neuron, Integrator_neuron, Poisson_neuron, FixedSynapse
from model_OM_LG_mainScripts.populations import Stoppinput1, Cortex_S, Cortex_G, STR_D1, STR_D2, STN, SNr, GPe_Proto, Thal, Integrator, IntegratorStop, GPeE, SNrE, STNE, STR_FSI, STRE, GPe_Arky, TestThalnoise, population_size, GPe_Proto2
from model_OM_LG_mainScripts.projections import Stoppinput1STN, Cortex_GSTR_D1, Cortex_GSTR_D2, Cortex_GSTR_FSI, Cortex_GThal, Cortex_SGPe_Arky, STR_D1SNr, STR_D2GPe_Proto, STNSNr, \
                                    STNGPe_Proto, GPe_ProtoSTN, GPe_ProtoSNr, SNrThal, ThalIntegrator, GPeEGPe_Proto, GPEGPe_Arky, SNrESNr, STNESTN, \
                                    STR_FSISTR_D1, STR_FSISTR_D2, STRESTR_D1, STRESTR_D2, GPe_ArkySTR_D1, GPe_ArkySTR_D2, TestThalnoiseThal,STRESTR_FSI, \
                                    STR_FSISTR_FSI, GPe_ArkySTR_FSI, GPe_ArkyGPe_Proto, GPe_ProtoGPe_Arky, STR_D2GPe_Arky, \
                                    GPe_ProtoSTR_FSI, STR_D1STR_D1, STR_D1STR_D2, STR_D2STR_D1, STR_D2STR_D2, Cortex_GSTR_D2, Cortex_SGPe_Proto, STNGPe_Arky, GPe_ArkyCortex_G, \
                                    ThalSD1, ThalSD2, ThalFSI, \
                                    GPe_ProtoGPe_Proto2, GPe_Proto2GPe_Proto, STR_D2GPe_Proto2, STR_D1GPe_Proto2, STNGPe_Proto2, Cortex_SGPe_Proto2, GPe_ArkyGPe_Proto2, GPe_Proto2STR_D1, GPe_Proto2STR_D2, GPe_Proto2STR_FSI, GPe_Proto2GPe_Arky, GPe_Proto2IntegratorStop, EProto1GPe_Proto, EProto2GPe_Proto2, EArkyGPe_Arky, \
                                    Cortex_SGPe_Arky2, STR_D2GPe_Arky2, GPe_ProtoGPe_Arky2, STNGPe_Arky2, GPe_Proto2GPe_Arky2, EArkyGPe_Arky2, GPe_Arky2STR_D1, GPe_Arky2STR_D2, GPe_Arky2STR_FSI

quit()
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

def changeArkyStopOutput(fD1,fD2,fFSI):
    """
    change Arky-Str weights but compansate it with second Arky population, thus only stop input effect is changed (second Arky doesn't get stop input)
    """
    GPe_ArkySTR_D1.mod_factor = wf['Arky_D1']*fD1
    GPe_ArkySTR_D2.mod_factor = wf['Arky_D2']*fD2
    GPe_ArkySTR_FSI.mod_factor =  wf['Arky_FSI']*fFSI
    GPe_Arky2STR_D1.mod_factor     = wf['Arky_D1']*int(params['Arky2'])*(1-fD1)#np.clip((1-fD1),0,None)
    GPe_Arky2STR_D2.mod_factor     = wf['Arky_D2']*int(params['Arky2'])*(1-fD2)#np.clip((1-fD2),0,None)
    GPe_Arky2STR_FSI.mod_factor    = wf['Arky_FSI']*int(params['Arky2'])*(1-fFSI)#np.clip((1-fFSI),0,None)


Integrator.tau = params['tau_Int']
IntegratorStop.tau = 30.0

#Cortex_GSTR_D2.mod_factor = wf['Go_D2']
#Cortex_GThal.mod_factor = wf['CortexG_Thal']
#STR_D1SNr.mod_factor = wf['D1_Snr']
#STR_D2GPe_Proto.mod_factor = wf['D2_Gpe']
#STR_D2GPe_Arky.mod_factor = wf['D2_Arky']



# Parameters:
# - Simulation: 
# Stop-signal delay (SSD) - fixed, 170ms
# Latency between "pause" and "cancel" signal - fixed, 10ms
# GO input rates (mean / variability)
# STOP input rates (mean / variability)
# 
# - Network:
# Weights (lower/upper bound of uniform dist.; mean/std. of Gaussian dist.)
# Connection probabilities ?!
# Stop1toSTN : w_lo, w_up
# CortexGo_StrD1 : w_lo, w_up
# CortexGo_Thal : w_lo, w_up
# CortexStop_GPeArky : w_lo, w_up
# StrD1_SNr : prob, w
# STN_SNr : prob, w
# STN_GPe : prob, w
# GPe_STN : prob, w
# GPe_SNr : prob, w
# SNr_Thal : prob, w
# Thal_Integrator : w
# GPeArky_StrD1 : w
# GPePoi_GPe : w
# GPePoi_Arky : w
# SNrPoi_SNr : w
# STNPoi_STN : w




t = 0
#general parameters
trials = 400 # 25 # 300 # 100 # Caution - memory consumption gets critical! 300 is OK.
t_smooth_ms = 20.0 # 5.0 # 1.0 # Errors for 5.0-20.0 ?!
t_decay = 300 # 500 #
ratesProto1vs2 = params['ratesProto1vs2']


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
