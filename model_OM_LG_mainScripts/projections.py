from ANNarchy import*

from .synapsemodels import FactorSynapse
from .populations import Stoppinput1, Cortex_S, Cortex_G, STR_D1, STR_D2, STN, SNr, GPe_Proto, Thal, Integrator, IntegratorStop, GPeE, SNrE, STNE, STR_FSI, STRE, GPe_Arky, GPe_Arky2, TestThalnoise, GPe_Proto2, EProto1, EProto2, EArky, ED1, ED2, EFSI
from .sim_params import params


### CortexGo outputs
Cortex_GSTR_D1 = Projection (
    pre = Cortex_G,
    post = STR_D1,    
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

Cortex_GSTR_D2 = Projection (
    pre = Cortex_G,
    post = STR_D2,    
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

Cortex_GThal = Projection (
    pre = Cortex_G,
   post = Thal,
   target = 'ampa',
   synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

Cortex_GSTR_FSI = Projection (
    pre = Cortex_G,
    post = STR_FSI,    
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'],  weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### CortexStop outputs
Cortex_SGPe_Arky = Projection (
    pre = Cortex_S,
    post = GPe_Arky,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

Cortex_SGPe_Proto = Projection (
    pre = Cortex_S,
    post = GPe_Proto,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

Cortex_SGPe_Proto2 = Projection (
    pre = Cortex_S,
    post = GPe_Proto2,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

Cortex_SGPe_Arky2 = Projection (
    pre = Cortex_S,
    post = GPe_Arky2,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### CortexPause outputs
Stoppinput1STN = Projection (
    pre = Stoppinput1,
    post = STN,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConInput'], weights = 1, delays = 0)

### StrD1 outputs
STR_D1SNr = Projection (
    pre = STR_D1,
    post = SNr,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_D1GPe_Proto2 = Projection (
    pre = STR_D1,
    post = GPe_Proto2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_D1STR_D1 = Projection(
    pre  = STR_D1,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_D1STR_D2 = Projection(
    pre  = STR_D1,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### StrD2 outputs
STR_D2GPe_Proto = Projection (
    pre = STR_D2,
    post = GPe_Proto,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_D2GPe_Arky = Projection (
    pre = STR_D2,
    post = GPe_Arky,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_D2STR_D1 = Projection(
    pre  = STR_D2,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_D2STR_D2 = Projection(
    pre  = STR_D2,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_D2GPe_Proto2 = Projection (
    pre = STR_D2,
    post = GPe_Proto2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_D2GPe_Arky2 = Projection (
    pre = STR_D2,
    post = GPe_Arky2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### StrFSI outputs
STR_FSISTR_D1 = Projection(
    pre  = STR_FSI,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_FSISTR_D2 = Projection(
    pre  = STR_FSI,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STR_FSISTR_FSI = Projection(
    pre  = STR_FSI,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### STN outputs
STNSNr = Projection (
    pre = STN,
    post = SNr,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = 10)

STNGPe_Proto = Projection (
    pre = STN,
    post = GPe_Proto,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STNGPe_Arky = Projection (
    pre = STN,
    post = GPe_Arky,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STNGPe_Proto2 = Projection (
    pre = STN,
    post = GPe_Proto2,
    target = 'ampa',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STNGPe_Arky2 = Projection (
    pre = STN,
    post = GPe_Arky2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### GPeProto outputs
GPe_ProtoSTN = Projection (
    pre = GPe_Proto,
    post = STN,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_ProtoSNr = Projection (
    pre = GPe_Proto,
    post = SNr,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_ProtoGPe_Arky = Projection (
    pre = GPe_Proto,
    post = GPe_Arky,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_ProtoSTR_FSI = Projection (
    pre = GPe_Proto,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse 
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_ProtoGPe_Proto2 = Projection (
    pre = GPe_Proto,
    post = GPe_Proto2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_ProtoGPe_Arky2 = Projection (
    pre = GPe_Proto,
    post = GPe_Arky2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### GPeArky outputs
GPe_ArkySTR_D1 = Projection (
    pre = GPe_Arky,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_ArkySTR_D2 = Projection (
    pre = GPe_Arky,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays'])) 

GPe_ArkyGPe_Proto = Projection (
    pre = GPe_Arky,
    post = GPe_Proto,
    target = 'gaba',
    synapse = FactorSynapse  
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_ArkySTR_FSI = Projection (
    pre = GPe_Arky,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_ArkyGPe_Proto2 = Projection (
    pre = GPe_Arky,
    post = GPe_Proto2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### GPeArkyCopy outputs
GPe_Arky2STR_D1 = Projection (
    pre = GPe_Arky2,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_Arky2STR_D2 = Projection (
    pre = GPe_Arky2,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays'])) 

GPe_Arky2STR_FSI = Projection (
    pre = GPe_Arky2,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### GPeCp outputs
GPe_Proto2GPe_Proto = Projection (
    pre = GPe_Proto2,
    post = GPe_Proto,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_Proto2STR_D1 = Projection (
    pre = GPe_Proto2,
    post = STR_D1,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_Proto2STR_D2 = Projection (
    pre = GPe_Proto2,
    post = STR_D2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_Proto2STR_FSI = Projection (
    pre = GPe_Proto2,
    post = STR_FSI,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_Proto2GPe_Arky = Projection (
    pre = GPe_Proto2,
    post = GPe_Arky,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_Proto2IntegratorStop = Projection (
    pre = GPe_Proto2,
    post = IntegratorStop,
    target = 'ampa',
    synapse = FactorSynapse
).connect_all_to_all(weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPe_Proto2GPe_Arky2 = Projection (
    pre = GPe_Proto2,
    post = GPe_Arky2,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### SNr outputs
SNrThal = Projection (
    pre = SNr,
    post = Thal,
    target = 'gaba',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### Thal outputs
ThalIntegrator = Projection (
    pre = Thal,
    post = Integrator,
    target = 'ampa',
    synapse = FactorSynapse
).connect_all_to_all(weights = 1, delays = Uniform(0.0, params['general_synDelays']))

ThalSD1 = Projection (
    pre = Thal,
    post = STR_D1,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

ThalSD2 = Projection (
    pre = Thal,
    post = STR_D2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

ThalFSI = Projection (
    pre = Thal,
    post = STR_FSI,
    target = 'ampa',
    synapse = FactorSynapse
).connect_fixed_number_pre(number = params['general_NrConIntrinsic'], weights = 1, delays = Uniform(0.0, params['general_synDelays']))

### Noise/Baseline inputs
GPeEGPe_Proto = Projection(
    pre  = GPeE,
    post = GPe_Proto,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))

GPEGPe_Arky = Projection(
    pre  = GPeE,
    post = GPe_Arky,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
       
SNrESNr = Projection(
    pre  = SNrE,
    post = SNr,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STNESTN = Projection(
    pre  = STNE,
    post = STN,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STRESTR_D1 = Projection(
    pre  = ED1,
    post = STR_D1,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))

STRESTR_D2 = Projection(
    pre  = ED2,
    post = STR_D2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))
 
TestThalnoiseThal = Projection(     
    pre = TestThalnoise,
    post = Thal,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1)

STRESTR_FSI = Projection(
    pre  = EFSI,
    post = STR_FSI,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))

EProto1GPe_Proto = Projection(
    pre  = EProto1,
    post = GPe_Proto,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))

EProto2GPe_Proto2 = Projection(
    pre  = EProto2,
    post = GPe_Proto2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))

EArkyGPe_Arky = Projection(
    pre  = EArky,
    post = GPe_Arky,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))

EArkyGPe_Arky2 = Projection(
    pre  = EArky,
    post = GPe_Arky2,
    target = 'ampa',
    synapse = FactorSynapse
).connect_one_to_one( weights = 1, delays = Uniform(0.0, params['general_synDelays']))











































































