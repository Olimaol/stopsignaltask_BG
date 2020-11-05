# Works on gyrus (ANNarchy 4.6.2)

from ANNarchy import *
import pylab as plt
import numpy as np
import random
from timeit import default_timer as timer
import sys
import scipy.optimize as sciopt
import scipy.stats as st

from fit_arky_params import read_arky_data
from arky_params_test import mean_response_rate, plot_vm_traces, calc_arky_response, Hybrid_neuron_inp_res

# TODO:
# Add neuron model parameters as inputs to the main simulation loop / mean_response_rate
# Calculate a fitting error (RMSE) based on data by Abdi et al. 2015
# Use scipy.optimize on the neuron model parameters



setup(dt=0.1)
#setup(num_threads=4 )

random.seed()
np.random.seed()



#General parameters
population_size = 10 # 100


#Layers
GPe      = Population(population_size, Hybrid_neuron_inp_res)
GPe_arky = Population(population_size, Hybrid_neuron_inp_res)

#GPe parameters
GPe.a = 0.005
GPe.b = 0.585
GPe.c = -65
GPe.d = 4
GPe.I = 0
GPe.R_input_megOhm = 450.0 # See Abdi, Magill et al. 2015, Table 1

#GPe Arky parameters

GPe_arky.a = 0.005 # 0.004 # 
GPe_arky.b = 0.1 # 0.585 # 
GPe_arky.c = -65
GPe_arky.d = 4
GPe_arky.I = 0
GPe_arky.R_input_megOhm = 560.0 # 560.0  # See Abdi, Magill et al. 2015, Table 1


# Individual params n0, n1, n2 varied:


# Result (use these values):
GPe_arky.a = 0.005 # 0.004
GPe_arky.b = 0.3 # 0.1 # 0.585
GPe_arky.c = -65
GPe_arky.d = 8
GPe_arky.I = 0

arky_params = [GPe_arky.a, GPe_arky.b, GPe_arky.c, GPe_arky.d, GPe_arky.n0, GPe_arky.n1, GPe_arky.n2]

#main loop
compile()

m_gpe  = Monitor(GPe,['spike','v','g_ampa', 'I'])
m_arky  = Monitor(GPe_arky,['spike','v','g_ampa', 'I'])

dt_step_ms          = 0.1
trial_length_ms     = 1500.0 
T_smoothing_ms = 250.0

n_trials            = 12 # 9 # 10 # 1
simlength_ms        = n_trials * trial_length_ms
sim_steps           = simlength_ms / dt_step_ms
timesteps           = plt.arange(0, simlength_ms, dt_step_ms)

injected_current_simul_nA    = np.nan *np.ones(n_trials)

#arky_params =  [ 4.19639566e-03,  3.13669521e-01, -7.19445597e+01,  9.78428300e+00,  1.33018996e+02,  4.45487555e+00,  3.51801768e-02] # Final result (abcd, n0n1n2), method=Nelder-Mead

rmse, mean_response_GPe, mean_response_arky_simul_Hz, gpe_spdata, arky_spdata, gpe_vdata, arky_vdata = calc_arky_response(arky_params, [GPe_arky, True]) # Without fitting

GPe_arky.a, GPe_arky.b, GPe_arky.c, GPe_arky.d, GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = arky_params


plt.ion()
fsize = 9
plt.figure(figsize=(9,12)) # 15,12

rmax_gpe = m_gpe.population_rate(gpe_spdata, smooth=T_smoothing_ms).max()
rmax_arky = m_arky.population_rate(arky_spdata, smooth=T_smoothing_ms).max()
rmax = max(rmax_gpe, rmax_arky)


plt.subplot(325)
#plt.subplot(323)
plt.plot(1e3 * injected_current_simul_nA, mean_response_GPe, 'b.-', ms=10)
plt.plot(1e3 * injected_current_simul_nA, mean_response_arky_simul_Hz, 'r.-', ms=10)
plt.xlabel('Injected current [pA]', fontsize=fsize)
plt.ylabel('Firing rate [spk/s]', fontsize=fsize)
plt.legend(('GPe proto', 'Arky'), fontsize=fsize)
ax3=plt.gca()
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontsize(fsize)
plt.axis([ax3.axis()[0], ax3.axis()[1], 0, ax3.axis()[3]])
plt.title('Responses to step currents', fontsize=1.2*fsize)


plot_vm_traces(gpe_vdata, arky_vdata, trial_length_ms, dt_step_ms, fsize, 321, 323)


# Additional simulations - driven firing: 
GPe.u = 0.0
GPe.v = -70.0
GPe.I = 0.0

GPe_arky.u = 0.0
GPe_arky.v = -70.0
GPe_arky.I = 0.0
simulate(500.0)

GPe.I = 0.1
GPe_arky.I = GPe.I
simulate(2000.0)

GPe.I = 0.0
GPe_arky.I = 0.0
simulate(500.0)


gpe_vdata = m_gpe.get('v')
arky_vdata = m_arky.get('v')
arky_Idata = m_arky.get('I')
plot_vm_traces(gpe_vdata, arky_vdata, 3000.0, dt_step_ms, fsize, 322, 324, arky_Idata)

plt.subplot(326)
inj_current_Arky_expdata_pA, rate_Arky_expdata_Hz = read_arky_data()
plt.plot(inj_current_Arky_expdata_pA, rate_Arky_expdata_Hz, 'g.-', ms=10, label='Arky (Abdi et al.)')
plt.plot(1e3 * injected_current_simul_nA, mean_response_arky_simul_Hz, 'r.-', ms=10, label='Arky fit')
slope, intercept, r_value, p_value, std_err = st.linregress(1e3 * injected_current_simul_nA, mean_response_arky_simul_Hz)
plt.plot(1e3 * injected_current_simul_nA, slope * 1e3 * injected_current_simul_nA + intercept, '--', label='linear fit')
plt.legend()
plt.axis([plt.axis()[0], plt.axis()[1], -5, 75])


plt.subplots_adjust(bottom=0.08, top=0.9, wspace=0.5, hspace=0.35, left=0.12)#, right=0.85 )

plt.savefig('plots/compare_arky_param_versions.png')#, dpi=300)

plt.ioff()
plt.show()












































