# Works on gyrus (ANNarchy 4.6.2)

from ANNarchy import *
import pylab as plt
import numpy as np
import random
from timeit import default_timer as timer
import sys
import scipy.optimize as sciopt
import scipy.stats as st

from fit_arky_params import read_arky_data, read_arky_data_Bogacz2016


def mean_response_rate(poprate, tot_trials, simsteps, dt):
    mean_response = plt.zeros(tot_trials)
    for trial in range(tot_trials):
        ti_last_in_trial = int( min(simsteps, (trial+1) / float(tot_trials) * simsteps - 1) )
        ti_last100ms_firstind = int( ti_last_in_trial - 100.0 / dt + 1 )
        mean_response[trial] = poprate[ti_last100ms_firstind : ti_last_in_trial].mean()
    return mean_response


def raster_rate_plot(mon, spdata, popname, subplot_idx, rate_color, Tsmooth_ms, maxrate, fs):
    mon_te, mon_ne = mon.raster_plot(spdata)
    fr_mon = mon.population_rate(spdata, smooth=Tsmooth_ms)
    ax_wrapper = []
    ax_wrapper.append(plt.subplot(subplot_idx))
    plt.plot(mon_te, mon_ne, 'k.')
    plt.ylabel(popname + ' neuron index', fontsize=fsize)
    plt.xlabel('t [ms]', fontsize=fsize)
    ax3 = plt.gca()
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontsize(fs)
    ax_wrapper.append(ax_wrapper[0].twinx())
    plt.plot(timesteps, fr_mon, lw=2, color=rate_color)
    plt.ylabel(popname + ' firing rate [spk/s]', color=rate_color, fontsize=fsize)
    ax_twin = plt.gca()
    for label in ax_twin.get_yticklabels():
        label.set_color(rate_color)
        label.set_fontsize(fs)
    ax_twin.axis([ax_twin.axis()[0], ax_twin.axis()[1], ax_twin.axis()[2], maxrate])
    plt.title(popname + ' responses to step currents', fontsize=1.2*fsize)


def plot_vm_traces(gpe_vdata, arky_vdata, length_ms, dt, fs, i_gpe_fig, i_arky_fig, I_arky=[], i_trial_arky=0):
    fs=fs+2
    #plt.subplot(i_gpe_fig)
    ax_wrapper = []
    ax_wrapper.append(plt.subplot(i_gpe_fig))
    end = int(length_ms / dt)
    #plt.plot(timesteps[i_trial * end : (i_trial+1) * end], gpe_vdata[i_trial * end : (i_trial+1) * end, 0], 'b', lw=0.5)
    #plt.plot(timesteps[0:end], gpe_vdata[0:end, 0], 'b', lw=0.5)
    t_init = timesteps[i_trial_arky * end]     
    plt.plot(timesteps[i_trial_arky * end : (i_trial_arky+1) * end] - t_init, gpe_vdata[i_trial_arky * end : (i_trial_arky+1) * end, 0], color=np.array([3,67,223])/255., lw=0.5)        
    print('Proto vm min = ', min(gpe_vdata[i_trial_arky * end : (i_trial_arky+1) * end, 0]))
    print('Arky vm min = ', min(arky_vdata[i_trial_arky * end : (i_trial_arky+1) * end, 0]))    
    ax5 = plt.gca()
    plt.xlabel('t [ms]', fontsize=fs)
    plt.ylabel('v [mV]', fontsize=fs)
    plt.title('GPe-Proto voltage traces', fontsize=1.2*fs)
    ax5.axis([ax5.axis()[0], ax5.axis()[1], -100, 40]) #  -80 
    ax5.set_yticks(range(-100, 45, 20)) # -80
    #ax5.set_yticklabels(['-80', '', '-60', '', '-40', '', '-20', '', '0', '', '20', '', '40'])    
    
    if len(I_arky) > 0:
        ax_wrapper.append(ax_wrapper[0].twinx())
        plt.plot(timesteps[0:end], I_arky, lw=2, color='0.5')
        ax_twin = plt.gca()
        plt.ylabel('Injected current', color='0.5')
        ax_twin.axis([ax_twin.axis()[0], ax_twin.axis()[1], ax_twin.axis()[2], 1.0])
        for label in ax_twin.get_yticklabels():
            label.set_color('0.5')


    ax_wrapper = []
    ax_wrapper.append(plt.subplot(i_arky_fig))
    #plt.subplot(i_arky_fig)
    #plt.plot(timesteps[0:end], arky_vdata[0:end, 0], 'r', lw=0.5)
    t_init = timesteps[i_trial_arky * end]     
    plt.plot(timesteps[i_trial_arky * end : (i_trial_arky+1) * end] - t_init, arky_vdata[i_trial_arky * end : (i_trial_arky+1) * end, 0], color=np.array([0,255,255])/255., lw=0.5)    
    ax6 = plt.gca()
    plt.xlabel('t [ms]', fontsize=fs)
    plt.ylabel('v [mV]', fontsize=fs)
    vmin = min(ax5.axis()[2], ax6.axis()[2])
    vmax = max(ax5.axis()[3], ax6.axis()[3])
    #ax5.axis([ax5.axis()[0], ax5.axis()[1], vmin, vmax])
    #ax6.axis([ax6.axis()[0], ax6.axis()[1], vmin, vmax])
    ax6.axis([ax5.axis()[0], ax5.axis()[1], -100, 40])    
    ax6.set_yticks(range(-100, 45, 20)) # , 10

    plt.title('GPe-Arky voltage traces', fontsize=1.2*fs)
    if len(I_arky) > 0:
        ax_wrapper.append(ax_wrapper[0].twinx())
        #plt.plot(timesteps[0:end], I_arky, lw=2, color='0.5')
        plt.plot(timesteps[i_trial_arky * end : (i_trial_arky+1) * end], I_arky, lw=2, color='0.5')        
        ax_twin = plt.gca()
        plt.ylabel('Injected current', color='0.5')
        ax_twin.axis([ax_twin.axis()[0], ax_twin.axis()[1], ax_twin.axis()[2], 1.0])
        for label in ax_twin.get_yticklabels():
            label.set_color('0.5')



#def calc_arky_response(arky_params, GPe_arky, full_output=False):
#def calc_arky_response(arky_params, input_args):    
def calc_arky_response(arky_proto_params, input_args):        

    GPe_arky = input_args[0]
    arky_params = arky_proto_params[0:7]
    GPe_arky.a = arky_params[0] # 0.005
    GPe_arky.b = arky_params[1] # 0.3 
    GPe_arky.c = arky_params[2] # -65
    GPe_arky.d = arky_params[3] # 8
    GPe_arky.n0 = arky_params[4]
    GPe_arky.n1 = arky_params[5]
    GPe_arky.n2 = arky_params[6]   

    full_output = input_args[1]

    print("arky_params = ", arky_params)

    for trial in range(n_trials):
        start = timer()
        if trial >= 7:    
            GPe_Proto.I = (trial-7) * 0.025 # 0.01 # Allow for negative currents to be injected
        else:
            GPe_Proto.I = (trial-7) * 0.01                        
        GPe_arky.I = GPe_Proto.I
        injected_current_simul_nA[trial] = GPe_Proto.I
        simulate(trial_length_ms)

    #T_smoothing_ms = 250.0

    gpe_spdata = m_GPe_Proto.get('spike')
    gpe_vdata = m_GPe_Proto.get('v')
    gpe_gAMPAdata = m_GPe_Proto.get('g_ampa')
    gpe_Idata = m_GPe_Proto.get('I')
    fr_gpe = m_GPe_Proto.population_rate(gpe_spdata, smooth=T_smoothing_ms) # 100.0

    arky_spdata = m_arky.get('spike')
    arky_vdata = m_arky.get('v')
    arky_gAMPAdata = m_arky.get('g_ampa')
    arky_Idata = m_arky.get('I')
    fr_arky = m_arky.population_rate(arky_spdata, smooth=T_smoothing_ms) # 100.0

    reset(populations=True)

    # Calculate firing rate for the last 100 ms of each stimulation period:
    mean_response_Proto_simul_Hz = mean_response_rate(fr_gpe, n_trials, sim_steps, dt_step_ms)
    mean_response_arky_simul_Hz = mean_response_rate(fr_arky, n_trials, sim_steps, dt_step_ms)

    # Calculate a fitting error relative to the Abdi et al. data
    #inj_current_Arky_expdata_pA, rate_Arky_expdata_Hz = read_arky_data()
    inj_current_Arky_expdata_pA, rate_Arky_expdata_Hz, sem_Arky_Hz, rate_Proto_expdata_Hz, sem_Proto_Hz = read_arky_data_Bogacz2016()

    i_trial_autonomous = 7
    vm_min = min(arky_vdata[i_trial_autonomous * int(trial_length_ms / dt_step_ms) : (i_trial_autonomous+1) * int(trial_length_ms / dt_step_ms), 0])  # dt_step_ms # 0.1
    print('Arky vm min = ', vm_min)

    #rmse_arky = np.sqrt( np.sum( (np.array(mean_response_arky_simul_Hz) - np.array(rate_Arky_expdata_Hz))**2 ) ) 
    rmse_arky = np.sqrt( np.sum( (np.array(mean_response_arky_simul_Hz) - np.array(rate_Arky_expdata_Hz))**2 ) + 10*(vm_min - (-71))**2) # Adding vm constraint
    
    print('rmse_arky = ', rmse_arky)

    #print('arky_spdata[0][0] = ', arky_spdata[0][0])
    #print('GPe_arky.a, GPe_arky.b, GPe_arky.c, GPe_arky.d = ', GPe_arky.a, GPe_arky.b, GPe_arky.c, GPe_arky.d)

    if full_output == False:
        return rmse_arky
    else:    
        return rmse_arky, mean_response_arky_simul_Hz,arky_spdata, arky_vdata

def calc_proto_response(proto_params, input_args):        
    GPe_Proto = input_args[0]
    GPe_Proto.a = proto_params[0]
    GPe_Proto.b = proto_params[1]
    GPe_Proto.c = proto_params[2]
    GPe_Proto.d = proto_params[3]
    GPe_Proto.n0 = proto_params[4]
    GPe_Proto.n1 = proto_params[5]
    GPe_Proto.n2 = proto_params[6]   

    full_output = input_args[1]

    print("proto_params = ", proto_params)

    for trial in range(n_trials):
        start = timer()
        if trial >= 7:    
            GPe_Proto.I = (trial-7) * 0.025 # 0.01 # Allow for negative currents to be injected
        else:
            GPe_Proto.I = (trial-7) * 0.01                        
        GPe_arky.I = GPe_Proto.I
        injected_current_simul_nA[trial] = GPe_Proto.I
        simulate(trial_length_ms)

    #T_smoothing_ms = 250.0

    gpe_spdata = m_GPe_Proto.get('spike')
    gpe_vdata = m_GPe_Proto.get('v')
    gpe_gAMPAdata = m_GPe_Proto.get('g_ampa')
    gpe_Idata = m_GPe_Proto.get('I')
    fr_gpe = m_GPe_Proto.population_rate(gpe_spdata, smooth=T_smoothing_ms) # 100.0

    arky_spdata = m_arky.get('spike')
    arky_vdata = m_arky.get('v')
    arky_gAMPAdata = m_arky.get('g_ampa')
    arky_Idata = m_arky.get('I')
    fr_arky = m_arky.population_rate(arky_spdata, smooth=T_smoothing_ms) # 100.0

    reset(populations=True)

    # Calculate firing rate for the last 100 ms of each stimulation period:
    mean_response_Proto_simul_Hz = mean_response_rate(fr_gpe, n_trials, sim_steps, dt_step_ms)
    mean_response_arky_simul_Hz = mean_response_rate(fr_arky, n_trials, sim_steps, dt_step_ms)

    # Calculate a fitting error relative to the Abdi et al. data
    #inj_current_Arky_expdata_pA, rate_Arky_expdata_Hz = read_arky_data()
    inj_current_Arky_expdata_pA, rate_Arky_expdata_Hz, sem_Arky_Hz, rate_Proto_expdata_Hz, sem_Proto_Hz = read_arky_data_Bogacz2016()

    i_trial_autonomous = 7
    vm_min = min(gpe_vdata[i_trial_autonomous * int(trial_length_ms / dt_step_ms) : (i_trial_autonomous+1) * int(trial_length_ms / dt_step_ms), 0])  # dt_step_ms # 0.1
    print('Proto vm min = ', vm_min)    

    #rmse_proto = np.sqrt( np.sum( (np.array(mean_response_Proto_simul_Hz[0:14]) - np.array(rate_Proto_expdata_Hz[0:14]))**2 ) )
    rmse_proto = np.sqrt( np.sum( (np.array(mean_response_Proto_simul_Hz[0:14]) - np.array(rate_Proto_expdata_Hz[0:14]))**2 ) + 10*(vm_min - (-66))**2) # Adding vm constraint    
    
    
    print('rmse_proto = ', rmse_proto)

    if full_output == False:
        return rmse_proto
    else:    
        return rmse_proto, mean_response_Proto_simul_Hz, gpe_spdata, gpe_vdata

setup(dt=0.1)
#setup(dt=0.01)
#setup(dt=1)
#setup(num_threads=4 )

random.seed()
np.random.seed()



#General parameters
population_size = 10 # 100

#Neuron models



Hybrid_neuron_inp_res = Neuron(
parameters="""

    a = 0.0   : population
    b = 0.0   : population
    c = 0.0   : population
    d = 0.0   : population
    n0 = 140. : population
    n1 = 5.0  : population
    n2 = 0.04 : population    
    I = 0.0   : population
    tau_ampa = 80 : population
    tau_gaba = 80 : population
    E_ampa = 0.0 : population
    E_gaba = -90.0 : population
    R_input_megOhm = 10.0 : population
""", 

equations="""
    dg_ampa/dt = -g_ampa/tau_ampa : init = 0
    dg_gaba/dt = -g_gaba/tau_gaba : init = 0
    dv/dt = n2*v*v+n1*v+n0 - u  + R_input_megOhm * I - g_ampa*(v-E_ampa) - g_gaba*(v-E_gaba): init = -70.
    du/dt = a*(b*(v)-u) : init = -18.55
""", # megOhm * nanoAmp = millivolt


spike = """v>=30""",
reset = """v = c    
           u = u+d""",
refractory = 5.0
) # 5.0


#Layers
GPe_Proto      = Population(population_size, Hybrid_neuron_inp_res)
GPe_arky = Population(population_size, Hybrid_neuron_inp_res)

#GPe parameters
GPe_Proto.a = 0.005
GPe_Proto.b = 0.585
GPe_Proto.c = -65
GPe_Proto.d = 4
GPe_Proto.I = 0
GPe_Proto.R_input_megOhm = 450.0 # See Abdi, Magill et al. 2015, Table 1

#GPe Arky parameters

GPe_arky.a = 0.005 # 0.004 # 
GPe_arky.b = 0.1 # 0.585 # 
GPe_arky.c = -65
GPe_arky.d = 4
GPe_arky.I = 0
GPe_arky.R_input_megOhm = 560.0 # 560.0  # See Abdi, Magill et al. 2015, Table 1


# Individual params n0, n1, n2 varied:
'''#
GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = GPe_Proto.n0 * 2, GPe_Proto.n1, GPe_Proto.n2          # Very high rates
GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = GPe_Proto.n0 * 0.9, GPe_Proto.n1, GPe_Proto.n2        # Somewhat lower rates, similar f-I slope

GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = GPe_Proto.n0, GPe_Proto.n1 * 1.05, GPe_Proto.n2       # Somewhat lower rates, similar f-I slope
GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = GPe_Proto.n0, GPe_Proto.n1 * 0.9, GPe_Proto.n2        # higher rates

GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = GPe_Proto.n0, GPe_Proto.n1, GPe_Proto.n2 * 1.05       # higher rates
GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = GPe_Proto.n0, GPe_Proto.n1, GPe_Proto.n2 * 0.9        # lower rates

GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = GPe_Proto.n0 * 0.95, GPe_Proto.n1 * 1.02, GPe_Proto.n2 # somewhat lower rates
'''


#GPe_arky.d, GPe_arky.a, paramset = GPe_Proto.d * 2.0, GPe_Proto.a * 0.5, 1      # Strongly reduced rates - good! A bit too strong when considering input resistance.
#GPe_arky.d, GPe_arky.a, paramset = GPe_Proto.d * 2.0, GPe_Proto.a * 0.25, 2     # Rates too low
GPe_arky.d, GPe_arky.a, paramset = GPe_Proto.d * 2.0, GPe_Proto.a * 0.8, 3       # My version: a=0.004, b=0.585
GPe_arky.d, GPe_arky.a, paramset = GPe_Proto.d * 2.0, 0.005, 4       # Ilianas version: a=0.005, b=0.1
GPe_arky.d, GPe_arky.a, paramset = GPe_Proto.d * 2.0, 0.005, 5       # New version: a=0.005, b=0.3

# Result (use these values):
GPe_arky.a = 0.005 # 0.004
GPe_arky.b = 0.3 # 0.1 # 0.585
GPe_arky.c = -65
GPe_arky.d = 8
GPe_arky.I = 0

arky_params = [GPe_arky.a, GPe_arky.b, GPe_arky.c, GPe_arky.d, GPe_arky.n0, GPe_arky.n1, GPe_arky.n2]
proto_params = [GPe_Proto.a, GPe_Proto.b, GPe_Proto.c, GPe_Proto.d, GPe_Proto.n0, GPe_Proto.n1, GPe_Proto.n2]

#main loop
compile('Annarchy99')

m_GPe_Proto  = Monitor(GPe_Proto,['spike','v','g_ampa', 'I'])
m_arky  = Monitor(GPe_arky,['spike','v','g_ampa', 'I'])

dt_step_ms          = dt() # 0.1
trial_length_ms     = 1500.0 
T_smoothing_ms = 250.0

n_trials            = 17 # 12 # 9 # 10 # 1
simlength_ms        = n_trials * trial_length_ms
sim_steps           = simlength_ms / dt_step_ms
timesteps           = plt.arange(0, simlength_ms, dt_step_ms)

injected_current_simul_nA    = np.nan *np.ones(n_trials)

#arky_params =  [ 5.04768174e-03,  3.27611185e-01, -7.17911993e+01,  8.68578568e+00,  1.17574208e+02,  4.41107716e+00,  3.76000533e-02] # 5ms refractory period - great! rmse_arky = 2.41027 (manual Abdi data)
#arky_params =  [ 5.02471952e-03,  3.27597679e-01, -7.30001054e+01,  8.96429028e+00,  1.17881446e+02,  4.42449605e+00,  3.74269283e-02] # Bogacz et al. data. rmse_arky =  2.102311282480598
#arky_params =  [ 5.38612347e-03,  3.36811868e-01, -71,  9.81279870e+00,  1.12989223e+02,  4.47315863e+00,  3.98586667e-02] # -71
arky_params =  [ 0.0054,          0.34,           -71,  9.81,            113,             4.47,            0.04] # rounded

#proto_params =  [5.40489759e-03, 5.84110289e-01, -6.87516180e+01,  3.53245611e+00, 1.36701406e+02,  5.20660168e+00,  4.16693675e-02] #  good enough (5ms refr.), rmse_proto = 3.5426
#proto_params =  [ 5.75941038e-03,  5.58540551e-01, -6.58954513e+01,  3.79816288e+00,  1.17164834e+02,  4.85649725e+00,  4.25927423e-02] # Proto vm min =  -65.89, rmse_proto =  3.50
#proto_params =  [ 5.75941038e-03,  5.58540551e-01, -65,  3.79816288e+00,  1.17164834e+02,  4.85649725e+00,  4.25927423e-02] # -64
proto_params =  [ 0.0058,          0.56,           -65,  3.8,             117,             4.86,            0.043] # rounded

GPe_arky.a, GPe_arky.b, GPe_arky.c, GPe_arky.d, GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = arky_params
GPe_Proto.a, GPe_Proto.b, GPe_Proto.c, GPe_Proto.d, GPe_Proto.n0, GPe_Proto.n1, GPe_Proto.n2 = proto_params



param_bounds = ((0, np.inf), (0, np.inf), (-80, -40), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)) # a,b,c,d,n0,n1,n2

'''#
res = sciopt.minimize(calc_arky_response, arky_params, [GPe_arky, False], bounds=param_bounds, method='Nelder-Mead') # Fit only Arky
res = sciopt.minimize(calc_proto_response, proto_params, [GPe_Proto, False], bounds=param_bounds, method='Nelder-Mead')  # Fit only Proto

rmse_arky, mean_response_arky_simul_Hz, arky_spdata, arky_vdata = calc_arky_response(res.x, [GPe_arky, GPe_Proto, True]) # After fitting#
rmse_proto, mean_response_Proto_simul_Hz, gpe_spdata, gpe_vdata = calc_proto_response(res.x, [GPe_Proto, True]) # After fitting#
'''

rmse_arky, mean_response_arky_simul_Hz, arky_spdata, arky_vdata = calc_arky_response(arky_params, [GPe_arky, True]) # Without fitting
rmse_proto, mean_response_Proto_simul_Hz, gpe_spdata, gpe_vdata = calc_proto_response(proto_params, [GPe_Proto, True]) # Without fitting

GPe_arky.a, GPe_arky.b, GPe_arky.c, GPe_arky.d, GPe_arky.n0, GPe_arky.n1, GPe_arky.n2 = arky_params
GPe_Proto.a, GPe_Proto.b, GPe_Proto.c, GPe_Proto.d, GPe_Proto.n0, GPe_Proto.n1, GPe_Proto.n2 = proto_params

plt.ion()
fsize = 9
plt.figure(figsize=(9,12)) # 15,12

rmax_gpe = m_GPe_Proto.population_rate(gpe_spdata, smooth=T_smoothing_ms).max()
rmax_arky = m_arky.population_rate(arky_spdata, smooth=T_smoothing_ms).max()
rmax = max(rmax_gpe, rmax_arky)

#raster_rate_plot(m_gpe, gpe_spdata, 'GPe proto', 321, 'b', T_smoothing_ms, rmax, fsize)
#raster_rate_plot(m_arky, arky_spdata, 'GPe Arky', 322, 'r', T_smoothing_ms, rmax, fsize)

#plt.subplot(324)
#ax4=plt.gca()
#plt.axis([ax4.axis()[0], ax4.axis()[1], 0, ax4.axis()[3]])


plot_vm_traces(gpe_vdata, arky_vdata, trial_length_ms, dt_step_ms, fsize, 321, 323, i_trial_arky=8)
#plot_vm_traces(gpe_vdata, arky_vdata, trial_length_ms, dt_step_ms, fsize, 321, 325)


# Additional simulations - driven firing: 
GPe_Proto.u = 0.0
GPe_Proto.v = -70.0
GPe_Proto.I = 0.0

GPe_arky.u = 0.0
GPe_arky.v = -70.0
GPe_arky.I = 0.0
simulate(500.0)

GPe_Proto.I = 0.1
GPe_arky.I = GPe_Proto.I
simulate(2000.0)

GPe_Proto.I = 0.0
GPe_arky.I = 0.0
simulate(500.0)

# New - hyperpolarization:
GPe_Proto.I = -0.1
GPe_arky.I = GPe_Proto.I
simulate(500.0)
GPe_Proto.I = 0.0
GPe_arky.I = 0.0
simulate(500.0)
# - end new
GPe_Proto.I = -0.5
GPe_arky.I = GPe_Proto.I
simulate(500.0)
GPe_Proto.I = 0.0
GPe_arky.I = 0.0
simulate(500.0)


gpe_vdata = m_GPe_Proto.get('v')
arky_vdata = m_arky.get('v')
arky_Idata = m_arky.get('I')
#plot_vm_traces(gpe_vdata, arky_vdata, 3000.0, dt_step_ms, fsize, 322, 324, arky_Idata) # old - only depolarization
#plot_vm_traces(gpe_vdata, arky_vdata, 4000.0, dt_step_ms, fsize, 322, 324, arky_Idata) # new - add'l hyperpolarization
plot_vm_traces(gpe_vdata, arky_vdata, 5000.0, dt_step_ms, fsize, 322, 324, arky_Idata) # new - 2x hyperpolarization

plt.subplot(325)
inj_current_Arky_expdata_pA, rate_Arky_expdata_Hz, sem_Arky_Hz, rate_Proto_expdata_Hz, sem_Proto_Hz = read_arky_data_Bogacz2016()
plt.plot(inj_current_Arky_expdata_pA, rate_Arky_expdata_Hz, '.-', color=np.array([255,64,0])/255., ms=10, label='GPe-Arky (experimental data)')
plt.plot(1e3 * injected_current_simul_nA, mean_response_arky_simul_Hz, '.-', color=np.array([0,255,255])/255., ls='--', ms=10, label='GPe-Arky (simulated data)')
plt.plot(inj_current_Arky_expdata_pA[0:14], rate_Proto_expdata_Hz[0:14],'.-', color=np.array([223,98,3])/255., ms=10, label='GPe-Proto (experimental data)')
plt.plot(1e3 * injected_current_simul_nA[0:14], mean_response_Proto_simul_Hz[0:14],'.-', color=np.array([3,67,223])/255., ls='--', ms=10, label='GPe-Proto (simulated data)')
plt.legend()
plt.axis([plt.axis()[0], plt.axis()[1], -5, 100]) # 75
plt.xlabel('Injected current [pA]')
plt.ylabel('Firing rate [spk/sec]')
plt.title('Responses to step currents', fontsize=1.2*fsize)


# Add further simulations (hyperpolarization) here:


#plt.subplots_adjust(bottom=0.08, top=0.9, wspace=0.5, hspace=0.35, left=0.12)#, right=0.85 )
plt.subplots_adjust(bottom=0.08, top=0.9, wspace=0.3, hspace=0.35, left=0.12)#, right=0.85 )

#plt.savefig('plots/compare_proto_arky_paramset'+str(paramset)+'.png')#, dpi=300)
resultsDir='../results'
try:
    os.makedirs(resultsDir)
except:
    if os.path.isdir(resultsDir):
        print(resultsDir+' already exists')
    else:
        print('could not create '+resultsDir+' folder')
        quit()
plt.savefig(resultsDir+'/compare_proto_arky_fit_expdata_dt.svg', dpi=300)

plt.ioff()
plt.show()












































