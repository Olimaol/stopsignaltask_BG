import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from .analysis import get_syn_mon

def get_and_plot_syn_mon(m_syn_D1, m_syn_D2, m_syn_FSI, m_syn_STN, m_syn_Proto, m_syn_Arky, m_syn_SNr, m_syn_Thal, param_id):
    #plt.ion()                
    plt.figure()
    ampa_stats, gaba_stats = get_syn_mon(m_syn_D1, 'Str. D1')
    plot_syn_stats(ampa_stats, gaba_stats, 'Str. D1', 'g')
    ampa_stats, gaba_stats = get_syn_mon(m_syn_D2, 'Str. D2')                
    plot_syn_stats(ampa_stats, gaba_stats, 'Str. D2', 'purple')                
    ampa_stats, gaba_stats = get_syn_mon(m_syn_FSI, 'Str. FSI')  
    plot_syn_stats(ampa_stats, gaba_stats, 'FSI', 'k')                                
    ampa_stats, gaba_stats = get_syn_mon(m_syn_STN, 'STN')
    plot_syn_stats(ampa_stats, gaba_stats, 'STN', 'orange')                                
    ampa_stats, gaba_stats = get_syn_mon(m_syn_Proto, 'Proto')
    plot_syn_stats(ampa_stats, gaba_stats, 'Proto', 'blue')                      
    ampa_stats, gaba_stats = get_syn_mon(m_syn_Arky, 'Arky')
    plot_syn_stats(ampa_stats, gaba_stats, 'Arky', 'cyan')                      
    ampa_stats, gaba_stats = get_syn_mon(m_syn_SNr, 'SNr')
    plot_syn_stats(ampa_stats, gaba_stats, 'SNr', 'tomato')                      
    ampa_stats, gaba_stats = get_syn_mon(m_syn_Thal, 'Thal')
    plot_syn_stats(ampa_stats, gaba_stats, 'Thal', '0.6') # yellow # pink
    plt.legend()                
    plt.savefig('plots/syn_stats_id'+str(param_id)+'.png')
    plt.close()                
    plt.ioff()                


def plot_syn_stats(ampa_stats, gaba_stats, str_popname, str_color):
    #percentiles = [75, 90, 95, 99, 100]
    percentiles = [90, 95, 99, 100]    
    plt.plot(percentiles, ampa_stats, label = str_popname+' Ampa', color=str_color)
    plt.plot(percentiles, gaba_stats, '--', label = str_popname+' Gaba', color=str_color)    
   


