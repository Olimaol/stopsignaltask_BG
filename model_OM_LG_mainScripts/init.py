# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 09:56:48 2020

@author: lorenz
"""

import numpy as np

def init_neuronmodels(pop, popsize):
    pop.v = np.random.normal(-65,10,popsize)
    pop.u = np.random.normal(-15,1.5,100)
    return pop.v, pop.u