# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 21:43:20 2016

Bayesian Solver

@author: pierre
"""
from __future__ import print_function
from __future__ import division
import numpy as np
pi=3.14


class BayesSolver(object):
    
    def __init__(self,dim):
        self.dim=dim
    
    def adviseNewSample(self):
        return np.linspace(0,1,self.dim)