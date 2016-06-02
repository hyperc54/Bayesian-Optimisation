# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 21:43:20 2016

Bayesian Solver
The library scikit learn performs the gaussian process regression

Parameters :

- dim: the number of dimensions of the blackbox the solver will try to solve

- bounds: [[x1min,x1max],[x2min,x2max],...,[xnmin,xnmax]] where n=dim

- acq_strategy: the acquisition strategy, proposed are :
    -- "basic" :
    -- "EI" : expected improvement
    
- ini_samp_strategy: the initial sampling strategy
    -- "basic" : regular hypercube -> 2**dimensions samples
    
@author: pierre
"""


from __future__ import print_function
from __future__ import division
from sklearn import gaussian_process
import numpy as np
pi=3.14


class BayesSolver(object):
    
    def __init__(self,dim,bounds,acq_strategy,ini_samp_strategy):
        self.dim=dim
        self.bounds=bounds
        if not dim==len(bounds):
            raise ValueError('dim and bounds size must match !')
        
        
        
        
        #iter represents the number of sampled queried
        self.iter=0
        
        #Acquisition strategy
        self.acq_strategy=acq_strategy
        
        #Initial Sampling strategy
        self.initial_sampling=perform_regular_initial_sampling(self.bounds)
        
        print("done")
        

    def adviseNewSample(self):
        
        res=[]           
        #initial sampling !!
        if (self.iter<len(self.initial_sampling)):
            self.iter=self.iter+1
            res=self.initial_sampling[self.iter]
        
        #After initial sampling not implemented yet        
        
        return res
        



#Translates a number to its binary equivalent in an array
#the array has taillemin minimal size
def bitfield(n,taillemin):
    
    res=np.array([1 if digit=='1' else 0 for digit in bin(n)[2:]])
    
    if len(res)<taillemin:
        res=np.concatenate((np.zeros(taillemin-len(res)),res),axis=0)
        
    return res

#Outputs the list of initial samples to take with these bounds
def perform_regular_initial_sampling(bounds):
    
    n=len(bounds)
    bound_l=np.array(bounds).reshape(-1,2)[:,0]
    bound_h=np.array(bounds).reshape(-1,2)[:,1]

    vec1=0.33*bound_l+0.66*bound_h
    vec2=0.66*bound_l+0.33*bound_h
    res=[]
    
    for i in range(2**n):
        res.append([vec1[k] if bitfield(i,n)[k]==0 else vec2[k] for k in range(n)])
        
    return np.array(res)