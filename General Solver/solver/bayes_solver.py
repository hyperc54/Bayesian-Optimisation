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
from scipy.optimize import minimize
import numpy as np
pi=3.14


#%%Class Solver, this is where the magic happens
class BayesSolver(object):
    
    def __init__(self,dim,bounds,acq_strategy,ini_samp_strategy):
        self.dim=dim
        self.bounds=bounds
        if not dim==len(bounds):
            raise ValueError('dim and bounds size must match !')
        
        self.is_at_initial_sampling=True
                       
        #iter represents the number of sampled queried
        self.iter=0
        
        #Acquisition strategy
        self.acq_strategy=acq_strategy
        
        #Initial Sampling strategy
        self.initial_sampling=perform_regular_initial_sampling(self.bounds)
        
        #Create GP
        self.gp = gaussian_process.GaussianProcess()
        
        #List of inputs and outputs known of the bbox
        #Empty at first
        self.inputs_real=[]
        self.output_real=[]
        
        print("done creating the solver")
        


    #Given a new input and new output of the black-box, this function will update the model of the gp
    def updateModel(self,new_inputs,new_output):
        self.inputs_real.append(new_inputs)
        self.output_real.append(new_output)
        
        if not len(self.output_real)==1: #Otherwise it raises an error...
            self.gp.fit(np.array(self.inputs_real),self.output_real)
        


    #Will ask the user a new sample to take to optimize the search
    #At first it only asks for samples folloing the samplng strategy
    #Then, it maximises an acquisition function
    def adviseNewSample(self):
        
        res=[]

        #initial sampling !!
        if (self.iter<len(self.initial_sampling)):
            res=self.initial_sampling[self.iter]
            self.iter=self.iter+1
            
            if(self.iter==len(self.initial_sampling)):
                self.is_at_initial_sampling=False
            
        else:
            #To globally find the max of the acqu function, we used local solvers (minimize with LBFGSB method) with a lot of starting points
            b=np.array(self.bounds)
            starting_points_list=np.random.uniform(b[:,0],b[:,1],size=(50,len(b))) #Took 100 starting points
            res=b[:,0] #random best at first
            best_acq_value=None
            
            for starting_point in starting_points_list:
                
                    
                
                mini=minimize(lambda x:acquisition_basic(self.gp,x.reshape(1, -1)),
                             starting_point.reshape(1, -1),
                             bounds=b,
                             method="L-BFGS-B")
                             
                if (best_acq_value is None or mini.fun[0]<best_acq_value):
                    res=mini.x
                    best_acq_value=mini.fun[0]
                
            
        return res
        



#%%Helpers functions

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
    

def acquisition_basic(gp,point):
    k=2
    mean,variance=gp.predict(point,eval_MSE=True)
    return mean-k*variance