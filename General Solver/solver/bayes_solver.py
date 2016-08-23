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
from scipy.stats import norm
from operator import itemgetter, attrgetter, methodcaller
import math
import numpy as np
import utilities.lhsmdu as lhs
import copy

pi=3.14


#%%Class Solver, this is where the magic happens
class BayesSolver(object):
    
    def __init__(self,dim,bounds,acq_strategy,ini_samp_strategy,**kwargs):
        self.dim=dim
        self.bounds=bounds
        if not dim==len(bounds):
            raise ValueError('dim and bounds size must match !')
            
        self.verbose = kwargs.get('verbose', 0)
        
        self.bbox = kwargs.get('bb', None)
        self.bbox_cons = copy.deepcopy(kwargs.get('bb_cons', None))

        #Constraints
        self.nb_black_constraints = kwargs.get('nb_black_constraints', 0)
        self.white_constraints = kwargs.get('constraints_white', [])
        
        self.is_at_initial_sampling=True
                       
        #iter represents the number of sampled queried
        self.iter=0
        
        #Acquisition strategy
        self.acq_strategy=acq_strategy
        self.ini_samp_strategy=ini_samp_strategy
        
        #choos right acq strategy
        if (self.acq_strategy=="basic"):
            self.acq_func=acquisition_basic
        else:
            self.acq_func=acquisition_EI
        
        #Initial Sampling strategy
        if (self.ini_samp_strategy=="basic"):
            self.initial_sampling=perform_regular_initial_sampling(self.bounds)
        elif(self.ini_samp_strategy=="latin"):
            self.initial_sampling=np.array(lhs.sample(self.dim, 2*self.dim)).transpose()
        else:
            self.initial_sampling=perform_minimal_sampling(self.bounds)
        
        
        #Create GP
            #Objective Function
        self.gp = gaussian_process.GaussianProcess(nugget=0.000001)
        self.gp_focused = gaussian_process.GaussianProcess()
            #Blackbox Constraints
        self.gp_bb_constraints=[]
        for s in range(self.nb_black_constraints):
            self.gp_bb_constraints.append(gaussian_process.GaussianProcess(nugget=0.000001))

        
        #List of inputs and outputs known of the bbox - Empty at first
        self.inputs_real=[]
        self.output_real=[]
        
        self.output_bb_cons_real=[[] for s in range(self.nb_black_constraints)]
        
        #shift is used to sample constraints with robustness (abandonned hence set to 0)
        self.shift=0
        
        
        print("done creating the solver")
        


    #Given a new input and new output of the black-box, this function will update the model of the gp
    def updateModel(self,new_inputs,new_output,**kwargs):
        self.inputs_real.append(new_inputs)
        
        #objective Outputs Update
        self.output_real.append(new_output)
        
        #Normalisation of variables ? abandonned
        #self.output_normalised=list(self.output_real)
        #self.output_normalised=map(lambda x:x-min(self.output_normalised),self.output_normalised)
        #self.output_normalised=map(lambda x:x/max(self.output_normalised),self.output_normalised)
        
        #BlackBox Output constraint
        cons_out=kwargs.get('output_cons', []) 
        for i in range(self.nb_black_constraints):
            self.output_bb_cons_real[i].append(cons_out[i])
        
        if not len(self.output_real)==1: #Otherwise it raises an error...
            self.gp.fit(np.array(self.inputs_real),self.output_real)
            for i in range(self.nb_black_constraints):
                self.gp_bb_constraints[i].fit(np.array(self.inputs_real),self.output_bb_cons_real[i])
        




    #Will ask the user a new sample to take to optimize the search
    #At first it only asks for samples folloing the samplng strategy
    #Then, it maximises an acquisition function
    def adviseNewSample(self,**kwargs):
        
        strat=kwargs.get('strategy', "0")
        
        #initial sampling !!
        if (self.iter<len(self.initial_sampling)):
            res=self.initial_sampling[self.iter]
            self.iter=self.iter+1
            
            if(self.iter==len(self.initial_sampling)):
                self.is_at_initial_sampling=False
            
        elif(strat=="0"):
            print("Strategy 1 -", end="")
            res,best_acq_value=self.takeOptimumAcquisitionValue(self.gp,self.bounds,self.acq_func)
                
        elif(strat=="1"):
            print("Strategy 2 -", end="")
            #refined gp
            self.gp.nugget=0.0000000001            
            self.shift=0
            
            new_set_input,new_set_output,dist = keepNClosestToBest(self.inputs_real,self.output_real,np.array(self.output_bb_cons_real).transpose(),self.dim*4)
            #prevent from going out of bounds
            new_bounds=boundInter(self.bounds,[[new_set_input[0][i]-0.05,new_set_input[0][i]+0.05] for i in range(self.dim)])
            
            new_set_output=map(lambda x:np.log(x-min(new_set_output)+0.00000001),new_set_output) 
            
            
            self.gp_focused.fit(np.array(new_set_input),new_set_output)

            res,best_acq_value=self.takeOptimumAcquisitionValue(self.gp,new_bounds,acquisition_EI)
            
        else:
            print("Strategy 3 -", end="")
            best_input,best_output,dist = keepNClosestToBest(self.inputs_real,self.output_real,np.array(self.output_bb_cons_real).transpose(),1)
            
            #we perform SLSQP method with a verygood starting point
            mini=minimize(lambda x:self.bbox.queryAt(x.reshape(-1,1)),
             np.array(best_input).reshape(1, -1),
             bounds=self.bounds,
             method='SLSQP',
             constraints=convertConsListToTuple(self.bbox_cons),
             options={'disp':1}
             )
            
             
             #or put
                #method="Nelder-Mead",
             #options={'disp':1,'maxfev':3},                
            
            
            res=mini.x
            
            #Try simplex nelder mead method as well
             
        return res
        

    #This function globally optimises the acq value and outputs best locations and values
    def takeOptimumAcquisitionValue(self,GP,bounds,acq,**kwargs):
        b=np.array(bounds)     
        res=[]
          
        
        #To globally find the max of the acqu function, we used local solvers (minimize with LBFGSB method) with a lot of starting points
        starting_points_list=np.random.uniform(b[:,0],b[:,1],size=(self.dim*50,len(b))) #Took dim*50 starting points
        starting_points_list[0]=(self.inputs_real[self.output_real.index(min(self.output_real))])
        res=b[:,0] #random best at first
        best_acq_value=None
        
        tar=self.bestFeasibleOutputSoFar()

        for starting_point in starting_points_list:
            
            
            mini=minimize(lambda x:acq(GP,x.reshape(1, -1),
                                                 target=tar,
                                                 gp_constraints=self.gp_bb_constraints,
                                                 white_constraints=self.white_constraints,
                                                 inputs=self.inputs_real,
                                                 shift=self.shift),
                         starting_point.reshape(1, -1),
                         bounds=b,
                         method="L-BFGS-B")
            
     
            #Treatment done in order to work with all local solvers
            if type(mini.fun) is list:
                mini.fun=mini.fun[0]    
            if (len(mini.x) != self.dim): #case where mini.x is a list with useless elements
                mini.x=mini.x[0] 
            
            
            if (best_acq_value is None or mini.fun<best_acq_value):
                res=mini.x
                best_acq_value=mini.fun
       
        #Info printing
        if (self.verbose):
            balance=acquisition_EI_balance(GP,mini.x.reshape(1, -1),target=tar)
            print("Expected Improvement : {0}  -  Exploration : {1}  -  Exploitation : {2}".format(best_acq_value,balance[2],balance[1]))
        
        #avoid sampling the same point        
        while (list(res) in map(list,self.inputs_real)):
            print('I almost sampled the same point twice ! It was {0}, the acq value was {1}'.format(res,best_acq_value))
            self.shift=self.shift/2
            print(self.shift)
            res=self.bestExplorationPoint()
            #or We take a random point
            #res=np.random.uniform(b[:,0],b[:,1])
        
        return res,best_acq_value



    #Output the bestFeasibleOutput had so far given a list of outputs value (with constraints value)   
    def bestFeasibleOutputSoFar(self):
        
        l2=np.array(self.output_bb_cons_real).transpose()
        cong=zip(self.output_real,[list(s) for s in list(l2)])
        cong=filter(lambda x:all(item < 0 for item in x[1]),cong)
        
        if(self.nb_black_constraints==0):
            target=min(self.output_real)
        elif(cong==[]):
            target=max(self.output_real)
        else:
            output_real_filtered,output_bb_cons_filtered=zip(*cong)          
            target=min(output_real_filtered)
    
        return target
            
    #Output the point with the biggest variance      
    def bestExplorationPoint(self):
    
        b=np.array(self.bounds)
        starting_points_list=np.random.uniform(b[:,0],b[:,1],size=(50,len(b))) 
        best_acq_value=None
        
        for starting_point in starting_points_list:
            
            
            mini=minimize(lambda x:acquisition_var(self.gp,x.reshape(1, -1)),
                         starting_point.reshape(1, -1),
                         bounds=b,
                         method="L-BFGS-B")
            
     
            #Treatment done in order to work with all local solvers
            if type(mini.fun) is list:
                mini.fun=mini.fun[0]    
            if (len(mini.x) != self.dim): #case where mini.x is a list with useless elements
                mini.x=mini.x[0] 
            
            
            if (best_acq_value is None or mini.fun<best_acq_value):
                res=mini.x
                best_acq_value=mini.fun
            
            
            
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

#Minimal sampling is composed of two points only
def perform_minimal_sampling(bounds):
    
    n=len(bounds)
    bound_l=np.array(bounds).reshape(-1,2)[:,0]
    bound_h=np.array(bounds).reshape(-1,2)[:,1]

    vec1=0.33*bound_l+0.33*bound_h
    vec2=0.66*bound_l+0.66*bound_h
    res=[vec1,vec2]
    
        
    return np.array(res)
    
    
def acquisition_basic(gp,point,**kwargs):
    k=2
    #Retrieve gp constraints and white constraints
    gp_constraints = kwargs.get('gp_constraints', [])
    white_constraints = kwargs.get('white_constraints', [])
    
    #Predictions
    mean,variance=gp.predict(point,eval_MSE=True)
    if len(gp_constraints)>0:
        mean_bb_constraints,variance_bb_constraints=(list(zip(*[s.predict(point,eval_MSE=True) for s in gp_constraints])[0])
                                                    ,list(zip(*[s.predict(point,eval_MSE=True) for s in gp_constraints])[1]))

    #Objective Acauisition
    ress = mean-k*variance
    
    #Constraints handling
    if len(gp_constraints)>0:
        for meean in mean_bb_constraints:
            if meean>0:
                ress*=0
        
    for constraint in white_constraints:
        if constraint(point)>0:
            ress*=0

    return ress


def acquisition_EI(gp,point,**kwargs):
    target = kwargs.get('target', None)
    inputs = kwargs.get('inputs', None)
    
    shift=kwargs.get('shift', 0)
    
    
    #Retrieve gp constraints and white constraints
    gp_constraints = kwargs.get('gp_constraints', [])
    white_constraints = kwargs.get('white_constraints', [])
    
    #Predictions
    mean,variance=gp.predict(point,eval_MSE=True)
    if len(gp_constraints)>0:
        mean_bb_constraints,variance_bb_constraints=(list(zip(*[s.predict(point,eval_MSE=True) for s in gp_constraints])[0])
                                                    ,list(zip(*[s.predict(point,eval_MSE=True) for s in gp_constraints])[1]))

    if np.any(inputs==point):
        return 0
        

    
    #Objective Acauisition
    if (variance!=0):
        z=(target-mean)/np.sqrt(variance)
    else:
        z=0
    
    
    ress=-np.sqrt(variance)*(z*norm.cdf(z)+norm.pdf(z))


    #Constraints handling
    if len(gp_constraints)>0:
        for meean,vaar in zip(mean_bb_constraints,variance_bb_constraints):
            if vaar!=0:
                ress*=(norm.cdf(-(meean+shift)/vaar))
            elif meean>0:
                ress*=0
        
    for constraint in white_constraints:
        if constraint(point)>0:
            ress*=0
        
    
    return ress


def acquisition_var(gp,point,**kwargs):
   
    #Predictions
    mean,variance=gp.predict(point,eval_MSE=True)

    return -variance



#the function below is used to know how much we explored or exploited
def acquisition_EI_balance(gp,point,**kwargs):
    target = kwargs.get('target', None)
    mean,variance=gp.predict(point,eval_MSE=True)

    if (variance!=0):
        z=(target-mean)/np.sqrt(variance)
    else:
        z=(target-mean)
    
    normal=(z*norm.cdf(z)+norm.pdf(z))
        
    return z,(z*norm.cdf(z)),(norm.pdf(z))
    
    
#Keep N closest to best samples    
def keepNClosestToBest(inputs,outputs,outputs_cons,n):
    if (list(outputs_cons)==[]):
        outputs_cons=[[-1] for i in range(len(inputs))]
    best_inputs,best_outputs,best_outputs_cons=zip(*sorted(zip(inputs,outputs,outputs_cons),key=itemgetter(1)))
    new_set_input=list(best_inputs)
    new_set_output=list(best_outputs)
    
    #We stop at the best and feasible output
    i=0    
    while (i<len(outputs_cons) and np.any(best_outputs_cons[i]>0)):
        i=i+1
    if(i==len(outputs_cons)):
        best=best_inputs[0]
    else:
        best=best_inputs[i]
  
  
    distance_to_best=list(new_set_input)
    distance_to_best=map(lambda x : compute2Norm(x-best),distance_to_best)    
    

    
    new_set_input,new_set_output,distance_to_best=zip(*sorted(zip(new_set_input,new_set_output,distance_to_best),key=itemgetter(2)))
    
    new_set_input=list(new_set_input)[0:n]
    new_set_output=list(new_set_output)[0:n]    
    
    
    return new_set_input,new_set_output,distance_to_best[n-1]



#Utility straightforward functions

def compute2Norm(array):
    return math.sqrt(sum(array*array))

def boundInter(boundlist1,boundlist2):
    reslist=[]
    for i in range(len(boundlist1)):
        reslist.append([max(boundlist1[i][0],boundlist2[i][0]),min(boundlist1[i][1],boundlist2[i][1])])
    
    return reslist


def compose2(f, g):
    return lambda x: f(g(x))

def neg(x):
    return (-1)*(x+0.0000001)

def convertConsListToTuple(list_cons):
    res=[{'type': 'ineq', 'fun':compose2(neg,box.queryAt)} for box in list_cons]
    res=tuple(res)   
    
    return res

