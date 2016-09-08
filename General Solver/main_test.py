# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 21:30:44 2016

Main program used for testing purposes.

It uses the blackbox testing module, and calls the bayesian solver in order to
test it on the "real" black-box.

@author: pierre
"""

#%% IMPORTS

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import box.box_test as boxt
import solver.bayes_solver as bayes
import plot.plot_helpers as ploth


import multiprocessing
import time

import inspect
import pickle

pi=3.14



#%% SETUP PARAMETERS

#Display graph or not
GRAPHE=0
#Displqy results qt the end
SHOW_RESULTS=True


#Objective Function
OBJ_NAME="minlp1_obj"
IS_OBJ_BLACK=True

#Constraints
    #Black - Ex ["1d_d","1d_b"]
BBOX_CONS_NAMES=["minlp1_c1","minlp1_c2"]
    #White
WBOX_CONS_NAMES=[]

NEW_BOUNDS_SIZE_LIMIT=0.15

#Choose either basic, latin,or min
SAMP_STRAT="latin"

#Input original bounds - Have to be compatible with the problem dimension
bounds=[[1,5.5],[1,5.5]]

#Sampling budget
BUDGET_I=80

history=[]

#%% Main function
def main():
    
    #Problem Setup
    #Objective Function
    if IS_OBJ_BLACK:
        bbox=boxt.BlackBox(OBJ_NAME)
    else:
        bbox=boxt.WhiteBox(OBJ_NAME)
        
    dim_bbox=bbox.getDim()
    
    
    #Constraints-for now make sure dimensions match
        #BlackBox Constraints
    black_box_constraints=[boxt.BlackBox(s) for s in BBOX_CONS_NAMES]
    nb_bb_cons=len(black_box_constraints)
        #WhiteBox Constraints
    white_box_constraints=[boxt.WhiteBox(s).getFunc() for s in WBOX_CONS_NAMES]
    
    
    #We create the right interface for the user for the dimension of the black box
    if GRAPHE:
        ##GrapheObj=ploth.ObjectGraph(dim_bbox)
        plotter = ploth.ObjectGraph(dim_bbox)
    
    #Solver Setup
    solver_b=bayes.BayesSolver(dim_bbox,
                             bounds,
                             "EI",
                             SAMP_STRAT,
                             constraints_white=white_box_constraints,
                             nb_black_constraints=nb_bb_cons,
                             verbose=1,
                             bb=bbox,
                             bb_cons=black_box_constraints,
                             new_bounds_size_limit=NEW_BOUNDS_SIZE_LIMIT)
                             
    
    #Budget Setup
    best=1e9
    age_best=0
    budget=BUDGET_I #Number of allowed samples
    budget_ini=budget
    var="0"

    while (budget>0):
        #We query the good advice of our wise solver
        new_samp_inputs=solver_b.adviseNewSample(strategy=var)
        
        #We print info to the user staring at his screen letting the pc do the work
        print_information_input(solver_b,bbox,new_samp_inputs,budget) #Console
        
        #Since we trust him, we query the black_box at that exact point
            #Objective
        new_samp_output=bbox.queryAt(new_samp_inputs)
            #Constraints
        new_samp_output_cons=[s.queryAt(new_samp_inputs) for s in black_box_constraints]
        new_samp_output_cons_wb=[func(new_samp_inputs) for func in white_box_constraints]        
        
        history.append([np.array(new_samp_inputs),new_samp_output,new_samp_output_cons,new_samp_output_cons_wb])
        
        #We reward him with the new information
        solver_b.updateModel(new_samp_inputs,new_samp_output,output_cons=new_samp_output_cons)
        
        #We print info to the user staring at his screen letting the pc do the work
        if GRAPHE==1 and budget<(budget_ini-1):
            plotter.updateInterface(solver_b,bbox,history,new_samp_inputs,new_samp_output,bounds)
            #plotter.updateInterface(solver_b,bbox,history,new_samp_inputs,new_samp_output)
        print_information_output(solver_b,bbox,new_samp_inputs,new_samp_output,new_samp_output_cons,white_box_constraints) #Console
        
        
        
        '''
        #Used to manually set the strategy to use   - 0 -> Constrained Bayesian Optimisation
                                                    - 1 -> Constrained Bayesian Optimisation on reduces GP to refine optimums
                                                    - 2 -> Quasi-Newton local solver
        
        var = raw_input("Next ? ")
        '''
        
        
        '''        
        #Used to automatically switch strategy if required
        #age_best is the number of unsuccessful tries threshold before switching 
        if (np.all(np.array(new_samp_output_cons)<0) and new_samp_output<best):
            best=new_samp_output
            age_best=0
        else:
            age_best=age_best+1
        
        if age_best>dim_bbox*10:
            #switch strategy
            var="1" if var=="0" else "0"
            age_best=0
        '''
        
        
        budget=budget-1 #deinc the budget
    

    
    if(SHOW_RESULTS):
        l=map(list, zip(*history))
        res_outputs=l[1]
        res_outputs_cons=l[2]
        res_outputs_cons_wb=l[3]
        
        for i in range(len(res_outputs)):
            if(np.any(np.array(res_outputs_cons[i])>0) or np.any(np.array(res_outputs_cons_wb[i])>0)):
                res_outputs[i]=1e9
            res_outputs[i]=min(res_outputs[0:i+1])
            
    
    
    #Output results in log files, + save python object
    pickle.dump(res_outputs, open( "results_logs/"+SAMP_STRAT+OBJ_NAME+".p", "wb" ))
    open( "results_logs/"+SAMP_STRAT+"_"+OBJ_NAME+"string.p", "wb" ).write("["+",".join(map(str,res_outputs))+"]")




#%% Helpers

def print_information_input(solver,bbox,new_samp_inputs,budget):
    print("\x1b[1;31m[{0}]\x1b[0m - Point sampled : \x1b[1;31m{1}\x1b[0m - ".format(budget,new_samp_inputs),end="")


def print_information_output(solver,bbox,new_samp_inputs,new_samp_output,new_samp_output_cons,white_box_constraints):
    print("Output : \x1b[1;31m{0}\x1b[0m  - ".format(new_samp_output),end="")

    wb_cons=[func(new_samp_inputs) for func in white_box_constraints]
    
    for ind,out in enumerate(new_samp_output_cons):
        print("BB-C",end=""),print(ind,end=""),print(" : ",end=""),print(out,end=""),print(" - ",end="")
    for ind,out in enumerate(wb_cons):
        print("WB-C",end=""),print(ind,end=""),print(" : ",end=""),print(out,end=""),print(" - ",end="")
    if any(i>0 for i in new_samp_output_cons) or any(i>0 for i in wb_cons):
        print("  INFEASIBLE")
    else:
        print(" ") #newline

def create_bounds_uniform_0_1(dim):
    return [[0,1] for i in range(dim)]

#%% Call main


if __name__ == "__main__":
    main()



