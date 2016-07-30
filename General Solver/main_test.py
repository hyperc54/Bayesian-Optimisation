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

pi=3.14



#%% SETUP PARAMETERS

#Display graph or not
GRAPHE=1

#Objective Function
OBJ_NAME="minlp1_obj"
IS_OBJ_BLACK=True

#Constraints
    #Black - Ex ["1d_d","1d_b"]
BBOX_CONS_NAMES=["minlp1_c1","minlp1_c2"]
    #White - Ex [wboxt.1d_d,wboxt.1d_b]
WBOX_CONS_NAMES=[]

#bounds=[[1,5.5],[1,5.5]]


#%% Main function
def main():
    
    #Problem Setup
    #Objective Function
    if IS_OBJ_BLACK:
        bbox=boxt.BlackBox(OBJ_NAME)
    else:
        bbox=boxt.WhiteBox(OBJ_NAME)
        
    dim_bbox=bbox.getDim()
    
    
    bounds=[[1,5.5],[1,5.5]]
    #bounds=create_bounds_uniform_0_1(dim_bbox)
        
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
                             "basic",
                             constraints_white=white_box_constraints,
                             nb_black_constraints=nb_bb_cons,
                             verbose=1,
                             bb=bbox)
                             
    
    #Budget Setup
    history=[]
    budget=50 #Number of allowed samples
    budget_ini=budget
    var=0

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
        history.append([np.array(new_samp_inputs),new_samp_output])    
        #print(history)
        
        #We reward him with the new information
        solver_b.updateModel(new_samp_inputs,new_samp_output,output_cons=new_samp_output_cons)
        
        #We print info to the user staring at his screen letting the pc do the work
        if GRAPHE==1 and budget<(budget_ini-1):
            plotter.updateInterface(solver_b,bbox,history,new_samp_inputs,new_samp_output,bounds)
            #plotter.updateInterface(solver_b,bbox,history,new_samp_inputs,new_samp_output)
        print_information_output(solver_b,bbox,new_samp_output,new_samp_output_cons) #Console
        
        var = raw_input("Next ? ")
        
        budget=budget-1 #deinc the budget




#%% Helpers

def print_information_input(solver,bbox,new_samp_inputs,budget):
    print("[{0}] - Point sampled : {1} - ".format(budget,new_samp_inputs),end="")


def print_information_output(solver,bbox,new_samp_output,new_samp_output_cons):
    print("Output : {0}  - ".format(new_samp_output),end="")

    for ind,out in enumerate(new_samp_output_cons):
        print("C",end=""),print(ind,end=""),print(" : ",end=""),print(out,end=""),print(" - ",end="")
    if any(i>0 for i in new_samp_output_cons):
        print("  INFEASIBLE")

def create_bounds_uniform_0_1(dim):
    return [[0,1] for i in range(dim)]

#%% Call main


if __name__ == "__main__":
    main()



