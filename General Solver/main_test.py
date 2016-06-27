# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 21:30:44 2016

Main program used for testing purposes.

It uses the blackbox testing module, and calls the bayesian solver in order to
test it on the "real" black-box.

@author: pierre
"""

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import bbox.blackbox_test as bboxt
import solver.bayes_solver as bayes
import plot.plot_helpers as ploth
pi=3.14

#Display graph or not
GRAPHE=1



#%% Main function
def main():
    #Create the black-box - choosing the function inside 2d_a
    bbox=bboxt.BlackBox("1d_b")
    dim_bbox=bbox.getDim()
    
    bounds=create_bounds_uniform_0_1(dim_bbox)    
    
    #We create the right interface for the user for the dimension of the black box
    if GRAPHE:
        fig,ax,ax2=ploth.create_interface(dim_bbox,GRAPHE)


    solver_b=bayes.BayesSolver(dim_bbox,
                             bounds,
                             "EI",
                             "basic",
                             verbose=1)
                             
    
    history=[]
    budget=30 #Number of allowed samples
    budget_ini=budget
                 
    #Initial sampling part
    while (budget>0):
        #We query the good advice of our wise solver
        new_samp_inputs=solver_b.adviseNewSample()
        
        #We print info to the user staring at his screen letting the pc do the work
        print_information_input(solver_b,bbox,new_samp_inputs,budget) #Console
        
        #Since we trust him, we query the black_box at that exact point
        new_samp_output=bbox.queryAt(new_samp_inputs)
        history.append([new_samp_inputs,new_samp_output])
        #We reward him with the new information
        solver_b.updateModel(new_samp_inputs,new_samp_output)
        
        #We print info to the user staring at his screen letting the pc do the work
        if GRAPHE:
            ploth.update_interface(dim_bbox,budget,budget_ini,ax,ax2,solver_b,bbox,history,new_samp_inputs,new_samp_output)
        print_information_output(solver_b,bbox,new_samp_output) #Console
        
        var = raw_input("Next ? ")
        print("")
        
        budget=budget-1 #deinc the budget




#%% Helpers

def print_information_input(solver,bbox,new_samp_inputs,budget):
    print("----------------")
    print(budget)
    print("Point sampled : "),
    print(new_samp_inputs)


def print_information_output(solver,bbox,new_samp_output):
    print("Output : "),
    print(new_samp_output)
    print("----------------")

def create_bounds_uniform_0_1(dim):
    return [[0,1] for i in range(dim)]

#%% Call main


if __name__ == "__main__":
    main()



