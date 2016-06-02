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
import numpy as np
import bbox.blackbox_test as bboxt
import solver.bayes_solver as bayes
pi=3.14


def main():
    #Create the black-box - choosing the function inside 2d_a
    bbox=bboxt.BlackBox("2d_a")
    dim_bbox=bbox.getDim()
    
    solver=bayes.BayesSolver(dim_bbox)
    
    print(bbox.queryAt([2,1]))
    
if __name__ == "__main__":
    main()