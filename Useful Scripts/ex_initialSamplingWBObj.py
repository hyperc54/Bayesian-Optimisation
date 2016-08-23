# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:03:11 2016

This script is an example of an efficient initial sampling method for a problem
with black-box functions involved but with a white-box objective function.

@author: pt914
"""

#%% IMPORTS

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


import multiprocessing
import time

from scipy.optimize import minimize

import inspect
import pickle

pi=3.14


#%% Function definition

def f(x):
    return -(1.5*np.exp(-3*((x[0]-0.8)**2+(x[1]-0.2)**2))+2*np.exp(-10*((x[0]-0.2)**2+(x[1]-0.7)**2)))


def distanceToBounds(point,bounds):#1-norm
    res=1e9
    for ind,bound in enumerate(bounds):
        res=min(abs(point[ind]-bound[0]),res)
        res=min(abs(point[ind]-bound[1]),res)
    
    threshold=0.1*(bounds[0][1]-bounds[0][0])
    if res>threshold:
        res=threshold
    
    return res


def distanceMinToOtherSamples(point,samples_list):#
    res=1e9

    
    for ind,sample in enumerate(samples_list):
        res=min(distanceFromTo(point,sample),res)
    
    return res

def distanceFromTo(point,point2):#
    res=0
    for ind,coord in enumerate(point):
        res+=(point[ind]-point2[ind])**2
    
    return np.sqrt(res)



b=np.array([[0,1],[0,1]])

samples=[]

def funcToMinimize(point):
    return f(point)*(1-np.exp(-1*(distanceToBounds(point,b))**2))*(1-np.exp((-1*distanceMinToOtherSamples(point,samples)))**2)
    
    



#%% Actual optimisation

for i in range(4):
       
    starting_points_list=np.random.uniform(b[:,0],b[:,1],size=(100,len(b))) #Took 100 starting points
    best_acq_value=None

    for starting_point in starting_points_list:
        
        mini=minimize(lambda x:funcToMinimize(x),
                     starting_point.reshape(1, -1),
                     bounds=b,
                     method="L-BFGS-B")
                     
                     
        
        if (best_acq_value is None or mini.fun<best_acq_value):
            res=mini.x
            best_acq_value=mini.fun
            
                
    samples.append(mini.x)
    


#%% Plot for 2D 
x = np.linspace(b[0][0], b[0][1], 150)
y = np.linspace(b[1][0], b[1][1], 150)
x,y=np.meshgrid(x, y)
xx=x.ravel()
yy=y.ravel()


z=zip(xx,yy)
z=map(f,z)

fig=plt.figure()

ax=fig.add_subplot(111)

ax.contourf(x,y,np.array(z).reshape(-1,len(x[0])),20)  
ax.scatter(np.array(samples).transpose()[0,:],np.array(samples).transpose()[1,:],s=30,color='red')

ax.scatter([0.33,0.33,0.66,0.66],[0.33,0.66,0.33,0.66],s=30,color='green')






