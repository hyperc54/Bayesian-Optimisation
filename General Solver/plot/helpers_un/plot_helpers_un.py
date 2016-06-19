# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 20:00:46 2016

Plot functions 1D

@author: pierre
"""


from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn import gaussian_process

def create_interface_1d(idfig):
    fig = plt.figure(idfig)
    ax = fig.add_subplot(211)#Obj
    ax2 = fig.add_subplot(212)#Acquisition
    plt.show()
    plt.draw()
    
    return fig,ax,ax2
    
    
def update_interface_1D(ax,ax2,solver,bbox,history):
    x = np.linspace(0, 1, 80)
    z=map(lambda y:bbox.queryAt([y]),x)
    xx=np.atleast_2d(x).T
    z_pred, sigma2_pred = solver.gp.predict(xx, eval_MSE=True)
    ax.clear()
    ax2.clear()
    ax.plot(x,np.array(z))
    ax.scatter(np.array(history),map(lambda y:bbox.queryAt([y]),history))
    ax2.plot(x,np.array(z_pred))
    ax2.fill_between(x,np.array(z_pred)+np.array(np.sqrt(sigma2_pred)),np.array(z_pred)-np.array(np.sqrt(sigma2_pred)),alpha=0.2)
    ax.set_xlim([0,1])
    ax2.set_xlim([0,1])
    ax2.plot(x)
    target=min(history)
    mean,variance=solver.gp.predict(xx,eval_MSE=True)
    z=(target-mean)/np.sqrt(variance)
    ax2.plot(x,np.sqrt(variance)*(z*norm.cdf(z)+norm.pdf(z)))