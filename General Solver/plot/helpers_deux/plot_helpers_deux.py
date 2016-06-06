# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:40:24 2016

Plot functions 2D

@author: pierre
"""

from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def create_interface_2d(idfig):
    fig = plt.figure(idfig)
    ax = fig.add_subplot(211)#Obj
    ax2 = fig.add_subplot(212)#Acquisition
    plt.show()
    plt.draw()
    
    return fig,ax,ax2
    
    
def update_interface_2D(ax,ax2,solver,bbox,history):
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    x,y=np.meshgrid(x, y)
    xx=x.ravel()
    yy=y.ravel()
    z=map(lambda x:bbox.queryAt(x),[np.array(i) for i in zip(xx,yy)])
    points_pred= np.array(map(lambda s:np.array(s),zip(xx,yy)))
    z_pred, sigma2_pred = solver.gp.predict(points_pred, eval_MSE=True)
    ax.clear()
    ax2.clear()
    c1=ax.contourf(x,y,np.array(z).reshape(-1,len(x[0])))
    ax.scatter(np.array(history).reshape(-1,2)[:,0],np.array(history).reshape(-1,2)[:,1])
    c2=ax2.contourf(x,y,np.array(z_pred).reshape(-1,len(x[0])))
    c1.set_clim(min(z),max(z))
    c2.set_clim(min(z),max(z))