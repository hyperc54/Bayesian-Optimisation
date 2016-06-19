# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:33:28 2016

Interface creation for some dimensions (1D,2D,3D)


@author: pierre
"""


from __future__ import print_function
from __future__ import division
import matplotlib.pyplot as plt
import helpers_un.plot_helpers_un as ploth1D
import helpers_deux.plot_helpers_deux as ploth2D
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
pi=3.14


def create_interface(dim_bbox,GRAPHE):
    fig=None
    ax=None
    ax2=None
    
    #Cas 1D
    if dim_bbox==1 and GRAPHE:
        fig,ax,ax2=ploth1D.create_interface_1d(1) 
        show_plot(1)
       
    
    #Cas 2D
    if dim_bbox==2 and GRAPHE:
        fig,ax,ax2=ploth2D.create_interface_2d(1) 
        show_plot(1)
    
    return fig,ax,ax2
    



def update_interface(dim_bbox,budget,budget_ini,ax,ax2,solver_b,bbox,history,new_samp_inputs,new_samp_output):
    if dim_bbox==1 and budget<budget_ini:
        ploth1D.update_interface_1D(ax,ax2,solver_b,bbox,history)
    if dim_bbox==2 and budget<budget_ini:
        ploth2D.update_interface_2D(ax,ax2,solver_b,bbox,history)

    show_plot(1)


def show_plot(figure_id=None):    
    if figure_id is not None:
        fig = plt.figure(num=figure_id)
    else:
        fig = plt.gcf()

    plt.show()
    plt.pause(1e-9)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()
