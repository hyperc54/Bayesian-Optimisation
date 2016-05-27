
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:12:38 2016

@author: pierre
"""

import numpy as np
from sklearn import gaussian_process
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
pi=3.14


X=np.atleast_2d([0])
y=np.array([0])
x=np.atleast_2d([0])
ind2=0

PARAMETER_ACQUISITION_VARIANCE=10



#Real function points
def compute_real_points(f):
    x_real=np.linspace(0,1,100)
    y_real=f(x_real).ravel()
    
    return x_real,y_real


#Four points sampled
def initial_sampling(f):
    #X = np.atleast_2d([0.,.2,.5,1.,.39]).T
    X = np.atleast_2d([0.25,.75]).T
    y = f(X).ravel()
    
    return X,y



def GP_call(X,y):
    #For the approximation
    x = np.atleast_2d(np.linspace(0, 1, 200)).T
    #GP call
    gp = gaussian_process.GaussianProcess()
    gp.fit(X, y)  
    y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)
    
    return x,y_pred,sigma2_pred

def GP_Plus_Acquisition(X,y):
    #regression with GP
    x,y_pred,sigma2_pred=GP_call(X,y)
    
    #acquisitoon function
    y_acquisition=y_pred-PARAMETER_ACQUISITION_VARIANCE*np.sqrt(sigma2_pred)
    mx=max(y_acquisition)
    ind=list(y_acquisition).index(mx)
    mn=min(y_acquisition)
    ind2=list(y_acquisition).index(mn)
    
    return x,y_pred,sigma2_pred,y_acquisition,mx,ind,mn,ind2


#Plots
def update_plot(ax,X,y,x,y_pred,sigma2_pred,y_acquisition,mx):
    ax.clear()
    ax2.clear()
    ax.scatter(X,y,s=400) #Sampled points
    ax.set_xlim([0,1])
    ax.plot(x,y_pred) #Approximation
    ax.fill_between(x.ravel(),y_pred-5*np.sqrt(sigma2_pred),y_pred+5*np.sqrt(sigma2_pred),color='black',alpha=0.1) #Confidence intervals
    ax.plot(x_real,y_real) #True function
    ax2.fill_between(x.ravel(),np.zeros(len(x)),(-y_acquisition+mx),color='green',alpha=0.1)


def plus_one(X,y,x,ind2):
    #new sample is at the max of acquisition func
    X=np.append(X,x[ind2]).reshape(-1,1)
    y = f(X).ravel()
    
    return X,y

    
class Index(object):
    
    def next(self, event):
        global X
        global y
        global x
        global ind2
        X,y=plus_one(X,y,x,ind2)
        x,y_pred,sigma2_pred,y_acquisition,mx,ind,mn,ind2=GP_Plus_Acquisition(X,y)
        update_plot(ax,X,y,x,y_pred,sigma2_pred,y_acquisition,mx)

#Figure
def create_interface():
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    callback = Index()
    axnext = plt.axes([0.81, 0.01, 0.05, 0.05])
    bnext = Button(axnext, '+')
    bnext.on_clicked(callback.next)
    
    return fig,ax,ax2,callback,bnext




#################################################
###############    MAIN         #################
#################################################

#Function definition
def f(x):
    return 3*((1.65*x-0.5)**2)*((np.cos(1.65*pi*x))**2)+0.2*np.sin(1.65*5*x)
    #return np.tan(x)
    #return np.tan(x)*(1.5+np.cos(3*pi*x))
    #return 0.5*(1-np.exp(-((4*x-3)**2)/2))-np.exp(-(46*x-40)**2)

##INITIALISATION##

#Create Interface
fig,ax,ax2,callback,bnext=create_interface()

#Compute real function f values
x_real,y_real=compute_real_points(f)

#sample few points on f
X,y=initial_sampling(f)

#regression plus acquistion
x,y_pred,sigma2_pred,y_acquisition,mx,ind,mn,ind2=GP_Plus_Acquisition(X,y)

#FIRST PLOT
update_plot(ax,X,y,x,y_pred,sigma2_pred,y_acquisition,mx)

##INITIALISATION FIN##
        






