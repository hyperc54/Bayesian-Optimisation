# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:12:38 2016

This script performs an unconstrained bayesian optimisation on a 1D objective function.
The algorithm uses Gaussian Process ith a squared exponential kernel, points that do not satisfy the approximate constraints
are out of the acquisition function.

@author: pierre
"""

import numpy as np
from sklearn import gaussian_process
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from scipy.stats import norm
pi=3.14


#%% Some parameters to tweak
PARAMETER_ACQUISITION_VARIANCE=10 #Strength of the variance for the acquisition function, for basic acquisition function only
INITIAL_SAMPLING=[0.25,.75] #Initial sampling points, can't be empty
#Box constraints
BOUND_L=0.0
BOUND_U=1.0
#CHOOSE YOUR ACQUISITION FUNCTION , 0=basic, 1=EI
ACQUIS=1



#%% Function Class
#Handles the real value, the sampled points, and the GP prediction
class Function(object): 
    
    def __init__(self,sub_plot,black_box_function):
        self.x_real=[]
        self.y_real=[]
        
        self.x_sample=[]
        self.y_sample=[]
        
        self.x_pred=[]
        self.y_pred=[]
        self.sigma2_pred=[]
        
        self.x_acquis=[]
        self.y_acquis=[]
        
        self.setAxis(sub_plot)
        self.computeRealPoints(black_box_function)
        self.performInitialSampling(black_box_function)
        
        self.bbfonc=black_box_function

    def setAxis(self,ax):
        self.ax=ax

    def computeRealPoints(self,f):
        self.x_real=np.linspace(BOUND_L,BOUND_U,100)
        self.y_real=f(self.x_real).ravel()

    def performInitialSampling(self,f):
        self.x_sample = np.atleast_2d(INITIAL_SAMPLING).T
        self.y_sample = f(self.x_sample).ravel()
        
    def computeGaussianProcessApproximation(self):
        #For the approximation
        self.x_pred = np.atleast_2d(np.linspace(BOUND_L, BOUND_U, 200)).T
        #GP call
        gp = gaussian_process.GaussianProcess()
        gp.fit(self.x_sample, self.y_sample)  
        self.y_pred, self.sigma2_pred = gp.predict(self.x_pred, eval_MSE=True)

    def sampleNewPoint(self,indice):
        #new sample is located at indice
        self.x_sample=np.append(self.x_sample,self.x_pred[indice]).reshape(-1,1)
        self.y_sample = ((self.bbfonc)(self.x_sample)).ravel()

    def computeAcquisitionFunction(self):
        #acquisitoon function
        self.x_acquis=self.x_pred
        self.y_acquis=self.y_pred-PARAMETER_ACQUISITION_VARIANCE*np.sqrt(self.sigma2_pred)
        self.y_acquis=-self.y_acquis+max(self.y_acquis)

    def computeAcquisitionFunction_EI(self,target):
        #acquisitoon function
        self.x_acquis=self.x_pred
        z=(target-self.y_pred)/np.sqrt(self.sigma2_pred)
        self.y_acquis=np.sqrt(self.sigma2_pred)*(z*norm.cdf(z)+norm.pdf(z))

        
    def updatePlot(self):
        self.ax.clear()
        self.ax.plot(self.x_real,self.y_real)
        self.ax.scatter(self.x_sample,self.y_sample,s=400)       
        self.ax.plot(self.x_pred,self.y_pred)
        
        self.ax.fill_between(self.x_pred.ravel(),self.y_pred-2*np.sqrt(self.sigma2_pred),self.y_pred+2*np.sqrt(self.sigma2_pred),color='black',alpha=0.1) #Confidence intervals
        ax2.clear()
        ax2.fill_between(self.x_acquis.ravel(),np.zeros(len(self.y_acquis)),self.y_acquis,color='green',alpha=0.1)

    def update_plot_add_acquisition(self):
        a=1
        #donothing for now

#%% Index
#Button class for interactive simulation
class Index(object):
    
    def next(self, event):
        #new sample
        ind=list(f_obj.y_acquis).index(max(f_obj.y_acquis))
        f_obj.sampleNewPoint(ind)

        f_obj.computeGaussianProcessApproximation()
        if ACQUIS:
            f_obj.computeAcquisitionFunction_EI(min(f_obj.y_sample))
        else:
            f_obj.computeAcquisitionFunction()
       
        #plot everything
        f_obj.updatePlot()
        ax.set_xlim(BOUND_L,BOUND_U)




#%% Interface creation
def create_interface():
    fig = plt.figure()
    ax = fig.add_subplot(211)#Obj
    ax2 = fig.add_subplot(212)#Acquisition

    
    ax.set_xlim(BOUND_L,BOUND_U)
    ax2.set_xlim(BOUND_L,BOUND_U)
   
    callback = Index()
    axnext = plt.axes([0.81, 0.01, 0.05, 0.05])#ButtonPosition
    bnext = Button(axnext, '+')#Button
    bnext.on_clicked(callback.next)#ButtonCallbackset
    
    return fig,ax,ax2,axg1,axg2,callback,bnext


    
#%% Functions definition
def f(x): #Objective
    return 3*((1.65*x-0.5)**2)*((np.cos(1.65*pi*x))**2)+0.2*np.sin(1.65*5*x)


#%% Main

    # Initialisation
fig,ax,ax2,axg1,axg2,callback,bnext=create_interface()

f_obj=Function(ax,f)

    # First Round

f_obj.computeGaussianProcessApproximation()
if ACQUIS:
    f_obj.computeAcquisitionFunction_EI(min(f_obj.y_sample))
else:
    f_obj.computeAcquisitionFunction()
f_obj.updatePlot()
ax.set_xlim(BOUND_L,BOUND_U)




