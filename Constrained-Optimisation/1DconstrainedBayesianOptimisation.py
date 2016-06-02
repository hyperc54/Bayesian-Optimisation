
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:12:38 2016

This script performs a bayesian optimisation on a 1D objective function over 2 1D constraints.
The algorithm uses Gaussian Process ith a squared exponential kernel.

Two options are provided for the acquisition function :
 ACQUIS=0 : -mean+k*standarddev for the obj function (you can tweak k=PARAMETER_ACQUISITION_VARIANCE)
             + points that are presumed infeasible are set to 0

 ACQUIS=1 : Expected Improvement acquisition function
             + multiplied by the probability of being feasible (constraints presumed independant)


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
PARAMETER_ACQUISITION_VARIANCE=10 #Strength of the variance for the acquisition function
INITIAL_SAMPLING=[0.25,.75] #Initial sampling points, can't be empty
#Box constraints
BOUND_L=0.0
BOUND_U=1.0
#ACQUIS FUNCTION 0=basic 1=EI
ACQUIS=1



#%% Function Class
#Handles the real value, the sampled points, and the GP prediction
class Function(object): 
    
    def __init__(self,sub_plot,black_box_function,**kwargs):
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
        self.constraint=kwargs.get('constraint')

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

    def computeAcquisitionFunction(self,infeasibility_vec):
        #acquisitoon function
        self.x_acquis=self.x_pred
        self.y_acquis=self.y_pred-PARAMETER_ACQUISITION_VARIANCE*np.sqrt(self.sigma2_pred)
        self.y_acquis=-self.y_acquis+max(self.y_acquis)
        self.y_acquis=np.multiply(infeasibility_vec,self.y_acquis)

    def computeAcquisitionFunction_EI(self,target,c1,c2):
        #acquisitoon function
        self.x_acquis=self.x_pred
        z=(target-self.y_pred)/np.sqrt(self.sigma2_pred)
        self.y_acquis=np.sqrt(self.sigma2_pred)*(z*norm.cdf(z)+norm.pdf(z))  
        self.y_acquis=self.y_acquis*norm.cdf(-c1.y_pred/np.sqrt(c1.sigma2_pred))
        self.y_acquis=self.y_acquis*norm.cdf(-c2.y_pred/np.sqrt(c2.sigma2_pred))
        
    def updatePlot(self):
        self.ax.clear()
        self.ax.plot(self.x_real,self.y_real)
        self.ax.scatter(self.x_sample,self.y_sample,s=400)       
        self.ax.plot(self.x_pred,self.y_pred)
        
        if self.constraint:
            self.ax.plot(self.x_real,np.zeros(len(self.x_real)))
        else:
            self.ax.fill_between(self.x_pred.ravel(),self.y_pred-5*np.sqrt(self.sigma2_pred),self.y_pred+5*np.sqrt(self.sigma2_pred),color='black',alpha=0.1) #Confidence intervals
            ax2.clear()
            if ACQUIS==0:
                ax2.fill_between(self.x_acquis.ravel(),np.zeros(len(self.x_acquis)),[(1-x)*(max(self.y_acquis)) for x in infeasibility],color='red',alpha=0.1)
            ax2.fill_between(self.x_acquis.ravel(),np.zeros(len(self.y_acquis)),self.y_acquis,color='green',alpha=0.1)


#%% Index
#Button class for interactive simulation
class Index(object):
    
    def next(self, event):
        global infeasibility
        #new sample
        ind=list(f_obj.y_acquis).index(max(f_obj.y_acquis))
        f_obj.sampleNewPoint(ind)
        g1_cons.sampleNewPoint(ind)
        g2_cons.sampleNewPoint(ind)
        
        g1_cons.computeGaussianProcessApproximation()
        g2_cons.computeGaussianProcessApproximation()
        
        infeas_g1=map(lambda s:1 if s<0 else 0, g1_cons.y_pred)
        infeas_g2=map(lambda s:1 if s<0 else 0, g2_cons.y_pred)
        infeasibility=[a*b for a,b in zip(infeas_g1,infeas_g2)]

        f_obj.computeGaussianProcessApproximation()
        if ACQUIS==0:
            f_obj.computeAcquisitionFunction(infeasibility)
        else:
            f_obj.computeAcquisitionFunction_EI(min(f_obj.y_sample),g1_cons,g2_cons)
       
        #plot everything
        f_obj.updatePlot()
        g1_cons.updatePlot()
        g2_cons.updatePlot()
        
        ax.set_xlim(BOUND_L,BOUND_U)
        ax2.set_xlim(BOUND_L,BOUND_U)
        axg1.set_xlim(BOUND_L,BOUND_U)
        axg2.set_xlim(BOUND_L,BOUND_U)



#%% Interface creation
def create_interface():
    fig = plt.figure()
    ax = fig.add_subplot(231)#Obj
    ax2 = fig.add_subplot(234)#Acquisition
    axg1 = fig.add_subplot(232)#Constraint1
    axg2 = fig.add_subplot(233)#Constraint2
    
    ax.set_xlim(BOUND_L,BOUND_U)
    ax2.set_xlim(BOUND_L,BOUND_U)
    axg1.set_xlim(BOUND_L,BOUND_U)
    axg2.set_xlim(BOUND_L,BOUND_U)
   
    callback = Index()
    axnext = plt.axes([0.81, 0.01, 0.05, 0.05])#ButtonPosition
    bnext = Button(axnext, '+')#Button
    bnext.on_clicked(callback.next)#ButtonCallbackset
    
    return fig,ax,ax2,axg1,axg2,callback,bnext


    
#%% Functions definition
def f(x): #Objective
    return 3*((1.65*x-0.5)**2)*((np.cos(1.65*pi*x))**2)+0.2*np.sin(1.65*5*x)

def g1(x): #Constraint 1
    return np.sin(6.28*(x-0.4))

def g2(x): #Constraint 2
    return (x-0.7)*np.exp(x)


#%% Main

    # Initialisation
fig,ax,ax2,axg1,axg2,callback,bnext=create_interface()

f_obj=Function(ax,f,constraint=False)
g1_cons=Function(axg1,g1,constraint=True)
g2_cons=Function(axg2,g2,constraint=True)

    # First Round
g1_cons.computeGaussianProcessApproximation()
g2_cons.computeGaussianProcessApproximation()

g1_cons.updatePlot()
g2_cons.updatePlot()

infeas_g1=map(lambda s:1 if s<0 else 0, g1_cons.y_pred)
infeas_g2=map(lambda s:1 if s<0 else 0, g2_cons.y_pred)
infeasibility=[a*b for a,b in zip(infeas_g1,infeas_g2)]

f_obj.computeGaussianProcessApproximation()
if ACQUIS==0:
    f_obj.computeAcquisitionFunction(infeasibility)
else:
    f_obj.computeAcquisitionFunction_EI(min(f_obj.y_sample),g1_cons,g2_cons)
f_obj.updatePlot()

ax.set_xlim(BOUND_L,BOUND_U)
ax2.set_xlim(BOUND_L,BOUND_U)
axg1.set_xlim(BOUND_L,BOUND_U)
axg2.set_xlim(BOUND_L,BOUND_U)

