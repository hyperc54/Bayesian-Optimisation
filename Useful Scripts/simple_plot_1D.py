
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:28:26 2016

@author: pierre

Simple plot of a 1D function + its surrogate GP model

"""

import numpy as np
from sklearn import gaussian_process
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
pi=3.14


#%% Initialisation
#Function definition that we will consider as a black-box
def f(x):
    return x*np.sin(x)

#We take 4 samples of the black_box function
X = np.atleast_2d([2,4,6,6.5]).T
Y = f(X).ravel()

#%% Gaussian Process Regression
#We create the gaussian process and make it fit our 4 samples
gp = gaussian_process.GaussianProcess(nugget=0.000000001)
gp.fit(X, Y)  

#We then predict thanks to this interpolation a grid of points in the space
x = np.atleast_2d(np.linspace(2, 7, 1000)).T #grid
y_pred, sigma2_pred = gp.predict(x, eval_MSE=True) #y_pred is the mean function of the distribution, sigma2_pred is its MSE


#%% Plotting of the results

#Creation of the figure
fig = plt.figure()
ax = fig.add_subplot(111)

#True Function
x_real=np.linspace(2,7,100)

#y_real=map(lambda x:0,x_real)
y_real=f(x_real).ravel()
#ax.plot(x_real,y_real,alpha=0.4)

#Approximation
ax.scatter(X,Y,s=20) #Samples
ax.plot(x,y_pred) #Mean
ax.fill_between(x.ravel(),y_pred-3*np.sqrt(sigma2_pred),y_pred+3*np.sqrt(sigma2_pred),color='black',alpha=0.1) #Confidence intervals


#y_1=(np.array([y_pred<0])*1).transpose()
#y_2=[norm.cdf(-y_pred[ind]/(10*sigma2_pred[ind])) for ind,val in enumerate(y_pred)]

#ax.plot(x,y_1,linewidth=5) 
#ax.plot(x,y_2,linewidth=2,color="blue") 