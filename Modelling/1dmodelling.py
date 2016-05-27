# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:28:26 2016

@author: pierre
"""

import numpy as np
from sklearn import gaussian_process
import matplotlib.pyplot as plt
pi=3.14


#%% Initialisation
#Function definition that we will consider as a black-box
def f(x):
    return 3*((x-0.5)**2)*((np.cos(pi*x))**2)+0.2*np.sin(5*x)

#We take 4 samples of the black_box function
X = np.atleast_2d([0.,.2,.5,1.]).T
Y = f(X).ravel()



#%% Gaussian Process Regression
#We create the gaussian process and make it fit our 4 samples
gp = gaussian_process.GaussianProcess()
gp.fit(X, Y)  

#We then predict thanks to this interpolation a grid of points in the space
x = np.atleast_2d(np.linspace(0, 1, 200)).T #grid
y_pred, sigma2_pred = gp.predict(x, eval_MSE=True) #y_pred is the mean function of the distribution, sigma2_pred is its MSE


#%% Plotting of the results

#Creation of the figure
fig = plt.figure()
ax = fig.add_subplot(111)

#True Function
x_real=np.linspace(0,1,100)
y_real=f(x_real).ravel()
ax.plot(x_real,y_real)

#Approximation
ax.scatter(X,Y,s=400) #Samples
ax.plot(x,y_pred) #Mean
ax.fill_between(x.ravel(),y_pred-2*np.sqrt(sigma2_pred),y_pred+2*np.sqrt(sigma2_pred),color='black',alpha=0.1) #Confidence intervals
