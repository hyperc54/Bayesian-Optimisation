# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 23:17:27 2016

@author: pt914
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("qt4agg")
from mpl_toolkits.mplot3d import Axes3D
pi=3.14

def f2d_a(x):
    return 3*((1.65*x[0]-0.5)**2)*((np.cos(1.65*pi*x[0]))**2)+0.2*np.sin(1.65*5*x[0])*x[1]

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


x = np.linspace(0,1, 50)
y = np.linspace(0,1, 50)
x,y=np.meshgrid(x, y)

xx=x.ravel()
yy=y.ravel()

zz=np.array(map(lambda s:-f(s),zip(xx,yy)))

fig=plt.figure()

ax=fig.add_subplot(111)

ax.contourf(x,y,zz.reshape(50,50))