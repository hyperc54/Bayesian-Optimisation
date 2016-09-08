# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:33:28 2016

Interface creation for some dimensions (1D,2D,3D)


@author: pierre
"""


from __future__ import print_function
from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("qt4agg")
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import numpy as np
import multiprocessing
import time
pi=3.14

class ObjectGraph(object):
    def __init__(self,dim,**kwargs):
        self.plotnb=1
        self.dim=dim
        
        self.fig = plt.figure()        
        
        plt.ion()
        
        if self.dim==1:
            self.ax_scat = self.fig.add_subplot(311)#Obj
            self.ax_approx = self.fig.add_subplot(312)#Approximation
            self.ax_eimap = self.fig.add_subplot(313)#Acquisition
        else:
            self.ax_scat = self.fig.add_subplot(321)#Obj
            self.ax_approx = self.fig.add_subplot(323)#Approximation
            self.ax_eimap = self.fig.add_subplot(325)#Acquisition
            self.ax_scat3d = self.fig.add_subplot(322,projection='3d')
            self.ax_approx3d = self.fig.add_subplot(324,projection='3d')       
            
        show_plot(self.plotnb)
        

    def threadme(self, solver_b,bbox,history,new_samp_inputs,new_samp_output):

        thread_plot = threading.Thread(target=self.updateInterface,
                                      args=(solver_b,bbox,history,new_samp_inputs,new_samp_output,))
        thread_plot.start()
        thread_plot.join()       
        
    def updateInterface(self,solver_b,bbox,history,new_samp_inputs,new_samp_output,bounds):
        if self.dim==1:
            self.updateInterface1D(solver_b,bbox,history,bounds)
        if self.dim==2:
            self.updateInterface2D(solver_b,bbox,history,bounds)
            
        show_plot(self.plotnb)
        
    def updateInterface1D(self,solver,bbox,history,bounds):
        x = np.linspace(0, 1, 80)
        z=map(lambda y:bbox.queryAt([y]),x)
        xx=np.atleast_2d(x).T
        z_pred, sigma2_pred = solver.gp.predict(xx, eval_MSE=True)
        self.ax_scat.clear()
        self.ax_approx.clear()
        self.ax_eimap.clear()
        
        self.ax_scat.plot(x,np.array(z))
        self.ax_scat.scatter(np.array(history)[:,0],np.array(history)[:,1])
        self.ax_approx.plot(x,np.array(z_pred))
        self.ax_approx.fill_between(x,np.array(z_pred)+np.array(np.sqrt(sigma2_pred)),np.array(z_pred)-np.array(np.sqrt(sigma2_pred)),alpha=0.2)

        self.ax_approx.plot(x)
        target=min(np.array(history)[:,1])
        mean,variance=solver.gp.predict(xx,eval_MSE=True)
        z=(target-mean)/np.sqrt(variance)
        self.ax_approx.plot(x,np.sqrt(variance)*(z*norm.cdf(z)+norm.pdf(z)))


    def updateInterface2D(self,solver,bbox,history,bounds):
        x = np.linspace(bounds[0][0], bounds[0][1], 150)
        y = np.linspace(bounds[1][0], bounds[1][1], 150)
        x,y=np.meshgrid(x, y)
        xxx = np.linspace(bounds[0][0], bounds[0][1], 30)
        yyy = np.linspace(bounds[1][0], bounds[1][1], 30)
        xxx,yyy=np.meshgrid(xxx, yyy)
        xx=x.ravel()
        yy=y.ravel()
        xxxx=xxx.ravel()
        yyyy=yyy.ravel()
        #z=map(lambda x:bbox.queryAt(x),[np.array(i) for i in zip(xx,yy)])
        points_pred= np.array(map(lambda s:np.array(s),zip(xx,yy)))
        z_pred, sigma2_pred = solver.gp.predict(points_pred, eval_MSE=True)
        
        points_pred2= np.array(map(lambda s:np.array(s),zip(xxxx,yyyy)))
        z_pred2, sigma2_pred2 = solver.gp.predict(points_pred2, eval_MSE=True)
        #target=min(map(lambda x:bbox.queryAt(x),[np.array(i) for i in history]))
        target=min(list(np.array(history)[:,1]))
        u=(target-z_pred)/np.sqrt(sigma2_pred)
        self.ax_scat.clear()
        self.ax_scat3d.clear()
        self.ax_approx3d.clear()
        self.ax_approx.clear()
        self.ax_eimap.clear()
        
        self.ax_scat.set_xlim([0,1])
        self.ax_scat.set_ylim([0,1])        
        #c1=ax.contourf(x,y,np.array(z).reshape(-1,len(x[0])))
        tt=np.array(map(np.asarray,np.array(history).reshape(-1,len(history[0]))[:,0]))
        tt_out=np.array(map(np.asarray,np.array(history).reshape(-1,len(history[0]))[:,1]))
        self.ax_scat.scatter(tt[:,0],tt[:,1])
        self.ax_scat3d.scatter(tt[:,0],tt[:,1],tt_out)

        self.ax_approx3d.scatter(xxxx,yyyy,z_pred2) 
        self.ax_approx3d.scatter(tt[:,0],tt[:,1],tt_out,color="red")
        
        self.ax_approx.contourf(x,y,np.array(z_pred).reshape(-1,len(x[0])))
        self.ax_eimap.contourf(x,y,np.array(np.sqrt(sigma2_pred)*(u*norm.cdf(u)+norm.pdf(u))).reshape(-1,len(x[0])))
        self.ax_approx.scatter(tt[:,0],tt[:,1])
        self.ax_eimap.scatter(tt[:,0],tt[:,1])
        #c1.set_clim(min(z),max(z))
        #c2.set_clim(min(z),max(z))

def show_plot(figure_id=None):    
    if figure_id is not None:
        fig = plt.figure(num=figure_id)
    else:
        fig = plt.gcf()



    plt.ion()
    plt.show()
    plt.pause(1e-9)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()

    plt.show()
    plt.pause(1e-9)
    plt.draw()
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()