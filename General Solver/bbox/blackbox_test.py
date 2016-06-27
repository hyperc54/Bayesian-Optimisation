# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:52:02 2016

Black-box class used for testing purposes.
It simply simulates a black-box.
You can choose the dimension and the exact form of the function inside the bbox.

@author: pierre
"""

from __future__ import print_function
from __future__ import division
import numpy as np
pi=3.14




class BlackBox(object):
    

    #Will create the black-box taken from the set of examples below
    def __init__(self,name):
        
        if name not in func_dic:
            err = "Wooooooops !" \
                  "This black-box name doesn't exist " \
                  "please choose 2d_a or 2d_b".format(name)
            raise NotImplementedError(err)
        else:
            self.func = func_dic[name]
            self.dim = dim_dic[name]
        
    #Simple query of the black box at a specific point, no other info given, that's the point!
    def queryAt(self,point):
        return self.func(point)
        
    def getDim(self):
        return self.dim
       
       
       
       
#%%Functions definitions
func_dic = {}
dim_dic = {}

#1D
def f1d_a(x):
    return 3*((1.65*x[0]-0.5)**2)*((np.cos(1.65*pi*x[0]))**2)+0.2*np.sin(1.65*5*x[0])

func_dic["1d_a"]=f1d_a
dim_dic["1d_a"]=1

def f1d_b(x):
    return (x[0]-0.4)**2
    
func_dic["1d_b"]=f1d_b
dim_dic["1d_b"]=1

#2D
def f2d_a(x):
    return 3*((1.65*x[0]-0.5)**2)*((np.cos(1.65*pi*x[0]))**2)+0.2*np.sin(1.65*5*x[0])*x[1]

func_dic["2d_a"]=f2d_a
dim_dic["2d_a"]=2

def f2d_b(x):
    return (x[0]-0.5)**2+(x[1]-0.5)**2

func_dic["2d_b"]=f2d_b
dim_dic["2d_b"]=2

def f2d_c(x):
    return (3/2)-x[0]-2*x[1]-(1/2)*np.sin(2*pi*(-2*x[1]+x[0]**2))

func_dic["2d_c"]=f2d_c
dim_dic["2d_c"]=2

#3D
    
#4D

#...