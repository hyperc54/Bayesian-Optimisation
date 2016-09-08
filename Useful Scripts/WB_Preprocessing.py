# -*- coding: utf-8 -*-
"""
Created on Fri Aug 05 15:52:26 2016

@author: pierre
"""


from __future__ import print_function
from __future__ import division

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
import time
from operator import mul
import copy
import sys

''' SETUP '''
bounds=[[1,5.5],[1,5.5]]

def f1(x):
    return 0.25*x[0]-0.0625*(x[0]*x[0])-0.0625*(x[1]*x[1])+0.5*x[1]-1

def f2(x):
    return 0.0714285714285714*(x[0]*x[0]) + 0.0714285714285714*(x[1]*x[1]) - 0.428571428571429*x[0]- 0.428571428571429*x[1]+1

list_cons=[f1,f2]

'''
Function union_contraintes 

Combines multiples constraints into 1 function
'''
def union_contraintes(x,list_cons):
    list_outputs=[f(x) for f in list_cons]
    
    res=1
    
    for inp in list_outputs:
        res*=(abs(inp)-inp)
    
    res/=2**(len(list_outputs)-1)
    
    res=-res
    
    res+=abs(reduce(mul, list_outputs))
    
    return res



'''
Function bitfield 

converts integer to binary array
'''
def bitfield(n,taillemin):
    
    res=np.array([1 if digit=='1' else 0 for digit in bin(n)[2:]])
    
    if len(res)<taillemin:
        res=np.concatenate((np.zeros(taillemin-len(res)),res),axis=0)
        
    return res



'''
Function getOptimum 

Calls many local sovlers to globally optimise a function
if isMin=True -> We are looking for the minimum, else maximum
'''
def getOptimum(func,bounds,nbLocalSolvers=100,isMin=True):
 
    best_acq_value=None
    
    
    if isMin:
        coeff=1
    else:
        coeff=-1

    bounds=np.array(bounds)
    starting_points_list=np.random.uniform(
                                            bounds[:,0],bounds[:,1],
                                            size=(nbLocalSolvers,len(bounds))
                                            ) #Took 100 starting points
    

    for starting_point in starting_points_list:
                
        opti=minimize(lambda x:coeff*func(x),
                     starting_point,
                     bounds=bounds,
                     method="L-BFGS-B")
  
        
        if (best_acq_value is None or opti.fun<best_acq_value):
            res=opti.x
            best_acq_value=opti.fun

        
    return res,coeff*best_acq_value


def computeVolume(bounds):
    res=1
    for b in bounds:
        res*=b[1]-b[0]
    return res

def computeVolumeList(bounds_list):
    res=0
    for b in bounds_list:
        res+=computeVolume(b[0])
    return res




'''
Function divideRegion 

divides the region (an hypercube) in 2**dim hypercubes
'''
def divideRegion(bounds):  
    n=len(bounds)
    bound_l=np.array(bounds).reshape(-1,2)[:,0]
    bound_h=np.array(bounds).reshape(-1,2)[:,1]
       
    res=[]
    
    for i in range(2**n):
        res.append([[bound_l[k],(bound_l[k]+bound_h[k])/2] if bitfield(i,n)[k]==0 else [(bound_l[k]+bound_h[k])/2,bound_h[k]] for k in range(n)])
        
    
    return res


    
'''
Function treatRegion 

Calls the optimiser and assigns if possible a flag 1/-1 to the region
'''
def treatRegion(region_bounds,func,nbLocalSolve=100): 
    
    mini_x,mini_fun=getOptimum(func,region_bounds,nbLocalSolvers=nbLocalSolve)
    if(mini_fun>0):#then the region is totally infeasible
        res=-1
    else:
        maxi_x,maxi_fun=getOptimum(func,region_bounds,nbLocalSolvers=nbLocalSolve,isMin=False)
        if(maxi_fun<=0):#then the region is totally feasible
            res=1
        else:
            res=0
    
    return res
    


'''
Computes bound sizes
'''
def tailleBound(bound):
    return min([b[1]-b[0] for b in bound])

def tailleMaxBound(bound_list):
    return max([tailleBound(bound) for bound in bound_list])



'''
Function refineBounds = main "Dividing" function

It handles the queue of regions that need to be computed
to check feasibility or infeasibility properties
'''
def refineBounds(bounds,func,taille_min,nbLocalSolversINI=10000):
    fini=[]
    to_compute=[bounds]
    undef=[]
    inc=0
    
    max_size=computeVolume(bounds)
        
    
    
    while(to_compute!=[]): #while there are still regions in the queue to compute
        
        #We pop the first region of the queue
        reg=to_compute.pop(0)
        size=computeVolume(reg)
        
        #Info Printing
        sys.stdout.write("\r Computed boxes: {0} - Still to compute: {1} - Size% of last box: {2} ".format(inc,len(to_compute),size/max_size))
        sys.stdout.flush()
        
        #try to check properties in it
        res=treatRegion(reg,func,nbLocalSolve=int(np.floor(size*nbLocalSolversINI/max_size))+1)
        if res==1:
            fini.append((reg,1))
        elif res==-1:
            fini.append((reg,-1))   
        else: #no property verified, we slice it and put new regions in the queue
            if tailleBound(reg)>taille_min:
                to_compute=to_compute+divideRegion(reg)
            else:
                undef.append((reg,0))
        inc=inc+1

    
    return fini,undef 



def getFeasibleSet(hypercubes_final):
    return filter(lambda x:x[1]==1,hypercubes_final)

def getInfeasibleSet(hypercubes_final):
    return filter(lambda x:x[1]==-1,hypercubes_final)

def verifySame(b1,b2,dim):
    b11=copy.deepcopy(b1)
    b22=copy.deepcopy(b2)
    b11.pop(dim)
    b22.pop(dim)
    return b11==b22

def lookForNeighbours(b,hypercubes_final,dim):
    res=-1
    for ind,tuple_bound in enumerate(hypercubes_final[:(int(len(hypercubes_final)/100))]):
        if ((tuple_bound[0][dim][0]==b[0][dim][1] or tuple_bound[0][dim][1]==b[0][dim][0]) and verifySame(b[0],tuple_bound[0],dim)):
            res=ind
    
    return res


'''
mergeSameDim

Merges matching neighbours along 1 dimension
'''
def mergeSameDim(hypercubes,dim):
    hypercubes_final=copy.deepcopy(hypercubes)
    fin=[]

    while(hypercubes_final!=[]):
        ind=-1
        b=hypercubes_final.pop(0)
        ind=lookForNeighbours(b,hypercubes_final,dim)   
                    
        if ind==-1:#no match for him
            fin.append(b)
        else:
                b2=hypercubes_final.pop(ind)
                
                b[0][dim][0]=min(b[0][dim][0],b2[0][dim][0])
                b[0][dim][1]=max(b[0][dim][1],b2[0][dim][1])
    
                hypercubes_final.append(b)

    return fin


'''
mergeGlobal = main "Merging" function

Merges matching neighbours along all dimensions 1 by 1
(not optimal but efficient to reduce the number of boxes considered)
'''
def mergeGlobal(hypercubes):

    if hypercubes==[]:
        return []
    
    res=copy.deepcopy(hypercubes)
    
    limit=len(hypercubes[0][0])
    for i in range(limit):
        
        #Info Printing
        sys.stdout.write("\r Dimension: {0}/{1} ".format(i+1,limit))
        sys.stdout.flush()
        
        res=mergeSameDim(res,i)
    
    return res





'''
Plotting utilities functions
'''
def fillSquare(bounds,ax):
    ax.fill([bounds[0][0],bounds[0][1],bounds[0][1],bounds[0][0]],
            [bounds[1][0],bounds[1][0],bounds[1][1],bounds[1][1]],
             'b',
             
             alpha=0.2,
             edgecolor='r')



def fillSquaresFromList(list_of_bounds,ax):
    for bounds_tuple in list_of_bounds:
        if bounds_tuple[1]==-1:
            color='r'
            edgecolor='black'
        elif(bounds_tuple[1]==0):
            color='green'
            edgecolor='b'
        else:
            color='b'
            edgecolor='b'
            
        bounds=bounds_tuple[0]
        ax.fill([bounds[0][0],bounds[0][1],bounds[0][1],bounds[0][0]], [bounds[1][0],bounds[1][0],bounds[1][1],bounds[1][1]], color, alpha=0.2, edgecolor=edgecolor)


def fillSquaresFromList2(list_of_bounds,ax):
    for bounds in list_of_bounds:

        ax.fill([bounds[0][0],bounds[0][1],bounds[0][1],bounds[0][0]], [bounds[1][0],bounds[1][0],bounds[1][1],bounds[1][1]], 'green', alpha=0.4)
             
     


def computeCentersList(hyperlist):
    minilist=[a[0] for a in hyperlist]
    
    res=[[(a[1]+a[0])/2 for a in b] for b in minilist]
    
    return res

def removeElementCopied(liste,ele):
    liste2=copy.deepcopy(liste)
    liste2.remove(ele)

    return liste2


def computeMinDistance(centers_list):
    return [min(map(compute2NormDistance,removeElementCopied(centers_list,a))) for a in centers_list]

def compute2NormDistance(l):
    return np.linalg.norm(np.array(l[0])-np.array(l[1]))
     





'''
MAIN FUNCTION

- list_func = the list of whitebox functions considered - functions objects
- bounds = the bounds of the original search space - list of [Min,Max] per dim
- taille_min_percent = the min size of the new boxes according to original bounds size

'''

def preprocFeasibleSpace(list_func,bounds,taille_min_percent,GRAPHE=0,nbLocalSolversInitial=10000):
    
    t = time.time()
    
    print("\x1b[1;31m Preprocessing the feasible space given by white-box constraints... \x1b[0m")
    taille_min_boxes=min([taille_min_percent*(b[1]-b[0]) for b in bounds])

    
    #Initialize the combined constraints function
    func=lambda x:union_contraintes(x,list_func)    
 
   
    #Dividing part
    fin,toc=refineBounds(bounds,func,taille_min_boxes,nbLocalSolversInitial)

    new_feas_set=getFeasibleSet(fin)
    fin5=getInfeasibleSet(fin)

    print(" ")
    print("Number of total boxes before merging: {0}".format(len(fin)+len(toc)))
    print("Number of total feasible boxes before merging: {0}".format(len(new_feas_set)))
    print("Number of total undefined boxes before merging: {0}".format(len(toc)))


    #Merging part
    print("Merging boxes...")
    fin3=copy.deepcopy(new_feas_set)
    toc2=copy.deepcopy(toc)
    
    #new_feas_set_merged=mergeGlobal(fin3)    
    #toc2=mergeGlobal(toc2)
    new_feas_set_merged=fin3    
    
    print("----")
    print("Number of total feasible boxes after merging: {0}".format(len(new_feas_set_merged)))
    print("Number of total undefined boxes after merging: {0}".format(len(toc2)))
    
    print("----")
    elapsed = time.time() - t
    print("Elapsed time: {0}s".format(elapsed))
    
    
    ## PLOT will prob crash if not 2Dimensional
    if GRAPHE:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
        
        grid=200
        
        x = np.linspace(bounds[0][0], bounds[0][1], grid)
        y = np.linspace(bounds[1][0], bounds[1][1], grid)
        x,y=np.meshgrid(x, y)
        
        
        z=list_func[0]((x,y))
        z1=list_func[1]((x,y))
        z2=union_contraintes((x,y),list_func) 
        
        levels=[-100,0,100]
        cs=ax1.contourf(x, y, z,levels)
        cs2=ax2.contourf(x, y, z1,levels)
        cs2=ax3.contourf(x, y, z2,levels)    
    
        fillSquaresFromList(new_feas_set_merged,ax4)
        fillSquaresFromList(fin5,ax4)                     
                
        fillSquaresFromList(toc2,ax4)
        
    print("----")
    volIni=computeVolume(bounds)
    volFeas= computeVolumeList(new_feas_set_merged)   
    volUndef=computeVolumeList(toc2)    
    
    print("Percentage of infeasible space dropped out: {0}%".format(100*(1-(volFeas+volUndef)/volIni)))
    print("Percentage of feasible space: {0}%".format(100*(volFeas/volIni)))
    print("Percentage of undefined space: {0}%".format(100*(volUndef/volIni)))

    new_feas_set_merged.sort(key=lambda x:computeVolume(x[0]),reverse=True)
    toc2.sort(key=lambda x:computeVolume(x[0]),reverse=True)

    return new_feas_set_merged,toc2


## MAIN

preprocFeasibleSpace(list_cons,bounds,0.1,GRAPHE=1,nbLocalSolversInitial=10000)


