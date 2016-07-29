# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 20:52:02 2016

Some white-boxes used for testing purposes

@author: pierre
"""

from __future__ import print_function
from __future__ import division
import numpy as np

pi=3.14


#%%

class Box(object):
    

    #Will create the black-box taken from the set of examples below
    def __init__(self,name):
            
        #Check name       
        if name not in func_dic:
            err = "Wooooooops !" \
                  "This box name doesn't exist ".format(name)
            raise NotImplementedError(err)
        else:
            self.func = func_dic[name]
            self.dim = dim_dic[name]
        
        
    #Simple query of the box at a specific point, no other info given, that's the point!
    def queryAt(self,point):
        return self.func(point)
    
    
    def getDim(self):
        return self.dim

    #Function we only have for a white box
    def getFunc(self):
        raise NotImplementedError("not implemented here")
        
    #Function we only have for a white box
    def getType(self):
        raise self.type

#%%       

class BlackBox(Box):
    
    #Function we only have for a white box
    def getFunc(self):
        raise NotImplementedError("Not available for a black box")


#%%

class WhiteBox(Box):
    
    #Function we only have for a white box
    def getFunc(self):
        return self.func     
       

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


def f1d_c(x):
    return (-x[0]+1)
    
func_dic["1d_c"]=f1d_c
dim_dic["1d_c"]=1


def f1d_d(x):
    return (-1)
    
func_dic["1d_d"]=f1d_d
dim_dic["1d_d"]=1


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

def f2d_d(x):
    return 1000*((4*(x[0]-0.3)**2)+(x[1]-0.4)**2)

func_dic["2d_d"]=f2d_d
dim_dic["2d_d"]=2

def f2d_e(x):
    return 1000*(((x[0]-0.25123)**2)+(x[1]-0.6812)**2)

func_dic["2d_e"]=f2d_e
dim_dic["2d_e"]=2

def f2d_f(x):
    return -(-158.086*x[0]*x[0]+48.541*x[1]*x[1]-206.248*x[0]*x[1]+161.552)

func_dic["2d_f"]=f2d_f
dim_dic["2d_f"]=2

def f2d_g(x):
    x1 = x[0]
    x2 = x[1]
    return (4 - 2.1*(x1*x1) + (x1*x1*x1*x1)/3.0)*(x1*x1) + x1*x2 + (-4 + 4*(x2*x2))*(x2*x2)

func_dic["2d_g"]=f2d_g
dim_dic["2d_g"]=2

def f2d_h(x):
    """Two Dimensional Shubert Function"""  
    j = np.arange(1, 6)
    
    tmp1 = np.dot(j, np.cos((j+1)*x[0] + j))
    tmp2 = np.dot(j, np.cos((j+1)*x[1] + j))
    
    return tmp1 * tmp2

func_dic["2d_h"]=f2d_h
dim_dic["2d_h"]=2




#5D
def f5d_a(x):
    return (x[0]-0.5)**2+(x[1]-0.5)**2+(x[2]-0.5)**2+(x[3]-0.5)**2+(x[4]-0.5)**2

func_dic["5d_a"]=f5d_a
dim_dic["5d_a"]=5
#4D

#...
def f8d_a(x):
    return (x[0]-0.5)**2+(x[1]-0.5)**2+(x[2]-0.5)**2+(x[3]-0.5)**2+(x[4]-0.5)**2+(x[5]-0.5)**2+(x[6]-0.5)**2+(x[7]-0.5)**2

func_dic["8d_a"]=f8d_a
dim_dic["8d_a"]=8


'''MINLP INSTANCES'''

#http://www.gamsworld.org/minlp/minlplib2/html/prob06.html

def f2d_minlp1_obj(x):
    """Prob06
    bounds
    [1,5.5][1,5.5]"""  
    return x[0]

func_dic["minlp1_obj"]=f2d_minlp1_obj
dim_dic["minlp1_obj"]=2


def f2d_minlp1_c1(x):
    """Prob06"""  
    return 0.25*x[0]-0.0625*(x[0]*x[0])-0.0625*(x[1]*x[1])+0.5*x[1]-1

func_dic["minlp1_c1"]=f2d_minlp1_c1
dim_dic["minlp1_c1"]=2

def f2d_minlp1_c2(x):
    """Prob06"""  
    return 0.0714285714285714*(x[0]*x[0]) + 0.0714285714285714*(x[1]*x[1]) - 0.428571428571429*x[0]- 0.428571428571429*x[1]+1

func_dic["minlp1_c2"]=f2d_minlp1_c2
dim_dic["minlp1_c2"]=2

#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/mathopt6.html

def f2d_minlp2_obj(x):
    """MathOpt6
    bounds
    [[-3,3][-3,3]]
    """  
    return np.exp(np.sin(50*x[0]))-np.sin(60*np.exp(x[1]))-np.sin(70*np.sin(x[0]))-np.sin(np.sin(80*x[1]))+np.sin(10*x[0]+10*x[1])+0.25*((x[0]**2)+x[1]**2)

func_dic["minlp2_obj"]=f2d_minlp2_obj
dim_dic["minlp2_obj"]=2


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/ex3_1_4.html

def f2d_minlp3_obj(x):
    """ex3_1_4
    bounds
    [[0,2][0,4][0,3]]
    """  
    return -2*x[0]+x[1]-x[2]

func_dic["minlp3_obj"]=f2d_minlp3_obj
dim_dic["minlp3_obj"]=3


def f2d_minlp3_c1(x):
    """ex3_1_4
    """  
    return -(x[0]*(4*x[0]-2*x[1]+2*x[2])+x[1]*(2*x[1]-2*x[0]-x[2])+x[2]*(2*x[0]-x[1]+2*x[2])+24)

func_dic["minlp3_c1"]=f2d_minlp3_c1
dim_dic["minlp3_c1"]=3



def f2d_minlp3_c2(x):
    """ex3_1_4
    """  
    return x[0]+x[1]+x[2]-4

func_dic["minlp3_c2"]=f2d_minlp3_c2
dim_dic["minlp3_c2"]=3


def f2d_minlp3_c3(x):
    """ex3_1_4
    """  
    return 3*x[1]+x[2]-6

func_dic["minlp3_c3"]=f2d_minlp3_c3
dim_dic["minlp3_c3"]=3


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/st_e08.html

def f2d_minlp4_obj(x):
    """st_e08
    bounds
    [[0,3][0,3]]
    """  
    return 2*x[0]+x[1]

func_dic["minlp4_obj"]=f2d_minlp4_obj
dim_dic["minlp4_obj"]=2


def f2d_minlp4_c1(x):
    """st_e08
    """  
    return (-4*x[0]**2)-(4*x[1]**2)+1

func_dic["minlp4_c1"]=f2d_minlp4_c1
dim_dic["minlp4_c1"]=2



def f2d_minlp4_c2(x):
    """st_e08
    """  
    return -16*x[0]*x[1]+1

func_dic["minlp4_c2"]=f2d_minlp4_c2
dim_dic["minlp4_c2"]=2



#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/st_e09.html

def f2d_minlp5_obj(x):
    """st_e09
    bounds
    [[0,1][0,1]]
    """  
    return -2*x[0]*x[1]

func_dic["minlp5_obj"]=f2d_minlp5_obj
dim_dic["minlp5_obj"]=2


def f2d_minlp5_c1(x):
    """st_e09
    """  
    return 4*x[0]*x[1]+2*x[0]+2*x[1]-3

func_dic["minlp5_c1"]=f2d_minlp5_c1
dim_dic["minlp5_c1"]=2

#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/st_e09.html

def f2d_minlp5_obj(x):
    """st_e09
    bounds
    [[0,1][0,1]]
    """  
    return -2*x[0]*x[1]

func_dic["minlp5_obj"]=f2d_minlp5_obj
dim_dic["minlp5_obj"]=2


def f2d_minlp5_c1(x):
    """st_e09
    """  
    return 4*x[0]*x[1]+2*x[0]+2*x[1]-3

func_dic["minlp5_c1"]=f2d_minlp5_c1
dim_dic["minlp5_c1"]=2

#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/st_e24.html

def f2d_minlp6_obj(x):
    """st_e24
    bounds
    [[0,4][1,5]]
    """  
    return ((5+x[0]-x[1])*(-1+x[0]+x[1])+x[0])

func_dic["minlp6_obj"]=f2d_minlp6_obj
dim_dic["minlp6_obj"]=2


def f2d_minlp6_c1(x):
    """st_e24
    """  
    return -2*x[0]-3*x[1]+9

func_dic["minlp6_c1"]=f2d_minlp6_c1
dim_dic["minlp6_c1"]=2


def f2d_minlp6_c2(x):
    """st_e24
    """  
    return 3*x[0]-x[1]-8

func_dic["minlp6_c2"]=f2d_minlp6_c2
dim_dic["minlp6_c2"]=2


def f2d_minlp6_c3(x):
    """st_e24
    """  
    return -x[0]+2*x[1]-8

func_dic["minlp6_c3"]=f2d_minlp6_c3
dim_dic["minlp6_c3"]=2


def f2d_minlp6_c4(x):
    """st_e24
    """  
    return x[0]+2*x[1]-12

func_dic["minlp6_c4"]=f2d_minlp6_c4
dim_dic["minlp6_c4"]=2


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/st_cqpjk2.html

def f2d_minlp7_obj(x):
    """st_e11
    bounds
    [[0,1][0,1][0,1]]
    """  
    return (9*x[0]*x[0]-15*x[0]+9*x[1]*x[1]-12*x[1]+9*x[2]*x[2]-9*x[2])

func_dic["minlp7_obj"]=f2d_minlp7_obj
dim_dic["minlp7_obj"]=3


def f2d_minlp7_c1(x):
    """st_e11
    """  
    return x[0]+x[1]+x[2]-10000000000

func_dic["minlp7_c1"]=f2d_minlp7_c1
dim_dic["minlp7_c1"]=3


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/st_miqp1.html

def f2d_minlp8_obj(x):
    """st_miqp1.
    bounds
    [[0,1][0,1][0,1][0,1][0,1]]
    """  
    return 50*x[0]*x[0]+42*x[0]+50*x[1]*x[1]+44*x[1]+50*x[2]*x[2]+45*x[2]+50*x[3]*x[3]+47*x[3]+50*x[4]*x[4]+47.5*x[4]

func_dic["minlp8_obj"]=f2d_minlp8_obj
dim_dic["minlp8_obj"]=5


def f2d_minlp8_c1(x):
    """st_miqp1.
    """  
    return -20*x[0]-12*x[1]-11*x[2]-7*x[3]-4*x[4]+40

func_dic["minlp8_c1"]=f2d_minlp8_c1
dim_dic["minlp8_c1"]=5


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/ex7_2_4.html

def f2d_minlp9_obj(x):
    """st_miqp1.
    bounds
    [[0.1,10][0.1,10][0.1,10][0.1,10][0.1,10][0.1,10][0.1,10][0.1,10]]
    """  
    return 10+((0.4*(x[0]**0.67)/(x[6]**0.67))+((0.4*x[1]**0.67)/(x[7]**0.67))-x[0]-x[1])

func_dic["minlp9_obj"]=f2d_minlp9_obj
dim_dic["minlp9_obj"]=8


def f2d_minlp9_c1(x):
    """st_miqp1.
    """  
    return 0.0588*x[4]*x[6]+0.1*x[0]-1

func_dic["minlp9_c1"]=f2d_minlp9_c1
dim_dic["minlp9_c1"]=8

def f2d_minlp9_c2(x):
    """st_miqp1.
    """  
    return 0.0588*x[5]*x[7]+0.1*x[0]+0.1*x[1]-1

func_dic["minlp9_c2"]=f2d_minlp9_c2
dim_dic["minlp9_c2"]=8

def f2d_minlp9_c3(x):
    """st_miqp1.
    """  
    return (4*x[2]/x[4])+(2/(x[4]*x[2]**0.71))+0.0588*(x[6]/(x[2]**1.3))-1

func_dic["minlp9_c3"]=f2d_minlp9_c3
dim_dic["minlp9_c3"]=8

def f2d_minlp9_c4(x):
    """st_miqp1.
    """  
    return (4*x[3]/x[5])+(2/(x[5]*x[3]**0.71))+0.0588*(x[3]**1.3)*x[7]-1

func_dic["minlp9_c4"]=f2d_minlp9_c4
dim_dic["minlp9_c4"]=8

#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/ex4_1_3.html

def f2d_minlp10_obj(x):
    """st_miqp1.
    bounds
    [[0,10]]
    """  
    x1=x[0]
    
    return (0.000089248*x1 - 0.0218343*(x1*x1) + 0.998266*x1**3 - 1.6995*x1**4 + 0.2*x1**5)

func_dic["minlp10_obj"]=f2d_minlp10_obj
dim_dic["minlp10_obj"]=1


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/ex2_1_1.html

def f2d_minlp11_obj(x):
    """ex2_1_1.
    bounds
    [[0,1][0,1][0,1][0,1][0,1]]
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    
    return (42*x1 - 0.5*(100*x1*x1 + 100*x2*x2 + 100*x3*x3 + 100*x4*x4 + 100*x5*x5)+ 44*x2 + 45*x3 + 47*x4 + 47.5*x5)

func_dic["minlp11_obj"]=f2d_minlp11_obj
dim_dic["minlp11_obj"]=5


def f2d_minlp11_c1(x):
    """ex2_1_1.
    """  
    return 20*x[0] + 12*x[1] + 11*x[2] + 7*x[3] + 4*x[4] - 40

func_dic["minlp11_c1"]=f2d_minlp11_c1
dim_dic["minlp11_c1"]=5


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/ex4_1_3.html

def f2d_minlp12_obj(x):
    """ex4_1_3
    bounds
    [[0.1,10][0.1,10][0.1,10][0.1,10][0.1,10][0.1,10][0.1,10][0.1,10]]
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    
    return (42*x1 - 0.5*(100*x1*x1 + 100*x2*x2 + 100*x3*x3 + 100*x4*x4 + 100*x5*x5)+ 44*x2 + 45*x3 + 47*x4 + 47.5*x5)

func_dic["minlp12_obj"]=f2d_minlp12_obj
dim_dic["minlp12_obj"]=5


def f2d_minlp12_c1(x):
    """ex4_1_3
    """  
    return 20*x[0] + 12*x[1] + 11*x[2] + 7*x[3] + 4*x[4] - 40

func_dic["minlp12_c1"]=f2d_minlp12_c1
dim_dic["minlp12_c1"]=5


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/st_jcbpaf2.html

def f2d_minlp13_obj(x):
    """st_jcbpaf2
    bounds
    [[0,100][0,100][0,100][0,100][0,100][0,100][0,100][0,100][0,100][0,100]]
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return (x1*x6 - x1 - x6 + x2*x7 - 2*x2 - 2*x7 + x3*x8 - 3*x3 - 3*x8 + x4*x9 - 4*x4 - 4*x9 + x5*x10 - 5*x5 - 5*x10)

func_dic["minlp13_obj"]=f2d_minlp13_obj
dim_dic["minlp13_obj"]=10


def f2d_minlp13_c1(x):
    """st_jcbpaf2
    """ 
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return x1 + 7*x2 + 5*x3 + 5*x4 - 6*x6 - 3*x7 - 3*x8 + 5*x9 - 7*x10 - 80

func_dic["minlp13_c1"]=f2d_minlp13_c1
dim_dic["minlp13_c1"]=10

def f2d_minlp13_c2(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return -3*x1 + 3*x2 + 8*x3 + 7*x4 - 9*x5 - 7*x6 - 9*x7 + 8*x9 - 7*x10 - 57

func_dic["minlp13_c2"]=f2d_minlp13_c2
dim_dic["minlp13_c2"]=10

def f2d_minlp13_c3(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return x1 + x3 + 3*x4 + 8*x5 + 9*x6 + 9*x8 - 7*x9 - 8*x10 - 92

func_dic["minlp13_c3"]=f2d_minlp13_c3
dim_dic["minlp13_c3"]=10

def f2d_minlp13_c4(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return -x1 - 2*x2 + 2*x3 + 9*x5 + 5*x6 - 3*x7 + x8 - x9 - 5*x10 - 55

func_dic["minlp13_c4"]=f2d_minlp13_c4
dim_dic["minlp13_c4"]=10


def f2d_minlp13_c5(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return -5*x1 + 8*x2 - 8*x3 + 3*x5 + 4*x7 - 5*x8 - 2*x9 + 9*x10 - 76

func_dic["minlp13_c5"]=f2d_minlp13_c5
dim_dic["minlp13_c5"]=10


def f2d_minlp13_c6(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return 4*x1 - x2 + 6*x3 - 4*x4 - 7*x5 - 8*x6 - 7*x7 + 6*x8 - 2*x9 - 9*x10- 14

func_dic["minlp13_c6"]=f2d_minlp13_c6
dim_dic["minlp13_c6"]=10

def f2d_minlp13_c7(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return 7*x2 + 4*x3 + 9*x5 - 6*x8 - 5*x9 - 5*x10 -47

func_dic["minlp13_c7"]=f2d_minlp13_c7
dim_dic["minlp13_c7"]=10


def f2d_minlp13_c8(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return -5*x1 - x2 + 7*x4 - x5 + 2*x6 + 5*x7 - 8*x8 - 5*x9 + 2*x10- 51;

func_dic["minlp13_c8"]=f2d_minlp13_c8
dim_dic["minlp13_c8"]=10

def f2d_minlp13_c9(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return -4*x1 - 7*x2 - 9*x4 + 2*x5 + 6*x6 - 9*x7 + x8 - 5*x9 -36

func_dic["minlp13_c9"]=f2d_minlp13_c9
dim_dic["minlp13_c9"]=10


def f2d_minlp13_c10(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return -2*x1 + 6*x2 + 8*x4 - 6*x5 + 8*x6 + 8*x7 + 5*x8 + 2*x9 - 7*x10 - 92;

func_dic["minlp13_c10"]=f2d_minlp13_c10
dim_dic["minlp13_c10"]=10

def f2d_minlp13_c11(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return x1 + x2 + x3 - 2*x4 + x5 + x6 + x7 + 4*x8 + x9 + 3*x10 - 200;

func_dic["minlp13_c11"]=f2d_minlp13_c11
dim_dic["minlp13_c11"]=10

def f2d_minlp13_c12(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return -(x1 + x2 + x3 + x4 + x5 - 1)
    
func_dic["minlp13_c12"]=f2d_minlp13_c12
dim_dic["minlp13_c12"]=10

def f2d_minlp13_c13(x):
    """st_jcbpaf2
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]
    x6=x[5]
    x7=x[6]
    x8=x[7]
    x9=x[8]
    x10=x[9]
    
    return -(x6 + x7 + x8 + x9 + x10 - 2)
    
func_dic["minlp13_c13"]=f2d_minlp13_c13
dim_dic["minlp13_c13"]=10


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/st_e41.html

def f2d_minlp14_obj(x):
    """st_e41
    bounds
    [[0.5,1][0.5,1][0.5,1][0.5,1]]
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    
    return (200*x1**0.6 + 200*x2**0.6 + 200*x3**0.6 + 300*x4**0.6)

func_dic["minlp14_obj"]=f2d_minlp14_obj
dim_dic["minlp14_obj"]=4


def f2d_minlp14_c1(x):
    """st_e41
    """
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]    
    
    return -(-((1 - x1)**2)*x3*((1 - x4)**2) - ((1 - (1 - (1 - x1)*(1 - x4))*x2)**2)*(1 - x3)) - 0.1

func_dic["minlp14_c1"]=f2d_minlp14_c1
dim_dic["minlp14_c1"]=4

def f2d_minlp14_c2(x):
    """st_e41
    """
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]    
    
    return (-((1 - x1)**2)*x3*((1 - x4)**2)) - ((1 - (1 - (1 - x1)*(1 - x4))*x2)**2)*(1 - x3)

func_dic["minlp14_c2"]=f2d_minlp14_c2
dim_dic["minlp14_c2"]=4


#############################################


##http://www.gamsworld.org/minlp/minlplib2/html/pointpack02.html

def f2d_minlp15_obj(x):
    """pointpack02
    bounds
    [[0.5,1][0,1][0,1][0,1]]
    """  
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    
    return -2*x1*x2 + x1*x1 + x2*x2 + x3*x3 - 2*x3*x4 + x4*x4

func_dic["minlp15_obj"]=f2d_minlp15_obj
dim_dic["minlp15_obj"]=4


def f2d_minlp15_c1(x):
    """pointpack02
    """
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]    
    
    return - x3 + x4

func_dic["minlp15_c1"]=f2d_minlp15_c1
dim_dic["minlp15_c1"]=4

def f2d_minlp15_c2(x):
    """pointpack02
    """
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]    
    
    return -x1+x2

func_dic["minlp15_c2"]=f2d_minlp15_c2
dim_dic["minlp15_c2"]=4



#############################################


##Toy example from Bayesian Opt with constraints Sim1

def f2d_paper1_obj(x):
    """
    [[0,6][0,6]]
    """  
    x1=x[0]
    x2=x[1]
    
    return np.cos(2*x1)*np.cos(x2)+np.sin(x1)
    
func_dic["paper1_obj"]=f2d_paper1_obj
dim_dic["paper1_obj"]=2


def f2d_paper1_c1(x):
    """pointpack02
    """
    x1=x[0]
    x2=x[1]
    
    return np.cos(x1)*np.cos(x2)-np.sin(x1)*np.sin(x2)-0.5
    
func_dic["paper1_c1"]=f2d_paper1_c1
dim_dic["paper1_c1"]=2


#############################################


##Toy example from Bayesian Opt with constraints Sim2

def f2d_paper2_obj(x):
    """
    [[0,6][0,6]]
    """  
    x1=x[0]
    x2=x[1]
    
    return np.sin(x1)+x2
    
func_dic["paper2_obj"]=f2d_paper2_obj
dim_dic["paper2_obj"]=2


def f2d_paper2_c1(x):
    """pointpack02
    """
    x1=x[0]
    x2=x[1]
    
    return np.sin(x1)*np.sin(x2)+0.95
    
func_dic["paper2_c1"]=f2d_paper2_c1
dim_dic["paper2_c1"]=2

       