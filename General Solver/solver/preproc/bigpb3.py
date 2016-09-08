from __future__ import print_function
from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.optimize import minimize
import numpy as np
import multiprocessing
import time
from operator import mul
import copy


def comb_cons(x):
    x0=x[0]
    x1=0
    x2=0
    x3=x[1]
    x4=x[2]
    x5=x[3]
    x6=x[4]
    x7=x[5]
    x8=1
    x9=x[6]
    x10=x[7]
    x11=x[8]
    x12=x[9]
    x13=1
    x14=1
    x15=1
    x16=x[10]
    x17=x[11]
    x18=x[12]
    x19=x[13]
    x20=x[14]
    x21=x[15]
    x22=x[16]
    x23=x[17]
    x24=x[18]
    x25=x[19]
    x26=x[20]

    e1= -np.sqrt((x3 - x1)**2 + (x4 - x2)**2) - x0 
    
    e2= -np.sqrt((x5 - x1)**2 + (x6 - x2)**2) - x0 
    
    e3= -np.sqrt((x5 - x3)**2 + (x6 - x4)**2) - x0 
    
    e4= -np.sqrt((x7 - x1)**2 + (x8 - x2)**2) - x0 
    
    e5= -np.sqrt((x7 - x3)**2 + (x8 - x4)**2) - x0 
    
    e6= -np.sqrt((x7 - x5)**2 + (x8 - x6)**2) - x0 
    
    e7= -np.sqrt((x9 - x1)**2 + (x10 - x2)**2) - x0 
    
    e8= -np.sqrt((x9 - x3)**2 + (x10 - x4)**2) - x0 
    
    e9= -np.sqrt((x9 - x5)**2 + (x10 - x6)**2) - x0 
    
    e10= -np.sqrt((x9 - x7)**2 + (x10 - x8)**2) - x0 
    
    e11= -np.sqrt((x11 - x1)**2 + (x12 - x2)**2) - x0 
    
    e12= -np.sqrt((x11 - x3)**2 + (x12 - x4)**2) - x0 
    
    e13= -np.sqrt((x11 - x5)**2 + (x12 - x6)**2) - x0 
    
    e14= -np.sqrt((x11 - x7)**2 + (x12 - x8)**2) - x0 
    
    e15= -np.sqrt((x11 - x9)**2 + (x12 - x10)**2) - x0 
    
    e16= -np.sqrt((x13 - x1)**2 + (x14 - x2)**2) - x0 
    
    e17= -np.sqrt((x13 - x3)**2 + (x14 - x4)**2) - x0 
    
    e18= -np.sqrt((x13 - x5)**2 + (x14 - x6)**2) - x0 
    
    e19= -np.sqrt((x13 - x7)**2 + (x14 - x8)**2) - x0 
    
    e20= -np.sqrt((x13 - x9)**2 + (x14 - x10)**2) - x0 
    
    e21= -np.sqrt((x13 - x11)**2 + (x14 - x12)**2) - x0 
    
    e22= -np.sqrt((x15 - x1)**2 + (x16 - x2)**2) - x0 
    
    e23= -np.sqrt((x15 - x3)**2 + (x16 - x4)**2) - x0 
    
    e24= -np.sqrt((x15 - x5)**2 + (x16 - x6)**2) - x0 
    
    e25= -np.sqrt((x15 - x7)**2 + (x16 - x8)**2) - x0 
    
    e26= -np.sqrt((x15 - x9)**2 + (x16 - x10)**2) - x0 
    
    e27= -np.sqrt((x15 - x11)**2 + (x16 - x12)**2) - x0 
    
    e28= -np.sqrt((x15 - x13)**2 + (x16 - x14)**2) - x0 
    
    e29= -np.sqrt((x17 - x1)**2 + (x18 - x2)**2) - x0 
    
    e30= -np.sqrt((x17 - x3)**2 + (x18 - x4)**2) - x0 
    
    e31= -np.sqrt((x17 - x5)**2 + (x18 - x6)**2) - x0 
    
    e32= -np.sqrt((x17 - x7)**2 + (x18 - x8)**2) - x0 
    
    e33= -np.sqrt((x17 - x9)**2 + (x18 - x10)**2) - x0 
    
    e34= -np.sqrt((x17 - x11)**2 + (x18 - x12)**2) - x0 
    
    e35= -np.sqrt((x17 - x13)**2 + (x18 - x14)**2) - x0 
    
    e36= -np.sqrt((x17 - x15)**2 + (x18 - x16)**2) - x0 
    
    e37= -np.sqrt((x19 - x1)**2 + (x20 - x2)**2) - x0 
    
    e38= -np.sqrt((x19 - x3)**2 + (x20 - x4)**2) - x0 
    
    e39= -np.sqrt((x19 - x5)**2 + (x20 - x6)**2) - x0 
    
    e40= -np.sqrt((x19 - x7)**2 + (x20 - x8)**2) - x0 
    
    e41= -np.sqrt((x19 - x9)**2 + (x20 - x10)**2) - x0 
    
    e42= -np.sqrt((x19 - x11)**2 + (x20 - x12)**2) - x0 
    
    e43= -np.sqrt((x19 - x13)**2 + (x20 - x14)**2) - x0 
    
    e44= -np.sqrt((x19 - x15)**2 + (x20 - x16)**2) - x0 
    
    e45= -np.sqrt((x19 - x17)**2 + (x20 - x18)**2) - x0 
    
    e46= -np.sqrt((x21 - x1)**2 + (x22 - x2)**2) - x0 
    
    e47= -np.sqrt((x21 - x3)**2 + (x22 - x4)**2) - x0 
    
    e48= -np.sqrt((x21 - x5)**2 + (x22 - x6)**2) - x0 
    
    e49= -np.sqrt((x21 - x7)**2 + (x22 - x8)**2) - x0 
    
    e50= -np.sqrt((x21 - x9)**2 + (x22 - x10)**2) - x0 
    
    e51= -np.sqrt((x21 - x11)**2 + (x22 - x12)**2) - x0 
    
    e52= -np.sqrt((x21 - x13)**2 + (x22 - x14)**2) - x0 
    
    e53= -np.sqrt((x21 - x15)**2 + (x22 - x16)**2) - x0 
    
    e54= -np.sqrt((x21 - x17)**2 + (x22 - x18)**2) - x0 
    
    e55= -np.sqrt((x21 - x19)**2 + (x22 - x20)**2) - x0 
    
    e56= -np.sqrt((x23 - x1)**2 + (x24 - x2)**2) - x0 
    
    e57= -np.sqrt((x23 - x3)**2 + (x24 - x4)**2) - x0 
    
    e58= -np.sqrt((x23 - x5)**2 + (x24 - x6)**2) - x0 
    
    e59= -np.sqrt((x23 - x7)**2 + (x24 - x8)**2) - x0 
    
    e60= -np.sqrt((x23 - x9)**2 + (x24 - x10)**2) - x0 
    
    e61= -np.sqrt((x23 - x11)**2 + (x24 - x12)**2) - x0 
    
    e62= -np.sqrt((x23 - x13)**2 + (x24 - x14)**2) - x0 
    
    e63= -np.sqrt((x23 - x15)**2 + (x24 - x16)**2) - x0 
    
    e64= -np.sqrt((x23 - x17)**2 + (x24 - x18)**2) - x0 
    
    e65= -np.sqrt((x23 - x19)**2 + (x24 - x20)**2) - x0 
    
    e66= -np.sqrt((x23 - x21)**2 + (x24 - x22)**2) - x0 
    
    e67= -np.sqrt((x25 - x1)**2 + (x26 - x2)**2) - x0 
    
    e68= -np.sqrt((x25 - x3)**2 + (x26 - x4)**2) - x0 
    
    e69= -np.sqrt((x25 - x5)**2 + (x26 - x6)**2) - x0 
    
    e70= -np.sqrt((x25 - x7)**2 + (x26 - x8)**2) - x0 
    
    e71= -np.sqrt((x25 - x9)**2 + (x26 - x10)**2) - x0 
    
    e72= -np.sqrt((x25 - x11)**2 + (x26 - x12)**2) - x0 
    
    e73= -np.sqrt((x25 - x13)**2 + (x26 - x14)**2) - x0 
    
    e74= -np.sqrt((x25 - x15)**2 + (x26 - x16)**2) - x0 
    
    e75= -np.sqrt((x25 - x17)**2 + (x26 - x18)**2) - x0 
    
    e76= -np.sqrt((x25 - x19)**2 + (x26 - x20)**2) - x0 
    
    e77= -np.sqrt((x25 - x21)**2 + (x26 - x22)**2) - x0 
    
    e78= -np.sqrt((x25 - x23)**2 + (x26 - x24)**2) - x0 

    l=[e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36,e37,e38,e39,e40,e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,e51,e52,e53,e54,e55,e56,e57,e58,e59,e60,e61,e62,e63,e64,e65,e66,e67,e68,e69,e70,e71,e72,e73,e74,e75,e76,e77,e78]
    
    res=1
    
    for inp in l:
        res*=(abs(inp)-inp)

    
    res/=2**(len(l)-1)
    
    res=-res
    
    res+=abs(reduce(mul, l))
    
    return res
