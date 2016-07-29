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


class WhiteBox(Box):
    
    #Function we only have for a white box
    def getFunc(self):
        return self.func     
       