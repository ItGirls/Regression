#!/usr/local/bin/python
# -*-coding:utf-8 -*-

"""
Define some basis function
@author: Tingtingzhu
@email:tingtingzhu93@gmail.com
@Date:2018.8.11 9:45
@Last modified by:Tingtingzhu 2018.8.11
"""
import numpy as np
class BasisFunction():

    def __init__(self):
        pass

    def gaussianBasis(self,x,mu,s=1.0):
        mu = mu/10.0
        tempFun = np.exp(-np.power(x-mu,2)/(2*np.power(s,2)))
        return tempFun
    def sigmoidalBasis(self,x,mu,s=1.0):
        mu = mu/10.0
        tempFun = 1.0/(1+np.exp(-(x-mu)/s))
        return tempFun

    def powerBasis(self,x,j):
        tempFun = np.power(x,j)
        return tempFun
