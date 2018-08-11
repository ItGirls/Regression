#!/usr/local/bin/python
# -*-coding:utf-8 -*-
"""
Define some measures
@author: Tingtingzhu
@email:tingtingzhu93@gmail.com
@Date:2018.8.11 15:29
@Last modified by:Tingtingzhu 2018.8.11
"""
import numpy as np
import math
class ErrorMeasure():
    def __init__(self):
        pass

    def sse(self,realY,hatY,testNum):
        tempError = map(lambda x: np.power(x, 2), np.array(realY) - np.array(hatY))
        sumError = sum(tempError)
        print "MSE: %f" % (sumError)

    def mse(self,realY,hatY,testNum):
        '''

        :param realY:list
        :param hatY:list
        :return:
        '''

        tempError = map(lambda x:np.power(x,2),np.array(realY)-np.array(hatY))
        sumError = sum(tempError)
        print "MSE: %f"%(sumError/testNum*1.0)
        # print "MSE: %f" % (self.sse(realY,hatY,testNum) / testNum * 1.0)

    def rmse(self,realY,hatY,testNum):
        tempError = map(lambda x: np.power(x, 2), np.array(realY) - np.array(hatY))
        sumError = sum(tempError)
        mse = sumError / testNum * 1.0
        print mse
        print math.sqrt(mse)
        print "RMSE: %f"%(np.sqrt(mse))

