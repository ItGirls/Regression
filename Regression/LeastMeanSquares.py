#!/usr/local/bin/python
# -*-coding:utf-8 -*-

"""
@author: Tingtingzhu
@email:tingtingzhu93@gmail.com
@Date:2018.8.11 19:00
@Last modified by:Tingtingzhu 2018.8.11
"""

import random
import numpy as np

class MLRegression1():
    def __init__(self,dataMat,wholeNum,numberBasis,alpha=0.1):
        '''

        :param dataMat: The whole data
        :param wholeNum: how many datas
        :param alpha: learning rate
        '''
        self.wholedata = dataMat
        self.wholeNum = wholeNum
        self.numberBasis = numberBasis
        self.alpha = alpha

    def initialize(self):
        self.wLMS = np.ones((self.numberBasis,1))

    def getDesignMatrix(self,trains, tempTrainIndicies,tempTestIndicies, basisFun):
        '''

        :param DataMat: the whole original data
        :param trains: number of training
        :param numberBasis: number of basis function i.e., M(the first is identity function)
        :param basisFun: can be gaussian, sigmoidal or power
        :return:
        '''
        # 目标
        self.wholeT = self.wholedata[:,-1]
        # 特征
        self.wholeF = np.tile(np.reshape(self.wholedata[:, 0], (self.wholeNum, 1)), (1, self.numberBasis))
        for j in range(1,self.numberBasis):
            #DataMat[:, j] = map(lambda x: np.power(x, 2), DataMat[:, j])
            self.wholeF[:, j] = map(lambda x:basisFun(x,j), self.wholeF[:, j])

        self.wholeF[:,0] = 1

        #split train and test features and targets
        self.N = trains
        self.t = self.wholeT[tempTrainIndicies]
        self.data = self.wholeF[tempTrainIndicies]

        self.testData = self.wholeF[tempTestIndicies]
        self.testT = self.wholeT[tempTestIndicies]

        # print self.testData
        # print self.testT

        # print self.t
        # print self.data


    def esitimateParameters(self,tol= 1e-06):
        '''

        :param tol:
        :return:
        '''
        count =0
        while True:
            #random choose one data
            indicies = random.randint(0, self.N - 1)
            temp = np.mat(self.data[indicies])

            #error
            tHat = np.dot(self.wLMS.T, temp.T)[0, 0]
            error =self.t[indicies] - tHat
            tempW = self.wLMS + self.alpha * error * temp.T

            #if convergence
            tempDelta = np.array(tempW.T)[0] - np.array(self.wLMS.T)[0]
            tempDelta = map(abs, tempDelta)
            if min(tempDelta) <= tol:
                self.wLMS = tempW
                break
            else:
                count += 1
        print "total iteration : %d" % (count)

    def predict(self):
        # print self.wLMS
        self.predictT = np.dot(self.testData,self.wLMS)




