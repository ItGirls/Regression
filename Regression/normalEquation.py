#!/usr/local/bin/python
# -*-coding:utf-8 -*-

"""
@author: Tingtingzhu
@email:tingtingzhu93@gmail.com
@Date:2018.8.11 9:45
@Last modified by:Tingtingzhu 2018.8.11
"""
# import BasisFunction
import numpy as np
import numpy.linalg as LA
class MLRegression():
    def __init__(self,dataMat,wholeNum,numberBasis):
        '''

        :param dataMat: The whole data
        :param wholeNum: how many datas
        '''
        self.wholedata = dataMat
        self.wholeNum = wholeNum
        self.numberBasis = numberBasis

    def initialize(self):
        self.tempPTP = np.mat(np.dot(self.data.T, self.data))

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

        print self.t
        print self.data



    def esitimateParameters(self,mylambda=0):
        '''
        :param tol:
        :return:
        '''
        self.wHat =np.dot(np.dot((mylambda*np.eye(self.numberBasis)+np.mat(self.tempPTP)).I,self.data.T),self.t).T
        # print self.wHat
    def predict(self):
        self.predictT = np.dot(self.testData,self.wHat)

        # self.predictT = np.dot(self.data, self.wHat)


