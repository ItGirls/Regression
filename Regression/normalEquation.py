#!/usr/local/bin/python
# -*-coding:utf-8 -*-

"""
Maximum Likelihood Solution for Linear Regression Model
Using normal Equation
@author: Tingtingzhu
@email:tingtingzhu93@gmail.com
@Date:2018.8.11 9:45
@Last modified by:Tingtingzhu 2018.8.11
"""

import numpy as np
import numpy.linalg as LA

class MLRegression():
    def __init__(self,dataMat,wholeNum,numberBasis):
        '''

        :param dataMat: The whole data
        :param wholeNum: how many datas
        :param numberBasis: how many basis functions
        '''
        self.wholedata = dataMat
        self.wholeNum = wholeNum
        self.numberBasis = numberBasis

    def initialize(self):
        self.tempPTP = np.mat(np.dot(self.data.T, self.data))

    def getDesignMatrix(self,trains, tempTrainIndicies,tempTestIndicies, basisFun):
        '''
        
        :param trains: number of training
        :param tempTrainIndicies:the indicies of the training data
        :param tempTestIndicies: the indicies of the testing data
        :param basisFun: can be gaussian, sigmoidal or power
        :return:
        '''
        # target
        self.wholeT = self.wholedata[:,-1]
        # features extraction
        self.wholeF = np.tile(np.reshape(self.wholedata[:, 0], (self.wholeNum, 1)), (1, self.numberBasis))
        for j in range(1,self.numberBasis):
            self.wholeF[:, j] = map(lambda x:basisFun(x,j), self.wholeF[:, j])

        self.wholeF[:,0] = 1

        #split train and test & features and targets
        #train data
        self.N = trains
        self.t = self.wholeT[tempTrainIndicies]
        self.data = self.wholeF[tempTrainIndicies]
        
        #test data
        self.testData = self.wholeF[tempTestIndicies]
        self.testT = self.wholeT[tempTestIndicies]

    def esitimateParameters(self,mylambda=0):
        '''
        :param mylambda: if not equals 0 then this model is converted to regularized least squares otherwise
        just Maximum Likelihood or Least squares
        :return:
        '''
        self.wHat =np.dot(np.dot((mylambda*np.eye(self.numberBasis)+np.mat(self.tempPTP)).I,self.data.T),self.t).T
        
    def predict(self):
        self.predictT = np.dot(self.testData,self.wHat)

   

