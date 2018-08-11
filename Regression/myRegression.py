#!/usr/local/bin/python
# -*-coding:utf-8 -*-

"""
@author: Tingtingzhu
@email:tingtingzhu93@gmail.com
@Date:2018.8.11 11:23
@Last modified by:Tingtingzhu 2018.8.11
"""
import numpy as np
import random
from BayesianRegression import *
from BasisFunction import *
from measures import *
from normalEquation import *
from LeastMeanSquares import *

MeasureChoice = ErrorMeasure()
basisfun = BasisFunction()

class MyRegression():

    def __init__(self,f):
        '''

        :param f: file of the data
        :return:
        '''
        tempdata = f.readlines()
        self.N = len(tempdata)
        self.wholeData = []
        for i in tempdata:
            self.wholeData.append(map(float,i.strip().split('\t')))

    def splitData(self,trainingRatio):
        '''

        :param trainingRatio:
        what percentage of data used to training
        :return:
        '''
        self.dataDim = len(self.wholeData[0])
        self.wholeData = np.reshape(self.wholeData, (self.N, self.dataDim))

        self.trainsNum = np.int(np.ceil(self.N*trainingRatio))
        self.testsNum = self.N - self.trainsNum
        #training data
        self.tempTrainIndicies = random.sample(range(self.N), self.trainsNum)
        #testing data
        self.tempTestIndicies = list(set(range(self.N))^set(self.tempTrainIndicies))

        # print tempTrainIndicies
        # print tempTestIndicies
        # self.trainData = self.wholeData[self.tempTrainIndicies]
        # self.testData = self.wholeData[self.tempTestIndicies]

        # print self.trainData
if __name__ == '__main__':
    DataFile = '/Users/tingtingzhu/学习书籍/prml/practice/data.txt'

    with open(DataFile) as f:
        #split data
        myDemo = MyRegression(f)
        myDemo.splitData(0.7)

        # #1.bayesis regression
        # #esitimate
        # mybayesis = BayesianRegression(myDemo.wholeData,myDemo.N)
        #
        # mybayesis.getDesignMatrix(myDemo.trainsNum, myDemo.tempTrainIndicies,
        #                           myDemo.tempTestIndicies,10,basisfun.powerBasis)
        # mybayesis.initialize()
        # mybayesis.esitimateParameters()
        #
        # #predict
        # mybayesis.predict()
        # # mybayesis.error()
        # hatY = np.array(mybayesis.predictT)[:, 0]
        # realY = mybayesis.testT
        #
        # # measure
        # MeasureChoice.rmse(realY, hatY, myDemo.testsNum)

        # #2.normal equeation + regularized least squares
        # mymodel = MLRegression(myDemo.wholeData, myDemo.N,12)
        #
        # mymodel.getDesignMatrix(myDemo.trainsNum, myDemo.tempTrainIndicies,
        #                           myDemo.tempTestIndicies, basisfun.powerBasis)
        # mymodel.initialize()
        # mymodel.esitimateParameters(mylambda=2)
        #
        # # predict
        # mymodel.predict()
        # # mybayesis.error()
        # hatY = np.array(mymodel.predictT)[:, 0]
        # realY = mymodel.testT
        # # realY = mymodel.t
        #
        # # measure
        # MeasureChoice.rmse(realY, hatY, myDemo.testsNum)


        #3. least squares with stochastic gradient decent
        mymodel = MLRegression1(myDemo.wholeData, myDemo.N,10)

        mymodel.getDesignMatrix(myDemo.trainsNum, myDemo.tempTrainIndicies,
                                  myDemo.tempTestIndicies, basisfun.powerBasis)
        mymodel.initialize()
        mymodel.esitimateParameters()

        # predict
        mymodel.predict()
        # mybayesis.error()
        hatY = np.array(mymodel.predictT)[:, 0]
        realY = mymodel.testT
        # realY = mymodel.t

        # measure
        MeasureChoice.rmse(realY, hatY, myDemo.testsNum)