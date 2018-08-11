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
def is_complex(x):
    # print isinstance(x,complex)
    return not isinstance(x,complex)

class BayesianRegression():
    def __init__(self,dataMat,wholeNum):
        '''

        :param dataMat: The whole data
        :param wholeNum: how many datas
        '''
        self.wholedata = dataMat
        self.wholeNum = wholeNum

    def getDesignMatrix(self,trains, tempTrainIndicies,tempTestIndicies,numberBasis, basisFun):
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
        self.wholeF = np.tile(np.reshape(self.wholedata[:, 0], (self.wholeNum, 1)), (1, numberBasis))
        for j in range(1,numberBasis):
            #DataMat[:, j] = map(lambda x: np.power(x, 2), DataMat[:, j])
            self.wholeF[:, j] = map(lambda x:basisFun(x,j), self.wholeF[:, j])

        self.wholeF[:,0] = 1

        #split train and test features and targets
        self.N = trains
        self.t = self.wholeT[tempTrainIndicies]
        self.data = self.wholeF[tempTrainIndicies]

        self.testData = self.wholeF[tempTestIndicies]
        self.testT = self.wholeT[tempTestIndicies]

        print self.testData
        print self.testT



    def initialize(self,alpha=0.1,beta=0.1):
        '''

        :return:
        '''
        self.alpha = alpha
        self.beta = beta
        self.tempPTP =  np.mat(np.dot(self.data.T,self.data))
        self.lambdaBasis,_ =np.linalg.eig(self.tempPTP)#ARRAY array([,,,...])
        # print type(self.lambdaBasis)
        self.lambdaBasis = np.array(filter(is_complex,self.lambdaBasis))

        # print type(np.array(self.lambdaBasis))

    def esitimateParameters(self,tol=0.001):
        '''
        to esitimate the parameters alpha and beta by iteration
        until convergence
        :param tol:
        :return:
        '''
        count =1
        while True:

            tempAlpha = self.alpha
            tempBeta = self.beta

            #get the real lambda
            # print type(self.lambdaBasis),'ok'
            tempLambda = self.lambdaBasis*self.beta

            #compute gama
            tempGama = sum(map(lambda x:x*1.0/(self.alpha+x),tempLambda))

            #reesitmate alpha and beta
            self.A = self.alpha+self.beta*self.tempPTP

            self.mN = self.beta*(np.dot(np.dot(self.A.I,self.data.T),self.t))

            self.mN = self.mN.T

            self.alpha = tempGama/np.dot(self.mN.T,self.mN)[0,0]

            self.beta = np.power(LA.norm((self.t-np.dot(self.data,self.mN)),2),2)/(self.N-tempGama)

            if min(map(lambda x:abs(x),[self.alpha-tempAlpha,self.beta-tempBeta]))<=tol:
                break
            else:
                count+=1
        print "total iteration : %d"%(count)
    def predict(self):
        self.predictT = np.dot(self.testData,self.mN)


    # def error(self):
    #     temp = np.array(np.reshape(self.predictT-np.reshape(self.testT,(60,1)),(1,60)))[0]
    #     temp = map(lambda x:np.power(x,2),temp)
    #     print sum(temp)/60
    #     pass
