# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:34:56 2020

@author: Rajiv

This program trains a Neural Network on the Bachelier formula
TO DO: add a regularisation function to the weights 

"""

import numpy as np
from RegressionHelper import ActivationFunctions
from RegressionHelper import readWeightsAndBiases
from RegressionHelper import writeWeightsAndBiases
from RegressionHelper import writeGradientCheck
from RegressionHelper import writeFeedForwardCheck

# One of my first Neural Networks
class NeuralNetwork:
    
    def __init__(self, N, alpha = 1):
        # L is the number of layers in the neural network
        # N is an array containing the number of nodes in each layer
        self.L = N.shape[0]
        self.N = N
        self.af = ActivationFunctions('sigmoid', alpha)
        self.phi = self.af.phi 
        self.dphi = self.af.dphi

        
    def initialise(self, weightsAndBiasesFile=''):
    
        # initialise weights and biases
        self.w = []
        self.b = []
        
        # if weights file supplied, read them in
        if weightsAndBiasesFile:
            self.w, self.b = readWeightsAndBiases(weightsAndBiasesFile)
            return
            
        # otherwise set randome weights
        # always important to set the seed for comparability
        np.random.seed(0)
        
        # these are members of the NN class
        self.w.append(np.zeros((self.N[0],self.N[0])))
        self.b.append(np.zeros(self.N[0]))
        
        # the initial weights and biases are set to a random number taken from a (standard) normal distribution, i.e.
        # with mean 0 and variance 1
        for l in range(1,self.L):
            self.w.append(np.random.normal(0.0,1.0,(self.N[l-1],self.N[l])))
            self.b.append(np.random.normal(0.0,1.0,self.N[l]))
               
    def feedForward(self, X):                
        
        # calculates the perceptron value (z) and activated value (a) through the activation function
        X = X if X.ndim > 1 else X.reshape(1,X.shape[0])
        
        # note, no activation for input layer, a = X
        a = [X]
        z=[np.zeros((X.shape[0],X.shape[1]))]
        for l in range(0,self.L-1):            
            z.append(a[l].dot(self.w[l+1]) + self.b[l+1])
            a.append(self.phi(z[l+1]))
            
        return z, a
    
    
    def calc_dcdz(self, a, z, y):
        # derivative of the cost function (c) w.r.t. perceptron value (z), i.e. dcdz 
        
        # special treatment for last layer as this derivative is from the cost function itself.
        # for the remainder of the layers following recursion formulae
        dcdz = [(2.0 / self.N[self.L-1]) * np.multiply( (a[self.L-1]-y) , self.dphi(z[self.L-1])) ]
        for l in reversed(range(0,self.L-1)):
              dcdz.insert(0,np.multiply( dcdz[0].dot((self.w[l+1]).T), self.dphi(z[l])))

        return dcdz

    # backpropogation of cost calculated at output layer through the network, updating the weights and biases as we go along
    # returns dcdz which is needed for derivative
    def backProp(self, eta, z, a, y):
                
        # derivative of the cost function (c) w.r.t. perceptron value (z), i.e. dcdz 
        dcdz = self.calc_dcdz(a, z, y)

        # calculating derivatives of the cost (c) function w.r.t weights (w), i.e. (dcdw) and bias (b), i.e. (dcdb) 
        # and updating weights MUST be done AFTER the derivatives are calculated,
        # as the latter depends on the former             
        for l in reversed(range(1,self.L)):
            dcdw = a[l-1].T.dot(dcdz[l])
            dcdb = np.sum(dcdz[l], axis = 0)
            self.w[l] -= eta * dcdw
            self.b[l] -= eta * dcdb
            
    
    def calcLoss(self, X, Y):
        zTemp, aTemp = self.feedForward(X)
        loss =  np.sum(np.square(Y-aTemp[self.L-1]))
            
        return loss
    
     
    def fit(self, eta, epochs, X, Y, batchSize, loss, weightsAndBiasesFile='', diagnosticsFile=''):
        rgen = np.random.RandomState(1)
        
        for epoch in range(epochs):
            
            # shuffle
            r = rgen.permutation(len(Y))

            
            # loop over entire randomised set
            for n in range(0, len(Y) - batchSize + 1, batchSize):
            
                # determine indices to use
                indices = r[n : n + batchSize]
            
                # feedforward
                z, a = self.feedForward(X[indices])
            
                # backprop
                self.backProp(eta, z, a, Y[indices])                
        
            # check error
            if epoch % 100 == 0:
                temp = self.calcLoss(X,Y)
                print('epoch = ', epoch, 'loss = ',temp, 'eta = ', eta)
                loss.append(temp)
                eta = eta * 0.995
                

        if weightsAndBiasesFile:
            writeWeightsAndBiases(self.w, self.b, weightsAndBiasesFile)
            
        if diagnosticsFile:
            error_w, error_b = self.GradientCheck(X,Y)
            writeGradientCheck(error_w, error_b, diagnosticsFile) 
            error_a = self.feedForwardCheck(X[0])
            writeFeedForwardCheck(error_a, diagnosticsFile)


    def backPropGradientCheck(self, eta, z, a, y):
                
        # derivative of the cost function (c) w.r.t. perceptron value (z), i.e. dcdz 
        dcdz = self.calc_dcdz(a, z, y)

        # calculating derivatives of the cost (c) function w.r.t weights (w), i.e. (dcdw) and bias (b), i.e. (dcdb) 
        # and updating weights MUST be done AFTER the derivatives are calculated,
        # as the latter depends on the former             
        dcdw = [a[self.L-2].T.dot(dcdz[self.L-1])]
        dcdb = [np.sum(dcdz[self.L-1], axis = 0)]
        for l in reversed(range(1,self.L-1)):
            dcdw.insert(0, a[l-1].T.dot(dcdz[l]))
            dcdb.insert(0, np.sum(dcdz[l], axis = 0))

        
        dcdw.insert(0,0)
        dcdb.insert(0,0)
        
        return dcdw, dcdb
    


    def GradientCheck(self, X, Y):        
        
        # base case
        eps = 1e-06
        if self.w == []:
            self.initialise()
        
        # calculate derivative
        z, a = self.feedForward(X)
        dcdw, dcdb = self.backPropGradientCheck(1.0, z, a, Y)
        
        # calculate error in weights
        for n in range(1, self.L):
            for lm1 in range(self.N[n-1]):
                for l in range(self.N[n]):
                    
                    # up
                    self.w[n][lm1][l] += eps
                    C_up = self.calcLoss(X, Y)
                    
                    # down
                    self.w[n][lm1][l] -= 2.0*eps
                    C_down = self.calcLoss(X, Y)
                    
                    # error in deriv
                    dcdw[n][lm1][l] -= (C_up - C_down) / (2.0 * eps)
                    
                    # restore original value
                    self.w[n][lm1][l] += eps
                                      
     

        # calculate error in bias
        for n in range(1, self.L):    
            for l in range(self.N[n]):
                
                # up
                self.b[n][l] += eps
                C_up = self.calcLoss(X, Y)
                
                # down
                self.b[n][l] -= 2.0 * eps
                C_down = self.calcLoss(X, Y)
                
                # error in deriv
                dcdb[n][l] -= (C_up - C_down) / (2.0 * eps)
                
                # restore original value
                self.b[n][l] += eps
            
        return dcdw, dcdb
            
        
    def feedForwardCheck(self, X):         
        
        # temporary override of class members
        import copy
        w_ = copy.deepcopy(self.w)
        b_ = copy.deepcopy(self.b)
        name = self.af.name
        alpha = self.af.alpha
        self.af = ActivationFunctions('linear')
        self.phi = self.af.phi 
        self.dphi = self.af.dphi
            
        for l in range(1,self.L):
            for i in range(self.N[l-1]):
                for j in range(self.N[l]):
                    self.w[l][i][j] = 1.0/(self.N[l-1])
                   
        for l in range(1,self.L):        
            for j in range(self.N[l]):
                self.b[l][j] = 0.0
            
        result = np.full(self.N[self.L-1], np.average(X))
        

        z, a = self.feedForward(X)
        error = result - a[self.L-1]
        
        # restore class members to original values
        self.w = copy.deepcopy(w_)
        self.b = copy.deepcopy(b_)
        self.af = ActivationFunctions(name, alpha)
        self.phi = self.af.phi 
        self.dphi = self.af.dphi
        
        return error
        