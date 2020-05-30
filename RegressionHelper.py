# -*- coding: utf-8 -*-
"""
Created on Tue May 26 19:36:04 2020

@author: Rajiv
"""


import numpy as np


class ActivationFunctions:
    
    def __init__(self, name = 'sigmoid', alpha = 1):
        self.alpha = alpha
        self.name = name
        self.bind()
        
    def bind(self):
        
        if self.name == 'sigmoid':
            self.phi = self.sigmoid    
            self.dphi = self.dsigmoid
        elif self.name == 'softplus':
            self.phi = self.softplus    
            self.dphi = self.dsoftplus
        elif self.name == 'linear':
            self.phi = self.linear
            self.dphi = self.dlinear
        else:
            raise RuntimeError('unknown activation function')

    def sigmoid(self, z):
        # activation function as a function of z
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))      
    
    def dsigmoid(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))    
    
    def softplus(self, z):
        return np.log(1+np.exp(np.clip(self.alpha*z, -250, 250)))      
        
    def dsoftplus(self, z):
        return self.alpha*self.sigmoid(z*self.alpha)
        
    
    def linear(self, z):
        return z
    
    def dlinear(self, z):
        return 1.0


def writeWeightsAndBiases(w, b, fileName):
    
    file = open(fileName,'w',newline='')

    # flattens the weights and records the dimension of each weight matrix
    weights = []
    weight_sizes = []
    for i in range(len(w)):
        weight_sizes.append(w[i].shape[0])
        weight_sizes.append(w[i].shape[1])
        weights.append(np.array(w[i]).flatten())

    flattened_weights = np.concatenate(weights)
    
    # flatten the biases
    flattened_biases = np.concatenate(b)


    # writes out the dimension and the flattened weights to a .csv file
    np.savetxt(file, [weight_sizes], delimiter=',', fmt='%i')
    np.savetxt(file, [flattened_weights], delimiter=',')
    np.savetxt(file, [flattened_biases], delimiter=',')

    file.close()


def readWeightsAndBiases(fileName):

    file = open(fileName,'r',newline='')
    
    # reads in the weights sizes and flattened weights and flattened biases
    weight_sizes = np.loadtxt(file,max_rows=1, delimiter=',',dtype=int)
    flattened_weights = np.loadtxt(file, skiprows = 0, max_rows=1, delimiter=',',dtype=float)
    flattened_biases = np.loadtxt(file, skiprows = 0, max_rows=1, delimiter=',',dtype=float)
    
    # unpack the weights into 2d arrays
    L = int(len(weight_sizes) / 2)
    c = 0
    d = 0
    weights = []
    biases = []
    for l in range(L):
        
        weights.append(np.zeros((weight_sizes[l*2], weight_sizes[l*2+1])))
        biases.append(np.zeros((weight_sizes[l*2+1])))
        
        for j in range(weight_sizes[l*2+1]):
            biases[l][j] = flattened_biases[d]
            d += 1
            
        for i in range(weight_sizes[l*2]):
            for j in range(weight_sizes[l*2+1]):
                weights[l][i][j] = flattened_weights[c]
                c += 1            
                
    file.close()
    
    return weights, biases


def writeGradientCheck(error_w, error_b, fileName):
    
    file = open(fileName,'w',newline='')

    for i in range(1,len(error_w)):     
        np.savetxt(file, error_w[i], delimiter=',')
        
        
    for i in range(1,len(error_b)):         
        np.savetxt(file, error_b[i], delimiter=',')
                
    file.close()
    
    
    
def writeFeedForwardCheck(error, fileName):
    
    file = open(fileName,'a',newline='')

    for i in range(len(error)):     
        np.savetxt(file, error[i], delimiter=',')
        
                
    file.close()