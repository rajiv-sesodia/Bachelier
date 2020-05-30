# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:34:56 2020

@author: Rajiv

This program trains a Neural Network on the Bachelier formula
TO DO: add a regularisation function to the weights 

"""

import pandas as pd
import numpy as np
from regressiondnn import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data - you;ll have to change this path of course depending on where you put your data
df = pd.read_excel("C:/Users/Rajiv/Google Drive/Science/Python/sandbox/DNN/Regression/Bachelier/Generator_v0.2.xlsx",usecols="M:U")

#remove zero values of call
df = df.drop(df[df['Call Price'] < 1e-05].index)

# remove too large values of d
df = df.drop(df[df['d'] < -10].index)
df = df.drop(df[df['d'] > 10].index)
df = df.drop(df[df['var'] < 1e-04].index)

#extract the training sample
X, Y = df.iloc[:, [0,1,2,3]].values, df.iloc[:, [4]].values

# split into training batch and test batch
# the training batch is used to train the neural network and the test batch is used to test the trained network
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle = False, random_state=0)

# this is a crucial step - it scales each training sample by the mean and variance of the sample. 
# scaling features dramatically improves the convergence of the network
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
        
# Noq create the basic structure of the Neural Network, essentially the number of nodes at each layer. Size of N is the number of layers
N = np.array([4,10,1]) #number of nodes in each layer

# basic error checking on inputs. Should have more checks here I guess
if N[N.shape[0]-1] != Y.shape[1]:
    raise RuntimeError('Last layer must be equal to the number of class variables')
    
# this is the learning rate. The higher the rate, the faster it learns, but the more noisy and unstable the convergence is.
# higher learning rates can miss minima which is essentially what the NN is trying to find
eta = 1.0

# now create the neural network class
NN = NeuralNetwork(N)
NN.initialise('weights.csv')

# loss is a vector showing how the loss varies with each iteration of the algorithm (epoch)
loss = []

# we do tshe calculation in batches as it is more efficient
batch = 20
epochs = 20

# fit the data
NN.fit(eta, epochs, X_train_std, Y_train, batch, loss, '', 'diagnostics.csv')

# # check how well we fitted the training data and test data
file_output = open('output.csv','w')
file_output.write("F,K,vol,T,pred,actual: \n")
zOut, aOut = NN.feedForward(X_train_std)
np.savetxt(file_output, np.c_[X_train, aOut[NN.L-1], Y_train], delimiter=',')
zOut, aOut = NN.feedForward(X_test_std)
np.savetxt(file_output, np.c_[X_test, aOut[NN.L-1], Y_test], delimiter=',')
file_output.close()

# check loss convergence
file_loss = open('loss.csv','w')
np.savetxt(file_loss, loss)
file_loss.close()


    

    


    













