# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:34:56 2020

@author: Rajiv

This program trains a Neural Network on the Bachelier formula
TO DO: add a regularisation function to the weights 

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from RegressionDNN import NeuralNetwork
from Bachelier import BachelierPricer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data - you;ll have to change this path of course depending on where you put your data
df = pd.read_excel("C:/Users/Rajiv/Google Drive/Science/Python/sandbox/DNN/Regression/Bachelier/Generator.xlsx",usecols="M:U")

# remove data with no explanatory power
# df = df.drop(df[df['Call Price'] < 1e-05].index)
# df = df.drop(df[df['d'] < -10].index)
# df = df.drop(df[df['d'] > 10].index)
# df = df.drop(df[df['var'] < 1e-04].index)

#extract the training sample
X, Y = df.iloc[:, [0,1,2,3]].values, df.iloc[:, [4,5,6]].values

# split into training batch and test batch
# the training batch is used to train the neural network and the test batch is used to test the trained network
X_train, X_test, Y_train_all, Y_test_all = train_test_split(X, Y, test_size=0.2, shuffle = False, random_state=0)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
Y_train = Y_train_all[:,[0]]
Y_test = Y_test_all[:,[0]]
        
# Now create the basic structure of the Neural Network, essentially the number of nodes at each layer. Size of N is the number of layers
N = np.array([4,10,1]) #number of nodes in each layer

# basic error checking on inputs. Should have more checks here I guess
if N[N.shape[0]-1] != Y_train.shape[1]:
    raise RuntimeError('Last layer must be equal to the number of class variables')
    
# this is the learning rate. The higher the rate, the faster it learns, but the more noisy and unstable the convergence is.
# higher learning rates can miss minima which is essentially what the NN is trying to find
eta = 1.2

# now create the neural network class
L2 = 0.0#1 / X_train.shape[0]
NN = NeuralNetwork(N, L2)
NN.initialise('weights.csv')

# loss is a vector showing how the loss varies with each iteration of the algorithm (epoch)
loss = []

# we do the calculation in batches as it is more efficient
batch = 50
epochs = 0

# fit the data
NN.fit(eta, L2, epochs, X_train_std, Y_train, batch, loss, stdsc, 'weights.csv', 'diagnostics.csv')

# check how well we fitted the training data 
file_output = open('output_train.csv','w')
file_output.write("F,K,vol,T,pv_p,pv,bp_error,delta_p,delta,vega_p,vega \n")
z_train, a_train = NN.feedForward(X_train_std)
bperror = 10000*np.abs(a_train[NN.L-1]-Y_train)
dyda_train = NN.gradient(eta, X_train_std, Y_train, stdsc)
delta_train = np.zeros(X_train_std.shape[0])
vega_train = np.zeros(X_train_std.shape[0])
for m in range(X_train_std.shape[0]):
    delta_train[m] = dyda_train[m][0][0]
    vega_train[m] = dyda_train[m][0][2]


np.savetxt(file_output, np.c_[X_train, a_train[NN.L-1], Y_train, bperror, delta_train, Y_train_all[:,1], vega_train, Y_train_all[:,2]], delimiter=',')
file_output.close()

# check how well we fitted the test data 
file_output = open('output_test.csv','w')
file_output.write("F,K,vol,T,pv_p,pv,bp_error,delta_p,delta,vega_p,vega \n")
z_test, a_test = NN.feedForward(X_test_std)
bperror = 10000*np.abs(a_test[NN.L-1]-Y_test)
dyda_test = NN.gradient(eta, X_test_std, Y_test, stdsc)
delta_test = np.zeros(X_test_std.shape[0])
vega_test = np.zeros(X_test_std.shape[0])
for m in range(X_test_std.shape[0]):
    delta_test[m] = dyda_test[m][0][0]
    vega_test[m] = dyda_test[m][0][2]


np.savetxt(file_output, np.c_[X_test, a_test[NN.L-1], Y_test, bperror, delta_test, Y_test_all[:,1], vega_test, Y_test_all[:,2]], delimiter=',')
file_output.close()

# visual check on slices
F = np.arange(-0.01, 0.05,0.001)
V = np.linspace(0.0010, 0.0150, num = 4)
T = np.linspace(0.1, 5, num = 5)

fig, axis = plt.subplots(V.shape[0], T.shape[0], sharex=True, sharey=True, squeeze=True)
for i in range(V.shape[0]):
    for j in range(T.shape[0]):
        
        # calculation
        X_slice = np.array([F, np.full(F.shape[0],0.01), np.full(F.shape[0],V[i]), np.full(F.shape[0],T[j])]).T
        X_slice_std = stdsc.transform(X_slice)
        zOut, aOut = NN.feedForward(X_slice_std)
        b = BachelierPricer('call').valueSlice(X_slice)

        #plot the result  
        axis[i,j].tick_params(axis='both', which='major', labelsize=5)
        axis[i,j].plot(F,aOut[NN.L-1], linewidth=1, label = '{0:.2f},{1:.1f}'.format(V[i]*100.0,T[j]))
        axis[i,j].plot(F,b, linewidth=1)
        axis[i,j].xaxis.set_ticks(np.arange(-0.01, 0.06, 0.02))
        axis[i,j].yaxis.set_ticks(np.arange(0, 0.06, 0.01))
        axis[i,j].legend(loc='upper left', fontsize='xx-small')

    













