# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:30:44 2020

@author: Rajiv
"""


import numpy as np

def writeWeights(w, fileName):
    
    file = open(fileName,'w',newline='')

    weights = []
    weight_sizes = []
    for i in range(len(x)):
        weight_sizes.append(w[i].shape[0])
        weight_sizes.append(w[i].shape[1])
        weights.append(np.array(w[i]).flatten())

    flattened_weights = np.concatenate(weights)


    np.savetxt(file, [weight_sizes], delimiter=',', fmt='%i')
    np.savetxt(file, [flattened_weights], delimiter=',')

    file.close()


def readWeights(fileName):

    file = open(fileName,'r',newline='')
    weight_sizes = np.loadtxt(file,max_rows=1, delimiter=',',dtype=int)
    flattened_weights = np.loadtxt(file,skiprows=0, delimiter=',',dtype=float)
    
    L = int(len(weight_sizes) / 2)
    c = 0
    weights = []
    for l in range(L):
        weights.append(np.zeros((weight_sizes[l*2], weight_sizes[l*2+1])))
        for i in range(weight_sizes[l*2]):
            for j in range(weight_sizes[l*2+1]):
                weights[l][i][j] = flattened_weights[c]
                c += 1

    file.close()
    
    return weights

np.random.seed(0)
x = []
x.append(np.random.normal(0.0,1.0,(2,6)))
x.append(np.random.normal(0.0,1.0,(6,3)))
x.append(np.random.normal(0.0,1.0,(3,5)))
x.append(np.random.normal(0.0,1.0,(5,2)))

# write the weights
print(x)
writeWeights(x, 'test.csv')

# read the weights
x_in = readWeights('test.csv')
print(x_in)









