# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:03:53 2020

@author: Rajiv
"""


import numpy as np
from Bachelier import BachelierPricer

file = open('linearData.csv',"w")
file.write('F, K, V, T, Call, Delta, Vega, Var, d \n')

N = 10
F = np.linspace(-0.01, 0.05, N)
K = np.linspace(-0.01, 0.05, N)
V = np.linspace(0.0050, 0.0150, N)
T = np.linspace(0.01,5.0, N)

for f in F:
    for k in K:
        for v in V:
            for t in T:
                var = v*v*t
                d = (f-k)/var
                call = BachelierPricer().value(f,k,v,t)
                delta = BachelierPricer().delta(f,k,v,t)
                vega = BachelierPricer().vega(f,k,v,t)
                np.savetxt(file, np.c_[f, k, v, t, call, delta, vega, var, d], delimiter=',')
    
file.close()