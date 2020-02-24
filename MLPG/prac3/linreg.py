#!/usr/bin/env python3

import numpy as np
import pystan

# Fix size of data
N = 30
# Generate data from uniform distribution over [0,10]
x = 10 * np.random.random_sample((N,))
# Define a 'true' intercept, slope and standard deviation
a = -2
b = 3.5
sigma = 2
# Generate data from P(y|x) where y ~ ax + b + N(0,sigma^2) 
# Sampling from N(0,1) and multiplying by sigma is the same as sampling
# from N(0,sigma^2)
y = []
for xi in x:
    y.append(a + b*xi + sigma * np.random.randn())

datadkt = {'N':N, 'x':x, 'y':y}
sm = pystan.StanModel(file='regression.stan')
fit = sm.sampling(data=datadkt,iter=1000, chains=4)
print(fit)

