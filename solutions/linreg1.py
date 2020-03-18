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
# Generate data from P(y|x) where y ~ a + bx + N(0,sigma^2) 
# Sampling from N(0,1) and multiplying by sigma is the same as sampling
# from N(0,sigma^2)
y = []
for xi in x:
    y.append(a + b*xi + sigma * np.random.randn())

datadkt = {'N':N, 'x':x, 'y':y}
sm = pystan.StanModel(file='regression1.stan')
fit = sm.sampling(data=datadkt,iter=1000, chains=4)
print(fit)
import matplotlib.pyplot as plt
fit.plot()
plt.show()

explanation = '''

With the original priors we get:

        mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
alpha  -2.74    0.03   0.77  -4.21  -3.25  -2.75  -2.21  -1.26    866    1.0
beta    3.68  4.5e-3   0.13   3.41   3.59   3.67   3.76   3.94    871    1.0
sigma   2.02  8.3e-3   0.28   1.57   1.82   1.99   2.18   2.67   1143    1.0
lp__  -35.18    0.05   1.24 -38.48 -35.76 -34.87 -34.28 -33.79    730   1.01


Just setting the mean of the priors for alpha and beta to the true values gives us posterior means
for alpha and beta closer to the true values (of -2 and 3.5). (See commented out lines in regression1.stan.)
The posterior is also a little more concentrated around its mean (smaller sd): 

        mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
alpha  -2.25    0.02   0.62  -3.47  -2.66  -2.27  -1.84  -1.01    697    1.0
beta    3.52  4.1e-3   0.12   3.28   3.44   3.52    3.6   3.75    791    1.0
sigma   1.87  7.5e-3   0.25   1.46    1.7   1.84   2.02   2.46   1144    1.0
lp__  -32.53    0.05    1.3 -35.86 -33.13 -32.21 -31.57 -31.03    562    1.0

To get even better posteriors (i.e. cheat even more!)
we reduce the prior variances for alpha and beta to 0.1. This gives us these results:

        mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
alpha   -2.0  2.4e-3    0.1  -2.19  -2.07   -2.0  -1.93   -1.8   1720    1.0
beta    3.51  1.2e-3   0.05   3.42   3.48   3.51   3.54   3.61   1629    1.0
sigma   1.56  5.3e-3   0.21   1.21   1.41   1.53   1.67   2.07   1618    1.0
lp__   -27.8    0.04   1.28 -31.16 -28.38 -27.42 -26.85 -26.34    877    1.0

So there is a greater concentration on the true values of alpha and beta as expected.

Note that the lp__ value is also increasing.

We could have also improved results by increasing the value of N.

'''
