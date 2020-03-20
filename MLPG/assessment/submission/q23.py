import q2dat
import pystan
import numpy as np
import matplotlib.pyplot as plt

margin = q2dat.L.index(1)
L0 = q2dat.L[0:margin]
S0 = q2dat.S[0:margin]
A0 = q2dat.A[0:margin]
P0 = q2dat.P[0:margin]

L1 = q2dat.L[margin:]
S1 = q2dat.S[margin:]
A1 = q2dat.A[margin:]
P1 = q2dat.P[margin:]

datadkt_0 = {
    'S': S0,
    'A': A0,
    'P': P0,
    'N': len(S0)
}
datadkt_1 = {
    'S': S1,
    'A': A1,
    'P': P1,
    'N': len(S1)
}

stan_model = """
data {
    int<lower=0> N; // number of data items
    vector[N] A; // predictor Age
    vector[N] S; // predictor Size
    vector[N] P; // outcome vector
}
parameters {
    real alpha; // intercept
    real beta_A;
    real<lower=0> beta_S; 
    real<lower=0> sigma; // error scale
}
model {
    P ~ normal(A * beta_A + S * beta_S + alpha, sigma); // likelihood
}
"""

def main():
    sm = pystan.StanModel(model_code=stan_model)
    fit_L0 = sm.sampling(data=datadkt_0, iter=1000, chains=4)
    fit_L1 = sm.sampling(data=datadkt_1, iter=1000, chains=4)
    print("THE FIT FOR LOCALE 0:")
    print(fit_L0)
    fit_L0.plot()
    plt.show()

    print("THE FIT FOR LOCALE 1:")
    print(fit_L1)
    fit_L1.plot()
    plt.show()


if __name__ == '__main__':
    main()
