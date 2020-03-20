import q2dat
import pystan
import numpy as np
import matplotlib.pyplot as plt

stan_model = """
data {
    int<lower=0> N; // number of data items
    vector[N] A; // predictor Age
    vector<lower = 0, upper = 1>[N] L; // predictor Locality
    vector[N] S; // predictor Size
    vector[N] P; // outcome vector
}
parameters {
    real alpha; // intercept
    real beta_L;
    real beta_A;
    real<lower=0> beta_S; 
    real<lower=0> sigma; // error scale
}
model {
    P ~ normal(A * beta_A + S * beta_S + L * beta_L + alpha, sigma); // likelihood
}
"""


def main():
    sm = pystan.StanModel(model_code=stan_model)
    fit = sm.sampling(data=q2dat.datadkt, iter=1000, chains=4)
    print("THE FIT:")
    print(fit)
    fit.plot()
    plt.show()

if __name__ == '__main__':
    main()
