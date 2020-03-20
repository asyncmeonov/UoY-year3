import q2dat
import pystan
import numpy as np
import matplotlib.pyplot as plt

stan_model = """
data {
    int<lower=0> N; // total number of houses
    int<lower = 1, upper = 2> L[N];
    vector[N] A; // predictor Age
    vector[N] S; // predictor Size
    vector[N] P; // outcome vector, Price
}
parameters {
    real alpha; // intercept

    // slopes and their variance
    vector[2] beta_A;
    vector<lower=0>[2] beta_S;

    // //slope noise
    real<lower=0> sigma_beta_A;
    real<lower=0> sigma_beta_S;

    real sigma_y; // general noise
}
transformed parameters {
    vector[N] y_hat;
    for (i in 1:N)
      y_hat[i] = alpha + A[i]*beta_A[L[i]] + S[i]*beta_S[L[i]];
}

model {

    beta_S ~ normal(0, sigma_beta_S);
    beta_A ~ normal(0, sigma_beta_A);
    P ~ normal(y_hat, sigma_y); // likelihood
}
"""

def main():
    sm = pystan.StanModel(model_code=stan_model)
    # formatting locales to be [1,2] so that we can use them as indices in STAN
    mod_L = np.add(q2dat.datadkt['L'],1)
    print(mod_L)
    q2dat.datadkt['L'] = mod_L
    fit = sm.sampling(data=q2dat.datadkt, iter=1000, chains=4)
    print("THE FIT")
    print(fit)
    fit.plot(['alpha','beta_A','beta_S','sigma_y'])
    plt.show()

if __name__ == '__main__':
    main()
