import q2dat
import pystan
import seaborn
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
#    fit.plot(['alpha','sigma_a','beta_A','beta_S','sigma_y'])
    fit.plot(['alpha','beta_A','beta_S','sigma_y'])
    plt.show()

    # alpha = fit['alpha']
    # beta_A = fit['beta_A']
    # beta_S = fit['beta_S']
    # beta_L = fit['beta_L']
    # sigma = fit['sigma']
    # lp = fit['lp__']
    # plot_trace(alpha, "intercept")
    # plt.show()
    # plot_trace(beta_A, "beta_Age")
    # plt.show()
    # plot_trace(beta_S, "beta_Size")
    # plt.show()
    # plot_trace(beta_L, "beta_Location")
    # plt.show()
    # plot_trace(sigma, "noise")
    # plt.show()


def plot_trace(param, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""

    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)

    # Plotting
    plt.subplot(2, 1, 1)
    plt.plot(param)
    plt.xlabel('samples')
    plt.ylabel(param_name)
    plt.axhline(mean, color='r', lw=2, linestyle='--')
    plt.axhline(median, color='c', lw=2, linestyle='--')
    plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
    plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
    plt.title('Trace and Posterior Distribution for {}'.format(param_name))

    plt.subplot(2, 1, 2)
    plt.hist(param, 30, density=True)
    seaborn.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--', label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--', label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)

    plt.gcf().tight_layout()
    plt.legend()


if __name__ == '__main__':
    main()
