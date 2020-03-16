import q2dat
import pystan
import seaborn
import numpy as np
import matplotlib.pyplot as plt

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
    fit_L0 = sm.sampling(data=q2dat.datadkt_0, iter=1000, chains=4)
    fit_L1 = sm.sampling(data=q2dat.datadkt_1, iter=1000, chains=4)
    print("THE FIT FOR LOCALE 0:")
    print(fit_L0)
    fit_L0.plot()
    plt.show()

    print("THE FIT FOR LOCALE 1:")
    print(fit_L1)
    fit_L1.plot()
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
