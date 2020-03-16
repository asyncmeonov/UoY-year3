import q2dat
import pystan
import seaborn
import numpy as np
import matplotlib.pyplot as plt

def main():
     X = np.array([q2dat.L, q2dat.S, q2dat.A]).T
     K = 3

     print("[[-L---S---A-]]")
     print(X)

     print("P")
     print(q2dat.P)

    datadkt = {'N':len(q2dat.L), 'K':K, 'x':X, 'y':q2dat.P}
    datadkt = q2dat.datadkt
    sm = pystan.StanModel(model_code=stan_model)
    fit = sm.sampling(data=datadkt,iter=1000, chains=4)
    print("THE FIT:")
    print(fit)

    alpha = fit['alpha']
    beta_A = fit['beta_A']
    beta_S = fit['beta_S']
    beta_L = fit['beta_L']
    sigma = fit['sigma']
    lp = fit['lp__']
    plot_trace(alpha,"intercept")
    plt.show()
    plot_trace(beta_A,"beta_Age")
    plt.show()
    plot_trace(beta_S,"beta_Size")
    plt.show()
    plot_trace(beta_L,"beta_Location")
    plt.show()
    plot_trace(sigma,"noise")
    plt.show()

def plot_trace(param, param_name='parameter'):
    """Plot the trace and posterior of a parameter."""
    
    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 2.5), np.percentile(param, 97.5)
    
    # Plotting
    plt.subplot(2,1,1)
    plt.plot(param)
    plt.xlabel('samples')
    plt.ylabel(param_name)
    plt.axhline(mean, color='r', lw=2, linestyle='--')
    plt.axhline(median, color='c', lw=2, linestyle='--')
    plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
    plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
    plt.title('Trace and Posterior Distribution for {}'.format(param_name))

    plt.subplot(2,1,2)
    plt.hist(param, 30, density=True); seaborn.kdeplot(param, shade=True)
    plt.xlabel(param_name)
    plt.ylabel('density')
    plt.axvline(mean, color='r', lw=2, linestyle='--',label='mean')
    plt.axvline(median, color='c', lw=2, linestyle='--',label='median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.2, label='95% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.2)
    
    plt.gcf().tight_layout()
    plt.legend()

if __name__ == '__main__':
    main()

