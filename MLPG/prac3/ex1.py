import pystan


sm = pystan.StanModel(file='./bernoulli.stan')
fit = sm.sampling(data={'N':5,'y':[0,0,0,1,1]},iter=1000, chains=4)
print(fit)
plt = fit.plot()
plt.show()
