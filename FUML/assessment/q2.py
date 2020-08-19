import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

# print("python q2.py TRAINING_SET_PATH TESTING_SET_PATH")
# raw_train_data = sys.argv[1]
# raw_secret_data = sys.argv[2]


def data_loader(data):
    f = data[:, 0:10]
    t = data[:, [10]]
    return f, t


data = np.genfromtxt('continuous.csv', delimiter=',')
secret = np.genfromtxt('continuous.csv', delimiter=',')
#data = np.genfromtxt(raw_train_data, delimiter=',')
#secret = np.genfromtxt(raw_secret_data, delimiter=',')
features, target = data_loader(data)
f_secret, t_secret = data_loader(secret)


def print_exam(model, t_features, t_target):
    print("R^2 value for the secret test %.3f" % model.score(t_features, t_target))
    print("Predictions for the secret test:")
    print(model.predict(t_features))


def evaluate(model, Xt, Yt, has_plot=False):
    print("=" * 50)
    name = type(model).__name__
    print("Metrics for", name)
    print("R^2 value %.3f" % model.score(Xt, Yt))
    print("CrossValidated R^2 values at 3 folds:", cross_val_score(model, Xt, Yt, scoring='r2', cv=3))
    print("Coefficients:", model.coef_)
    print("Intercept i.e. beta0", model.intercept_)
    if has_plot:
        for i in range(len(Xt[0])):
            sorted_index = np.argsort(Xt[:, i])
            sX = np.sort(Xt[:, i])
            sY = Yt[sorted_index]
            plot(sX, sY, model.predict(Xt), ("X" + str(i + 1), "Target"))
    print("=" * 50)


def plot(X, Y, pred=None, labels=("Feature", "Target")):
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.plot(X, Y, 'o', label='data')
    if pred is not None: plt.plot(X, pred, color='blue', linewidth=3)
    plt.show()

r2_features = (features[:, [6, 7, 9]])  # derived from LassoCV previous LassoCV
r2_lassoCV = Lasso(alpha=0.2627152448819272).fit(r2_features, target)
#evaluate(r2_lassoCV, r2_features, target, False)

print_exam(r2_lassoCV, f_secret[:, [6, 7, 9]], t_secret)
