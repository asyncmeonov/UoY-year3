#!/usr/bin/env python
import sys

print ("python p1.py DISCRETE_SET")
raw_data = sys.argv[1]

#raw_data = "discrete.csv"


def data_loader(raw_data):
    with open(raw_data, "r") as data:
        X1 = []
        X2 = []
        X3 = []
        Y = []
        arr = data.readlines()[1:]  # remove header
        for datapoint in arr:
            f = datapoint[:-1].split(',')  # -1 indexing to remove \n
            X1.append(int(f[0]))
            X2.append(int(f[1]))
            X3.append(int(f[2]))
            Y.append(int(f[3]))
        return X1, X2, X3, Y


X1, X2, X3, Y = data_loader(raw_data)


# P(X=x)
def marginal(X, x):
    return X.count(x) / len(X)


def matcher(X, x, Y, y):
    return [i for i, j in zip(X, Y) if i == x and j == y]


def marginal_bayes(X, x):
    alpha = 1
    beta = 1
    r = X.count(x)
    n = len(X)
    return (alpha + r) / (alpha + beta + n)


# P(X=x,Y=y)
def joint(X, x, Y, y):
    matches = matcher(X,x,Y,y)
    return len(matches) / len(X)

# joint with priors
def bs_joint(X, x, Y, y):
    alpha = 1
    beta = 1
    jXY = matcher(X, x, Y, y)
    r = len(jXY)
    n = len(Y)
    return (alpha + r) / (alpha + beta + n)


# P(X=0|Y=0) as viewed by frequentists, i.e. MLE
# P(X=0|Y=0) = P(X=0,Y=0)/P(Y=0)
def fr_conditional(X, x, Y, y):
    return joint(X, x, Y, y) / marginal(Y, y)


def bs_conditional(X, x, Y, y):
    return bs_joint(X, x, Y, y) / marginal_bayes(Y, y)


### MLE
def print_MLE(X1, X2, X3, Y):
    print("MLE")
    print("P(Y=0) = %.3f" % marginal(Y, 0))
    print("P(X1=0|Y=0) = %.3f" % fr_conditional(X1, 0, Y, 0))
    print("P(X1=0|Y=1) = %.3f" % fr_conditional(X1, 0, Y, 1))
    print("P(X2=0|Y=0) = %.3f" % fr_conditional(X2, 0, Y, 0))
    print("P(X2=0|Y=1) = %.3f" % fr_conditional(X2, 0, Y, 1))
    print("P(X3=0|Y=0) = %.3f" % fr_conditional(X3, 0, Y, 0))
    print("P(X3=0|Y=1) = %.3f" % fr_conditional(X3, 0, Y, 1))
    print("\n")

def print_bayesian(X1, X2, X3, Y):
    print("Bayesian")
    print("P(Y=0) = %.3f" % marginal_bayes(Y, 0))
    print("P(X1=0|Y=0) = %.3f" % bs_conditional(X1, 0, Y, 0))
    print("P(X1=0|Y=1) = %.3f" % bs_conditional(X1, 0, Y, 1))
    print("P(X2=0|Y=0) = %.3f" % bs_conditional(X2, 0, Y, 0))
    print("P(X2=0|Y=1) = %.3f" % bs_conditional(X2, 0, Y, 1))
    print("P(X3=0|Y=0) = %.3f" % bs_conditional(X3, 0, Y, 0))
    print("P(X3=0|Y=1) = %.3f" % bs_conditional(X3, 0, Y, 1))


print_MLE(X1, X2, X3, Y)
print_bayesian(X1, X2, X3, Y)