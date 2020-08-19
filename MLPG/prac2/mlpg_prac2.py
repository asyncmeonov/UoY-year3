# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
#       % (X_test.shape[0], (y_test != y_pred).sum()))
# print(gnb.theta_)


# import numpy as np
# rng = np.random.RandomState(1)
# X = rng.randint(5, size=(6, 100))
# y = np.array([1, 2, 3, 4, 5, 6])
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB()
# clf.fit(X, y)

# pred_for = X[2:3]
# print(np.unique(pred_for, return_counts=True))
# print(pred_for)
# print(clf.predict(pred_for))

import numpy as np
data = np.loadtxt("alarm_100.dat",skiprows=1)
Y = ["PULMEMBOLUS","PAP","KINKEDTUBE","INTUBATION","MINVOLSET","VENTMACH","DISCONNECT","VENTTUBE","VENTLUNG","SHUNT","VENTALV","FIO2","PVSAT","SAO2","INSUFFANESTH","ARTCO2","ANAPHYLAXIS","TPR","CATECHOL","HR","ERRCAUTER","HREKG","HYPOVOLEMIA","LVFAILURE","LVEDVOLUME","CVP","STROKEVOLUME","CO","BP","EXPCO2","ERRLOWOUTPUT","PRESS","HRBP","MINVOL","HISTORY","HRSAT","PCWP"]
arity = data[0]
X = data[1:]

print(Y)
print(arity)
print(X)

def matcher(X, x):
    return [i for i in X if i == x]

def MLE(y):
    Y_prior = 1/arity[y]
    Y_post = []
    for cat in np.nditer((np.unique(arity))):
        col = X[:,y]
        y_given_x = (matcher(col,cat))
        Y_post.append(Y_prior * (len(y_given_x)/len(col)))
    return Y_post


print(MLE(0))