import pystan
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = np.loadtxt("exam.dat",skiprows=1,dtype=int)
print(data)

should_use_saved = False
should_save_model = True


# # Get Features
x1 = data[:,0]
x2 = data[:,1]

n = data[:,2]
r = data[:,3]

print(x1, x2, n, r)

# # Fix size of data
N = len(data)

datadkt = { 'N':N, 'x1':x1, 'x2':x2, 'N': N, 'n':n, 'r':r}
if (should_use_saved):
    try:
        print("Opening saved model...")
        sm = pickle.load(open('model.pkl', 'rb'))
    except FileNotFoundError:
        print("No saved model, training new...")
        sm = pystan.StanModel(file='exam.stan')
else:
    print("Ignoring saved model, compiling new ...")
    sm = pystan.StanModel(file='exam.stan')

fit = sm.sampling(data=datadkt,iter=1500, chains=4)

if (should_save_model):
     with open('model.pkl', 'wb') as f:
         pickle.dump(sm, f)

print("THE FIT:")
print(fit)


# ex = fit.extract(dtypes={'y_new':int})
# print()
# all_preds = ex['y_new']
# preds = np.around(np.mean(all_preds, axis = 0))
# print(type(preds))
# print(y_test.shape)
# print(preds.shape)
# print(y_test)
# print(preds)
# #for x in preds:
# #    print(x)

# report = classification_report(y_test, preds, target_names=data.target_names)
# conf_matrix = confusion_matrix(y_test, preds, labels=[0,1])

# print(report)
# print(conf_matrix)

# tn, fp, fn, tp = conf_matrix.ravel()
# print((tn, fp, fn, tp))