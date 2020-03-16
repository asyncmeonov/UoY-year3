import pystan
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():
    data = load_breast_cancer()
    #print(data)
    print(len(data.feature_names))
    print(data.target_names)
    # print(data.feature_names)
    # print(data.target_names)
    # print(data.target[0])
    # print(data.target[[0,500]])
    # print(data.target)
    # print(data.data)
    # print(type(data.data))
    # print(data.data[...,3])

    should_use_saved = False
    should_save_model = not should_use_saved


    # Get Features
    x = data.data

    y = data.target

    (X_train, X_test, y_train, y_test ) = train_test_split(x,y,test_size=0.2)

    # Fix size of data
    N = len(X_train.data)
    N_new = len(X_test.data)

    datadkt = { 'N':N, 'x':X_train, 'y':y_train, 'N_new': N_new, 'x_new':X_test}
    if (should_use_saved):
        try:
            print("Opening saved model...")
            sm = pickle.load(open('model.pkl', 'rb'))
        except FileNotFoundError:
            print("No saved model, training new...")
            sm = pystan.StanModel(file='breast_cancer.stan')
    else:
        print("Disregarding saved mode, training new...")
        sm = pystan.StanModel(file='breast_cancer.stan')

    fit = sm.sampling(data=datadkt,iter=1500, chains=4)

    if (should_save_model):
        with open('model.pkl', 'wb') as f:
            pickle.dump(sm, f)

    print("THE FIT:")
    print(fit)


    ex = fit.extract(dtypes={'y_new':int})
    print()
    all_preds = ex['y_new']
    preds = np.around(np.mean(all_preds, axis = 0))
    print(type(preds))
    print(y_test.shape)
    print(preds.shape)
    print(y_test)
    print(preds)
    #for x in preds:
    #    print(x)

    report = classification_report(y_test, preds, target_names=data.target_names)
    conf_matrix = confusion_matrix(y_test, preds, labels=[0,1])

    print(report)
    print(conf_matrix)

    tn, fp, fn, tp = conf_matrix.ravel()
    print((tn, fp, fn, tp))

    # ex = fit.extract(permuted = True)
    # print()
    # preds = np.mean(ex['y_new'], axis = 0)
    # print(type(preds))
    # print(preds.shape)
    # print(preds == y_test)

if __name__ == '__main__':
    main()