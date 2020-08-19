# %%

import numpy as np
import pandas
import matplotlib.pyplot as plt
from pandas.plotting._matplotlib import scatter_matrix
from sklearn.decomposition import PCA


##accessing columns| ndarray[:,col_i]
##accessing rows   | ndarray[row_i,:]

# %%
def plot(X, Y, target, labels=("Features", "Target")):
    plt.figure(figsize=(10, 10), dpi=300)
    plt.scatter(X, Y, c=target)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


def plot_scatter_matrix(dataframe):
    scatter_matrix(dataframe, figsize=(15, 11))


# %%
def main():
    features = np.genfromtxt('pca_ex.csv', delimiter=',')
    target = np.genfromtxt('classes.txt', delimiter=' ')
    X1 = features[:, 0]
    X2 = features[:, 1]
    X3 = features[:, 2]
    print(features.shape)
    print(len(features[0]))
    print(target.shape)
    print("Rank of feature matrix:", np.linalg.matrix_rank(features))
    plot(X1, X2,target, ["X1", "X2"])
    plot(X1, X3,target, ["X1", "X3"])
    plot(X2, X3,target, ["X2", "X3"])
    pca = PCA(n_components=2).fit(features)
    pca_features = pca.transform(features)
    print("Variance ratio for 1st and 2nd pc: ", pca.explained_variance_ratio_)
    print("Directions", pca.components_)
    plot(pca_features[:, 0], pca_features[:, 1], target, ["FPC", "SPC"])


main()
