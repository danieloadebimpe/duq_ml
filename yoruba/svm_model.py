import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


from prepare import train_model, train_y
from prepare import xtrain_count, xvalid_count

from prepare import xtrain_tfidf, xvalid_tfidf
from prepare import xtrain_tfidf_ngram, xvalid_tfidf_ngram
from prepare import xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.svm import SVC


def plot_decision_regions(X, y, classifier, test_idx=None,
                                resolution=0.2):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.xlim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    #highlight test examples
    if test_idx:
        #plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')

svm_model = train_model(svm.SVC(kernel='poly', degree=3, coef0=1, C=15), xtrain_count, train_y, xvalid_count)
print("svm count vects: ", svm_model)

svm_model = train_model(svm.SVC(kernel='poly', degree=3, coef0=1, C=15), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("svm n-gram vectors: ", svm_model)

svm_model = train_model(svm.SVC(kernel='poly', degree=3, coef0=1, C=15), xtrain_tfidf, train_y, xvalid_tfidf)
print("svm n-gram word: ", svm_model)

svm_model = train_model(svm.SVC(kernel='poly', degree=3, coef0=1, C=15), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("svm n-gram char: ", svm_model)

# svm = SVC(kernel='linear', C=1.0, random_state=1)
# svm.fit(xtrain_tfidf_ngram, train_y)

# print(xtrain_tfidf_ngram)

# plot_decision_regions(xtrain_tfidf_ngram, train_y, classifier=svm, test_idx=range(105, 150))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.tight_layout()
# plt.show()


