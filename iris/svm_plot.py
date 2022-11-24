import matplotlib.pyplot as plt

import numpy as np
from sklearn import svm, datasets
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# we create 40 separable points
# X, y = make_blobs(n_samples=40, centers=2, random_state=6)
# print(X)

iris = datasets.load_iris()
X = iris['data'][:,(2,3)]
y = (iris['target'] == 2).astype(np.float64)

svm_clf = Pipeline([
                ('scaler', StandardScaler()),
                ('linear_svc', LinearSVC(C=1, loss='hinge'))
])

# fit the model, don't regularize for illustration purposes
# clf = svm.SVC(kernel="linear", C=1000)
clf = svm.SVC(kernel="linear", C=1000)
# clf.fit(X, y)

clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()