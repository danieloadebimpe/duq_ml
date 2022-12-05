import matplotlib.pyplot as plt

import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer

#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn import svm

from sklearn.inspection import DecisionBoundaryDisplay


#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.svm import LinearSVC


# we create 40 separable points
# X, y = make_blobs(n_samples=40, centers=2, random_state=6)
# print(X)


df = pd.read_csv("~/terminal-cpu/data/imdb_1k_set.csv") 

count = CountVectorizer()

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

tweets = df['review'].apply(preprocessor)

feature_cols = []
corpus = tweets.to_numpy()
bag = count.fit_transform(corpus)

vec_df = pd.DataFrame(count.vocabulary_, index=[0])
#print(vec_df.head())

vec_df = vec_df.sort_index(ascending=True, axis=1)
print(vec_df.head())
#print(vec_df.columns.tolist())


tweets = tweets.to_numpy()
tweet_vector = []
for tweet in tweets:
    tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
    np.set_printoptions(precision=2)
    # l2 = l2-normalization, which returns a vector of length 1
    vecs = tfidf.fit_transform(count.fit_transform([tweet])).toarray()
    tweet_vector.append(vecs)

df['tweet'] = tweet_vector

tweet_fv = [item for tweet in tweet_vector for item in tweet]
#tweet_fv = np.asarray(tweet_fv, dtype=object)

feature_vecs = np.empty(shape=(8552, 10))

for tweet in tweet_fv:
    fv = np.asarray(tweet)
    np.append(feature_vecs, fv, axis=None)

feature_vecs = np.where(np.isfinite(feature_vecs), feature_vecs, 0)
print(feature_vecs)

#print(feature_vecs[:8522][:, (2,3,4,5)])

X = feature_vecs[:8522][:, (2,3,4,5)]
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)


#X = iris['data'][:,(2,3)]
y = (df['sentiment'] == 'positive').astype(np.float64)
print(y.shape, X.shape)

# svm_clf = Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('linear_svc', LinearSVC(C=1, loss='hinge'))
# ])

#svm_clf.fit(X, y)

#fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=1000)
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