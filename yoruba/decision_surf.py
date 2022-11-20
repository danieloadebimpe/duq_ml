import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('/Users/femiadebimpe/terminal-cpu/data/yo_train.tsv', sep='\t', header=0) 

count = CountVectorizer()

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

tweets = df['tweet'].apply(preprocessor)

feature_cols = []
corpus = tweets.to_numpy()
bag = count.fit_transform(corpus)

vec_df = pd.DataFrame(count.vocabulary_, index=[0])
print(vec_df.head())

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
# for vec in tweet_vector[0]:
#     print(vec)
print(df.head())

X = df[:4276].values
y = df['label'][:4276].values

print(X)
#print(y.tolist())

# for class_value in range(2):

#     row_ix = np.where(y == class_value)

#     plt.scatter(X[row_ix, 'negative'], X[row_ix, 'neutral'])

# plt.show()



# X_train, X_test, y_train, y_test = train_test_split(
#                                                 X, y,
#                                                 test_size=0.25,
#                                                 random_state=0)

# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)