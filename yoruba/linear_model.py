from prepare import train_model, train_y
from prepare import xtrain_count, xvalid_count

from prepare import xtrain_tfidf, xvalid_tfidf
from prepare import xtrain_tfidf_ngram, xvalid_tfidf_ngram
from prepare import xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars

from sklearn import linear_model


linear_count = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, count vectors: ", linear_count)

linear_tfidf = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, tfidf words: ", linear_tfidf)

linear_ngram = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, tfidf ngram: ", linear_ngram)

linear_char = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, tfidf chars : ", linear_char)