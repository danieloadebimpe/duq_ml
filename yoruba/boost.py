from prepare import train_model, train_y
from prepare import xtrain_count, xvalid_count

from prepare import xtrain_tfidf, xvalid_tfidf
from prepare import xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars
from prepare import xtrain_tfidf_ngram, xvalid_tfidf_ngram

from sklearn import ensemble

import xgboost 

boost_count = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print("xgb count vectors: ", boost_count)

boost_tfidf_word = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print("xgb tfidf vectors: ", boost_tfidf_word)

boost_tfidf_ngram = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram.tocsc(), train_y, xvalid_tfidf_ngram.tocsc())
print("xgb tfidf ngram: ", boost_tfidf_word)

boost_tfidf_char = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
print("xgb tfidf-char : ", boost_tfidf_char)




