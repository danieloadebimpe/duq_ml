from prepare import train_model, train_y
from prepare import xtrain_count, xvalid_count

from prepare import xtrain_tfidf, xvalid_tfidf
from prepare import xtrain_tfidf_ngram, xvalid_tfidf_ngram
from prepare import xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars


from sklearn import ensemble


rf_count = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print('RF, Count vectors: ', rf_count)

rf_tfidf = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print('RF, tfidf words: ', rf_tfidf)

rf_tfidf_ngram = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print('RF, tfidf ngram: ', rf_tfidf_ngram)

rf_tfidf_chars = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print('RF, tfidf chars: ', rf_tfidf_chars)



