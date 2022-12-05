from prepare import train_model, train_y
from prepare import xtrain_count, xvalid_count

from prepare import xtrain_tfidf, xvalid_tfidf

from sklearn import ensemble


rf_count = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print('RF, Count vectors: ', rf_count)

rf_tfidf = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print('RF, tfidf words: ', rf_tfidf)


