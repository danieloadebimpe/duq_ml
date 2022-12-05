from prepare import train_model, train_y
from prepare import xtrain_count, xvalid_count

from prepare import xtrain_tfidf, xvalid_tfidf
from prepare import xtrain_tfidf_ngram, xvalid_tfidf_ngram
from prepare import xtrain_tfidf_ngram_chars, xvalid_tfidf_ngram_chars

from sklearn import naive_bayes


count_vecs = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: ", count_vecs)

tfidf_word = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, Tfidf word level: ", tfidf_word)

tfidf_ngram = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, Tfidf ngram level: ", tfidf_ngram)

tfidf_char = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, Tfidf ngram_char level: ", tfidf_ngram)






