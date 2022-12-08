import pandas as pd, re
from unidecode import unidecode
import tensorflow as tf
from keras.preprocessing import text, sequence 
from keras import layers, models, optimizers
import sklearn
import sklearn.model_selection

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble


yoruba_tweets = pd.read_csv('/Users/femiadebimpe/terminal-cpu/data/yo_train.tsv', sep='\t', header=0)
print(yoruba_tweets.head())


tweets = yoruba_tweets.pop('tweet')
labels = yoruba_tweets.pop('label')

# word = 'Ọ̀ọ̀ wo'
# print(unidecode(word))

def preprocess(tweet):
    tweet = unidecode(tweet)
    # patterns = [r'[@user]', r'[RT]', 
    # r'[:]', r'[http//|https//]', r'[>>]', r'[#]']

    # for pattern in patterns:
    #     tweet = re.sub(pattern, '', tweet)
        
    #re.findall(r'#([a-z][A-Z])', tweet)
    tweet = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", '', tweet)

    return tweet

tweets = tweets.apply(preprocess)
# for tweet in tweets:
#     print(tweet)

train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(tweets, labels)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# count vectors as features 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\W{1}')
count_vect.fit(tweets)

xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)
#print(xtrain_count)

# tf-idf vectors as features 
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(tweets)

# word level tf-idf
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)
#print('word', xtrain_tfidf)

# ngram level tf-idf
tfidf_vect_gram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_gram.fit(tweets)
xtrain_tfidf_ngram = tfidf_vect_gram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_gram.transform(valid_x)
#print('vect-gram', xtrain_tfidf_ngram)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(tweets)
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


