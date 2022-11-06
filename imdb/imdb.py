import tensorflow as tf
from collections import Counter
import pandas as pd

# changed this to test set, but originally used the entire set to get analysis for classifier 
imdb_data = pd.read_csv("~/terminal-cpu/data/imdb_train_set.csv") 
                
#imdb_reviews = imdb_data.pop('review')

print(imdb_data.head())

positive_sentiment = imdb_data[imdb_data['sentiment'] == 'positive']
#print(positive_sentiment)
positive_sentiment = positive_sentiment.pop('review')
#print(positive_sentiment)

negative_sentiment = imdb_data[imdb_data['sentiment'] == 'negative']
#print(negative_sentiment)
negative_sentiment = negative_sentiment.pop('review')
#print(negative_sentiment)


ds_positive = tf.data.Dataset.from_tensor_slices(positive_sentiment)
ds_negative = tf.data.Dataset.from_tensor_slices(negative_sentiment)


def preprocess(words):
    words = tf.strings.regex_replace(words, b"<br\\s*/?>", b" ")
    words = tf.strings.regex_replace(words, b"[^a-zA-Z]", b" ")
    words = tf.strings.lower(words)
    words = tf.strings.split(words)
    return words

postitive_reviews = list()
negative_reviews = list()


def get_letter_instances(reviews, ds):
    for element in ds.as_numpy_iterator():
        reviews.append(preprocess(element))

    vocab = Counter()
    for review in reviews:
        vocab.update(list(review.numpy()))

    word_list = list()
    for word in vocab:
        word_list.append(word.decode('utf-8'))

    letter_list = list()
    for letter in word_list:
        letter_list.append([x for x in letter])
        #print([x for x in letter])
    
    i = 0
    letter_vocab = Counter()
    for x in letter_list:
        #print(x)
        for item in x:
            letter_vocab.update(item)
            i += 1

    print(i)
    print(letter_vocab)
      

get_letter_instances(postitive_reviews, ds_positive)
get_letter_instances(negative_reviews, ds_negative)

