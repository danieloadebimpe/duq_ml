import pandas as pd
import tensorflow as tf
from collections import Counter


yourba_tweets = pd.read_csv('/Users/femiadebimpe/terminal-cpu/data/yo_train.tsv', sep='\t', header=0)
print(yourba_tweets.head())




positive_sentiment = yourba_tweets[yourba_tweets['label'] == 'positive']
#print(positive_sentiment)
positive_sentiment = positive_sentiment.pop('tweet')
#print(positive_sentiment)
print(positive_sentiment.shape[0])

negative_sentiment = yourba_tweets[yourba_tweets['label'] == 'negative']
#print(negative_sentiment)
negative_sentiment = negative_sentiment.pop('tweet')
#print(negative_sentiment)
print(negative_sentiment.shape[0])


ds_positive = tf.data.Dataset.from_tensor_slices(positive_sentiment)
ds_negative = tf.data.Dataset.from_tensor_slices(negative_sentiment)


def preprocess(words):
    words = tf.strings.regex_replace(words, b"<br\\s*/?>", b" ")
    words = tf.strings.regex_replace(words, b"[^a-zA-Z]", b" ")
    words = tf.strings.lower(words)
    words = tf.strings.split(words)
    return words

postitive_tweets = list()
negative_tweets = list()


def get_letter_instances(reviews, ds):
    for element in ds.as_numpy_iterator():
        reviews.append(preprocess(element))

    vocab = Counter()
    i = 0
    for review in reviews:
        vocab.update(list(review.numpy()))
        i += 1
    
    print(vocab.most_common()[:100])
    print(i)


get_letter_instances(postitive_tweets, ds_positive)
get_letter_instances(negative_tweets, ds_negative)


