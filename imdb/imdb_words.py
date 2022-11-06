import tensorflow as tf
from collections import Counter
import pandas as pd

# changed this to test set, but originally used the entire set to get analysis for classifier 
imdb_data = pd.read_csv("~/terminal-cpu/data/imdb_1k_set.csv") 
                
#imdb_reviews = imdb_data.pop('review')

#print(imdb_data.head())

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
    i = 0
    for review in reviews:
        vocab.update(list(review.numpy()))
        i += 1
    
    print(vocab.most_common()[:100])
    print(i)


get_letter_instances(postitive_reviews, ds_positive)
get_letter_instances(negative_reviews, ds_negative)

