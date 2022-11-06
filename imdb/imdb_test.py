import tensorflow as tf
from collections import Counter
import pandas as pd

# testing the classifier based on my analysis - that is, the strongest/ greatest difference of 
# ocurrences of the same word for positive and negative reviews. In my analysis, I evaluated that 
# the highest difference of letters between positives and negative reviews was 'h' which had a 
# .12% difference. 

# Therefore, I used the % of h instances in positive reviews - 2.57% as the decision, or 
# classification algorithm.

imdb_data = pd.read_csv("~/terminal-cpu/data/imdb_1k_set.csv") 

reviews = imdb_data.pop('review')
sentiment_scores = imdb_data.pop('sentiment')

reviews_ts = tf.data.Dataset.from_tensor_slices(reviews)

def preprocess(words):
    words = tf.strings.regex_replace(words, b"<br\\s*/?>", b" ")
    words = tf.strings.regex_replace(words, b"[^a-zA-Z]", b" ")
    words = tf.strings.lower(words)
    words = tf.strings.split(words)
    return words


#print(reviews_ts)

cleaned_reviews = list()

for words in reviews_ts.as_numpy_iterator():
    cleaned_review = preprocess(words)
    cleaned_reviews.append(cleaned_review)
    #print(cleaned_review)

parsed_words = list()
def get_review_stats(cleaned_review):
    for words in cleaned_review:
        byte_word = words.numpy()
        decoded_word = byte_word.decode('utf')
        #print(decoded_word)
        parsed_word = [x for x in decoded_word]
        parsed_words.append(parsed_word)
        #print(parsed_word)
    i = 0
    h_count = 0
    for x in parsed_words:
        #print(x)
        for item in x:
            i += 1
            if item == 'h':
                h_count += 1
            else:
                continue
    # print("number of letters in review: ", i,
    #       "number of h's: ", h_count)
    return i, h_count
   
positive_review_count = 0
negative_review_count = 0

sentiment_review_list = list()

for review in cleaned_reviews:
    letter_count, h_instances = get_review_stats(review)
    h_percentage = (h_instances / letter_count) * 100
    #print(h_percentage)
    if (h_percentage > 2.57):
        positive_review_count += 1
        #print('positive!')
        sentiment_review_list.append('positive')
    else:
        negative_review_count += 1
        #print('negative')
        sentiment_review_list.append('negative')


matches_count = 0
for classified_sentiment, real_sentiment in zip(sentiment_review_list, sentiment_scores):
    if classified_sentiment == real_sentiment:
        matches_count += 1
        print('match')
    else:
        print('no match')
        continue

print(matches_count)
total_reviews = 1000

classifier_accuracy = (matches_count / total_reviews) * 100
print(classifier_accuracy)


        








