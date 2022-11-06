'''
Daniel Adebimpe 
Moby Dick Reader
Assignment - Process the Moby Dick text file, and find the most common words. 
Determine what percentage they occur, and how many instances there are.

'''

import tensorflow as tf
from collections import Counter

moby = ''

file_in = open('moby_dick.txt', 'rt')
chunk = 100
while True:
    fragment = file_in.read(chunk)
    if not fragment:
        break
    moby += fragment

file_in.close()
#print(len(moby.split()))

moby_word_count = len(moby.split())

dataset = tf.data.TextLineDataset('./moby_dick.txt')

def preprocess(words):
    words = tf.strings.regex_replace(words, b"<br\\s*/?>", b" ")
    words = tf.strings.regex_replace(words, b"[^a-zA-Z]", b" ")
    words = tf.strings.split(words)
    return words

moby_words = list()

for element in dataset.as_numpy_iterator():
    moby_words.append(preprocess(element))

vocab = Counter()
for word in moby_words:
    vocab.update(list(word.numpy()))
    #print(word.numpy())

#print(vocab.most_common()[:3])

most_common1 = vocab.most_common()[0]
print("these are the top 5 words that occur in moby dick, with the respective percentage: ")
#print(type(most_common1))
print(most_common1[0].decode('utf-8'), ": ", most_common1[1], "occurences")
print(round(most_common1[1] / moby_word_count * 100, 2), "percent")

most_common2 = vocab.most_common()[1]

print(most_common2[0].decode('utf-8'), ": ", most_common2[1], "occurences")
print(round(most_common2[1] / moby_word_count * 100, 2), "percent")

most_common3 = vocab.most_common()[2]

print(most_common3[0].decode('utf-8'), ": ", most_common3[1], "occurences")
print(round(most_common3[1] / moby_word_count * 100, 2), "percent")

most_common4 = vocab.most_common()[3]

print(most_common4[0].decode('utf-8'), ": ", most_common4[1], "occurences")
print(round(most_common4[1] / moby_word_count * 100, 2), "percent")

most_common5 = vocab.most_common()[4]

print(most_common5[0].decode('utf-8'), ": ", most_common5[1], "occurences")
print(round(most_common5[1] / moby_word_count * 100, 2), "percent")









