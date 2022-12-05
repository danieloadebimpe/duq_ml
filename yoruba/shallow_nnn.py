import tensorflow as tf
from prepare import train_model, train_y
from prepare import xtrain_count, xvalid_count

from prepare import xtrain_tfidf_ngram, xvalid_tfidf_ngram

from keras import layers, models, optimizers


def create_model_architecture(input_size):
    input_layer = layers.Input((input_size, ), sparse=True)
    hidden_layer = layers.Dense(100, activation='relu')(input_layer)
    output_layer = layers.Dense(1, activation='sigmoid')(hidden_layer)

    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 


classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
nnn = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)