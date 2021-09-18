import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras


def make_seq(inputs):
    cnn_layer = keras.layers.Conv1D(128, 64)(inputs)
    cnn_layer = keras.layers.BatchNormalization()(cnn_layer)
    cnn_layer = keras.layers.Activation('relu')(cnn_layer)
    cnn_layer = keras.layers.Dropout(0.2)(cnn_layer)
    cnn_layer = keras.layers.Conv1D(128, 64)(cnn_layer)
    cnn_layer = keras.layers.Activation('relu')(cnn_layer)

    return cnn_layer


def attention_model():
    # query, value input
    query_input = keras.Input(shape=(600, 6))
    value_input = keras.Input(shape=(600, 6))

    # query, value seq
    query_seq = make_seq(query_input)
    value_seq = make_seq(value_input)

    # attention value
    attention_value = keras.layers.Attention()([query_seq, value_seq])

    # GAP
    query = keras.layers.GlobalAvgPool1D()(query_seq)
    attention = keras.layers.GlobalAvgPool1D()(attention_value)

    # concat
    input_layer = keras.layers.concatenate([query, attention])

    # output
    output = keras.layers.Dense(61, activation='softmax')(input_layer)

    # model
    model = keras.models.Model(inputs=[query_input, value_input], outputs=[output])

    return model


m = attention_model()
print(m.summary())
