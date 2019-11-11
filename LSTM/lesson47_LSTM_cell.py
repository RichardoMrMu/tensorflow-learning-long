# -*- coding: utf-8 -*-
# @Time    : 2019-11-11 19:50
# @Author  : RichardoMu
# @File    : lesson47_LSTM_cell.py
# @Software: PyCharm

import tensorflow as tf

try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras import optimizers, layers, datasets
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

tf.random.set_seed(22)
np.random.seed(22)
batchsz = 128
# the mose frequest words
total_words = 10000
max_review_len = 80
embedding_len = 100
(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=total_words)
x_train = keras.preprocessing.sequence.pad_sequence(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequence(x_train, maxlen=max_review_len)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.shuffle(1000).batch(batchsz, drop_remainder=True)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.batch(batchsz)


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        self.state0 = [tf.zeros([batchsz, units]), tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units]), tf.zeros([batchsz, units])]

        # transform text to embedding representation
        # [b,80]=> [b,80,100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)

        # [b, 80, 100] , h_dim: 64
        # RNN: cell1 ,cell2, cell3
        # SimpleRNN
        # self.rnn_cell0 = layers.SimpleRNNCell(units, dropout=0.5)
        # self.rnn_cell1 = layers.SimpleRNNCell(units, dropout=0.5)
        self.rnn_cell0 = layers.LSTMCell(units, dropout=0.5)
        self.rnn_cell1 = layers.LSTMCell(units, dropout=0.5)

        self.outlayer = layers.Dense(1)

    def call(self, inputs, training=None):
        """

        :param inputs:
        :param training:
        :return:
        """
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1

        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word,state0,training)
            out1 ,state1 = self.rnn_cell1(out0,state1,training)

        x = self.outlayer(out1)
        prob = tf.sigmoid(x)
        return  prob

def main():
    units = 64
    epochs = 4
    import time
    t0 = time.time()
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(lr=1e-3),
                  loss=tf.losses.binary_crossentropy(),
                  metric=['accuracy'])
    model.fit(train_db,epochs,validation_data=test_db)
    model.evaluate(test_db)
    t1 = time.time()
    print('total time cost :',t1-t0)
if __name__ == '__main__':
    main()