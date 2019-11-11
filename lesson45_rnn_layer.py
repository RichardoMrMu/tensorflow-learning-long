# -*- coding:utf-8 -*-
# @Time     : 2019-11-09 21:16
# @Author   : Richardo Mu
# @FILE     : lesson45_rnn_layer.PY
# @Software : PyCharm

import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
# from tensorflow.python import keras
from keras import optimizers,datasets,layers,Sequential
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
batch_size = 128
# the most frequest words
total_words = 10000
max_review_len = 100
embedding_len = 10
(x_train,y_train) , (x_test,y_test) = datasets.imdb.load_data(num_words=total_words)
# x_train [b,80]
# x_test [b,80]
x_train = keras.preprocessing.sequence.pad_sequence(x_train,maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequence(x_test,maxlen=max_review_len)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.shuffle(1000).batch(batch_size=batch_size,drop_remainder=True)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.batch(batch_size,drop_remainder=True)

class MyRNN(keras.model):
    def __init__(self,units):
        super(MyRNN, self).__init__()
        # transform text to embedding representation
        # [b,80] => [b,80,100]
        self.embedding = layers.Embedding(total_words,embedding_len,input_length=max_review_len)
        # [b,80,100] ,n_dim : 64
        self.rnn = Sequential([
            layers.SimpleRNN(units,dropout=0.5,return_sequences=True,unroll=True),
            layers.SimpleRNN(units,dropout=0.5,unroll=True)
        ])
        # fc ,pb,80,100] => [b,64] => [b,1]
        self.outlayer = layers.Dense(1)

    def call(self,inputs,training=None):
        """
        net(x) net(x,training=True):train mode
        net(x,training=False) :test
        :param inputs: [b,80]
        :param training:
        :return:
        """
        # [b,80]
        x = inputs
        # embedding : [b,80] => [b,80,100]
        x = self.embedding(x)
        # rnn cell compute
        # x: [b,80,100] => [b,64]
        x = self.rnn(x)
        # out :[b,64] => [b,1]
        x = self.outlayer(x)
        # p(yis pos|x)
        prob = tf.sigmoid(x)
        return prob
def main():
    units = 64
    epochs = 4
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam(lr=1e-3),
                  loss = tf.losses.categorical_crossentropy(),
                  metrics=['accuracy'])
    model.fit(train_db,epochs=epochs,validation_data=test_db,validation_freq=1)
    model.evaluate(test_db)

if __name__ == '__main__':
    main()