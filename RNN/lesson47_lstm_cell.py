# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 10:51
# @Author  : RichardoMu
# @File    : lesson47_lstm_cell.py
# @Software: PyCharm

# 情感分类
import os
import tensorflow as tf
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras import optimizers,layers,datasets,preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
# set seed
tf.random.set_seed(22)
# parameters
# most frequest words
total_words = 10000
batch_size = 128
max_word_len = 80
embedding_len = 100
# load data
(x_train,y_train),(x_test,y_test) = datasets.imdb.load_data(num_words=total_words)

x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=max_word_len)
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=max_word_len)
# db
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.shuffle(1000).batch(batch_size,drop_remainder=True)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.batch(batch_size,drop_remainder=True)

class MyRNN(keras.Model):
    # units = hidden_layer
    def __init__(self,units):
        super(MyRNN, self).__init__()
        self.embedding = layers.Embedding(total_words,embedding_len,input_length=max_word_len)
        self.state0 = [tf.zeros([batch_size,units]),tf.zeros([batch_size,units])]
        self.state1 = [tf.zeros([batch_size,units]),tf.zeros([batch_size,units])]

        self.rnncell0 = layers.LSTMCell(units=units,dropout=0.2)
        self.rnncell1 = layers.LSTMCell(units=units,dropout=0.2)

    #     [b,80,100] -> [b,64] -> [b,1]
        self.outlayer = layers.Dense(1)

    def call(self,inputs,training=None):
        x = self.embedding(inputs)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            # h1 = x*wxh + ht-1 * whh
            # out0:[b,64]
            out0, state0 = self.rnncell0(word,state0,training)
            out1, state1 = self.rnncell1(out0,state1,training)
        #     [b,64] => [b,1]
        x = self.outlayer(out1)
        x = tf.sigmoid(x)
        return x
def main():
    units = 64
    epochs = 10
    model = MyRNN(units)
    model.compile(optimizer=optimizers.Adam,loss=tf.losses.binary_crossentropy(),
                  metrics=['accuracy'])
    model.fit(train_db,epochs=epochs,validation_data=test_db)
    model.evaluate(test_db)
if __name__ == '__main__':
    main()