# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 10:50
# @Author  : RichardoMu
# @File    : lesson47_LSTM_Llayer.py
# @Software: PyCharm
# 情感分类
import os
import tensorflow as tf
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras import optimizers,layers
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

# imbd load data
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)
# limit word lenth as max_word_len
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_word_len)
x_test - tf.keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_word_len)
# x [b,80]
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.shuffle(1000).batch(batch_size=batch_size,drop_remainder=True)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.batch(batch_size,drop_remainder=True)
# units hidden_dim

class MyRNN(tf.keras.Model):
    def __init__(self,units):
        super(MyRNN, self).__init__()
        # state0 = [tf.zeros([batch_size,units])]
        # state1 = [tf.zeros([batch_size,units])]
        #  we need x [b,80,100]
        # Embedding(embeding数量，输出维度即embedding维度，输入单个单词的长度)
        self.embedding = layers.Embedding(total_words,embedding_len,input_length=max_word_len)

        self.rnn = tf.keras.Sequential([
            tf.keras.layers.LSTM(units,dropout=0.2,return_sequence=True,unroll=True),
            layers.LSTM(units,dropout=0.2,unroll=True)
        ])
    #   x [b,80,100] -> [b,units] -> [b,1]
        self.outlayer = layers.Dense(1)

    def call(self,inputs,training=None):
        x = self.embedding(inputs)
        x = self.rnn(x)
        x - self.outlayer(x)
        # 经过sigmoid层输出概率
        x = tf.sigmoid(x)
        return x


def main():
    units = 64
    model = MyRNN(units=units)
    epochs = 10
    model.compile(optimizer=optimizers.Adam(lr=1e-3),loss=tf.losses.binary_crossentropy(),
                  metrics=['accuracy'])
    model.fit(train_db,epochs=epochs,validation_data=test_db)
    model.evaluate(test_db)

if __name__ == '__main__':
    main()

