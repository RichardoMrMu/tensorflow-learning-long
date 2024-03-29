# -*- coding: utf-8 -*-
# @Time    : 2019-11-03 20:30
# @Author  : RichardoMu
# @File    : forward.py
# @Software: PyCharm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import tensorflow as tf
from  tensorflow import keras
try:
    import tensorflow.keras as keras

except:
    import tensorflow.python.keras as keras
from keras import datasets

(x,y) ,( x_test,y_test) = datasets.mnist.load_data()

# x [0,255] -> [0,1]
x = tf.convert_to_tensor(x,dtype=tf.float32)/255.
y = tf.convert_to_tensor(y,dtype= tf.int32)
x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)/255.
y_test = tf.convert_to_tensor(y_test,dtype= tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(128)
# train_iter = iter(train_db)
# sample = next(train_iter)
# print("batch:",sample[0].shape,sample[1].shape)

# [b,784] -> [b,256] -> [b,128] - > [b,10]
w1,b1  = tf.Variable(tf.random.truncated_normal([784,256],stddev=0)),tf.Variable(tf.random.truncated_normal(tf.zeros([256])))
w2,b2 = tf.Variable(tf.random.truncated_normal([784,128],stddev=0)),tf.Variable(tf.random.truncated_normal(tf.zeros([128])))
w3,b3 = tf.Variable(tf.random.truncated_normal([784,10],stddev=0)),tf.Variable(tf.random.truncated_normal(tf.zeros([10])))
lr = 1e-3

for epoch in range(100):
    for step ,(x,y) in enumerate(train_db):
        x = tf.reshape(x,[-1,28*28])
        with tf.GradientTape() as tape:
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = tf.nn.relu(h1@w2 + b2)
            out = tf.nn.relu(h2@w3 + b3)

            y_onehot = tf.one_hot(y,depth= 10)

            loss = tf.reduce_mean(tf.square(y_onehot-out))
            # loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        if step%100 == 0:
            print(epoch,step,"loss",float(loss))

    # test/validation
    # [w1,b1,w2,b2,w3,b3]
    total_correct ,total_num = 0,0
    for step ,(x ,y) in enumerate(test_db):
        # reshape x -> [b,28*28]
        x = tf.reshape(x,[-1,28*28])
        # [b,28*28] -> [b,256] -> [b,128] -> [b,10]
        h1 = tf.nn.relu(x@w1+b1)
        h2 = tf.nn.relu(h1@w2 + b2)
        out = h2@w3 + b3

        # out [b,10]
        prob = tf.nn.softmax(out,axis=1)
        pred = tf.argmax(prob,axis=1)
        pred = tf.cast(pred,dtype=tf.int32)

        correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_correct += int(correct)
        total_num += x.shape[0]
    acc = total_correct/total_num
    print("test acc ",acc )
