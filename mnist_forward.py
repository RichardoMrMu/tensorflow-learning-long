# -*- coding: utf-8 -*-
# @Time    : 2019-11-04 21:32
# @Author  : RichardoMu
# @File    : mnist_forward.py
# @Software: PyCharm

import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
from keras import optimizers,layers,datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(x,y):

    x = tf.cast(x,dtype=tf.float32)/255.
    x = tf.reshape(x,[-1,28*28])
    y = tf.cast(y,dtype=tf.int32)
    y = tf.one_hot(y,depth=10)
    return x,y

(x,y ) , (x_test,y_test) = datasets.mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(60000).batch(128).map(preprocess()).repeat(30)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.shuffle(10000).batch(128).map(preprocess())

def main():
    # learning rate
    lr = 1e-3
    # 784 - > 512
    w1,b1 = tf.Variable(tf.random.truncated_normal([784,512])), tf.Variable(tf.random.truncated_normal([512]))
    # 512- > 256
    w2,b2  = tf.Variable(tf.random.truncated_normal([512,256])) , tf.Variable(tf.random.truncated_normal([256]))
    # 256 -> 10
    w3 , b3 = tf.Variable(tf.random.truncated_normal([256,10])) , tf.Variable(tf.random.truncated_normal([10]))


    for step, (x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x@w1 + b1)
            h2= tf.nn.relu(h1@w2 + b2)
            out  = h2@w3 + b3

            # compute loss
            loss = tf.square(y-out)
            # [b , 10 ] -> [b]
            loss = tf.reduce_mean(loss ,axis =1)
            # loss -> scalar
            loss = tf.reduce_mean(loss)


        grads  = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])

        for p,g in zip([w1,b1,w2,b2,w3,b3],grads):
            p.assign_sub(lr*g)

        if step%100 ==0:
            print(step,"loss:",float(loss))

        # evaluate
        if step% 500 == 0:
            total,total_correct = 0.,0
            for step,(x,y) in enumerate(test_db):
                h1 = tf.nn.relu(x @ w1 + b1)
                h2 = tf.nn.relu(h1 @ w2 + b2)
                out = h2 @ w3 + b3

                pred = tf.argmax(out,axis=1)

                y = tf.argmax(y,axis=1)

                correct = tf.equal(pred,y)

                total_correct += tf.reduce_sum(tf.cast(correct,dtype=tf.int32))
                total += x.shape[0]

            print(step,"evaluate acc :",total_correct/total)
if __name__ == '__main__':
    main()


