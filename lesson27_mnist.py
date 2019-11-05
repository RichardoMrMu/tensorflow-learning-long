# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 20:22
# @Author  : RichardoMu
# @File    : lesson27_mnist.py
# @Software: PyCharm

import  tensorflow as tf
try :
    import tensorflow.python.keras as keras
except:
    import  tensorflow.keras as keras
from keras import layers,optimizers,datasets,Sequential

import  os

os.environ["TF_CPP_MIN_LOG"] ='2'
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.float32)
    return x,y
(x,y),(x_test,y_test )  = datasets.mnist.load_data()
print(x.shape,y.shape)

batchsz = 256

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess()).shuffle(10000).batch(batchsz)

test_db = tf.data.Dataset.from_tensor_slices((x,y))
test_db = test_db.map(preprocess()).batch(batchsz)

model = Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(10)
])
model.build(input_shape=[None,28*28])
model.summary()
optimizer = optimizers.SGD(learning_rate=1e-3)

def main():
    for epoch in range(1000):
        for step ,(x,y) in enumerate(train_db):
            # [b,28,28] ->> [b,756]
            x = tf.reshape(x,[-1,28*28])
            with tf.GradientTape() as tape:

                output = model(x)
                y_onehot = tf.one_hot(y,depth=10)
                # mean square error
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot,output))
                # crossentropy
                loss_ce = tf.losses.categorical_crossentropy(y_onehot,output,from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)
            grad = tape.gradient(loss_ce,model.trainable_variables)
            optimizer.apply_gradients(zip(grad,model.trainable_varibles))

            if step%100 == 0:
                print("epoch:",epoch,"step:",step,"loss:",float(loss_ce),float(loss_mse))

        # test
        #
        total ,total_correct = 0.,0
        for x,y in test_db:
            x = tf.reshape(x,[-1,28*28])

            logits  = model(x)
            # logits -> [b,10]
            pro = tf.nn.softmax(logits,axis=1)
            # pro [b,10] ->  [b]
            pred = tf.argmax(pro,axis=1)
            pred = tf.cast(pred,dtype=tf.int32)
            # pre [b]  y [b]
            total_correct += tf.equal(pred,y)
            total += y.shape[0]
        acc = total_correct/total
        print("epoch:",epoch,"test_acc:",acc)

if __name__ == '__main__':
    main()


