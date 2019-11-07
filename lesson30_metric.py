# -*- coding: utf-8 -*-
# @Time    : 2019-11-06 14:57
# @Author  : RichardoMu
# @File    : lesson30_metric.py
# @Software: PyCharm
import  tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.kears as keras
# from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from keras import datasets,layers,optimizers,Sequential,metrics
def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)

    return x,y

batch_size = 128
(x,y) ,(x_test,y_test) = datasets.mnist.load_data()

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(60000).batch(batch_size).repeat(10)

val_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
val_db = val_db.map(preprocess).batch(batch_size)

model = Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(10)
])
model.build(input_shape=[None,28*28])
model.summary()
optimizer = optimizers.Adam(lr=1e-3)
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

def main():
    for epoch in range(10000):
        for step , (x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y,depth=10)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits))
                loss_meter.update_state(loss)

            grads = tape.gradient(loss,model.trainable_weights)
            optimizer.apply_gradients(zip(grads,model.trainable_weights))

            if step%100 == 0:
                print(step, 'loss:', loss_meter.result().numpy())
                loss_meter.reset_states()


            # evaluate
            if step % 500 == 0:
                total,correct_total = 0.,0.
                for step,(x,y) in enumerate(val_db):
                    logits = model(x)
                    pred = tf.argmax(logits,axis=1)
                    pred = tf.cast(pred,dtype=tf.int32)

                    # correct =
                    correct_total += tf.reduce_sum(tf.equal(y,pred)).numpy()
                    total += x.shape[0]
                    acc_meter.update_state(y,pred)
                print(step, 'Evaluate Acc:', correct_total / total, acc_meter.result().numpy())
