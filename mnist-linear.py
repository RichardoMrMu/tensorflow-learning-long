# -*- coding: utf-8 -*-
# @Time    : 2019-11-03 16:37
# @Author  : RichardoMu
# @File    : mnist-linear.py
# @Software: PyCharm
# use linear to classify mnist dataset
# process
# 1. compute h1, h2 ,out
# 2. compute loss
# 3. compute gradient and update w1,b1,w2,b2,w3,b3
# 4. loop
import tensorflow as tf
import os
# from tensorflow import keras
# from tensorflow.keras import layers,optimizers,datasets
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras import datasets,layers,optimizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(x,y),(x_val,y_val) = datasets.mnist.load_data()
# 将像素值归一化到0-1范围内
x = tf.convert_to_tensor(x,dtype=tf.float32) /255.
y = tf.convert_to_tensor(y,dtype=tf.float32)
# 将y转化成one-hot，因为mnist有10类，因此depth为10
y = tf.one_hot(y,depth=10)
print(x.shape,y.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
train_dataset = train_dataset.batch(200)

model = keras.Squential([
    layers.Dense(512,activation='relu'),
    layers.Dense(256,activation='relu'),
    layers.Dense(10)
])
optimizer = optimizers.SGD(learning_rate=1e-3)


def train(epoch):
    for step , (x,y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            x = tf.reshape(x,(-1,28*28))
            out = model(x)
            loss = tf.reduce_sum(tf.square(out-y))/x.shape[0]


        grads = tape.gradient(loss,model.trainable_varibles)
        optimizer.apply_gradient(zip(grads,model.trainable_varibles))
        if step%100 == 0:
            print(epoch,step,loss.numpy())

def main():
    for epoch in range(30):
        train(epoch)
if __name__ == '__main__':
    main()