# -*- coding: utf-8 -*-
# @Time    : 2019-11-12 9:36
# @Author  : RichardoMu
# @File    : GAN.py
# @Software: PyCharm

import  tensorflow as tf
# from tensorflow.keras import layers
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
from keras import  layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # [b,100] -> [b,3*3*512] -> [b,3,3,512] -> [b,64,64,3]
        self.Dense1 = layers.Dense(3*3*512)
        self.conv1 = layers.Conv2DTranspose(256,kernel_size=3,strides=3,padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(128,kernel_size=5,strides=2,padding='valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(3, kernel_size=4, strides=3, padding='valid')
        # self.bn1 = layers.BatchNormalization()

    def call(self,inputs,training=None):
        # [b,100] - > [b,3*3*512] -> [b,3,3,512] -> [b,64,64,3]
        x = self.Dense1(inputs)
        x = tf.reshape(x,[-1,3,3,512])
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x)))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = tf.tanh(x)

        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b,64,64,3] -> [b,1]
        self.conv1 = layers.Conv2D(64,kernel_size=5,strides=3,padding='valid')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(128,kernel_size=5,strides=3,padding='valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256,kernel_size=5,strides=3,padding='valid')
        self.bn3 = layers.BatchNormalization()
        # [b,h,w,c] -> [b,-1]
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(1)

    def call(self,inputs,training=None):
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs)))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x)))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x)))
        # [b,h,w,c] - > [b,-1]
        x = self.flatten(x)
        # [b,-1] -> [b,1]
        logits = self.dense1(x)
        return logits

def main():
    d = Discriminator()
    g = Generator()

    # d input [b,64,64,3]
    z = tf.random.normal([2,64,64,3])

    # g input shape [b,100]
    x = tf.random.normal([2,100])
    prob = d(z)
    print(prob)

    x_hat = g(x)
    print(x_hat)
if __name__ == '__main__':
    main()