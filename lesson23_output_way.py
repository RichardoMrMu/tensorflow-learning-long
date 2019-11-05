# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 9:46
# @Author  : RichardoMu
# @File    : lesson23_output_way.py
# @Software: PyCharm

import tensorflow as tf
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras

# a  = tf.linspace(-6.,6.,10)
# print(a)
# a_sig = tf.sigmoid(a)
# print(a_sig)

x = tf.random.normal([1,28,28])*5
print("x:",x)
x_min = tf.reduce_min(x)
x_max = tf.reduce_max(x)
print("x_min:",x_min,"x_max:",x_max)

x = tf.sigmoid(x)
print("x_min:",tf.reduce_min(x),"x_max:",tf.reduce_max(x))

