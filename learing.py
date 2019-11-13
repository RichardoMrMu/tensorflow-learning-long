# -*- coding: utf-8 -*-
# @Time    : 2019-11-12 19:37
# @Author  : RichardoMu
# @File    : learing.py
# @Software: PyCharm

import tensorflow as tf
x = tf.random.normal([4,80,100])
xt0 = x[:,0,:]

cell = tf.keras.layers.SimpleRNNCell(64)

out,ht1 = cell(xt0,[tf.zeros([4,64])])
print(out.shape,ht1)
print(ht1[0])
# print(xt0)