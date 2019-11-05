# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 14:16
# @Author  : RichardoMu
# @File    : lesson24_loss_function.py
# @Software: PyCharm

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG'] = '2'
y = tf.constant([1,2,3,0,2])
y = tf.one_hot(y,depth=4)
y = tf.cast(y,dtype=tf.float32)

out = tf.random.normal([5,4])
loss1 = tf.reduce_mean(tf.square(y-out))
loss2 = tf.square(tf.norm(y-out))/(5*4)
loss3 = tf.reduce_mean(tf.losses.MSE(y,out))

print("loss1:",loss1,'\n',''"loss2:",loss2,'\n',''"loss3:",loss3)
