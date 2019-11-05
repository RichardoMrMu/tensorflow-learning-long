# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 14:31
# @Author  : RichardoMu
# @File    : lesson25_gradient_descent.py
# @Software: PyCharm

import tensorflow as tf
w = tf.constant(1.)
x = tf.constant(2.)
y = w+x
# with tf.GradientTape() as tape:
#     tape.watch([w])
#     y2 = x*w
# grad1 = tape.gradient(y,[w])
# print(grad1)
with tf.GradientTape() as tape:
    tape.watch([w])
    y2 = x*w
grad2 = tape.gradient(y2,[w])
print(grad2)