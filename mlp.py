# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 9:18
# @Author  : RichardoMu
# @File    : mlp.py
# @Software: PyCharm

import tensorflow as tf
try :
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
from keras import  layers,Sequential

x = tf.random.normal([2,3])

model = Sequential(
    layers.Dense(2,activation='relu'),
    layers.Dense(2,activation='relu'),
    layers.Dense(2)
)

model.build(input_shape=[None,3])

model.summary()
"""
打印结果
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                multiple                  8
_________________________________________________________________
dense_1 (Dense)              multiple                  6
_________________________________________________________________
dense_2 (Dense)              multiple                  6
=================================================================
Total params: 20
Trainable params: 20
Non-trainable params: 0
"""

for p in model.trainable_variables:
    print(p.name,p.shape)
"""
打印结果
dense/kernel:0 (3, 2)
dense/bias:0 (2,)
dense_1/kernel:0 (2, 2)
dense_1/bias:0 (2,)
dense_2/kernel:0 (2, 2)
dense_2/bias:0 (2,)
"""