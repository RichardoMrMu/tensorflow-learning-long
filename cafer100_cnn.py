# -*- coding:utf-8 -*-
# @Time     : 2019-11-07 22:47
# @Author   : Richardo Mu
# @FILE     : cafer100_cnn.PY
# @Software : PyCharm
import tensorflow as tf
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras import layers,optimizers,datasets,Sequential
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
class CNN(keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = Sequential([
            layers.Conv2D(64,kernel_si)
        ])