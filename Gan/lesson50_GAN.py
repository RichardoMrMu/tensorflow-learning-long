# -*- coding:utf-8 -*-
# @Time     : 2019-11-11 23:00
# @Author   : Richardo Mu
# @FILE     : lesson50_GAN.PY
# @Software : PyCharm
import tensorflow as tf
try:
    import tensorflow.keras as keras
except:
    import  tensorflow.pyhton.keras as keras
from tensorflow.keras import optimizer,layers

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

