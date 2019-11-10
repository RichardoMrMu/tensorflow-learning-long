# -*- coding:utf-8 -*-
# @Time     : 2019-11-09 21:16
# @Author   : Richardo Mu
# @FILE     : lesson45_rnn_layer.PY
# @Software : PyCharm

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import optimizers,datasets,layers
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
batch_size = 128
# the most frequest words
total_words = 10000
max_review_len = 100
embedding_len = 10
