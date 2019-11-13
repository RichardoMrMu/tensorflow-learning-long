# -*- coding: utf-8 -*-
# @Time    : 2019-11-12 9:36
# @Author  : RichardoMu
# @File    : dataset.py
# @Software: PyCharm
#
import multiprocessing
import tensorflow as tf

def make_anime_dataset(img_paths,batch_size,resize=64,drop_remainder=True,shuffle=True,repeat=1):
    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img,[resize,resize])
        img = tf.clip_by_value(img,0,255)
        img = img / 127.5 - 1
        return  img
    dataset = disk_img_batch_dataset(


    )

def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    of
    pass
def memory_data_batch_dataset():
    pass
def disk_image_batch_dataset():
    pass

