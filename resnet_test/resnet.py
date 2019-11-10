# -*- coding:utf-8 -*-
# @Time     : 2019-11-09 10:47
# @Author   : Richardo Mu
# @FILE     : resnet.PY
# @Software : PyCharm

import tensorflow as tf
# from tensorflow import keras
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
from tensorflow.python.keras import layers,Sequential

class ResnetBlock(layers.Layer):
    def __init__(self,filter_num,stride):
        super(ResnetBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num,kernel_size=[3,3],strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num,kernel_size=[3,3],
                                   strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        self.downsample = Sequential([
            layers.Conv2D(filter_num,kernel_size=[1,1],strides=stride),
            layers.BatchNormalization()
        ]) if stride != 1 else lambda x:x
        self.stride = stride
    def call(self,inputs,training=None):
        residual = self.downsample(inputs)
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        add = layers.add([bn2,residual])
        out = self.relu(add)
        return out
class ResNet(keras.Model):
    def __init__(self,layer_dim,num_class=100):
        super(ResNet, self).__init__()
        self.stem = Sequential([
            layers.Conv2D(64,kernel_size=[3,3],strides=(1,1)),
            layers.BatchNormalization(),
            layers.Activation('rele'),
            layers.MaxPool2D(pool_size=(2,2),strides=(1,1),padding='same')
        ])
        self.layer1 = self._build_resblock(64,layer_dim[0])
        self.layer2 = self._build_resblock(128,layer_dim[1],stride=2)
        self.layer3 = self._build_resblock(256,layer_dim[2],stride=2)
        self.layer4 = self._build_resblock(512,layer_dim[3],stride=2)
    #     output [b,512,h,w]
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_class)
    def call(self,inputs,training=None):
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x
    def _build_resblock(self,filter_num,blocks,stride=1):
        res_blocks = Sequential()
        res_blocks.add(ResnetBlock(filter_num,stride))
        for _ in range(1,blocks):
            res_blocks.add(ResnetBlock(filter_num,stride=1))
        return res_blocks
def resnet18():
    return ResNet([2,2,2,2])
def resnet34():
    return ResNet([3,4,6,3])