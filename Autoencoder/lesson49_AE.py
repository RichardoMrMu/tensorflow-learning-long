# -*- coding: utf-8 -*-
# @Time    : 2019-11-11 19:51
# @Author  : RichardoMu
# @File    : lesson49_AE.py
# @Software: PyCharm
import tensorflow as tf
try:
    import tensorflow.python,keras as keras
except:
    import tensorflow.keras as keras
from keras import layers,optimizers,datasets,Sequential
import os
import numpy as np
tf.random.set_seed(22)
np.random.seed(22)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
assert tf.__version__.startswith('2.')
from matplotlib import pyplot as plt
from PIL import Image

def save_images(imgs,name):
    new_im = Image.new('L',(280,280))

    index = 0
    for i in range(0,280,28):
        for j in range(0,280,28):
            im = imgs[index]
            im = Image.fromarray(im,mode='L')
            new_im.paste(im,(i,j))
            index += 1
    new_im.save(name)
def preprocess(x):
    x = tf.cast(x,dtype=tf.float32)/255.
    return x
h_dim = 20
batchsz = 512
lr = 1e-3

(x_train,y_train) ,(x_test,y_test) = datasets.fashion_mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.map(preprocess).shuffle(batchsz*5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.map(preprocess).batch(batchsz)

class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256,activation=tf.nn.relu),
            layers.Dense(128,activation=tf.nn.relu)
            layers.Dense(h_dim)
        ])
        # Decoders
        self.decoder = Sequential([
            layers.Dense(128,activation=tf.nn.relu),
            layers.Dense(256,activation=tf.nn.relu),
            layers.Dense(784)
        ])
    def call(self, inputs, training=None):
        # [b,784] => [b,10]
        h = self.encoder(inputs)
        # [b,10] => [b,784]
        x_hat = self.decoder(h)

        return x_hat

model = AE()
model.build(input_shape=[None,784])
model.summary()

optimizer = optimizers.Adam(1e-3)

def main():
    for epoch in range(100):
        for step in enumerate(train_db):
             # [b,28,28] => [b,784]
            x = tf.reshape(x,[-1,784])
            with tf.GradientTape() as tape:
                x_rec_logits = model(x)
                rec_loss = tf.losses.binary_crossentropy(x,x_rec_logits,from_logits=True)
                rec_loss = tf.reduce_mean(rec_loss)

            grad = tape.gradient(rec_loss,model.trainable_variables)
            optimizers.apply_gradient(zip(grad,model.trainable.variables))

            if step % 100 ==0:
                print("epoch:{0},step:{1},rec_loss:{2}".format(epoch,step,rec_loss))

#             evaluation
            x = next(iter(test_db))
            logits = model(tf.reshape(x,[-1,784]))
            x_hat = tf.sigmoid(logits)
            x_hat = tf.reshape(x_hat,[-1,28,28])
#             为什么要2倍
#             [b,28,28] => [2b,28,28]
            x_concat = tf.concat([x,x_hat],axis=0)
            x_concat = x_hat
            x_concat = x_concat.numpy()*255.
            x_concat = x_concat.astype(np.unit8)
            save_images(x_concat,'ae_image/rec_epoch_%d.png'%epoch)
if __name__ == '__main__':
    main()