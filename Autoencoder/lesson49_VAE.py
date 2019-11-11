# -*- coding: utf-8 -*-
# @Time    : 2019-11-11 19:52
# @Author  : RichardoMu
# @File    : lesson49_VAE.py
# @Software: PyCharm

import tensorflow as tf
try:
    import tensorflow.python,keras as keras
except:
    import tensorflow.keras as keras
from tensorflow.keras import layers,optimizers,datasets,Sequential
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

z_dim = 10


class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = layers.Dense(128)
        # 为什么此处为z-dim
        self.fc2 = layers.Dense(z_dim)
        self.fc3 = layers.Dense(z_dim)
        # decoder
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self,x):
        h = tf.nn.relu(self.fc1(x))
        # get mean
        mu = self.fc2(h)

        # get variance
        log_var = self.fc3(h)

        return h ,mu ,log_var
    def decoder(self,z):
        out = tf.nn.relu(self.fc4(z))
        out = tf.nn.relu(self.fc5(out))
        return out
    def reparameterize(self,mu,log_var):
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var*0.5)
        z = mu + std * eps
        return z
    def call(self, inputs, training=None):
        # [b,784] => [b,z_dim] [b,z_dim]
        mu,log_var = self.encoder(inputs)
        # reparameterize trick
        z = self.reparameterize(mu,log_var)
        x_hat = self.decoder(z)
        return x_hat ,mu , log_var

model = VAE()
model.build(input_shape=(4,784))
optimizer = tf.optimizers.Adam(lr=1e-3)
def main():
    for epoch in range(100):
        for step ,x in enumerate(train_db):
            x= tf.reshape(x,[-1,784])
            with tf.GradientTape() as tape:
                x_rec_logits,mu,log_var = model(x)
                rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x)
                rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]

                k1_div = -0.5*(log_var+1 -mu**2 - tf.exp(log_var))
                k1_div = tf.reduce_sum(k1_div) / x.shape[0]

                loss = rec_loss + 1.* k1_div

            grad = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradient(loss,model.trainable_variabel)

            if step% 100 == 0:
                print("epoch:{0},step:{1},k1_div:{2},rec_loss:{3}".format(
                    epoch,step,float(k1_div),float(loss)
                ))
        # evaluation
        z = tf.random.normal((batchsz,z_dim))
        logits = model.decoder(z)
        x_hat = tf.sigmoid(logits)
        x_hat = tf.reshape(x_hat,[-1,28,28]).numpy()*255.
        x_hat = x_hat.dtype(np.unit8)
        save_images(x_hat,"vae_images/sampled_epoch%d.png"%epoch)

        x = next(iter(test_db))
        x = tf.shape(x,[-1,784])
        x_hat_logits ,_,_ = model(x)
        x_hat = tf.sigmoid(x_hat_logits)
        x_hat = tf.reshape(x_hat,[-128,28]).numpy()*255.
        x_hat = x_hat.dtype(np.unit8)
        save_images(x_hat,"vae_images/rec_epoch%d.png"%epoch)
if __name__ == '__main__':
    main()



