# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 12:25
# @Author  : RichardoMu
# @File    : VAE.py
# @Software: PyCharm

import  os
import  tensorflow as tf
import  numpy as np
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras import optimizers,layers,datasets,preprocessing,Sequential
from    PIL import Image
from    matplotlib import pyplot as plt



tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)

batch_size = 128
h_dim = 20
lr = 1e-3
# load data
(x_train,y_train),(x_test,y_test) = datasets.fashion_mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batch_size*5).batch(batch_size,drop_remainder=True)

test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batch_size,drop_remainder=True)


class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(h_dim)
        self.fc3 = layers.Dense(h_dim)

        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self,x):
        h = tf.nn.relu(self.fc1(x))
        # get mean
        mu = self.fc2(h)
        # get variance 方差
        log_var = self.fc3(h)
        return mu,log_var
    def decoder(self,z):
        output = self.fc4(z)
        output = self.fc5(output)
        return output

    def reparameterize(self,mu,log_var):
        eps = tf.random.normal(log_var.shape)
        log_var = tf.exp(log_var*0.5)
        z = mu + log_var * eps
        return z
    def call(self,inputs,training=None):
        # [b,684] -> [b,z_dim]
        mu , log_var = self.encoder(inputs)
        z = self.reparameterize(mu,log_var)
        x_hat = self.decoder(z)

        return x_hat,mu,log_var


model = VAE()
model.build(input_shape=[4,784])
model.summary()

for epoch in range(1000):
    for step , x in enumerate(train_db):
        x = tf.reshape(x,[-1,784])
        with tf.GradientTape() as tape:
            x_rec_logits , mu ,log_var = model(x)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x,logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]

            loss = rec_loss + 1. * kl_div

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, 'kl div:', float(kl_div), 'rec loss:', float(rec_loss))

        # evaluation
    z = tf.random.normal((batchsz, z_dim))
    logits = model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/sampled_epoch%d.png' % epoch)

    x = next(iter(test_db))
    x = tf.reshape(x, [-1, 784])
    x_hat_logits, _, _ = model(x)
    x_hat = tf.sigmoid(x_hat_logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/rec_epoch%d.png' % epoch)
