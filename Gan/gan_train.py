# -*- coding: utf-8 -*-
# @Time    : 2019-11-12 9:36
# @Author  : RichardoMu
# @File    : gan_train.py
# @Software: PyCharm

import tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras
# from tensorflow.python import keras
from keras import optimizers,datasets,layers,Sequential
import os
from Gan.GAN import Generator,Discriminator
import numpy as np
from scipy.misc import toimage
import glob
from Gan.dataset import make_anime_dataset
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)


def CEloss_ones(d_real_logits):
    #     [b,1] -> [1,1,1,1...]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits,
                                                   labels=tf.ones_like(d_real_logits))
    return tf.reduce_mean(loss)


def CEloss_zeros(d_fake_logits):
    #     [0,0,0...]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,
                                                   labels=tf.zeros_like(d_fake_logits))
    return tf.reduce_mean(loss)


def d_loss_calcu(generator,discriminator,batch_z,batch_x,is_training):
    # treat real img as real
    # treat generated img as fake
    fake_img = generator(batch_z,is_training)
    d_fake_logits = discriminator(fake_img,is_training)
    d_real_logits = discriminator(batch_x,is_training)

    d_loss_real = CEloss_ones(d_real_logits)
    d_loss_fake = CEloss_zeros(d_fake_logits)
    loss = d_loss_fake + d_loss_real
    return loss


def g_loss_calcu(generator,discriminator,batch_z,is_training):
    fake_img = generator(batch_z,is_training)
    d_fake_logits = discriminator(fake_img,is_training)
    loss = CEloss_ones(d_fake_logits)
    return loss


def main():
    tf.random.set_seed(1234)
    np.random.seed(1234)
#     hyper parameters
    z_dim = 100
    epochs = 30000000
    batch_size = 512
    learding_rate = 1e-3
    is_training =True
    img_path = glob.glob(r'c:/img')
    dataset,img_shape,_ = make_anime_dataset(img_path,batch_size)
    print(dataset,img_shape)
    sample = next(iter(dataset))
    print(sample.shape,tf.reduce_max(sample).numpy(),tf.reduce_min(sample).numpy())
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=[None,z_dim])
    generator.summary()

    discriminator = Discriminator()
    discriminator.build(input_shape=[None,64,64,3])
    generator.summary()

    g_optimizer = optimizers.Adam(lr=learding_rate)
    d_optimizer = optimizers.Adam(lr=learding_rate)

    for epoch in range(epochs):
        # fake input g input
        batch_z = tf.random.uniform([batch_size,z_dim],minval=-1.,maxval=1.)
        # real img d input
        batch_x = next(db_iter)
        # train d
        with tf.GradientTape() as tape:
            d_loss = d_loss_calcu(generator,discriminator,batch_z,batch_x,is_training)
        grad = tape.gradient(d_loss,discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grad,discriminator.trainable_variables))

        # train g
        with tf.GradientTape() as tape:
            g_loss = g_loss_calcu(generator,discriminator,batch_z,is_training)
        grad = tape.gradient(g_loss)
if __name__ == '__main__':
    main()