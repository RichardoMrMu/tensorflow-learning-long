# -*- coding: utf-8 -*-
# @Time    : 2019-11-13 11:16
# @Author  : RichardoMu
# @File    : auto_encoder.py
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

class AE(keras.Model):
    def __init__(self):
        super(AE, self).__init__()

        # auto-encoder
        self.encoder = Sequential([
            layers.Dense(256,activation=tf.nn.relu),
            layers.Dense(128,activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])
        self.decoder = Sequential([
            layers.Dense(128,activation=tf.nn.relu),
            layers.Dense(256,activation=tf.nn.relu),
            layers.Dense(784)
        ])

    def call(self,inputs,training=None):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x


model = AE()
model.build(input_shape=(None, 784))
model.summary()
optimizer = optimizers.Adam(lr=lr)

def main():
    # epochs = 10
    # model = AE()
    # model.compile(optimizer=optimizers.Adam(lr=lr),
    #               loss=tf.losses.binary_crossentropy(),
    #               metrics=['accuracy'])
    # model.fit(train_db,epochs,validation_data=test_db)
    # model.evaluate(test_db)
    epochs = 10
    for epoch in range(epochs):
        for step,x in enumerate(train_db):
            # [b,28,28] -> [b,784]
            with tf.GradientTape() as tape:
                x = tf.reshape(x,[-1,784])
                logits = model(x)
                loss = tf.reduce_mean(tf.losses.binary_crossentropy(y_train,logits,from_logits=True))

            grad = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradient(zip(grad,model.trainable_variables))

            if step%100 ==0:
                print(epoch,step,float(loss))

#             evaluation
            x = next(iter(test_db))
            logits = model(x)
            x_hat = tf.sigmoid(x)
            x_hat = tf.reshape(x_hat,[-1,28,28])
            x_hat = x_hat.numpy()*255.
            x_hat = x_hat.dtype(np.unit8)
            save_images(x_hat,"img/rec_epoch_%d.png"%epoch)




if __name__ == '__main__':
    main()
