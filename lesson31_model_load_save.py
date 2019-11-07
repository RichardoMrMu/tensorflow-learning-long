# -*- coding: utf-8 -*-
# @Time    : 2019-11-07 20:13
# @Author  : RichardoMu
# @File    : lesson31_model_load_save.py
# @Software: PyCharm
import  tensorflow as tf
try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.kears as keras
# from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from keras import datasets,layers,optimizers,Sequential,metrics
def preprocess(x, y):

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)

    return x,y

batch_size = 128
(x,y) ,(x_test,y_test) = datasets.mnist.load_data()
x_train,x_val = tf.split(x,num_or_size_splits=[50000,10000])
y_train,y_val = tf.split(y,num_or_size_splits=[50000,10000])

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.map(preprocess).shuffle(50000).batch(batch_size)
val_db = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_db = val_db.map(preprocess).shuffle(10000).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(batch_size)

model = Sequential([
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(10)
])
model.build(input_shape=[None,28*28])
model.summary()
optimizer = optimizers.Adam(lr=1e-3)
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()


# save and load weights


# def main():
#     model.compile(optimizer=optimizer,loss=tf.losses.categorical_crossentropy(from_logits=True),
#                   metrics=['accuracy'])
#     model.fit(train_db,epochs=10,validation_data=val_db,validation_freq=1)
#     model.evaluate(test_db)
#     # save parameters of model
#     model.save_weights("weights.ckpt")
#     print("saved model!")
#     del model
#     model = Sequential([
#         layers.Dense(256, activation='relu'),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(32, activation='relu'),
#         layers.Dense(10)
#     ])
#     model.load_weights("weights.ckpt")
#     print("loaded weights!")
#
#     model.compile(optimizer=optimizer,loss=tf.losses.categorical_crossentropy(from_logits=True),
#                   metrics=['accuracy'])
#
#     model.evaluate(test_db)
# if __name__ == '__main__':
#     main()

# save and load model

def main():
    model.compile(optimizer=optimizer,loss=tf.losses.categorical_crossentropy(from_logits=True),
                  metrics=["accuracy"])
    model.fit(train_db,validation_data=val_db,epochs=10,validation_freq=1)
    model.evaluate(test_db)
    model.save("model.mdl")
    print("saved model")
    del model

    model  = tf.keras.models.load_model("model.mdl")
    print("load model")
    model.compile(optimizer=optimizer,loss=tf.losses.categorical_crossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_db,epochs=10,validation_data=val_db,validation_freq=1)
    model.evaluate(test_db)
if __name__ == '__main__':
    main()