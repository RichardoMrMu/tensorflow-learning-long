# -*- coding: utf-8 -*-
# @Time    : 2019-11-07 20:36
# @Author  : RichardoMu
# @File    : cifar10.py.py
# @Software: PyCharm

import  tensorflow as tf

# try:
#     import tensorflow.python.keras as keras
# except:
#     import tensorflow.keras as keras

try:
    import tensorflow.python.keras as keras

except:
    import tensorflow.keras as keras

from keras import optimizers,layers,Model,datasets,Sequential,metrics
def preprocess(x,y):
    # [0,255] -> [-1,1]
    x = 2*tf.cast(x,dtype=tf.float32)/255.-1
    y = tf.cast(y,dtype=tf.int32)
    return x,y

batch_size = 32

(x,y),(x_test,y_test) = datasets.cifar10.load_data()
x_train,x_val = tf.split(x,num_or_size_splits=[50000,10000])
y_train,y_val = tf.split(y,num_or_size_splits=[50000,10000])
y_train,y_val,y_test = tf.squeeze(y_train),tf.squeeze(y_val),tf.squeeze(y_test)
# y_test = tf.squeeze(y_test)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.map(preprocess).shuffle(50000).batch(batch_size)
val_db = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_db  = val_db.map(preprocess).shuffle(10000).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(batch_size)



class MyDense(keras.layer.Layer):
    def __int__(self,input_dim,output_dim):
        super(MyDense, self).__int__()
        self.kernel = self.add_variable('w',[input_dim,output_dim])

    def call(self,inputs,training=None):

        x = inputs @ self.kernel

        return x
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # self.layer1 = tf.nn.relu(MyDense(32*32*3,256))
        # self.layer2 = tf.nn.relu(MyDense(256,128))
        # self.layer3 = tf.nn.relu(MyDense(128,64))
        # self.layer4 = tf.nn.relu(MyDense(64,32))
        # self.layer5 = tf.nn.relu(MyDense(32,10))
        self.model = Sequential([
            tf.nn.relu(MyDense(32 * 32 * 3, 256)),
            tf.nn.relu(MyDense(256, 128)),
            tf.nn.relu(MyDense(128, 64)),
            tf.nn.relu(MyDense(64, 32)),
            tf.nn.relu(MyDense(32, 10))
        ])
    def call(self,x,training=None):
        """
        :param input: [b,32,32,3]
        :param training:
        :return:
        """
        x = tf.reshape(x,[-1,32*32*3])
        output = self.model(x)
        return output
model = MyModel()

def main():
    model.compile(optimizer=optimizer,loss=tf.losses.categorical_crossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_db,epochs=15,validation_data=val_db,validation_freq=1)
    model.evaluate(test_db)
    # save model weights
    model.save_weights('cifar10_weights.mdl')
    print("saved model weights")
    del model
    model = MyModel()
    model.load_weights("cifar10_weights.mdl")
    print("loaded model!")
    model.compile(optimizer=optimizer,loss=tf.losses.categorical_crossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.evaluate(test_db)

if __name__ == '__main__':
    main()

