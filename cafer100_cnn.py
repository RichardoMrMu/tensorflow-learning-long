# -*- coding:utf-8 -*-
# @Time     : 2019-11-07 22:47
# @Author   : Richardo Mu
# @FILE     : cafer100_cnn.PY
# @Software : PyCharm
import tensorflow as tf
try:
    import tensorflow.keras as keras
except:
    import tensorflow.python.keras as keras
from keras import layers,optimizers,datasets,Sequential
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
class CNN(keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = Sequential([
            # unit 1
            layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
            layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
            # unit 2
            layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
            layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
            # unit 3
            layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
            layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),
            # unit 4
            layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # unit 5
            layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
            layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
            layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        ])
        self.dense = Sequential([
            layers.Dense(256,activation=tf.nn.relu),
            layers.Dense(128,activation=tf.nn.relu),
            layers.Dense(100,activation=tf.nn.relu)
        ])
    def call(self,inputs):
        output = self.cnn(inputs)
        output = tf.reshape(output,[-1,512])
        output = self.dense(output)
        return output

def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y


# load datasets
batch_size =32
(x,y),(x_test,y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y,axis=1)
y_test = tf.squeeze(y_test,axis=1)
# x_train,x_val = tf.split()
# split data
lenth_x = x.shape[0]
lenth_y = y.shape[0]
alph = 0.7
x_train,x_val = x[:lenth_x*alph-1],x[lenth_x*alph:]
y_train,y_val = y[:lenth_y*alph-1],y[lenth_y*alph:]


train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.map(preprocess).shuffle(x_train.shape[0]).batch(batch_size)
val_db = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_db = val_db.map(preprocess).shuffle(x_val.shape[0]).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).shuffle(x_val.shape[0]).batch(batch_size)

def main():
    # build model

    model = CNN()
    model.build(input_shape=[None, 32, 32, 3])
    optimizer = optimizers.Adam(lr=1e-3)
    for epoch in range(10):
        for step,(x,y) in enumerate(train_db):
            x = tf.reshape(x,[-1,32*32*3])
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y,depth=100)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True))

            grad = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grad,model.trainable_variables))

            if step % 100 == 0:
                print("epoch:{0},step:{1},loss:{2}".format(epoch,step,float(loss)))

            if step%500 == 0:
                total_correct,total_sum = 0,0.
                for step,(x,y) in enumerate(val_db):
                    x = tf.reshape(x,[-1,32*32*3])
                    logits = model(x)
                    # y_onehot = tf.one_hot(y,depth=100)
                    prob = tf.nn.softmax(logits,axis=1)
                    pred = tf.argmax(prob,axis=1)
                    pred = tf.cast(pred,dtype=tf.int32)
                    total_correct += tf.reduce_sum(tf.equal(y_onehot,logits))
                    total_sum += x.shaep[0]
                print("epoch:{0},step:{1},accuracy:{2}".format(epoch,step,total_correct/total_sum))

if __name__ == '__main__':
    main()