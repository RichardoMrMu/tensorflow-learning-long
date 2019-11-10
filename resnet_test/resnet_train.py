# -*- coding:utf-8 -*-
# @Time     : 2019-11-09 10:47
# @Author   : Richardo Mu
# @FILE     : resnet_train.PY
# @Software : PyCharm

# x选择cifar100
import tensorflow as tf
from tensorflow.python.keras import optimizers,datasets
from tensorflow.python import keras
from resnet_test.resnet import resnet18
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置随机数，保证实验可重复
tf.random.set_random_seed(111)
def preprocess(x,y):
    """
    :param x: data img
    :param y: label
    :return:
    """
    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)
    return  x,y
# batchsize
batch_size = 128
(x,y),(x_test,y_test) = datasets.cifar100.load_data()

y,y_test = tf.squeeze(y,axis=1), tf.squeeze(y_test,axis=1)
# 将x ，y 分为 x_train 和x_val
x_train,x_val = tf.split(x,num_or_size_splits=[40000,10000])
y_train ,y_val = tf.split(y,num_or_size_splits=[40000,10000])
# 读取数据
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.map(preprocess).shuffle(40000).batch(batch_size)
val_db = tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_db = val_db.map(preprocess).shuffle(40000).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(batch_size)

optimizer = optimizers.Adam(lr=1e-3)
model = resnet18()
def validation(epoch,steps):
    correct, total_sum = 0, 0.
    for step, (x, y) in enumerate(val_db):
        logits = model(x)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        correct += tf.reduce_sum(tf.cast(tf.equal(y, pred), dtype=tf.int32))
        total_sum += x.shape[0]
    acc = correct / total_sum
    print("epoch:{0}    step:{1}    accuracy:{2}".format(epoch, steps,acc))
    return acc
def test():
    correct, total_sum = 0, 0.
    for step, (x, y) in enumerate(test_db):
        logits = model(x)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        correct += tf.reduce_sum(tf.cast(tf.equal(y, pred), dtype=tf.int32))
        total_sum += x.shape[0]
    acc = correct / total_sum
    print("test accuracy:{0}".format(acc))
    return acc

def main():
    best_epoch, best_val_acc = 0, 0.
    for epoch in range(10):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x)
                y_onehot = tf.one_hot(y, depth=100)
                loss = tf.losses.categorical_crossentropy(y_onehot, logits)
                loss = tf.reduce_mean(loss)
            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradient(zip(grad, model.trainable_variables))

            # validation
            if step %100 ==0:
                val_acc = validation(epoch,step)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    model.save_weights("best_weights.mdl")
                    print("saved model")

            #             test
            if step % 500 ==0:
                _ = test(epoch,step)
    print("best-epoch:{0}   best_val_accuracy:{1}".format(best_epoch,best_val_acc))
    model.load_weights("best_weights.mdl")
    print("loaded from weights")
    test_acc = test()
    print("test accuravy :{0}".format(test_acc))
if __name__ == '__main__':
    main()
