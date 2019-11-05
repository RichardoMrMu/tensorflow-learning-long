# -*- coding: utf-8 -*-
# @Time    : 2019-11-05 14:45
# @Author  : RichardoMu
# @File    : lesson26_himmelblau.py
# @Software: PyCharm
import  numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
def himmelblau(x):
    return (x[0] ** 2 + x[1] -11)**2 + ( x[0] + x[1] **2 -7 )**2
x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
print("x,y range:",x.shape,y.shape)
X,Y = np.meshgrid(x,y)
print("X,Y maps:",X.shape,Y.shape)
z = himmelblau([X,Y])

fig = plt.figure("himmelblau")
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,z)
ax.view_init(60,-30)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

x = tf.constant([-4.,0.])
for step in range(2000):
    with tf.GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)

    grad = tape.gradient(y,[x])[0]
    x -= 0.01*grad
