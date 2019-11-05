# -*- coding: utf-8 -*-
# @Time    : 2019-11-03 14:47
# @Author  : RichardoMu
# @File    : linear_regression.py
# @Software: PyCharm
# use numpy to get a linear regression
import numpy as np

def compute_error_by_given_data(b,w,points):

    # error
    totalError = 0
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        # compute mean-squared-error
        totalError += (y-(w*x+b))**2

    # avarage loss for each points
    return totalError/float(len(points))

def compute_gradient_and_update(b_current,w_current,points,lr):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
    #     grad_b = 2(w*x+b-y)
        b_gradient += (2/N)*(w_current*x + b_current - y )
    #     grad_w = 2(w*x+b-y)*x
        w_gradient += (2/N)*(w_current*x+b_current-y)*x

#     new_b = b_current-lr *b_gradient
#     new_w = w_current-lr*w_gradient
    new_b = b_current - lr * b_gradient
    new_w = w_current - lr * w_gradient
    return [new_b, new_w]

# set w'->w b'->b and loop
def gradient_descent_runner(points , strating_b,strating_w ,lr,num_iterations):

    b = strating_b
    w = strating_w
#     update for several time
    for i in range(num_iterations):
        b,w = compute_gradient_and_update(b_current=b,w_current=w,points=points,lr = lr)
        print("epoch : {}, b:{},w={},error = {}".format(i,b,w,compute_error_by_given_data(b,w,points)))
    return b,w
def main():
    points = np.genfromtxt("data.csv"
                           ,delimiter=","
                           )
    lr = 1e-4
    initial_b = 0
    initial_w = 0
    num_iteration = 1000
    print("Starting gradient descent at b = {0},w = {1},error = {2}"
          .format(initial_b,initial_w,compute_error_by_given_data(initial_b,initial_w,points)))
    print("running")
    [b,w] = gradient_descent_runner(points,initial_b, initial_w,lr,num_iteration)
    print("after {0} iteration b = {1},w = {2}, error = {3}".format(num_iteration,b,w,
                compute_error_by_given_data(b,w,points)))
    # print(points)
if __name__ == '__main__':
    main()

