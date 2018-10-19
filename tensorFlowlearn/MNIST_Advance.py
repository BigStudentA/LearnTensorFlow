#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : MNIST_Advance.py
import input_data
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.InteractiveSession() #启动计算图
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))  #权重
b = tf.Variable(tf.zeros([10]))  #偏置bias

sess.run(tf.global_variables_initializer())  #变量初始化

y = tf.nn.softmax(tf.matmul(x,W) + b) #softmax模型

cross_entropy = -tf.reduce_sum(y_*tf.log(y))  #代价函数

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(10000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #评估模型 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
