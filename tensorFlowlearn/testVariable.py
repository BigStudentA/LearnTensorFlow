#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : testVariable.py

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

state = tf.Variable(0,name='counter')  #变量用Variable定义，自定义初值和名字
print(state.name)
one = tf.constant(1)  #定义常量

new_value = tf.add(state,one)  #new_value =state+one
update = tf.assign(state,new_value) #state = new_value

init = tf.global_variables_initializer() #定义变量一定用这个初始化

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
