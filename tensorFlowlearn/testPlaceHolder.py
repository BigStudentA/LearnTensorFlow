#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : testPlaceHolder.py
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

input1 = tf.placeholder(tf.float32)  #传入值
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)  #乘法运算

with tf.Session() as sess:
    result = sess.run(output,feed_dict={input1:[3.5],input2:[2.1]})  #placeholder传入值实用feed_dict字典传入
    print(result)