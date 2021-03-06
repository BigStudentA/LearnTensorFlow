import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# creat data
x_data = np.random.rand(100).astype(np.float32)
y_data =x_data*0.1+0.3

# create tensorflow structure start#
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer= tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# initilize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
# Instructions for updating:
# Use `tf.global_variables_initializer` instead.
init = tf.global_variables_initializer() #初始化结构
# create tensorflow structure end#

sess = tf.Session() #会话
sess.run(init)#激活init初始化

for step in range(200):
    sess.run(train)
    if step%20 == 0:
        print(step,sess.run(Weights),sess.run(biases))

sess.close()
