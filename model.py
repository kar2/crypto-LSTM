import data

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



learning_rate = 0.1
num_steps = data.num_steps

 # Get train and test data
x_train = data.get('x_train')
y_train = data.get('y_train')
x_test = data.get('x_test')
y_test = data.get('y_test')

#Tensorflow stuff
weight = tf.Variable(0.5, dtype=tf.float64)
bias = tf.Variable(0.0, dtype=tf.float64)
scale = tf.Variable(5.0, dtype=tf.float64)

curr_input = tf.placeholder(tf.float64, shape=(), name='input')
target = tf.placeholder(tf.float64, shape=(), name='target')

prev_output = 0.0 # Output from last LSTM cell
prev_cell_state = 0.0 # Previous cell state

# LSTM layer values
def sigmoid_layer(input, weight, bias):
    return tf.sigmoid(tf.add(tf.multiply(weight, input), bias))

def tanh_layer(input, weight, bias):
    return tf.tanh(tf.add(tf.multiply(weight, input), bias))

def lstm(input, prev_out, prev_cell):
    forget_layer = sigmoid_layer(prev_out, weight, bias)
    input_layer = sigmoid_layer(input, weight, bias)
    candidate_layer = tanh_layer(input, weight, bias)
    output_layer = sigmoid_layer(input, weight, bias)
    # Values to update for next LSTM cell
    cell_state = tf.add(tf.multiply(forget_layer, prev_cell), tf.multiply(input_layer, candidate_layer))
    output = tf.multiply(output_layer, tf.tanh(cell_state))
    prev_output = output
    prev_cell_state = cell_state
    # Update state and previous output
    return output

prediction = tf.multiply(scale, lstm(curr_input, prev_output, prev_cell_state)) # Keep track of each prediction
cost = tf.square(prediction-target)
opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
minimize = opt.minimize(cost, var_list=[weight, scale])
prediction_list = []
cost_list = []
scale_list = []

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(num_steps):
        print("Step #: " + str(i))
        x = x_train[i]
        y = y_train[i]
        feed = {curr_input:x, target:y}
        curr_pred = sess.run(prediction, feed_dict=feed)
        curr_cost = sess.run(cost, feed_dict=feed)
        opt_op = sess.run(minimize, feed_dict=feed)
        prediction_list.append(curr_pred)
        cost_list.append(curr_cost)
        scale_list.append(scale.eval())
        print("Prediction: " + str(curr_pred))
        print("Cost: " + str(curr_cost))
        print("Weight: " + str(weight.eval()))
        print("Scale: " + str(scale.eval()))
plt.plot(cost_list)
# plt.plot(y_train)
# plt.plot(prediction_list)
plt.show()
