import data

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Learning rate for training
learning_rate = 0.0001

 # Get train and test data
x_train = data.get('x_train')
y_train = data.get('y_train')
x_test = data.get('x_test')
y_test = data.get('y_test')

# LSTM layer operations
def sigmoid_layer(input, weight, bias):
    return tf.sigmoid(tf.add(tf.multiply(weight, input), bias))

def tanh_layer(input, weight, bias):
    return tf.tanh(tf.add(tf.multiply(weight, input), bias))

# Represent operations in a single LSTM cell
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

# Training variables
weight = tf.Variable(1.0, dtype=tf.float64)
bias = tf.Variable(0.0, dtype=tf.float64)
scale = tf.Variable(4.0, dtype=tf.float64)

# Placeholders for training data
curr_input = tf.placeholder(tf.float64, shape=(), name='input')
target = tf.placeholder(tf.float64, shape=(), name='target')

# LSTM elements
prev_output = 0.0 # Output from last LSTM cell
prev_cell_state = 0.0 # Previous cell state

# Multiply LSTM cell output by scale to better fit target data
prediction = tf.multiply(scale, lstm(curr_input, prev_output, prev_cell_state)) # Keep track of each prediction

# Cost operations
cost = tf.square(prediction-target)
opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
minimize = opt.minimize(cost, var_list=[weight, scale])

# Session to be used for both training and testing
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

def train(num_steps):
    # List of predictions and cost to be used to visually represent data
    prediction_list = []
    cost_list = []
    print("Training.")
    for step in range(num_steps):
        print("Step #: " + str(step))
        # Create feed dict for current step
        feed = {curr_input:x_train[step], target:y_train[step]}
        # Evaluate prediction and cost with data for current step
        curr_pred = sess.run(prediction, feed_dict=feed)
        curr_cost = sess.run(cost, feed_dict=feed)
        # Run cost minimization operation
        opt_op = sess.run(minimize, feed_dict=feed)
        # Append prediction and cost into lists to be returned after training
        prediction_list.append(curr_pred)
        cost_list.append(curr_cost)
        print("Prediction: " + str(curr_pred))
    return {'Predictions': prediction_list, 'Costs': cost_list}

def test(num_steps):
    prediction_list = []
    cost_list = []
    # Resetting LSTM elements from training
    prev_output = 0.0
    prev_cell_state = 0.0
    print("Testing.")
    for step in range(num_steps):
        print("Step #: " + str(step))
        # Create feed dict for current step
        feed = {curr_input:x_test[step], target:y_test[step]}
        # Evaluate prediction and cost with data for current step
        curr_pred = sess.run(prediction, feed_dict=feed)
        curr_cost = sess.run(cost, feed_dict=feed)
        # Append prediction and cost into lists to be returned after training
        prediction_list.append(curr_pred)
        cost_list.append(curr_cost)
        print("Prediction: " + str(curr_pred))
    return {'Predictions': prediction_list, 'Costs': cost_list}
