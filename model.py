import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_ratio = 0.7
tf.device('cpu')
# Load data into dataframe
# Drop unneeded dataframe columns
df = pd.read_csv('bitcoin.csv').drop(['Volume_(Currency)','Weighted_Price','Volume_(BTC)'], axis=1)
dataset_size = df.shape[0]


# Train and test data
data = df['Close'].as_matrix().astype(float) # Using close data for learning
train_start = 0
train_end = int(np.floor(train_ratio*dataset_size))
test_start = train_end
test_end = dataset_size
train = data[np.arange(train_start, train_end)]
test = data[np.arange(test_start, test_end)]

# Split into x and y train and test data
x_train = train
y_train = np.append(train[1:], train[-1])

x_test = test
y_test = np.append(test[1:], test[-1])

#Tensorflow stuff

# Gate weights
forget_weight = tf.Variable(1.0, name='forget_weight')
input_weight = tf.Variable(1.0, name='input_weight')
candidate_weight = tf.Variable(1.0, name='candidate_weight')
output_weight = tf.Variable(1.0, name='output_weight')

# Gate biases
forget_bias = tf.Variable(0.0, name='forget_bias')
input_bias = tf.Variable(0.0, name='input_bias')
candidate_bias = tf.Variable(0.0, name='candidate_bias')
output_bias = tf.Variable(0.0, name='output_bias')

# Current input, previous output, and previous cell state tensors
curr_input = tf.placeholder(dtype=tf.float32, name='xt')
prev_output = 0.0
prev_cell_state = 0.0
# prev_output = tf.placeholder_with_default(input=0, shape=None, name='ht-1')
# prev_cell_state = tf.placeholder_with_default(input=0, shape=None, name='previous_cell_state')

# Placeholder for real values
target = tf.placeholder(dtype=tf.float32, name='yt')

# LSTM layer values
def sigmoid_layer(input, weight, bias):
    return tf.sigmoid(tf.add(tf.multiply(weight, input), bias))

def tanh_layer(input, weight, bias):
    return tf.tanh(tf.add(tf.multiply(weight, input), bias))

def lstm_cell(input, prev_out, prev_cell):
    layer_op_input = [input, prev_out]
    forget_layer = sigmoid_layer(layer_op_input, forget_weight, forget_bias)
    input_layer = sigmoid_layer(layer_op_input, input_weight, input_bias)
    candidate_layer = tanh_layer(layer_op_input, candidate_weight, candidate_bias)
    output_layer = sigmoid_layer(layer_op_input, output_weight, output_bias)

    # Values to update for next LSTM cell
    cell_state = tf.add(tf.multiply(forget_layer, prev_cell), tf.multiply(input_layer, candidate_layer))
    output = tf.multiply(output_layer, tf.tanh(cell_state))
    # Update state and previous output
    prev_cell_state = cell_state
    prev_output = output
    return output

# Define loss function
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(lstm_cell(curr_input,prev_output,prev_cell_state),
             feed_dict={curr_input: x_train})
