import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_ratio = 0.7
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

def normalize(list):
    normalized_list = []
    for i in range(len(list)):
        if i == 0:
            normalized_list.append(0)
        else:
            normalized_list.append((list[i]-list[i-1])/list[i-1])
    return normalized_list

# Split into x and y train and test data
x_train = normalize(train)
y_train = np.append(x_train[1:], x_train[-1])

x_test = normalize(test)
y_test = np.append(x_test[1:], x_test[-1])


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
    print("input: " + str(input))
    # print("prev_out: " + str(prev_out))
    # print("prev_cell: " + str(prev_cell))
    layer_op_input = input+prev_out
    # print("Layer op input: "+ str(layer_op_input))

    forget_layer = sigmoid_layer(layer_op_input, forget_weight, forget_bias)
    # print("Forget layer " + str(forget_layer.eval()))

    input_layer = sigmoid_layer(layer_op_input, input_weight, input_bias)
    # print("Input layer " + str(input_layer.eval()))

    candidate_layer = tanh_layer(layer_op_input, candidate_weight, candidate_bias)
    # print("Candidate layer " + str(candidate_layer.eval()))

    output_layer = sigmoid_layer(layer_op_input, output_weight, output_bias)
    # print("Output layer " + str(output_layer.eval()))

    # Values to update for next LSTM cell
    cell_state = tf.add(tf.multiply(forget_layer, prev_cell), tf.multiply(input_layer, candidate_layer))
    print("Cell state: " + str(cell_state.eval()))
    output = tf.multiply(output_layer, tf.tanh(cell_state))
    print("Output: " + str(output.eval()))
    # Update state and previous output
    return {'output': output.eval(), 'cell_state': cell_state.eval()}

# Define loss function
init = tf.global_variables_initializer()
output_list = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        print("Pass #" + str(i))
        cell = lstm_cell(x_train[i], prev_output, prev_cell_state)
        prev_output = cell['output']
        output_list.append(prev_output)
        prev_cell_state = cell['cell_state']

plt.plot(output_list)
plt.plot(x_train[0:100])
plt.plot(np.zeros(100))
plt.show()
