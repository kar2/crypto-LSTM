import tensorflow as tf
import tensorboard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_ratio = 0.7
learning_rate = 0.1
num_iters = 100

global prev_output
global prev_cell_state
global weight
global bias
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
    for i in range(1,len(list)):
        normalized_list.append((list[i]-list[i-1])/list[i])
    return normalized_list

# Split into x and y train and test data
x_train = np.array(normalize(train)[:num_iters])

y_train = np.append(x_train[1:], x_train[-1])[:num_iters]

x_test = np.array(normalize(test)[:num_iters])
y_test = np.append(x_test[1:], x_test[-1])[:num_iters]


#Tensorflow stuff
weight = tf.Variable(0.5, name='weight')
bias = tf.Variable(0.0, name='bias')

curr_input = tf.placeholder(tf.float32, name='xt')
target = tf.placeholder(tf.float32, name='yt')

prev_output = tf.Variable(0.0, name='prev_output') # Output from last LSTM cell
prev_cell_state = tf.Variable(0.0, name='prev_cell_state')

# prediction = lstm(curr_input, prev_output, prev_cell_state) # Keep track of each prediction
#
# # Placeholder for real values
# target = tf.placeholder(dtype=tf.float32, name='yt')
#
# cost = tf.square(prediction-target)
# opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
# minimize = opt.minimize(cost)

# LSTM layer values
def sigmoid_layer(input, weight, bias):
    return tf.sigmoid(tf.add(tf.multiply(weight, input), bias))

def tanh_layer(input, weight, bias):
    return tf.tanh(tf.add(tf.multiply(weight, input), bias))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
# cost = tf.square(prediction-target)
# opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
# minimize = opt.minimize(cost)

    sess.run(init)

    def lstm(input, prev_out, prev_cell):
        #print("input: " + str(input))
        layer_op_input = input + prev_out
        forget_layer = sigmoid_layer(layer_op_input, weight, bias)
        input_layer = sigmoid_layer(layer_op_input, weight, bias)
        candidate_layer = tanh_layer(layer_op_input, weight, bias)
        output_layer = sigmoid_layer(layer_op_input, weight, bias)
        # Values to update for next LSTM cell
        cell_state = tf.add(tf.multiply(forget_layer, prev_cell), tf.multiply(input_layer, candidate_layer))
        output = tf.multiply(output_layer, tf.tanh(cell_state))
        print("Output shape: " + str(output.shape))
        #print("Prediction")
        # prediction.assign(output)
        sess.run(prev_output.assign(output))
        sess.run(prev_cell_state.assign(cell_state))
        # Update state and previous output
        return output

    prediction = lstm(curr_input, prev_output, prev_cell_state) # Keep track of each prediction

    cost = tf.square(prediction-target)
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    minimize = opt.minimize(cost)

# Define loss function
# loss = tf.square()
    # init = tf.global_variables_initializer()
    # cost = tf.square(prediction-target)
    # opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    # minimize = opt.minimize(cost)
    # cost = tf.square(true_val-cell['output'])
    # feed = {curr_input: x_train, target: y_train}
    for i in range(num_iters):
        feed = {curr_input: x_train[i], target: y_train[i]}
        print((y_train[i]-x_train[i])**2)
    result = sess.run([cost,minimize], feed_dict=feed)
    print(result)

# plt.figure(0)
# plt.plot(y_train[:num_iters])
# plt.plot(output_list[:num_iters])
# plt.show()
#
# plt.figure(1)
# plt.plot(loss_list)
# plt.savefig('loss.png')
