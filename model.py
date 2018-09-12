import tensorflow as tf
import tensorboard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_ratio = 0.7
learning_rate = 0.5
num_iters = 100
keep_prob = 0.8 # Keep probability for forget layer

# Load data into dataframe
# Drop unneeded dataframe columns
df = pd.read_csv('bitcoin.csv').drop(['Volume_(Currency)','Weighted_Price','Volume_(BTC)'], axis=1)
dataset_size = df.shape[0]

def normalize(list):
    normalized_list = []
    for i in range(1,len(list)):
        normalized_list.append((list[i]-list[i-1])/list[i])
    return normalized_list


# Train and test data
data = df['Close'].as_matrix().astype(float) # Using close data for learning
train_start = 0
train_end = int(np.floor(train_ratio*dataset_size))
test_start = train_end
test_end = dataset_size
train = normalize(data[np.arange(train_start, train_end)])
test = normalize(data[np.arange(test_start, test_end)])


# Split into x and y train and test data
x_train = (train[:num_iters])
y_train = (train[1:num_iters+1])

x_test = (test[:num_iters])
y_test = (test[1:num_iters+1])


#Tensorflow stuff
weight = tf.Variable(0.5, dtype=tf.float64)
keep_prob_weight = tf.Variable(keep_prob, dtype=tf.float64)
bias = tf.Variable(0.0, dtype=tf.float64)

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

prediction = lstm(curr_input, prev_output, prev_cell_state) # Keep track of each prediction
cost = tf.square(prediction-target)
opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
minimize = opt.minimize(cost, var_list=[weight, bias, keep_prob_weight])
prediction_list = []
cost_list = []
feed = {curr_input:x_train, target:y_train}

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(num_iters):
        print("Step #: " + str(i))
        x = x_train[i]
        y = y_train[i]
        step = sess.run([prediction, cost], feed_dict={curr_input:x, target:y})
        prediction_list.append(prediction.eval())
        cost_list.append(cost.eval())
        print(step)

plt.plot(prediction_list)
# plt.plot(prediction_list)
plt.show()
