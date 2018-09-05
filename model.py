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
data = df['Close'].as_matrix() # Using close data for learning
train_start = 0
train_end = int(np.floor(train_ratio*dataset_size))
test_start = train_end
test_end = dataset_size
train = data[np.arange(train_start, train_end)]
test = data[np.arange(test_start, test_end)]

# Split into x and y train and test data
x_train = train
y_train = train[1:]

x_test = test
y_test = test[1:]

print(x_train,y_train)
#Tensorflow stuff
