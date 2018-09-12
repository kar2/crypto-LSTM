import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Number of iterations for training
num_steps = 3000

# Train vs test ratio
train_ratio = 0.7

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

# Return processed data based on input
def get(s):
    out = {
        'x_train': train[:num_steps],
        'y_train': train[1:num_steps+1],
        'x_test': test[:num_steps],
        'y_test': test[1:num_steps+1]
    }
    if s in out.keys():
        return out[s]
