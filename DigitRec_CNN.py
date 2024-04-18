import numpy as np
import tensorflow as tf
import json

from torch import autograd as grad
import torch.nn as nn
#import matplotlib.pyplot as plt



def clear_json_file(filename):
    """
    Clears the content of a JSON file by overwriting with an empty object.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    # Create an empty object (dictionary or list based on original data type)
    empty_data = type(data)()
    with open(filename, 'w') as f:
        json.dump(empty_data, f)


L_rate = 0.01
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#assert x_train.shape == (60000, 28, 28)
weights_1 = np.random.rand(784,784*5)
weights_2 = np.random.rand(784*5,10)


# Remember to vectorize this array ^
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)


for i in range(len(x_train)):
    x_i = np.reshape(x_train[i], 784)  # turns the image into an input layer
    
    """
    Forward propagation
    """
    hidden_layer = np.dot(np.transpose(x_i), weights_1)
    hidden_layer = nn.functional.relu(hidden_layer)
    y_hat = np.dot(np.transpose(hidden_layer), weights_2)
    y_hat = nn.functional.softmax(y_hat,dim=0)

    """
    Backwards propagation
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(y_hat, y_train[i])
    weights_2 = weights
    

    if(i%100 == 0):
        out_file = open("Weights.json", "w")
        clear_json_file("Weights.json")
        json.dump(weights_1,out_file,indet = 6)
        json.dump(weights_2,out_file,indet = 6)
        json.dump({"this is iteration number:",i},out_file,indet = 10)
        out_file.close() 
