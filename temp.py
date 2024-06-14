import numpy as np
import tensorflow as tf
import json

from torch import autograd as grad
import torch.nn as nn
import torch
import time
#import matplotlib.pyplot as plt


def is_json_empty(filepath):
  """Checks if a JSON file is empty by loading the data.

  Args:
      filepath: Path to the JSON file.

  Returns:
      True if the file is empty, False otherwise.

  Raises:
      JSONDecodeError: If the JSON file is corrupt.
  """
  try:
    with open(filepath, 'r') as f:
      data = json.load(f)
      return not data
  except "JSONDecodeError":
    # Handle corrupt JSON file here (optional)
    return False
  
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

def save_weights(out_file):
    if(not is_json_empty):
        clear_json_file("Weights.json")
    json.dump(weights_1.tolist(), out_file, indent=6)
    json.dump(weights_2.tolist(), out_file, indent=6)
    json.dump(i, out_file, indent=10)
    out_file.close()



start_time =time.time()
L_rate = 0.01
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#assert x_train.shape == (60000, 28, 28)
weights_1 = torch.rand(784,784*5, dtype=float)
weights_2 = torch.rand(784*5,10, dtype=float)


# Remember to vectorize this array ^
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

y_train = torch.tensor(y_train)

for i in range(25):
    x_i = torch.tensor(x_train[i],dtype=float)
    x_i = torch.flatten(x_i)
    """
    Forward propagation
    """
    hidden_layer = torch.matmul(torch.t(x_i), weights_1)
    hidden_layer = nn.functional.relu(hidden_layer)
    y_hat = torch.matmul(torch.t(hidden_layer), weights_2)
    y_hat = nn.functional.softmax(y_hat,dim=0)

    """
    Backwards propagation
    """
    loss = nn.CrossEntropyLoss()
    loss_val = loss(y_hat, y_train[i])
    

    if (i == 24):
        out_file = open("Weights.json", "w")
        save_weights(out_file)

print(time.time() - start_time)
