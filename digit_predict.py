import torch as t
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import json


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
    json.dump(["iteration",epoch], out_file, indent=10)
    json.dump(W1.tolist(), out_file, indent=6)
    json.dump(W2.tolist(), out_file, indent=6)
    out_file.close()



t.set_printoptions(10)
# Generate data (train and test set)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

W1 = t.rand([784, 784*20], requires_grad=True,dtype=float)  # Set requires_grad for gradient calculation
t.nn.init.xavier_uniform_(W1)
W2 = t.rand([784*20, 10], requires_grad=True,dtype=float)
t.nn.init.xavier_uniform_(W2)

# /\"for loop" Segment/\
start_time = time.time()
#grad_arr = []

#    -----Init-----
for epoch in range(len(x_train)) :
    L_rate = 0.01
    x_i = t.tensor(x_train[epoch],dtype=float)
    x_i = t.flatten(x_i)
    y = y_test[epoch]
    y = [1 if i == y-1 else 0 for i in range(10)]
    y = t.tensor(y,dtype=float)

    # --Forward Propagation--
    h1_layer = t.matmul(t.t(x_i), W1)
    h1_layer = t.nn.functional.relu(h1_layer)
    out_layer = t.matmul(t.t(h1_layer), W2)
    y_hat = t.nn.functional.softmax(out_layer, dim=0)

    # --Backward Propagtion--
    loss = t.nn.CrossEntropyLoss()  # Instantiate the loss function
    loss_value = loss(y_hat, y)  # Calculate loss with predicted probabilities and labels
    loss_value.backward()

    W1_update = t.sub(W1, W1.grad, alpha=L_rate)
    W2_update = t.sub(W2, W2.grad, alpha=L_rate)
    #grad_arr.append(torch.mean(W1.grad))
    W1.data = W1_update.data
    W2.data = W2_update.data
    
    W1.grad.zero_()
    W2.grad.zero_()
    #print(grad_arr[epoch])
    if(epoch%200 == 0):
        print(time.time() - start_time,'\n\nSaving Weights at iteration ',epoch ,end ='...\n\n ')
        save_weights(out_file=open("Weights.json", "w"))
        print("\ndone\n\n")
#print("\n\n\n", torch.median(torch.tensor(grad_arr)))


# --Test--
print(time.time() - start_time,end='\n\nye\n\n ')
#plt.plot(grad_arr, '.',color="blue")
plt.show()
