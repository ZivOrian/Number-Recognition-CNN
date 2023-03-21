import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#  ----------------FUNCTIONS-------------------#
Learning_Rate = 0.2

def number_expectation (iter1):
    temp_arr = round(0.1/9, 4)
    arr = [temp_arr]*10
    arr[iter1] = 0.99
    return arr

def sigmoid_output (out_y): 
    ret_y = np.empty(0)
    for out_val in out_y:
        ret_y = np.append( ret_y, ( 1 / (1 + np.exp(-out_val ))))
    return ret_y


def error_function (out_y, target_array):
    error_sum = 0
    for oc,tc in out_y,target_array :
        error_sum += 0.5*(tc-oc)**2
    return error_sum

def backprop_derivative(t1, out_h1, out_y1):
    return (out_y1 - t1) * (out_y1 * (1-out_y1))* out_h1

#___________________INITIALIZATION_____________________#
#      Input Layer --> hidden layer --> output layer

dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()


x_train = tf.keras.utils.normalize(x_train, axis =1)
x_test = tf.keras.utils.normalize(x_test, axis =1)


h1_weights = np.random.random(size = (28**2,28**2)) # c = 784, r = 784
out_weights = np.random.random(size = (28**2,10))# c = 784, r = 10

Bias_CONST = 30
#_______________BUILDING THE NEAURAL NETWORK___________________#

hidden_layer_1 = np.dot(np.reshape(x_train[0],-1), h1_weights)+Bias_CONST#  input layer --> hidden layer

output_layer = tf.keras.utils.normalize(np.dot(hidden_layer_1, out_weights))#  hidden layer --> output layer
output_layer = np.reshape(output_layer,-1).tolist()



output_layer = sigmoid_output(output_layer)#  applying logistic sigmoid to output layer

#_______________BACK-PROPAGATION___________________#
#print(output_layer)
for i in range(len(out_weights)):
    for j in range(10):#  len(out_weights[i]) = 10
        out_weights[i][j] -= Learning_Rate*backprop_derivative(t1= number_expectation(y_train[0])[j], out_h1= hidden_layer_1[i], out_y1= output_layer[j])



#for i in range(len(h1_weights)):
    #for j in range(784):#  len(out_weights[i]) = 784
        #h1_weights[i][j] -= Learning_Rate*backprop_derivative(t1= number_expectation(y_train[0])[j], out_h1= x_train[0][i], out_y1= hidden_layer_1[j])



np.set_printoptions(threshold=sys.maxsize)
print()


#--tasks
#   A.Apply back propagation to hidden layer weights.
#   B.Organize activation function.
#   C.Make the code a for loop for each instance in the training data.