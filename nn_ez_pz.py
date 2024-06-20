import torch
from torch import nn
from torch import optim
from torch import autograd as grad
import tensorflow as tf  # Use torchvision for MNIST dataset
import os
import time



FILE_PATH = "model.pth"

# Define the basic network architecture
class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()  # Proper class inheritance
        self.L1 = nn.Linear(28 * 28, 28**2*5)  # Adjust input size to 28x28
        self.relu = nn.ReLU()
        nn.init.kaiming_uniform_(self.L1.weight)
        self.L2 = nn.Linear(28**2*5, 10)  # Output layer for 10 classes
        self.softmax = nn.Softmax()
        nn.init.xavier_normal_(self.L2.weight)

    def forward(self, x):
        x=self.L1(x)
        x=self.relu(x)
        x=self.L2(x)
        x=self.softmax(x)
        return x



if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Hyperparamaeter zone
    L_rate = 1e-2

    # ---LEARNING LOOP--- { 1 iteration }
    for epoch in range(len(x_train)):
        start_time = time.time()    
        net = BasicNet()
        # Converting the TF input layer to a tensor of a single dimension
        inputX = torch.tensor(x_train[epoch],dtype=torch.float32)
        inputX = torch.flatten(inputX)
        targetY = torch.tensor(y_train[epoch])
        
        #initiating the optimization method and the loss function
        optimizer = optim.SGD(params=net.parameters() ,lr=L_rate, weight_decay=1e-5)# Utilizing momentum for gradient descent
        loss_func = nn.CrossEntropyLoss()

        pred = net.forward(inputX)

        # ---BackPropagation---
        
        loss = loss_func(pred,targetY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.save(net, FILE_PATH)
        if epoch%100==0:
            os.system("cls")
            print(f"example number {epoch}")
