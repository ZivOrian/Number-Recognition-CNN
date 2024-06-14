import torch
import torch.nn as nn
from nn_ez_pz import BasicNet
from torchvision import datasets, transforms


FILE = "model.pth"
model = torch.load(FILE)
model.eval()
testset = datasets.mnist('data/', transform=None)

#pred = model.forward(x_test[0])
#print(pred==y_test[0])