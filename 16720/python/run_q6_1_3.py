import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import sklearn.metrics
from nn import *

import torchvision

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_cifar10 = torchvision.datasets.CIFAR10('../data', train=True, download=True, transform=transform)
batches = torch.utils.data.DataLoader(train_cifar10, batch_size=16, shuffle=True)
num_examples = len(train_cifar10)
batch_num = len(batches)


#define model 
model = torch.nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding = 0),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
                            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
                            nn.Flatten(),
                            nn.Linear(400, 120),
                            nn.ReLU(),
                            nn.Linear(120, 84),
                            nn.ReLU(),
                            nn.Linear(84, 10),                            
                            nn.Softmax(1))

#define loss function
loss_fn = nn.CrossEntropyLoss()

#define optmizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


train_loss = []
train_acc = []
for itr in range(600):

    train_epoch_acc = 0   
    train_epoch_loss = 0 
    for xb, yb in batches:
        optimizer.zero_grad()    

        #forward pass
        probs = model(xb.view(-1,3,32, 32))
    
        #loss
        loss = loss_fn(probs, yb) 
        train_epoch_loss += loss.item()  

        #accuracy
        _, pred_values = torch.max(probs.data, 1) 
        

        train_epoch_acc += torch.sum(pred_values == yb).item()

        #backward propagation
        loss.backward()

        #update parameters
        optimizer.step()

    train_acc.append(train_epoch_acc/num_examples)
    train_loss.append(train_epoch_loss/batch_num)

   
    if itr % 2 == 0:
       print("itr: {:02d} \t train loss: {:.2f} \t train acc : {:.2f} ".format(itr,train_loss[-1],train_acc[-1]))
       


torch.save(model.state_dict(), './weights_6_1_3')

# plot loss curves
plt.clf()
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss)-1)
plt.ylim(0, 3)
plt.legend()
plt.grid()
plt.savefig("../writeup/figures/q_6_1_3_loss.png")
 
# plot accuracy curves
plt.clf()
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(0, len(train_acc)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.savefig("../writeup/figures/q_6_1_3_accuracy.png")
