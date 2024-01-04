import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import sklearn.metrics
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

batch_size = 16
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

batches_torch = []
for xb, yb in batches:
    xb_torch = torch.from_numpy(xb).to(torch.float32)
    yb_torch = torch.from_numpy(yb).to(torch.float32)
    batches_torch.append((xb_torch,yb_torch))


valid_x = torch.from_numpy(valid_x).to(torch.float32)
test_x = torch.from_numpy(test_x).to(torch.float32)

valid_y = torch.from_numpy(valid_y).to(torch.float32)
test_y = torch.from_numpy(test_y).to(torch.float32)

#define model 
model = torch.nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, stride = 1, padding = 0),
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
                            nn.Linear(84, 36),                            
                            nn.Softmax(1))

#define loss function
loss_fn = nn.CrossEntropyLoss()

#define optmizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []
for itr in range(600):

    train_epoch_acc = 0   
    train_epoch_loss = 0 
    for xb, yb in batches_torch:
        optimizer.zero_grad()    

        #forward pass
        probs = model(xb.view(-1, 1, 32, 32))
    
        #loss
        loss = loss_fn(probs, yb) 
        train_epoch_loss += loss.item()  

        #accuracy
        _, pred_values = torch.max(probs.data, 1) 
        _, true_values = torch.max(yb, 1)

        train_epoch_acc += torch.sum(pred_values == true_values).item()

        #backward propagation
        loss.backward()

        #update parameters
        optimizer.step()

    train_acc.append(train_epoch_acc/train_x.shape[0])
    train_loss.append(train_epoch_loss/batch_num)

    #compute validation loss and accuracy
    val_probs = model(valid_x.view(-1,1,32, 32))
    val_loss = loss_fn(val_probs, valid_y)
    valid_loss.append(val_loss.item())

    _, pred_values = torch.max(val_probs.data, 1) 
    _, true_values = torch.max(valid_y, 1)

    val_acc = torch.sum(pred_values == true_values).item()/valid_y.shape[0]
    valid_acc.append(val_acc)        

    if itr % 2 == 0:
       print("itr: {:02d} \t train loss: {:.2f} \t train acc : {:.2f} \t val loss: {:.2f} \t val acc : {:.2f}".format(itr,train_loss[-1],train_acc[-1], valid_loss[-1], valid_acc[-1]))
       

torch.save(model.state_dict(), './weights_6_1_2')

# plot loss curves
plt.clf()
plt.plot(range(len(train_loss)), train_loss, label="training")
plt.plot(range(len(valid_loss)), valid_loss, label="validation")
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(train_loss)-1)
plt.ylim(2, 4)
plt.legend()
plt.grid()
plt.savefig("../writeup/figures/q_6_1_2_loss.png")
 
# plot accuracy curves
plt.clf()
plt.plot(range(len(train_acc)), train_acc, label="training")
plt.plot(range(len(valid_acc)), valid_acc, label="validation")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(0, len(train_acc)-1)
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.savefig("../writeup/figures/q_6_1_2_accuracy.png")
