import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from mpl_toolkits.axes_grid1 import ImageGrid
from nn import *


import torchvision
from sklearn.preprocessing import OneHotEncoder

#############
#train data#
###########

#get train labels
with open('../../hw1/data/train_labels.txt') as f:
    train_labels = f.readlines()

train_y = []
for l in train_labels:
    train_y.append(int(l.split('\n')[0]))

train_y = np.vstack(train_y)

onehot_encoder = OneHotEncoder(sparse=False)
train_y = onehot_encoder.fit_transform(train_y)

#get train images
with open('../../hw1/data/train_files.txt') as f:
    train_files = f.readlines()

resize_transform = torchvision.transforms.Resize((32,32))

train_images =[]
for img in train_files:  
    t_img = image.imread('../../hw1/data/'+img.split('\n')[0]) /255.   
    t_img = t_img.reshape(3, t_img.shape[0], t_img.shape[1])  

    t_img  = torch.from_numpy(t_img)

    t_img_res = resize_transform(t_img)
    train_images.append(torch.flatten(t_img_res).detach().numpy())
    
train_x = np.vstack(train_images)

batch_size = 32
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

batches_torch = []
for xb, yb in batches:
    xb_torch = torch.from_numpy(xb).to(torch.float32)
    yb_torch = torch.from_numpy(yb).to(torch.float32)
    batches_torch.append((xb_torch,yb_torch))

############
#test data#
##########

#get test labels
with open('../../hw1/data/test_labels.txt') as f:
    test_labels = f.readlines()

test_y = []
for l in test_labels:
    test_y.append(int(l.split('\n')[0]))

test_y = np.vstack(test_y)
test_y =  torch.from_numpy(test_y)
#test_y = onehot_encoder.fit_transform(test_y)

#get test images
with open('../../hw1/data/test_files.txt') as f:
    test_files = f.readlines()

test_images =[]
for img in test_files:  
    t_img = image.imread('../../hw1/data/'+img.split('\n')[0]) /255. 

    if t_img.shape[2] == 4:
        t_img = t_img[:,:,0:3]

    t_img = t_img.reshape(3, t_img.shape[0], t_img.shape[1])  

    t_img  = torch.from_numpy(t_img)

    t_img_res = resize_transform(t_img)
    test_images.append(torch.flatten(t_img_res).detach().numpy())

test_images = np.vstack(test_images)

test_x_torch = torch.from_numpy(test_images).to(torch.float32)


#define model 
model = torch.nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1, padding = 0),
                            nn.BatchNorm2d(6),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
                            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0),
                            nn.BatchNorm2d(16),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0),
                            nn.Flatten(),
                            nn.Linear(400, 120),
                            nn.ReLU(),
                            nn.Linear(120, 84),
                            nn.ReLU(),
                            nn.Linear(84, 8),                            
                            nn.Softmax(1))


#define loss function
loss_fn = nn.CrossEntropyLoss()

#define optmizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_loss = []
train_acc = []
test_acc=[]
for itr in range(1000):

    train_epoch_acc = 0   
    train_epoch_loss = 0 
    for xb, yb in batches_torch:
        optimizer.zero_grad()    

        #forward pass
        probs = model(xb.view(-1,3,32, 32))
    
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

    probs = model(test_x_torch.view(-1,3,32, 32))
    _, pred_values = torch.max(probs.data, 1) 

    t_acc = torch.sum(test_y[:,0] == pred_values)/400
    
    test_acc.append(t_acc.item())    

    if itr % 2 == 0:
       print("itr: {:02d} \t train loss: {:.2f} \t train acc : {:.2f} \t test acc : {:.2f}".format(itr,train_loss[-1],train_acc[-1],test_acc[-1]))
       


test_acc = np.vstack(test_acc)
train_acc = np.vstack(train_acc)

np.max(np.vstack(test_acc))

train_acc[np.argmax(test_acc)]


torch.save(model.state_dict(), './weights_6_1_4')









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

