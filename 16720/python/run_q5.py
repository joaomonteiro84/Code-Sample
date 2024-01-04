import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string
from util import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(train_x.shape[1], 32, params, "layer1")
initialize_weights(32, 32, params, "layer2")
initialize_weights(32, 32, params, "layer3")
initialize_weights(32, train_x.shape[1], params, "output")

for l in ['layer1', 'layer2', 'layer3', 'output']:
    params['MW_'+l] = np.zeros(params['W'+l].shape)
    params['Mb_'+l] = np.zeros(params['b'+l].shape)

# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # forward        
        h1 = forward(xb, params,'layer1', relu)
        h2 = forward(h1, params,'layer2', relu)
        h3 = forward(h2, params,'layer3', relu)
        output_pred = forward(h3,params,'output', sigmoid)        

        # loss        
        loss = np.sum((xb-output_pred)**2)             
        total_loss += loss       

        # backward
        delta1 = output_pred - xb
        delta2 = backwards(delta1,params,'output',linear_deriv)        
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        _ = backwards(delta4,params,'layer1',relu_deriv)

        # apply gradient   
        for l in ['layer1', 'layer2', 'layer3', 'output']:
            params['MW_'+l] = 0.9*params['MW_'+l] - learning_rate*params['grad_W'+l]
            params['W'+l] += params['MW_'+l]
            params['Mb_'+l] = 0.9*params['Mb_'+l] - learning_rate*params['grad_b'+l]
            params['b'+l] += params['Mb_'+l]      
    
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()
plt.savefig("../writeup/figures/q_5_2_loss.png")

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
h1 = forward(visualize_x, params,'layer1', relu)
h2 = forward(h1, params,'layer2', relu)
h3 = forward(h2, params,'layer3', relu)
reconstructed_x = forward(h3,params,'output', sigmoid)    


# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()
plt.savefig("../writeup/figures/q_5_3_1_comparison.png")


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR

h1 = forward(valid_x, params,'layer1', relu)
h2 = forward(h1, params,'layer2', relu)
h3 = forward(h2, params,'layer3', relu)
reconstructed_x = forward(h3,params,'output', sigmoid) 

avg_psnr = 0
for i in range(reconstructed_x.shape[0]):
    avg_psnr += peak_signal_noise_ratio(valid_x[i,:], reconstructed_x[i,:])

avg_psnr /= reconstructed_x.shape[0]

print(avg_psnr)