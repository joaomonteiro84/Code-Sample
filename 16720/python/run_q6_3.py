import torch
from torchvision.models import resnet50, ResNet50_Weights
import collections
import matplotlib.pyplot as plt
import cv2
import numpy as np

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()

image_net_val = np.load('../data/val_data.npz', allow_pickle=True)

X = image_net_val['data']
y = image_net_val['labels']

#Rottweiler images
#in the validation set, it is label 64  (https://github.com/Evolving-AI-Lab/ppgn/blob/master/misc/map_clsloc.txt) 
# while in the training data the label is 234 (https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)

idx = np.where(y == 64)
idx = idx[0].tolist()

acc = 0
acc_top5 = 0
predictions = []
for i in idx:
    img = X[i,:].reshape((64, 64, 3), order='F').transpose(1,0,2)

    img = np.moveaxis(img, 2, 0)
    t_img = torch.from_numpy(img)

    batch = preprocess(t_img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()

    predictions.append(class_id)    

    acc += (class_id == 234)
    acc_top5 += torch.sum(torch.topk(prediction.flatten(), 5).indices == 234).item()

#top-1 accuracy
print(acc/len(idx))

#top-5 accuracy
print(acc_top5/len(idx))

np.unique(predictions)



frequency = collections.Counter(predictions)
print(dict(frequency))


#load a video
rott_video = cv2.VideoCapture('../data/1028449682-preview.mp4')

n_frames = int(rott_video.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(rott_video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(rott_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = 0
frames = []

while (i < n_frames):
    _, img = rott_video.read()    


    if i in [0, 89, 179, 269]:
        plt.clf()
        imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.savefig("../writeup/figures/q_6_3_rot"+str(i)+".png")


    frames.append(cv2.resize(img, (64, 64)))       
    i += 1

rott_video.release()

acc_video = 0
acc_video_top5 = 0
predictions_video = []
for img in frames:
    img = np.moveaxis(img, 2, 0)
    t_img = torch.from_numpy(img)

    batch = preprocess(t_img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()

    predictions_video.append(class_id)    

    acc_video += (class_id == 234)
    acc_video_top5 += torch.sum(torch.topk(prediction.flatten(), 5).indices == 234).item()


acc_video/len(predictions_video)
acc_video_top5/len(predictions_video)

frequency = collections.Counter(predictions_video)
print(dict(frequency))  