import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def cluster_letters(bboxes):
    idx_need_clusters = list(range(len(bboxes)))
   
    clusterId = 0
    clustering = {}
    row_avg = []
    n_need_clusters = len(idx_need_clusters)

    while n_need_clusters > 0:    
        leader_idx = idx_need_clusters[0]
        clustering[clusterId] = [leader_idx]
        row_avg.append(centroids[leader_idx][1])
        
        idx_need_clusters.remove(leader_idx)
        minr_leader, minc_leader, maxr_leader, maxc_leader = bboxes[leader_idx]
        
        idx_copy = idx_need_clusters.copy()
        
        for i in idx_copy:             
            
            minr_cand, minc_cand, maxr_cand, maxc_cand = bboxes[i]
            
            #cluster if candidate's box overlaps with leader's box
            if not((minr_leader > maxr_cand) | (minr_cand > maxr_leader)):
                clustering[clusterId].append(i)
                idx_need_clusters.remove(i)
                scluster = len(clustering[clusterId])
                row_avg[clusterId] = (row_avg[clusterId]*(scluster-1) + centroids[i][1])/scluster
                
        
        n_need_clusters = len(idx_need_clusters)
        clusterId += 1
    
    
    return clustering, row_avg


true_text = {}
txt_p1 ="TODOLIST1MAKEATODOLIST2CHECKOFFTHEFIRSTTHINGONTODOLIST"
txt_p2 = "3REALIZEYOUHAVEALREADYCOMPLETED2THINGS4REWARDYOURSELFWITHANAP"
true_text['01_list.jpg'] = txt_p1+txt_p2
true_text['02_letters.jpg'] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
true_text['03_haiku.jpg'] = "HAIKUSAREEASYBUTSOMETIMESTHEYDONTMAKESENSEREFRIGERATOR"
true_text['04_deep.jpg'] = "DEEPLEARNINGDEEPERLEARNINGDEEPESTLEARNING"


for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    centroids = []
    plt.imshow(bw, cmap='Greys')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        y = (bbox[0] + bbox[2])/2
        x = (bbox[1] + bbox[3])/2
        centroids.append(np.array([x,y]))
        
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    plt.savefig("../writeup/figures/q_4_3_img_"+img.split('.')[0]+".png")
    
    # find the rows via clustering    
    clustering, row_avg = cluster_letters(bboxes)
    
    #sort letters and numbers  
    sorted_boxes = []
    row_order = list(np.argsort(row_avg))    
    
    for r in row_order:        
        col_pos = []
        for l in clustering[r]:
            col_pos.append(centroids[l][0])
            
        col_order = list(np.argsort(col_pos))
        
        for c in col_order:            
            sorted_boxes.append(bboxes[clustering[r][c]])      
        

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    img_letters_prepd = []
    for b in sorted_boxes:
        sel_letter = bw[b[0]:b[2], b[1]:b[3]]
        
        #add padding 
        sel_letter_pad = np.pad(sel_letter, 30)
        
        #do multiple dilations to thick the letters
        for _ in range(5):
            sel_letter_pad = skimage.morphology.dilation(sel_letter_pad)     
        
        #resize img
        newsize = (32, 32)
        sel_letter_res = skimage.transform.resize(sel_letter_pad, (32, 32))  
        
       
        #flat array
        img_letters_prepd.append((1.0-sel_letter_res.T).flatten())        

    
    img_letters_prepd = np.vstack(img_letters_prepd)
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    
    #predict characters    
    h1 = forward(img_letters_prepd,params,'layer1')
    class_probs = forward(h1,params,'output',softmax)

    #print predicted characters    
    letter_pred_index = np.argmax(class_probs, axis=1)
    for l in letter_pred_index:
        print(letters[l], end =" ")
    
    
    #compute accuracy    
    chars_true_text = list(true_text[img])
    
    ncorrect = 0
    for i in range(len(letter_pred_index)):
        if letters[letter_pred_index[i]] == chars_true_text[i]:
            ncorrect += 1
    
    acc = ncorrect/len(chars_true_text)
    print(acc)