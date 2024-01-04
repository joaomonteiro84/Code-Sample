import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
    # returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []   
    bw = None
       
    #estimate noise
    estNoise = skimage.restoration.estimate_sigma(image = image,  
                                                  average_sigmas = True, 
                                                  multichannel = True, 
                                                  channel_axis = 2)        
    #denoise
    denoiseImg = skimage.restoration.denoise_wavelet(image = image,
                                                     sigma = estNoise,
                                                     channel_axis = 2)     
    #grayscale
    gs_img = skimage.color.rgb2gray(denoiseImg)
    
    #threshold    
    thresh = skimage.filters.threshold_otsu(gs_img)
    
    #morphology
    bw = skimage.morphology.closing(gs_img < thresh, 
                                    skimage.morphology.square(7))    
    
    cleared_image = skimage.segmentation.clear_border(bw)
    
    #label
    label_image = skimage.measure.label(cleared_image)    
    
    #distribution of region areas
    region_areas = []
    for region in skimage.measure.regionprops(label_image):
        region_areas.append(region.area)
     
    #get a threshold for "small" boxes
    avg_region = np.mean(np.array(region_areas))
    std_region = np.std(np.array(region_areas))
    small_area_threshold = avg_region - 2.5*std_region

    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas       
        if region.area >= small_area_threshold:
            bboxes.append(region.bbox)           
    
    return bboxes, bw.astype(float)


