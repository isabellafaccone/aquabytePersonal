# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 02:36:33 2018

@author: sramena1

optic flow based segmentation - test
"""

'''
Bryton notes: Tests out an optical flow-based segmenation method for computing the fish count. 
Lice are not detected, but rather the annotations are used to give a sense of lice count.
'''

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
import cv2
import re
import imageio
import skimage
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import hdbscan
from skimage.transform import resize
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.misc import imresize 

def compute_center_ori_annotation(anno):
    x = np.array([])
    y = np.array([])
    ptsList = anno[2]
    for pt in ptsList:
        x = np.append(x,pt[0])
        y = np.append(y,pt[1])
    mean_x, mean_y = np.mean(x), np.mean(y)
    #center = (np.round(mean_y), np.round(mean_x))
    center = (np.int(np.round(mean_y)), np.int(np.round(mean_x)))
    #print center
    ang_rad, intercept = np.polyfit(x,y,1)
    ang_deg = np.rad2deg(ang_rad)
    return center, ang_deg
    
def compute_lice_to_fish_ratio_frame(seg_img, labels, this_frame_annos):
    area_thresh = 10000
    ecc_thresh = 0.75
    fish_count = 0
    for label in labels:
        if label > 0:
            seg_copy = np.copy(seg_img)
            seg_copy[seg_copy != label] = 0
            seg_copy[seg_copy == label] = 1
            labeled_img = skimage.measure.label(seg_copy)
            regions = skimage.measure.regionprops(labeled_img)
            
            for region in regions:
                if region.area > area_thresh and region.eccentricity > ecc_thresh:
                    fish_count = fish_count + 1
                    
                    
                    #print label, len(regions)
    return fish_count
    #return frame_ratio, frame_nlice, frame_nfish
    
base_dir = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\flow_images"
base_dir_img = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\piece_00_frames\\color"
file_prefix = "fish_00_"
file_prefix_img = "piece_00_"

xml_annotations_filename = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Annotations\\testfile_piece_00_Lice_VIPER.xml"
parser = xmlp.Parser(xml_annotations_filename)
sorted_licelist = parser.parse()

mag_thresh = 20
all_frames_fish_count = 0
all_frames_lice_count = 0
plt.close("all")
for frame_num in range(1233):
#for i in range(1000,1001):
  
    file_name_ang = path.join(base_dir, file_prefix+str(frame_num)+'flow_ang.png')
    file_name_mag = path.join(base_dir, file_prefix+str(frame_num)+'flow_mag.png')
    #file_name_img = path.join(base_dir_img,file_prefix_img+str(frame_num)+'.png')
    
    mag = cv2.imread(file_name_mag)
    ang = cv2.imread(file_name_ang)
    im = cv2.imread(file_name_img)
    
    mag = cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY)
    ang = cv2.cvtColor(ang, cv2.COLOR_BGR2GRAY)
    
    _, mag_mask = cv2.threshold(mag, mag_thresh, 1, cv2.THRESH_BINARY)
    
    mag = np.multiply(mag, mag_mask)
    ang = np.multiply(ang, mag_mask)
    
    mag = np.float64(mag)
    ang = np.float64(ang)

    nr,nc = np.shape(mag)
    
    mag = resize(mag, (nr/2, nc/2))
    ang = resize(ang, (nr/2, nc/2))
    
    nr2, nc2 = np.shape(mag)
    
    data = [[mag[i][j], ang[i][j]] for i in range(nr2) for j in range(nc2) if mag[i][j] > 0]
    #data = [[mag[i][j]] for i in range(nr2) for j in range(nc2) if mag[i][j] > 0]
    #data = [[ang[i][j]] for i in range(nr2) for j in range(nc2) if mag[i][j] > 0]
    locs = [[i, j] for i in range(nr2) for j in range(nc2) if mag[i][j] > 0]
    #data = [[i, j, mag[i][j], ang[i][j]] for i in range(nr2) for j in range(nc2) if mag[i][j] > 0]
    
    data = np.float64(data)

    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=5000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)
    
    seg_img = np.zeros(np.shape(mag), dtype = np.uint8)
    
    for jj in range(len(labels)):
        seg_img[locs[jj][0], locs[jj][1]] = labels[jj]+1

    seg_img = imresize(seg_img, 4.0)
    
    this_frame_annos = []
    for ii in range(len(sorted_licelist)):
        if sorted_licelist[ii][0] == i:
            this_frame_annos.append(sorted_licelist[ii])
            
    this_frame_lice_count = 0
    
    for anno in this_frame_annos:
        if anno[-1] is 'lice':
            this_frame_lice_count += 1
    
    unique_labels = list(set(labels))
    
    this_frame_fish_count = compute_lice_to_fish_ratio_frame(seg_img, unique_labels, this_frame_annos)
    
    print frame_num, this_frame_lice_count, this_frame_fish_count
    
    all_frames_fish_count = all_frames_fish_count + this_frame_fish_count
    all_frames_lice_count = all_frames_lice_count + this_frame_lice_count
    
#    plt.figure()
#    plt.imshow(seg_img)
#    
#    plt.figure()
#    plt.imshow(mag)
#    plt.figure()
#    plt.imshow(ang)
#    plt.figure()
#    plt.imshow(im)

print all_frames_lice_count, all_frames_fish_count
lice_to_fish_ratio = all_frames_lice_count/(all_frames_fish_count+0.0)
print lice_to_fish_ratio

