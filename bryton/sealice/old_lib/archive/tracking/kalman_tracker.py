# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 15:42:12 2018

@author: sramena1
"""
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

    
def extract_sealice_candidate_locs(thresh_mask,bbox_halfwidth):
    area_min, area_max = 25, 150
    nr, nc = np.size(thresh_mask,0), np.size(thresh_mask,1) 
    ecc_min, ecc_max = 0.75, 0.999
    if np.max(thresh_mask,axis=None) == 255:
        thresh_mask=np.where(thresh_mask==255,1,0)
    
    labels = skimage.measure.label(thresh_mask,background=0)
    regions = skimage.measure.regionprops(labels)
    candidate_locs = []
    for region in regions:
        if region.area > area_min and region.area < area_max:
            if region.eccentricity > ecc_min and region.eccentricity < ecc_max:
                if region.centroid[0]-bbox_halfwidth > 0 and region.centroid[0]-bbox_halfwidth < nr and region.centroid[1]-bbox_halfwidth > 1 and region.centroid[1]-bbox_halfwidth < nc:
                    candidate_locs.append(region.centroid)
        
    return candidate_locs
    
def write_frames(vid,out_str,i):
    fr = vid.get_data(i)
    file_name = out_str+str(i)+'.png'
    imageio.imwrite(file_name, fr)
    
vid = imageio.get_reader("C:\Users\srame\Documents\Python Scripts\\aquabyte\\data\\Videos\\testfile_piece_00.mp4",'ffmpeg')
#vid = imageio.get_reader("C:\Users\srame\Documents\Python Scripts\\aquabyte\\data\\Videos\\testfile_piece_02.mp4",'ffmpeg')
vid_metadata = vid.get_meta_data()
nframes = vid_metadata['nframes']
fps = vid_metadata['fps']
print nframes, fps

start_frame = 0
#end_frame = nframes
end_frame = 2
out_str = 'fish_02_'
#mask_thresh = 100
mask_thresh = 100
bbox_halfwidth = 20
for i in range(start_frame, end_frame):
    #write_frames(vid,out_str,i)
    print i
    frame = vid.get_data(i)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,thresh_mask = cv2.threshold(gray_frame,mask_thresh,255,cv2.THRESH_BINARY)
    candidate_locs = extract_sealice_candidate_locs(thresh_mask,bbox_halfwidth)
    nr, nc = np.size(thresh_mask,0), np.size(thresh_mask,1) 
    
       
    for candidate_loc in candidate_locs:
        bbox_bl = np.uint16([candidate_loc[0] - bbox_halfwidth, candidate_loc[1] - bbox_halfwidth])
        bbox_tr = np.uint16([candidate_loc[0] + bbox_halfwidth, candidate_loc[1] + bbox_halfwidth])
#        print nr,nc
#        print bbox_bl
#        print bbox_tr
        #cv2.rectangle(frame,(bbox_bl[1] - 5,bbox_bl[0] - 5),(bbox_tr[1] + 5,bbox_tr[0] + 5),(255,0,0),3)
        cv2.rectangle(frame,(bbox_tr[1] - 5,bbox_tr[0] - 5),(bbox_bl[1] + 5,bbox_bl[0] + 5),(255,0,0),3)
        
    ii = 0
       
    cv2.imshow('frame',frame)
    cv2.imshow('mask',thresh_mask)
    k = cv2.waitKey(60) & 0xff
    if k== 27:
        break
  