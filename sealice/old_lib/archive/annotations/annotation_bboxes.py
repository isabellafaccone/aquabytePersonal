# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:18:35 2018

@author: sramena1
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 01:58:40 2018

@author: sramena1
"""

'''
Bryton: This seemed to be a test file to write the annotation boxes
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
import xml_parser_2 as xmlp


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
    
    
vid = imageio.get_reader("C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Videos\\testfile_piece_00.mp4",'ffmpeg')

vid_metadata = vid.get_meta_data()
nframes = vid_metadata['nframes']
fps = vid_metadata['fps']

start_frame = 0
end_frame = nframes-100 # 100 #

writer = imageio.get_writer("outvid_anno_bboxes.mp4", fps=fps)

xml_annotations_filename = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Annotations\\testfile_piece_00_Lice_VIPER.xml"
parser = xmlp.Parser(xml_annotations_filename)
sorted_licelist = parser.parse()
    
bbox_hwidth = 40
for frame_num in range(start_frame, end_frame):
    print frame_num
    this_frame_annos = []
    for ii in range(len(sorted_licelist)):
        if sorted_licelist[ii][0] == frame_num:
            this_frame_annos.append(sorted_licelist[ii])
    frame = vid.get_data(frame_num)
    nr, nc, nch = np.shape(frame)
    for anno in this_frame_annos:
        if anno[-1] is 'lice':
            center, ori = compute_center_ori_annotation(anno)
            #print center
            bbox_bl = np.uint16([center[0] - bbox_hwidth, center[1] - bbox_hwidth])
            bbox_tr = np.uint16([center[0] + bbox_hwidth, center[1] + bbox_hwidth])
            if center[0] - bbox_hwidth > 0 and center[1] - bbox_hwidth > 0 and center[0] + bbox_hwidth < nr and center[1] + bbox_hwidth < nc:
                cv2.rectangle(frame,(bbox_tr[1],bbox_tr[0]),(bbox_bl[1],bbox_bl[0]),(0,0,255),3)
    
    
    #cv2.imshow('frame',frame)
    writer.append_data(frame)
#    cv2.imshow('mask',thresh_mask)
#    k = cv2.waitKey(60) & 0xff
#    if k== 27:
#        break

writer.close()
            
    
    
       


  
