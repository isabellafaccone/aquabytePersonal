# -*- coding: utf-8 -*-
"""
Aquabyte sealice detection

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

def compute_farneback_OF(vid, curr_fr, n_frames_OF, n_frames_total):
    fr = vid.get_data(curr_fr)
    fr1 = cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    flow_avg = np.zeros_like(fr1, dtype = 'float')
    angle_avg = np.zeros_like(fr1, dtype = 'float')
    
    # sum the optic flow mag for robustness. n_frames_op
    for jj in range(curr_fr - n_frames_OF, curr_fr + n_frames_OF + 1):
        if(jj >=0 and jj < n_frames_total and jj != curr_fr):
            fr2 = vid.get_data(jj)
            fr2 = cv2.cvtColor(fr2,cv2.COLOR_BGR2GRAY)
            #flow = cv2.calcOpticalFlowFarneback(fr1,fr2,None,0.5,10,35,10,7,10,0)
            #flow = cv2.calcOpticalFlowFarneback(fr1,fr2,None,0.5,10,2,2,2,0.5,0)
            #flow = cv2.calcOpticalFlowFarneback(fr1,fr2,None, 0.9,5,3, 1,7, 1.2, 0)
            flow = cv2.calcOpticalFlowFarneback(fr1,fr2,None,0.5,3,15,3,5,1.2,0)
            mag, angle = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
            angle_avg += angle
            flow_avg += mag
    angle_avg = angle_avg/n_frames_OF
    flow_avg = flow_avg/n_frames_OF

    return flow_avg, angle_avg
    
def write_frames(vid,out_str,i):
    fr = vid.get_data(i)
    file_name = out_str+str(i)+'.png'
    imageio.imwrite(file_name, fr)
    
vid = imageio.get_reader("C:\Users\srame\Documents\Python Scripts\\aquabyte\\data\Videos\\testfile_piece_00.mp4",'ffmpeg')
vid_metadata = vid.get_meta_data()
nframes = vid_metadata['nframes']
fps = vid_metadata['fps']
print nframes, fps
out_str = 'fish_00_'
start_frame = 101
#end_frame = nframes
end_frame = 102

n_frames_OF = 1
for i in range(start_frame,end_frame):
    #write_frames(vid,out_str,i)
    fr = vid.get_data(i)
    fr2 = vid.get_data(i+1)
    plt.figure()
    plt.imshow(fr)
    #plt.imshow(fr1)
    flow_avg, angle_avg = compute_farneback_OF(vid,i,n_frames_OF,end_frame)
    
    plt.figure()
    plt.imshow(flow_avg)
    plt.figure()
    plt.imshow(angle_avg)    