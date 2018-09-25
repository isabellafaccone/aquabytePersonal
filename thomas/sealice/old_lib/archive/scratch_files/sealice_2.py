# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 15:42:12 2018

@author: sramena1
"""

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

def write_frames(vid,out_str,i):
    fr = vid.get_data(i)
    file_name = out_str+str(i)+'.png'
    imageio.imwrite(file_name, fr)
    
#cap = cv2.VideoCapture("C:\Users\srame\Documents\Python Scripts\\aquabyte\\data\\Videos\\testfile_piece_00.mp4")
#cap = cv2.VideoCapture("testfile_piece_00.mp4")

vid = imageio.get_reader("C:\Users\srame\Documents\Python Scripts\\aquabyte\\data\\Videos\\testfile_piece_00.mp4",'ffmpeg')
vid_metadata = vid.get_meta_data()
nframes = vid_metadata['nframes']
fps = vid_metadata['fps']
print nframes, fps
out_str = 'fish_00_'
start_frame = 0
#end_frame = nframes
end_frame = 1334

fgbg = cv2.createBackgroundSubtractorMOG2(25)
#fgbg3 = cv2.createBackgroundSubtractorKNN()
for i in range(start_frame, end_frame):
   # ret,frame = cap.read()
    frame = vid.get_data(i)
   # print ret
    #fgmask = fgbg.apply(frame)
    fgmask = fgbg3.apply(frame)
    cv2.imshow('frame',frame)
    cv2.imshow('fgmask', fgmask)
    
    k = cv2.waitKey(30) & 0xff
    if k== 27:
        break

cap.release()
cv2.destroyAllWindows()

    
    
    
    
    
    