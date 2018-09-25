# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 01:58:40 2018

@author: sramena1
"""

'''
Bryton notes: seems to just translate the frames from color to grey
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

vid = imageio.get_reader("C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Videos\\testfile_piece_00.mp4",'ffmpeg')

vid_metadata = vid.get_meta_data()
nframes = vid_metadata['nframes']
fps = vid_metadata['fps']
print nframes, fps

start_frame = 0
end_frame = nframes-100
out_str = 'piece_00_'
for i in range(start_frame, end_frame):
    print i
    frame = vid.get_data(i)
    frame = cv2.resize(frame,(0,0),fx=0.25, fy=0.25)
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame,(0,0),fx=0.25, fy=0.25)
    file_name = out_str+str(i)+'.png'
    imageio.imwrite(file_name, frame)
    file_name2 = out_str+str(i)+'_gray'+'.png'
    imageio.imwrite(file_name2, gray_frame)
       


  
