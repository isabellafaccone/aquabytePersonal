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

sys.path.append('/home/paperspace/sealice/pyflow')
import pyflow

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
            flow = cv2.calcOpticalFlowFarneback(fr1,fr2,0.5,10,35,10,7,10,0)
            #flow = cv2.calcOpticalFlowFarneback(fr1,fr2,0.5,10,2,2,2,0.5,0)
            #flow = cv2.calcOpticalFlowFarneback(fr1,fr2, 0.9,5,3, 1,7, 1.2, 0)
            #flow = cv2.calcOpticalFlowFarneback(fr1,fr2,0.5,3,15,3,5,1.2,0)
            mag, angle = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
            angle_avg += angle
            flow_avg += mag
    angle_avg = angle_avg/n_frames_OF
    flow_avg = flow_avg/n_frames_OF

    return flow_avg, angle_avg, flow

def compute_Coarse2Fine_OF(vid,curr_fr, n_frames_OF, n_frames_total,resize_frac_OF):
    
    ### Coarse2Fine OF params ###
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    ############################

    fr1 = vid.get_data(curr_fr)
    fr1 =   fr1.astype(float) / 255.
    nr, nc, nch = np.shape(fr1)
    flow_mag_avg = np.zeros((nr,nc), dtype='float')
    flow_ang_avg = np.zeros((nr,nc), dtype='float')
    
    flag = 0

    for jj in range(curr_fr - n_frames_OF, curr_fr + n_frames_OF + 1):
        if(jj >=0 and jj < n_frames_total and jj != curr_fr):
            fr2 = vid.get_data(jj)
            fr2 = fr2.astype(float) / 255.
            if resize_frac_OF < 1 and flag == 0:
                fr1 = cv2.resize(fr1, None, fx = resize_frac_OF, fy = resize_frac_OF)
                print np.shape(fr1)
                fr2 = cv2.resize(fr2, None, fx = resize_frac_OF, fy = resize_frac_OF)
                print np.shape(fr2)
                flow_mag_avg = np.zeros(np.shape(fr1)[0:2], dtype='float')
                flow_ang_avg = np.zeros(np.shape(fr2)[0:2], dtype='float')
                hsv = np.zeros_like(fr1,dtype = np.uint8)
                flag = 1
            elif resize_frac_OF==1 and flag == 0:
                hsv = np.zeros((nr,nc,nch),dtype = np.uint8)
                flag = 1
            u, v, im2W = pyflow.coarse2fine_flow(fr1, fr2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
            mag, ang = cv2.cartToPolar(u,v)
            flow_mag_avg += mag
            flow_ang_avg += ang
    flow_mag_avg = flow_mag_avg/n_frames_OF
    flow_ang_avg = flow_ang_avg/n_frames_OF
    
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    hsv[..., 0] = flow_ang_avg * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(flow_mag_avg, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb_gray   = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    flow_mag_img = cv2.normalize(flow_mag_avg, None, 0, 255, cv2.NORM_MINMAX)
    flow_ang_img = flow_ang_avg * 180 / np.pi / 2
    flow_ang_img = cv2.normalize(flow_ang_avg, None, 0, 255, cv2.NORM_MINMAX)
    return rgb, rgb_gray, flow_mag_img, flow_ang_img

def compute_Coarse2Fine_OF2(vid,curr_fr, n_frames_OF, n_frames_total):
    ### Coarse2Fine OF params ###
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    ############################
    fr1 = vid.get_data(curr_fr)
    fr1 =   fr1.astype(float) / 255.
   
    nr, nc, nch = np.shape(fr1)
    flow_mag_avg = np.zeros((nr,nc), dtype='float')
    flow_ang_avg = np.zeros((nr,nc), dtype='float')
    fr2 = vid.get_data(curr_fr+6)
    fr2 = fr2.astype(float) / 255.
    u, v, im2W = pyflow.coarse2fine_flow(fr1, fr2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    mag, ang = cv2.cartToPolar(u,v)
   
    hsv = np.zeros_like(fr1,dtype = np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb_gray   = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    flow_mag_img = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_ang_img = ang * 180 / np.pi / 2
    flow_ang_img = cv2.normalize(flow_ang_img, None, 0, 255, cv2.NORM_MINMAX)
    return rgb, rgb_gray, flow_mag_img, flow_ang_img
    
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
    
vid = imageio.get_reader("C:\Users\srame\Documents\Python Scripts\\aquabyte\\data\\Videos\\testfile_piece_33.mp4",'ffmpeg')
#vid = imageio.get_reader("/home/paperspace/sealice/dataaaaa/Videos/testfile_piece_02.mp4",'ffmpeg')
vid_metadata = vid.get_meta_data()
nframes = vid_metadata['nframes']
fps = vid_metadata['fps']
print nframes, fps

start_frame = 0
#end_frame = nframes
end_frame = 4
n_frames_OF = 1
resize_frac_OF = 1
out_str = 'testfile_piece_33_frame_'
#mask_thresh = 100
mask_thresh = 100
bbox_halfwidth = 20
for i in range(start_frame, end_frame):
    write_frames(vid,out_str,i)
    print i
#==============================================================================
#     frame = vid.get_data(i)
#     gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     ret,thresh_mask = cv2.threshold(gray_frame,mask_thresh,255,cv2.THRESH_BINARY)
#     candidate_locs = extract_sealice_candidate_locs(thresh_mask,bbox_halfwidth)
#     nr, nc = np.size(thresh_mask,0), np.size(thresh_mask,1) 
#     
#     #flow_avg, angle_avg, flow = compute_farneback_OF(vid,i,n_frames_OF,end_frame)
#     #rgb, rgb_gray,flow_mag, flow_ang = compute_Coarse2Fine_OF(vid,i,n_frames_OF,end_frame, resize_frac_OF)
#     rgb, rgb_gray,flow_mag, flow_ang = compute_Coarse2Fine_OF2(vid,i,n_frames_OF,end_frame)
#     
# #    for candidate_loc in candidate_locs:
# #        bbox_bl = np.uint16([candidate_loc[0] - bbox_halfwidth, candidate_loc[1] - bbox_halfwidth])
# #        bbox_tr = np.uint16([candidate_loc[0] + bbox_halfwidth, candidate_loc[1] + bbox_halfwidth])
# #        #cv2.rectangle(frame,(bbox_bl[1] - 5,bbox_bl[0] - 5),(bbox_tr[1] + 5,bbox_tr[0] + 5),(255,0,0),3)
# #        cv2.rectangle(frame,(bbox_tr[1] - 5,bbox_tr[0] - 5),(bbox_bl[1] + 5,bbox_bl[0] + 5),(255,0,0),3)
# #    ii = 0
# #    cv2.imshow('mag_angle',rgb)
# #    cv2.imshow('mag_angle_gray',rgb_gray)
# #    cv2.imshow('flow_mag',flow_mag)
# #    cv2.imshow('flow_ang',flow_ang)
#     
#     cv2.imwrite('flow_images/'+out_str+str(i)+'mag_angle'+'.png',rgb)
#     cv2.imwrite('flow_images/'+out_str+str(i)+'mag_angle_gray'+'.png',rgb_gray)
#     cv2.imwrite('flow_images/'+out_str+str(i)+'flow_mag'+'.png',flow_mag)
#     cv2.imwrite('flow_images/'+out_str+str(i)+'flow_ang'+'.png',flow_ang)
# 
#     k = cv2.waitKey(30) & 0xff
#     if k== 27:
#         break
#==============================================================================
  
