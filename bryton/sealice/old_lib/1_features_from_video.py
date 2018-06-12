# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:36:49 2018

@author: sramena1
"""
import sys
import os
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
import cv2
import imageio
import xml_parser_2 as xmlp
from scipy.misc import imrotate as imrotate


def flatten_ORB_descriptor_list(orb_list):
    orbs = []
    for i in range(len(orb_list)):
        if orb_list[i] is not None:
            for j in orb_list[i]:
                if j is not None:
                    orbs.append(j)
    return orbs

def compute_center_ori_annotation(anno):
    x = np.array([])
    y = np.array([])
    ptsList = anno[2]
    for pt in ptsList:
        x = np.append(x,pt[0])
        y = np.append(y,pt[1])
    mean_x, mean_y = np.mean(x), np.mean(y)
    center = (np.int(np.round(mean_y)), np.int(np.round(mean_x)))
    ang_rad, intercept = np.polyfit(x,y,1)
    ang_deg = np.rad2deg(ang_rad)
    return center, ang_deg
    
def get_patch_from_image(frame, center, ori, half_patch_size):
    patch_len_for_rot = np.round(np.sqrt(2)*(2*half_patch_size+1))
    half_patch_len_rot = np.int(np.floor(patch_len_for_rot/2))
    nr,nc,nch = np.shape(frame)
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if center[0]-half_patch_len_rot > 0 and center[0]+half_patch_len_rot+1 < nr \
        and center[1]-half_patch_len_rot > 0 and center[1]+half_patch_len_rot+1 < nc :
            larger_patch_rgb = np.asarray(frame[center[0]-half_patch_len_rot:center[0]+half_patch_len_rot+1, \
                                                center[1]-half_patch_len_rot:center[1]+half_patch_len_rot+1, 0:3])
            larger_patch_gray = cv2.cvtColor(larger_patch_rgb,cv2.COLOR_BGR2GRAY)
            nr_p, nc_p, nch = np.shape(larger_patch_rgb)
            lp_ctr = (np.int(np.floor(nr_p/2.)+1), np.int(np.floor(nc_p/2.)+1))
            patch_rgb = larger_patch_rgb[lp_ctr[0] - half_patch_size : lp_ctr[0] + half_patch_size+1, \
                                         lp_ctr[1] - half_patch_size : lp_ctr[1] + half_patch_size+1, :]
            patch_gray = cv2.cvtColor(patch_rgb,cv2.COLOR_BGR2GRAY)
            temp_rgb_rot = imrotate(larger_patch_rgb,ori)
            rot_patch_rgb = temp_rgb_rot[lp_ctr[0] - half_patch_size : lp_ctr[0] + half_patch_size+1, \
                                         lp_ctr[1] - half_patch_size : lp_ctr[1] + half_patch_size+1, :]
            rot_patch_gray = cv2.cvtColor(rot_patch_rgb,cv2.COLOR_BGR2GRAY)
    else:
        patch_rgb = []
        patch_gray = []
        rot_patch_rgb = []
        rot_patch_gray = []

    return patch_rgb, patch_gray, rot_patch_rgb, rot_patch_gray
    
def compute_features_for_sealice_classifier(vid, start_frame, end_frame, xml_annotations_filename, half_patch_size):
    
    # parse the XML annotation file and sort by frame number
    parser = xmlp.Parser(xml_annotations_filename)
    sorted_licelist = parser.parse()
   
    orb_descriptor = cv2.ORB_create()
    # haar_descriptor = # HAAR and LBP cascade classifier training requires 
    # data preparation in a special way, too time consuming for now. Will be 
    # done if required
    
    all_frames_lice_kps = []
    all_frames_nonlice_kps = []
    all_frames_lice_rgb_patches = [] 
    all_frames_nonlice_rgb_patches = [] 
    all_frames_lice_rotated_rgb_patches = []
    all_frames_nonlice_rotated_rgb_patches = []
    all_frames_lice_gray_patches = [] 
    all_frames_nonlice_gray_patches = [] 
    all_frames_lice_rotated_gray_patches = []
    all_frames_nonlice_rotated_gray_patches = []
    all_frames_lice_ORB_descr = []
    all_frames_nonlice_ORB_descr = []

    for frame_num in range(start_frame,end_frame):
        #print 'processing frame ' + str(frame_num) + ' of ' + str(end_frame) + ' frames'
        this_frame_annos = []
        for ii in range(len(sorted_licelist)):
            if sorted_licelist[ii][0] == frame_num:
                this_frame_annos.append(sorted_licelist[ii])
    
        frame = vid.get_data(frame_num)
        
        this_frame_lice_kps = []
        this_frame_nonlice_kps = []
        this_frame_lice_rgb_patches = [] 
        this_frame_nonlice_rgb_patches = [] 
        this_frame_lice_rotated_rgb_patches = []
        this_frame_nonlice_rotated_rgb_patches = []
        this_frame_lice_gray_patches = [] 
        this_frame_nonlice_gray_patches = [] 
        this_frame_lice_rotated_gray_patches = []
        this_frame_nonlice_rotated_gray_patches = []
    
        ii = 0
        for anno in this_frame_annos:
            if anno[1] > 2 and anno[1] < 5:
                center, ori = compute_center_ori_annotation(anno)
                patch_rgb, patch_gray, rot_patch_rgb, rot_patch_gray = get_patch_from_image(frame, center, ori, half_patch_size)

                if len(patch_rgb)>0:
                    if anno[-1] is 'lice':
                        temp_kp = cv2.KeyPoint()
                        temp_kp.pt = (center[1], center[0])
                        this_frame_lice_kps.append(temp_kp)
                        this_frame_lice_rgb_patches.append(patch_rgb)
                        this_frame_lice_rotated_rgb_patches.append(rot_patch_rgb)
                        this_frame_lice_gray_patches.append(patch_gray)
                        this_frame_lice_rotated_gray_patches.append(rot_patch_gray)
                    if anno[-1] is 'not_lice':
                        temp_kp = cv2.KeyPoint()
                        temp_kp.pt = (center[1], center[0])
                        this_frame_nonlice_kps.append(temp_kp)
                        this_frame_nonlice_rgb_patches.append(patch_rgb)
                        this_frame_nonlice_rotated_rgb_patches.append(rot_patch_rgb)
                        this_frame_nonlice_gray_patches.append(patch_gray)
                        this_frame_nonlice_rotated_gray_patches.append(rot_patch_gray)
                
                    ii = ii + 1

        nonlice_kps, nonlice_descr = orb_descriptor.compute(frame,this_frame_nonlice_kps)
        lice_kps, lice_descr = orb_descriptor.compute(frame,this_frame_lice_kps) # ORB descriptor class drops some key points, hence getting the new key points here
        all_frames_lice_kps.append(this_frame_lice_kps)
        all_frames_nonlice_kps.append(this_frame_nonlice_kps)
        all_frames_lice_rgb_patches.append(this_frame_lice_rgb_patches) 
        all_frames_nonlice_rgb_patches.append(this_frame_nonlice_rgb_patches) 
        all_frames_lice_rotated_rgb_patches.append(this_frame_lice_rotated_rgb_patches)
        all_frames_nonlice_rotated_rgb_patches.append(this_frame_nonlice_rotated_rgb_patches)
        all_frames_lice_gray_patches.append(this_frame_lice_gray_patches) 
        all_frames_nonlice_gray_patches.append(this_frame_nonlice_gray_patches) 
        all_frames_lice_rotated_gray_patches.append(this_frame_lice_rotated_gray_patches)
        all_frames_nonlice_rotated_gray_patches.append(this_frame_nonlice_rotated_gray_patches)
        
        all_frames_lice_ORB_descr.append(lice_descr)
        all_frames_nonlice_ORB_descr.append(nonlice_descr)
           
    return sorted_licelist, all_frames_lice_kps, all_frames_nonlice_kps, all_frames_lice_rgb_patches, all_frames_nonlice_rgb_patches, all_frames_lice_rotated_rgb_patches, all_frames_nonlice_rotated_rgb_patches, all_frames_lice_gray_patches, all_frames_nonlice_gray_patches, all_frames_lice_rotated_gray_patches, all_frames_nonlice_rotated_gray_patches, all_frames_lice_ORB_descr, all_frames_nonlice_ORB_descr
     
video_filename = "C:\Users\srame\Documents\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Videos\\testfile_piece_34.mp4"
[filepath, filename] = os.path.split(video_filename)
[filename_, ext] = filename.split(os.extsep)
xml_anno_filename  = "C:\Users\srame\Documents\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Annotations\\testfile_piece_34_Lice_VIPER.xml"

vid = imageio.get_reader(video_filename,'ffmpeg')
vid_metadata = vid.get_meta_data()
nframes = vid_metadata['nframes']

chunk_size = 500
nchunks = nframes/chunk_size
half_patch_size = 24

for i in range(0,nchunks+1):
    start_frame = i*chunk_size
    end_frame = (i+1)*chunk_size

    if end_frame > nframes:
        end_frame = start_frame + (nframes%chunk_size) - 200 # reading last 10-20 frames creates IO error, so dropping last 40
        
    sorted_licelist, all_frames_lice_kps, all_frames_nonlice_kps, all_frames_lice_rgb_patches, all_frames_nonlice_rgb_patches, \
    all_frames_lice_rotated_rgb_patches, all_frames_nonlice_rotated_rgb_patches, all_frames_lice_gray_patches, \
    all_frames_nonlice_gray_patches, all_frames_lice_rotated_gray_patches, all_frames_nonlice_rotated_gray_patches, \
    all_frames_lice_ORB_descr, all_frames_nonlice_ORB_descr = compute_features_for_sealice_classifier(vid, \
                                                    start_frame, end_frame, xml_anno_filename, half_patch_size)
    
    all_frames_lice_kps = [item for sublist in all_frames_lice_kps for item in sublist]
    all_frames_nonlice_kps = [item for sublist in all_frames_nonlice_kps for item in sublist]
    all_frames_lice_rgb_patches = [item for sublist in all_frames_lice_rgb_patches for item in sublist]
    all_frames_nonlice_rgb_patches = [item for sublist in all_frames_nonlice_rgb_patches for item in sublist]
    all_frames_lice_rotated_rgb_patches = [item for sublist in all_frames_lice_rotated_rgb_patches for item in sublist]
    all_frames_nonlice_rotated_rgb_patches = [item for sublist in all_frames_nonlice_rotated_rgb_patches for item in sublist]
    all_frames_lice_gray_patches = [item for sublist in all_frames_lice_gray_patches for item in sublist]
    all_frames_nonlice_gray_patches = [item for sublist in all_frames_nonlice_gray_patches for item in sublist]
    all_frames_lice_rotated_gray_patches = [item for sublist in all_frames_lice_rotated_gray_patches for item in sublist]
    all_frames_nonlice_rotated_gray_patches = [item for sublist in all_frames_nonlice_rotated_gray_patches for item in sublist]
    
    all_frames_lice_ORB_descr2 = flatten_ORB_descriptor_list(all_frames_lice_ORB_descr)
    all_frames_nonlice_ORB_descr2 = flatten_ORB_descriptor_list(all_frames_nonlice_ORB_descr)
    print len(all_frames_lice_ORB_descr2), len(all_frames_nonlice_ORB_descr2)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_kps', all_frames_lice_kps)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_kps', all_frames_nonlice_kps)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_rgb_patches', all_frames_lice_rgb_patches)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_rgb_patches', all_frames_nonlice_rgb_patches)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_rotated_rgb_patches', all_frames_lice_rotated_rgb_patches)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_rotated_rgb_patches', all_frames_nonlice_rotated_rgb_patches)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_gray_patches', all_frames_lice_gray_patches)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_gray_patches', all_frames_nonlice_gray_patches)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_rotated_gray_patches', all_frames_lice_rotated_gray_patches)
    #np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_rotated_gray_patches', all_frames_nonlice_rotated_gray_patches)
    np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_ORB_descr', all_frames_lice_ORB_descr)
    np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_ORB_descr', all_frames_nonlice_ORB_descr)
