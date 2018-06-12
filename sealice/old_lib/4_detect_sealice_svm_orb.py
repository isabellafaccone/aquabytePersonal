# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 19:21:41 2018

@author: sramena1
"""
import imageio
import cv2
import sys
import os
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
import skimage
from skimage.measure import label, regionprops

class detectSeaLiceSVMORB(object):
    mask_thresh = 100
    def __init__(self,vid_filepath, svm_mdl_filepath):
        self.vid_filepath = vid_filepath
        self.svm_mdl_filepath = svm_mdl_filepath
        self.mask_thresh = 80
        self.bbox_halfwidth = 24
        self.area_min = 100
        self.area_max = 200
        self.ecc_min = 0.7
        self.ecc_max = 0.99
        
    def extract_sealice_candidate_kps(self,thresh_mask):
        
        nr, nc = np.size(thresh_mask,0), np.size(thresh_mask,1) 
        if np.max(thresh_mask,axis=None) == 255:
            thresh_mask=np.where(thresh_mask==255,1,0)
            
        labels = skimage.measure.label(thresh_mask,background=0)
        regions = skimage.measure.regionprops(labels)
        sealice_candidate_kps = []
        for region in regions:
            if region.area > self.area_min and region.area < self.area_max:
                if region.eccentricity > self.ecc_min and region.eccentricity < self.ecc_max:
                    if region.centroid[0]-self.bbox_halfwidth > 0 and region.centroid[0]-self.bbox_halfwidth < nr \
                        and region.centroid[1]-self.bbox_halfwidth > 1 and region.centroid[1]-self.bbox_halfwidth < nc:
                        temp_kp = cv2.KeyPoint()
                        temp_kp.pt = (region.centroid[1], region.centroid[0])
                        sealice_candidate_kps.append(temp_kp)
        
        return sealice_candidate_kps
    
    def detect(self):
        
        vid = imageio.get_reader(self.vid_filepath,'ffmpeg')
        vid_metadata = vid.get_meta_data()
        nframes = vid_metadata['nframes']
        fps = vid_metadata['fps']
        
        # load the saved SVM model
        svm_mdl = cv2.ml.SVM_load(self.svm_mdl_filepath)
        # create a ORB instance
        orb_descriptor = cv2.ORB_create()
        # create an instance of video writer
        vidwriter = imageio.get_writer('outvid.mp4', fps=15)
        
        # for each frame detect keypoints
        for frame_num in range(nframes-200):
            print 'processing frame ' + str(frame_num) + ' of ' + str(nframes) + ' frames'
            frame = vid.get_data(frame_num)
            
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            ret,thresh_mask = cv2.threshold(gray_frame,self.mask_thresh,255,cv2.THRESH_BINARY)
            candidate_kps = self.extract_sealice_candidate_kps(thresh_mask)
            
            kps, orb_descrs = orb_descriptor.compute(frame,candidate_kps)
            orb_descrs = np.asfarray(orb_descrs, dtype = 'float32')
            if np.size(orb_descrs)>1:
                pred_labels = svm_mdl.predict(orb_descrs)
                pred_labels = pred_labels[1]
                for i, pred_label in enumerate(pred_labels):
                    if pred_label[0] == 1:
                        tl2 = (np.int(kps[i].pt[0]-28), np.int(kps[i].pt[1]-28)) 
                        br2 = (np.int(kps[i].pt[0]+28), np.int(kps[i].pt[1]+28))
                        cv2.rectangle(frame, tl2, br2, (0,0,255), 3)
            vidwriter.append_data(frame)
        vidwriter.close()
    
def main():
    vid_filepath = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Videos\\testfile_piece_00.mp4"
    svm_mdl_filepath = "sealice_detection_ORB_SVM_model.yml"
    
    sealice_detector = detectSeaLiceSVMORB(vid_filepath, svm_mdl_filepath)
    sealice_detector.detect()

if __name__ == '__main__':
    main()       
