# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:21:58 2018

@author: sramena1
"""
import sys
import os
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
import cv2
import fnmatch
#import xmlparser
import xml_parser_2 as xmlp
import imageio

class sealiceSVMpredictAnnotatedVideo(object):
    """ 
        update doc string later
    """
    def __init__(self, svm_mdl_filepath, video_filename, anno_xml_filename):
        """ 
        update doc string later
        """
        self.svm_mdl_filepath = svm_mdl_filepath
        self.video_filename = video_filename
        self.xml_anno = anno_xml_filename
        
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
        
    def predict_sealice(self):
        """ 
        update doc string later
        """
        n_lice = 0
        n_nonlice = 0
        lice_tp = 0
        lice_fn = 0
        nonlice_tn = 0
        nonlice_fp = 0
        # load the trained SVM model
        svm_mdl = cv2.ml.SVM_load(self.svm_mdl_filepath)
        # create an ORB instance
        orb_descriptor = cv2.ORB_create()
        # read the XML anno file and arrange properly
#        parser = xmlparser.dataFile(self.xml_anno)
#        lice = parser.parseFile()
#        licelist = parser.toList(lice)
#        sorted_licelist = sorted(licelist,key = itemgetter(0))

        parser = xmlp.Parser(self.xml_anno)
        sorted_licelist = parser.parse()
        
        # read the video and get frames
        vid = imageio.get_reader(self.video_filename,'ffmpeg')
        vid_metadata = vid.get_meta_data()
        nframes = vid_metadata['nframes']
        fps = vid_metadata['fps']
        # create a out video for writing the predicted frames
        #vidwriter = imageio.get_writer('outvid.mp4', fps=fps)
        vidwriter = imageio.get_writer('outvid.mp4', fps=15)
        
        # for each video frame, compute ORB descr and predict
        for frame_num in range(nframes-200):
        #for frame_num in range(260,270):
            print 'processing frame ' + str(frame_num) + ' of ' + str(nframes) + ' frames'
            this_frame_annos = []
            for ii in range(len(sorted_licelist)):
                if sorted_licelist[ii][0] == frame_num:
                    this_frame_annos.append(sorted_licelist[ii])
        
            frame = vid.get_data(frame_num)
            frame_copy = np.copy(frame)
            this_frame_lice_kps = []
            this_frame_nonlice_kps = []
            for anno in this_frame_annos:
                if anno[1] > 2 and anno[1] < 5:
                    center, ori = compute_center_ori_annotation(anno)
                    tl = (center[1]-20, center[0]-20)
                    br = (center[1]+20, center[0]+20)
                    
                    tl2 = (center[1]-35, center[0]-35) # larger rect for predicted label
                    br2 = (center[1]+35, center[0]+35)
                    #print anno[-1]
                    #if anno[-1] == 1:
                    if anno[-1] is 'lice':
                        n_lice = n_lice + 1
                        # first plot the GT bbox. For lice, red bbox for 48x48 pix.
                        color = (0,255,0)
                        opp_color = (255,0,0)
                        cv2.rectangle(frame_copy, tl, br, color, 3)
                        this_frame_lice_kps = []
                        #temp_kp = cv2.KeyPoint(0,0,1)
                        temp_kp = cv2.KeyPoint()
                        temp_kp.pt = (center[1], center[0])
                        #temp_kp.pt = (center[0], center[1])
                        this_frame_lice_kps.append(temp_kp)
                        #temp_kp.pt = (center[0], center[1])
                        #print anno[-1], temp_kp.pt, temp_kp.size
                        #kp_, descr = orb_descriptor.compute(frame, temp_kp)
                        kp_, descr = orb_descriptor.compute(frame, this_frame_lice_kps)
                        #print np.shape(descr), descr
                        if descr is not None:
                            descr = np.asfarray(descr, dtype = 'float32')
                            pred_label = svm_mdl.predict(descr)
                            #print frame_num, pred_label[1][0][0], anno[-1]
                            #if pred_label == anno[-1]:
                            if pred_label[1][0][0] == 1:
                                lice_tp = lice_tp + 1
                                cv2.rectangle(frame_copy, tl2, br2, color, 3)
                            #elif pred_label != anno[-1]:
                            elif pred_label[1][0][0] == 0:
                                lice_fn = lice_fn + 1
                                cv2.rectangle(frame_copy, tl2, br2, opp_color, 3)

                    #elif anno[-1] == 0:
                    elif anno[-1] is 'not_lice':
                        n_nonlice = n_nonlice + 1
                        # first plot the GT bbox. For lice, red bbox for 48x48 pix.
                        color = (255,0,0)
                        opp_color = (0,255,0)
                        cv2.rectangle(frame_copy, tl, br, color, 3)
                        this_frame_nonlice_kps = []
                        temp_kp = cv2.KeyPoint()
                        temp_kp.pt = (center[1], center[0])
                        #temp_kp.pt = (center[0], center[1])
                        this_frame_nonlice_kps.append(temp_kp)
                        #print anno[-1], temp_kp.pt, temp_kp.size
                        #kp_, descr = orb_descriptor.compute(frame, temp_kp)
                        kp_, descr = orb_descriptor.compute(frame, this_frame_nonlice_kps)
                        #print kp_, len(descr)
                        if descr is not None:
                            descr = np.asfarray(descr, dtype = 'float32')
                            pred_label = svm_mdl.predict(descr)
                            #print frame_num, pred_label[1][0][0], anno[-1]
                            #if pred_label == anno[-1]:
                            if pred_label[1][0][0] == 0:
                                nonlice_tn = nonlice_tn + 1
                                cv2.rectangle(frame_copy, tl2, br2, color, 3)
                            #elif pred_label != anno[-1]:
                            elif pred_label[1][0][0] == 1:
                                nonlice_fp = nonlice_fp + 1
                                cv2.rectangle(frame_copy, tl2, br2, opp_color, 3)
            
            # write video for visualization with ground-truth lice (red), 
            # non-lice(blue) bounding boxes and predicted lice (red) and 
            # non-line (blue) bounding boxes of slightly larger size. Color 
            # matching BBs means the classifier is doing a good job.  
            vidwriter.append_data(frame_copy)
            
            #print n_lice, n_nonlice, lice_tp, lice_fn, nonlice_tn, nonlice_fp
            
        vidwriter.close()
        lice_precision = (lice_tp)/(lice_tp + nonlice_fp + 0.0)
        lice_recall = (lice_tp)/(lice_tp + lice_fn + 0.0)
        lice_accuracy = (lice_tp + nonlice_tn)/(lice_tp + lice_fn + nonlice_tn + nonlice_fp + 0.0)
        print n_lice, n_nonlice, lice_tp, lice_fn, nonlice_tn, nonlice_fp
        print("Precision = {} Recall = {} Accuracy = {}".format(lice_precision, lice_recall, lice_accuracy))
            
def main():
    svm_mdl_filepath = "sealice_detection_ORB_SVM_model_2.yml"
    video_filename = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Videos\\testfile_piece_18.mp4"
    anno_xml_filename = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Annotations\\testfile_piece_18_Lice_VIPER.xml"
    
    svm_predict_anno_vid = sealiceSVMpredictAnnotatedVideo(svm_mdl_filepath, video_filename, anno_xml_filename)
    svm_predict_anno_vid.predict_sealice()

if __name__ == '__main__':
    main()
    