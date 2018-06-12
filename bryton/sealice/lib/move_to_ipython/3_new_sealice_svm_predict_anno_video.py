'''
Predicts sea lice using a pre-trained SVM
Gives precision and recall numbers
'''

import sys
import os
#from os import path
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
import cv2
#import fnmatch
#import xml_parser_2 as xmlp
#import imageio

import utils

# class sealiceSVMpredictAnnotatedVideo(object):
#     """ 
#         update doc string later
#     """
#     def __init__(self, svm_mdl_filepath, video_filename, anno_xml_filename):
#         """ 
#         update doc string later
#         """
#         self.svm_mdl_filepath = svm_mdl_filepath
#         self.video_filename = video_filename
#         self.xml_anno = anno_xml_filename
        
    # def compute_center_ori_annotation(self, anno):
        
    #     x = np.array([])
    #     y = np.array([])
    #     ptsList = anno[2]
    #     for pt in ptsList:
    #         x = np.append(x,pt[0])
    #         y = np.append(y,pt[1])
    #     mean_x, mean_y = np.mean(x), np.mean(y)
    #     center = (np.int(np.round(mean_y)), np.int(np.round(mean_x)))
    #     ang_rad, intercept = np.polyfit(x,y,1)
    #     ang_deg = np.rad2deg(ang_rad)
    #     return center, ang_deg
        
def predict_sealice(svm_mdl_filepath, annotations_file):
    num_lice = 0
    num_nonlice = 0
    lice_true_positive = 0
    lice_false_negative = 0
    nonlice_true_negative = 0
    nonlice_false_positive = 0

    # load the trained SVM model
    svm_mdl = cv2.ml.SVM_load(svm_mdl_filepath)

    # create an ORB instance
    orb_descriptor = cv2.ORB_create()

    annotations = utils.get_lice_annotations_from_file(annotations_file)

    # for each video frame, compute ORB descr and predict
    for annotation_index, annotation in enumerate(annotations):
        print 'Processing annotation %i of %i' % (annotation_index, len(annotations))

        image_filename, x1, y1, x2, y2, label = annotation

        center_x = int(np.round((x1 + x2) / 2.0))
        center_y = int(np.round((y1 + y2) / 2.0))

        frame = np.array(Image.open(image_filename))
        frame_copy = np.copy(frame)

        this_frame_lice_kps = []
        this_frame_nonlice_kps = []

        #if anno[1] > 2 and anno[1] < 5:
        #center, ori = self.compute_center_ori_annotation(anno)
        
        tl = (center_x - 20, center_y - 20)
        br = (center_x + 20, center_y + 20)
        
        tl2 = (center_x - 35, center_y - 35) # larger rect for predicted label
        br2 = (center_x + 35, center_y + 35)

        temp_kp = cv2.KeyPoint()
        temp_kp.pt = (center_x, center_y)
        kp_, descr = orb_descriptor.compute(frame, [ temp_kp ])

        if utils.is_lice(label):
            num_lice = num_lice + 1

            color = (0,255,0) # LICE is GREEN
            opp_color = (255,0,0)

            cv2.rectangle(frame_copy, tl, br, color, 3)

            if descr is not None:
                descr = np.asfarray(descr, dtype = 'float32')
                pred_label = svm_mdl.predict(descr)
                #print frame_num, pred_label[1][0][0], anno[-1]
                if pred_label[1][0][0] == 1:
                    lice_true_positive = lice_true_positive + 1
                    cv2.rectangle(frame_copy, tl2, br2, color, 3)
                elif pred_label[1][0][0] == 0:
                    lice_false_negative = lice_false_negative + 1
                    cv2.rectangle(frame_copy, tl2, br2, opp_color, 3)

        else:
            num_nonlice = num_nonlice + 1

            color = (255,0,0)
            opp_color = (0,255,0)

            cv2.rectangle(frame_copy, tl, br, color, 3)

             if descr is not None:
                descr = np.asfarray(descr, dtype = 'float32')
                pred_label = svm_mdl.predict(descr)
                #print frame_num, pred_label[1][0][0], anno[-1]
                if pred_label[1][0][0] == 0:
                    nonlice_true_negative = nonlice_true_negative + 1
                    cv2.rectangle(frame_copy, tl2, br2, color, 3)
                elif pred_label[1][0][0] == 1:
                    nonlice_false_positive = nonlice_false_positive + 1
                    cv2.rectangle(frame_copy, tl2, br2, opp_color, 3)

        # write video for visualization with ground-truth lice (red), 
        # non-lice(blue) bounding boxes and predicted lice (red) and 
        # non-line (blue) bounding boxes of slightly larger size. Color 
        # matching BBs means the classifier is doing a good job.  
        #vidwriter.append_data(frame_copy)

        # TODO: write this to file for visualization

    #vidwriter.close()
    lice_precision = (lice_true_positive) / (lice_true_positive + nonlice_false_positive + 0.0)
    lice_recall = (lice_true_positive) / (lice_true_positive + lice_false_negative + 0.0)
    
    print 'Precision = %0.2f, Recall = %0.2f' % (lice_precision, lice_recall)
    print n_lice, n_nonlice, lice_tp, lice_fn, nonlice_tn, nonlice_fp
            
if __name__ == '__main__':
    base_directory = '/root/bryton/aquabyte_sealice'

    annotations_file = '%s/annotations.csv' % (base_directory, )
    svm_mdl_filepath = '%s/models/sealice_detection_ORB_SVM_model.yml' % (base_directory, )

    #video_filename = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Videos\\testfile_piece_29.mp4"
    #anno_xml_filename = "C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\\Annotations\\testfile_piece_29_Lice_VIPER.xml"
    
    predict_sealice(svm_mdl_filepath, annotations_file)
    