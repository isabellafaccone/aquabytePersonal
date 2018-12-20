'''
Gets candidate points
'''

#import imageio
import cv2
#import sys
import os
#from os import path
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np

import utils
import features
    
def detect(mask_thresh, bbox_halfwidth, area_min, area_max, ecc_min, ecc_max, svm_mdl_filepath, annotations_file):
    # vid = imageio.get_reader(self.vid_filepath,'ffmpeg')
    #vid_metadata = vid.get_meta_data()
    #nframes = vid_metadata['nframes']
    #fps = vid_metadata['fps']

    # load the saved SVM model
    svm_mdl = cv2.ml.SVM_load(svm_mdl_filepath)
    # create a ORB instance
    orb_descriptor = cv2.ORB_create()
    # create an instance of video writer
    #vidwriter = imageio.get_writer('outvid.mp4', fps=15)

    annotations = utils.get_lice_annotations_from_file(annotations_file)

    # for each frame detect keypoints
    for annotation_index, annotation in enumerate(annotations):
        print 'Processing annotation %i of %i' % (annotation_index, len(annotations))

        image_filename, x1, y1, x2, y2, label = annotation

        frame = np.array(Image.open(image_filename))
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh_mask = cv2.threshold(gray_frame, mask_thresh, 255, cv2.THRESH_BINARY)

        candidate_kps = features.extract_sealice_candidate_kps(thresh_mask, bbox_halfwidth, area_min, area_max, ecc_min, ecc_max)
        
        print candidate_kps

        kps, orb_descrs = orb_descriptor.compute(frame, candidate_kps)
        orb_descrs = np.asfarray(orb_descrs, dtype = 'float32')

        if np.size(orb_descrs) > 1:
            pred_labels = svm_mdl.predict(orb_descrs)
            pred_labels = pred_labels[1]

            for i, pred_label in enumerate(pred_labels):
                if pred_label[0] == 1:
                    tl2 = (np.int(kps[i].pt[0] - 28), np.int(kps[i].pt[1] - 28)) 
                    br2 = (np.int(kps[i].pt[0] + 28), np.int(kps[i].pt[1] + 28))
                    cv2.rectangle(frame, tl2, br2, (0, 0, 255), 3)
        #vidwriter.append_data(frame)
    #vidwriter.close()

if __name__ == '__main__':
    mask_thresh = 100
    bbox_halfwidth = 24
    area_min = 100
    area_max = 200
    ecc_min = 0.7
    ecc_max = 0.99

    base_directory = '/root/bryton/aquabyte_sealice'

    annotations_file = '%s/annotations.csv' % (base_directory, )
    svm_mdl_filepath = '%s/models/sealice_detection_ORB_SVM_model.yml' % (base_directory, )
    
    detect(mask_thresh, bbox_halfwidth, area_min, area_max, ecc_min, ecc_max, svm_mdl_filepath, annotations_file)
