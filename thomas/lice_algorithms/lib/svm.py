import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
from PIL import Image

from datetime import datetime
import utils

''' 
    The SVM used to train the model
'''
class SealiceSVMTrainer(object):
    def __init__(self, model_directory, orb_output_directory, svm_output_directory, descriptor_type, train_indices):
        self.model_directory = model_directory
        self.orb_output_directory = orb_output_directory
        self.svm_output_directory = svm_output_directory
        self.descriptor_type = descriptor_type
        self.train_indices = train_indices
    
    def prepare_data_and_train(self, saveToFile):
        """ 
        update doc string later
        """
        if self.descriptor_type == 'ORB':
            lice_train_data = self.collect_lice_ORB_train_data(saveToFile)
            print '%i lice feature descriptors' % (np.shape(lice_train_data)[0], )

            nonlice_train_data = self.collect_nonlice_ORB_train_data(saveToFile)
            print '%i nonlice feature descriptors' % (np.shape(nonlice_train_data)[0], )

            train_data, train_labels = self.prepare_traindata_labels(lice_train_data, nonlice_train_data)

            return self.train_and_save_SVM_model(np.array(train_data, np.float32), np.array(train_labels, np.int32), saveToFile)
        elif self.descriptor_type == 'HAAR':
            dummy = 0
        
    def collect_lice_ORB_train_data(self, saveToFile = False):
        lice_SVM_train_data = np.zeros((1, 32), dtype=np.uint8) # 32 is the ORB descriptor size
        
        orb_files = os.listdir(self.orb_output_directory)
        
        for train_index in self.train_indices:
            for orb_file in orb_files:
                file_split = orb_file.split('_')

                if str(train_index) in file_split and 'lice' in file_split: # only looking for files with lice ORB
                    orb_descriptors = np.load(os.path.join(self.orb_output_directory, orb_file))

                    if np.shape(orb_descriptors)[0] > 0:
                        print 'Adding %i lice feature descriptors: %i -> %i' % (np.shape(orb_descriptors)[0], np.shape(lice_SVM_train_data)[0], np.shape(orb_descriptors)[0] + np.shape(lice_SVM_train_data)[0])

                        
                        lice_SVM_train_data = np.concatenate((lice_SVM_train_data, orb_descriptors), axis=0)

        if saveToFile:
            SVM_output_file = '%s/lice_SVM_train_data.npy' % (self.svm_output_directory,)

            np.save(SVM_output_file, lice_SVM_train_data)

        return lice_SVM_train_data  
        
    def collect_nonlice_ORB_train_data(self, saveToFile = False):
        nonlice_SVM_train_data = np.zeros((1, 32), dtype=np.uint8) # 32 is the ORB descriptor size

        orb_files = os.listdir(self.orb_output_directory)
        
        for train_index in self.train_indices:
            for orb_file in orb_files:
                file_split = orb_file.split('_')
                
                if str(train_index) in file_split and 'nonlice' in file_split:
                    orb_descriptors = np.load(os.path.join(self.orb_output_directory, orb_file))

                    if np.shape(orb_descriptors)[0] > 0:
                        print 'Adding %i nonlice feature descriptors: %i -> %i' % (np.shape(orb_descriptors)[0], np.shape(nonlice_SVM_train_data)[0], np.shape(orb_descriptors)[0] + np.shape(nonlice_SVM_train_data)[0])
                        
                        nonlice_SVM_train_data = np.concatenate((nonlice_SVM_train_data, orb_descriptors), axis=0)
    
        if saveToFile:
            SVM_output_file = '%s/nonlice_SVM_train_data.npy' % (self.svm_output_directory,)
            
            np.save(SVM_output_file, nonlice_SVM_train_data)
    
        return nonlice_SVM_train_data[1:]

    '''
    Make sure we have the same number of feature descriptors in each, and shuffle
    '''
    def prepare_traindata_labels(self, lice_SVM_train_data, nonlice_SVM_train_data):
        num_lice, num_dims = np.shape(lice_SVM_train_data)
        num_nonlice, num_dims = np.shape(nonlice_SVM_train_data)
        num_samples = np.min([num_lice, num_nonlice])
        
        if num_samples == num_lice:
            train_data = np.asfarray(lice_SVM_train_data, dtype = 'float')
            train_labels = np.ones((num_samples, 1))
            rand_indices = np.random.choice(np.arange(num_nonlice), num_samples, replace=False)
            train_data = np.concatenate((train_data, nonlice_SVM_train_data[rand_indices, :]), axis = 0)
            train_labels = np.concatenate((train_labels, np.zeros(np.shape(train_labels))), axis = 0)

        if num_samples == num_nonlice:
            train_data = np.asfarray(nonlice_SVM_train_data, dtype = 'float')
            train_labels = -1 * np.ones((num_samples, 1))
            rand_indices = np.random.choice(np.arange(num_lice), num_samples, replace=False)
            train_data = np.concatenate((train_data, lice_SVM_train_data[rand_indices, :]), axis = 0)
            train_labels = np.concatenate((train_labels, np.ones(np.shape(train_labels))), axis = 0)
            
        shuffle_indices = np.random.choice(np.arange(np.shape(train_data)[0]), np.shape(train_data)[0], replace = False)
        train_data = train_data[shuffle_indices, :]
        train_labels = train_labels[shuffle_indices]

        train_labels = train_labels.astype(int)
        
        return train_data, train_labels
    
    def train_and_save_SVM_model(self, train_data, train_labels, save = True):
        print 'Training model'
        
        # Set up SVM for OpenCV 3
        model = cv2.ml.SVM_create()
        # Set SVM type
        model.setType(cv2.ml.SVM_C_SVC)
        # Set SVM Kernel to Radial Basis Function (RBF) 
        model.setKernel(cv2.ml.SVM_RBF)
       
        C_grid = model.getDefaultGridPtr(cv2.ml.SVM_C)
        C_grid.minVal = 1e-10
        C_grid.maxVal = 1e10
        C_grid.logStep = 2

        Gamma_grid = model.getDefaultGridPtr(cv2.ml.SVM_GAMMA)
        Gamma_grid.minVal = 1e-10
        Gamma_grid.maxVal = 1e10
        Gamma_grid.logStep = 2

        model.trainAuto(train_data, cv2.ml.ROW_SAMPLE, train_labels, 10, C_grid, Gamma_grid)

        if save:
            # Save trained model 
            current_datetime = '{:%Y%m%d-%H%M%S}'.format(datetime.now())
            model_output_file = '%s/sealice_detection_ORB_SVM_model_%s.yml' % (self.model_directory, current_datetime)

            model.save(model_output_file)

            print 'Saving SVM model at %s' % (model_output_file, )
        
            return model_output_file
        
def predict_sealice(svm_model_filepath, annotations, indices, display = False):
    num_lice = 0
    num_nonlice = 0
    lice_true_positive = 0
    lice_false_negative = 0
    nonlice_true_negative = 0
    nonlice_false_positive = 0

    # load the trained SVM model
    svm_model = cv2.ml.SVM_load(svm_model_filepath)

    # create an ORB instance
    orb_descriptor = cv2.ORB_create()
    
    if display:
        f, ax = plt.subplots(5, 2, figsize=(20, 40))

    processed_index = -1

    # for each video frame, compute ORB descr and predict
    for annotation_index, annotation in enumerate(annotations):
        if annotation_index % 10 == 0:
            print 'Processing frame %i of %i' % (annotation_index, len(annotations))

        image_filename, x1, y1, x2, y2, label = annotation

        center_x = int(np.round((x1 + x2) / 2.0))
        center_y = int(np.round((y1 + y2) / 2.0))

        try:
            frame = np.array(Image.open(image_filename))
        except:
            print 'Image not found: %s' % (image_filename, )
            continue
            
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

        if descr is None:
            continue
        else:
            processed_index = processed_index + 1 

        is_correct = False

        if utils.is_lice(label):
            num_lice = num_lice + 1

            color = (0,255,0) # LICE is GREEN
            opp_color = (255,0,0)

            cv2.rectangle(frame_copy, tl, br, color, 3)

            descr = np.asfarray(descr, dtype = 'float32')
            pred_label = svm_model.predict(descr)
            #print frame_num, pred_label[1][0][0], anno[-1]
            if pred_label[1][0][0] == 1:
                lice_true_positive = lice_true_positive + 1
                cv2.rectangle(frame_copy, tl2, br2, color, 3)
                is_correct = True
            elif pred_label[1][0][0] == -1:
                lice_false_negative = lice_false_negative + 1
                cv2.rectangle(frame_copy, tl2, br2, opp_color, 3)
            else:
                print 'Found unknown value: %i' % (pred_label[1][0][0], )
        else:
            num_nonlice = num_nonlice + 1

            color = (255,0,0)
            opp_color = (0,255,0)

            cv2.rectangle(frame_copy, tl, br, color, 3)

            descr = np.asfarray(descr, dtype = 'float32')
            pred_label = svm_model.predict(descr)
            #print frame_num, pred_label[1][0][0], anno[-1]
            if pred_label[1][0][0] == -1:
                nonlice_true_negative = nonlice_true_negative + 1
                cv2.rectangle(frame_copy, tl2, br2, color, 3)
                is_correct = True
            elif pred_label[1][0][0] == 1:
                nonlice_false_positive = nonlice_false_positive + 1
                cv2.rectangle(frame_copy, tl2, br2, opp_color, 3)
            else:
                print 'Found unknown value: %i' % (pred_label[1][0][0], )

        if display: 
            if processed_index < 10:
                ax[processed_index / 2][processed_index % 2].imshow(frame_copy)
                if is_correct:
                    ax[processed_index / 2][processed_index % 2].set_title('Correct')
                else:
                    ax[processed_index / 2][processed_index % 2].set_title('Incorrect')

    lice_precision = (lice_true_positive) / (lice_true_positive + nonlice_false_positive + 0.0)
    lice_recall = (lice_true_positive) / (lice_true_positive + lice_false_negative + 0.0)

    if display:
        #plt.tight_layout()
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)
        plt.show()

        plt.close()
        
        print 'Wait for the images...'

    return {
        'precision': lice_precision,
        'recall': lice_recall,
        'num_lice': num_lice,
        'num_nonlice': num_nonlice,
        'lice_true_positive': lice_true_positive, 
        'lice_false_negative': lice_false_negative,
        'nonlice_true_negative': nonlice_true_negative,
        'nonlice_false_positive': nonlice_false_positive
    }