# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:24:59 2018

@author: sramena1

To read in ORB descriptor data and train a SVM classifier
"""
import sys
import os
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np
import cv2
import fnmatch

class sealiceSVMtrain(object):
    """ 
        update doc string later
    """
    def __init__(self,base_dir, video_str, video_list, descr_type):
        """ 
        update doc string later
        """
        self.base_dir = base_dir
        self.video_str = video_str
        self.video_list = video_list
        self.descr_type = descr_type
    
    def prepare_data_and_train(self):
        """ 
        update doc string later
        """
        if self.descr_type == 'ORB':
            lice_SVM_train_data = collect_lice_ORB_train_data(self)
            nonlice_SVM_train_data = collect_nonlice_ORB_train_data(self)
            train_data, train_labels = prepare_traindata_labels(self,lice_SVM_train_data, nonlice_SVM_train_data)
            trainSVMmodel(self,train_data,train_labels)
        elif self.descr_type == 'HAAR':
            dummy = 0
        
    def collect_lice_ORB_train_data(self):
        """ 
        update doc string later
        """
        lice_SVM_train_data = np.zeros((1,32),dtype=np.uint8) # 32 is the ORB descriptor size
        files = os.listdir(self.base_dir)
        for vid_num in self.video_list:
            for ffile in files:
                file_split = ffile.split('_')
                if str(vid_num) in file_split and 'lice' in file_split:
                    orb_descrs = np.load(os.path.join(self.base_dir,ffile))
                    print len(orb_descrs), len(orb_descrs[0])
                    print np.shape(lice_SVM_train_data), np.shape(orb_descrs)
                    lice_SVM_train_data = np.concatenate((lice_SVM_train_data,orb_descrs),axis=0)
                           
        return lice_SVM_train_data[1:]   
            
    def collect_lice_ORB_train_data_2(self):
        """ 
        update doc string later
        """
        lice_SVM_train_data = np.zeros((1,32),dtype=np.uint8) # 32 is the ORB descriptor size
        files = os.listdir(self.base_dir)
        for vid_num in self.video_list:
            for ffile in files:
                file_split = ffile.split('_')
                if str(vid_num) in file_split and 'lice' in file_split:
                    orb_descrs = np.load(os.path.join(self.base_dir,ffile))
                    for des in orb_descrs:
                        if des is not None:
                            lice_SVM_train_data = np.concatenate((lice_SVM_train_data,des),axis=0)
                           
        return lice_SVM_train_data[1:]
        
    def collect_nonlice_ORB_train_data(self):
        """ 
        update doc string later
        """
        nonlice_SVM_train_data = np.zeros((1,32),dtype=np.uint8) # 32 is the ORB descriptor size
        files = os.listdir(self.base_dir)
        for vid_num in self.video_list:
            for ffile in files:
                file_split = ffile.split('_')
                if str(vid_num) in file_split and 'nonlice' in file_split:
                    orb_descrs = np.load(os.path.join(self.base_dir,ffile))
                    nonlice_SVM_train_data = np.concatenate((nonlice_SVM_train_data,orb_descrs),axis=0)
    
        return nonlice_SVM_train_data[1:]
    
    def collect_nonlice_ORB_train_data_2(self):
        """ 
        update doc string later
        """
        nonlice_SVM_train_data = np.zeros((1,32),dtype=np.uint8) # 32 is the ORB descriptor size
        files = os.listdir(self.base_dir)
        for vid_num in self.video_list:
            for ffile in files:
                file_split = ffile.split('_')
                if str(vid_num) in file_split and 'nonlice' in file_split:
                    orb_descrs = np.load(os.path.join(self.base_dir,ffile))
                    for des in orb_descrs:
                        if des is not None:
                            nonlice_SVM_train_data = np.concatenate((nonlice_SVM_train_data, des),axis=0)
    
        return nonlice_SVM_train_data[1:]

    def prepare_traindata_labels(self,lice_SVM_train_data, nonlice_SVM_train_data):
        """ 
        update doc string later
        """        
        n_lice, ndims = np.shape(lice_SVM_train_data)
        n_nonlice, ndims = np.shape(nonlice_SVM_train_data)
        n_samples = np.min([n_lice,n_nonlice])
        
        if n_samples == n_lice:
            train_data = np.asfarray(lice_SVM_train_data, dtype = 'float')
            train_labels = np.ones((n_samples,1))
            #rand_indices = np.random.choice(np.arange(n_nonlice), n_samples, replace=False)
            #train_data = np.concatenate((train_data, nonlice_SVM_train_data[rand_indices, :]), axis = 0)
            #train_labels = np.concatenate((train_labels, np.zeros((n_samples, 1)), axis = 0)
            #labels2 = np.zeros(np.shape(train_labels))
            train_data = np.concatenate((train_data, nonlice_SVM_train_data), axis = 0)
            train_labels = np.concatenate((train_labels, -1*np.ones((n_nonlice,1))), axis = 0)
            #train_labels = np.concatenate((train_labels, np.zeros(np.shape(train_labels))), axis = 0)
            
        if n_samples == n_nonlice:
            train_data = np.asfarray(nonlice_SVM_train_data, dtype = 'float')
            #train_labels = np.zeros((n_samples,1))
            train_labels = -1*np.ones((n_samples,1))
            #rand_indices = np.random.choice(np.arange(n_lice), n_samples, replace=False)
            #train_data = np.concatenate((train_data, lice_SVM_train_data[rand_indices, :]), axis = 0)
            train_data = np.concatenate((train_data, lice_SVM_train_data), axis = 0)
            train_labels = np.concatenate((train_labels, np.ones((n_lice,1))), axis = 0)
            #train_labels = np.concatenate((train_labels, np.ones(np.shape(train_labels))), axis = 0)
            
#        shuffle_indices = np.random.choice(np.arange(np.shape(train_data)[0]), np.shape(train_data)[0], replace = False)
#        train_data = train_data[shuffle_indices,:]
#        train_labels = train_labels[shuffle_indices]
        print np.shape(train_data), np.shape(train_labels)
        return train_data, train_labels
    
    def trainSVMmodel(self,train_data,train_labels):
        """ 
        update doc string later
        """
        # Set up SVM for OpenCV 3
        mdl = cv2.ml.SVM_create()
        # Set SVM type
        mdl.setType(cv2.ml.SVM_C_SVC)
        # Set SVM Kernel to Radial Basis Function (RBF) 
        mdl.setKernel(cv2.ml.SVM_RBF)
        
        # Set parameter C
        #mdl.setC(C)
        # Set parameter Gamma
        #mdl.setGamma(gamma)
        # Train SVM on training data  
        #mdl.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
        mdl.trainAuto(train_data, cv2.ml.ROW_SAMPLE, train_labels)
        # Save trained model 
        mdl.save("sealice_detection_ORB_SVM_model_5.yml")
        
def main():
    base_dir = 'C:\\Users\\srame\\Documents\\Python Scripts\\aquabyte\\aquabyte_ml\\aquabyte_python\\algorithms\\sealice\\data\patches_and_descriptors\\ORB_data'
    video_str = 'testfile_piece_'
    video_list = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    descr_type = 'ORB'
    sealice_svm_trainer = sealiceSVMtrain(base_dir,video_str,video_list,descr_type)
    lice_train_data = sealice_svm_trainer.collect_lice_ORB_train_data_2()
    print np.shape(lice_train_data)
    nonlice_train_data = sealice_svm_trainer.collect_nonlice_ORB_train_data_2()
    print np.shape(nonlice_train_data)
    data, labels = sealice_svm_trainer.prepare_traindata_labels(lice_train_data,nonlice_train_data)
    labels = labels.astype(int)
    print np.shape(data), data.dtype, np.shape(labels), labels.dtype
    sealice_svm_trainer.trainSVMmodel(np.array(data, np.float32), np.array(labels, np.int32))

if __name__ == '__main__':
    main()