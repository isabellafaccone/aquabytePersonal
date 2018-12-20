'''
Reads in the ORB descriptor data and trains a SVM classifier
'''

import os
import svm

if __name__ == '__main__':
    base_directory = '/root/bryton/aquabyte_sealice'

    model_directory = '%s/models' % (base_directory, )
    orb_output_directory = '%s/orb_output' % (base_directory, )
    svm_output_directory = '%s/svm_output' % (base_directory, )

    try: 
        os.makedirs(svm_output_directory)
    except OSError:
        if not os.path.isdir(svm_output_directory):
            raise

    descr_type = 'ORB'
    train_start_index = 0
    train_end_index = 100
    
    sealice_SVM_trainer = svm.SealiceSVMTrainer(model_directory, orb_output_directory, svm_output_directory, descr_type, train_start_index, train_end_index)
    
    sealice_SVM_trainer.prepare_data_and_train(False)