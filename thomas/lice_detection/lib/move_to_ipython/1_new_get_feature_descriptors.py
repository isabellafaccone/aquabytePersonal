'''
    Computes all of the feature descriptors from the annotations file
'''

#import sys
import os
import csv
#from os import path
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import numpy as np

from utils import get_lice_annotations_from_file
from features import get_sealice_orb_descriptors

if __name__ == '__main__':
    base_directory = '/root/bryton/aquabyte_sealice'

    annotations_file = '%s/annotations.csv' % (base_directory, )

    annotations = get_lice_annotations_from_file(annotations_file)
    num_annotations = len(annotations)

    chunk_size = 500
    num_chunks = num_annotations / chunk_size

    half_patch_size = 24

    orb_output_directory = '%s/orb_output' % (base_directory, )

    try: 
        os.makedirs(output_directory)
    except OSError:
        if not os.path.isdir(output_directory):
            raise

    for chunk_index in range(0, num_chunks + 1):
        start_annotation_index = chunk_index * chunk_size
        end_annotation_index = (chunk_index + 1) * chunk_size
        if end_annotation_index > len(num_annotations):
            end_annotation_index = num_annotations

        annotations_subset = annotations[start_annotation_index : end_annotation_index]
          
        features = get_sealice_orb_descriptors(annotations_subset, half_patch_size)
        
        all_frames_lice_ORB_descr = features['all_frames_lice_ORB_descr']
        all_frames_nonlice_ORB_descr = features['all_frames_nonlice_ORB_descr']

        '''
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
        '''

        all_frames_lice_ORB_descr = [ orb_list if orb_list is not None for orb_list in all_frames_lice_ORB_descr ]
        all_frames_lice_ORB_descr2 = [ orb if orb is not None for orb in orb_list for orb_list in all_frames_lice_ORB_descr ]
        all_frames_nonlice_ORB_descr = [ orb_list if orb_list is not None for orb_list in all_frames_nonlice_ORB_descr ]
        all_frames_nonlice_ORB_descr2 = [ orb if orb is not None for orb in orb_list for orb_list in all_frames_nonlice_ORB_descr ]

        print len(all_frames_lice_ORB_descr2), len(all_frames_nonlice_ORB_descr2)
        
        '''
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_kps', all_frames_lice_kps)
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_kps', all_frames_nonlice_kps)
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_rgb_patches', all_frames_lice_rgb_patches)
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_rgb_patches', all_frames_nonlice_rgb_patches)
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_rotated_rgb_patches', all_frames_lice_rotated_rgb_patches)
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_rotated_rgb_patches', all_frames_nonlice_rotated_rgb_patches)
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_gray_patches', all_frames_lice_gray_patches)
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_gray_patches', all_frames_nonlice_gray_patches)
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_lice_rotated_gray_patches', all_frames_lice_rotated_gray_patches)
        np.save(filename_+'_chunk_'+str(i)+'_startFrame_'+str(start_frame)+'_to_endFrame_'+str(end_frame)+'_nonlice_rotated_gray_patches', all_frames_nonlice_rotated_gray_patches)
        '''

        lice_ORB_descr_out_file = '%s/chunk_%i_lice_ORB_descr' % (orb_output_directory, chunk_index)
        nonlice_ORB_descr_out_file = '%s/chunk_%i_nonlice_ORB_descr' % (orb_output_directory, chunk_index)

        np.save(lice_ORB_descr_out_file, all_frames_lice_ORB_descr)
        np.save(nonlice_ORB_descr_out_file, all_frames_nonlice_ORB_descr)
