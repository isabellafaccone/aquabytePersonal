import cv2

from PIL import Image
import numpy as np
import utils
import skimage
from skimage.measure import label, regionprops

#from scipy.misc import imrotate as imrotate

'''
This used to be relevant when we had polygonal annotations
Right now, it's no longer relevant because we use bounding boxes
As such, comment it out for now

It gets the center and degree of the annotation

def compute_center_ori_annotation(anno):s
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
'''

'''
This used to be relevant when we had polygonal annotations
Right now, it's no longer relevant because we use bounding boxes
As such, comment it out for now

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
'''

'''
Gets the patch from the image assuming a bounding box
'''
def get_patch_from_image(frame, center_x, center_y, half_patch_size):    
    patch_len_for_rot = np.round(np.sqrt(2) * (2 * half_patch_size + 1))
    half_patch_len_rot = np.int(np.floor(patch_len_for_rot / 2))
    
    num_rows, n_columns, n_channels = np.shape(frame)
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if center_x - half_patch_len_rot > 0 and center_x + half_patch_len_rot + 1 < num_rows \
        and center_y - half_patch_len_rot > 0 and center_y + half_patch_len_rot + 1 < n_columns :
            larger_patch_rgb = np.asarray(frame[center_x - half_patch_len_rot:center_x + half_patch_len_rot + 1, \
                                                center_y - half_patch_len_rot:center_y + half_patch_len_rot + 1, :])
            larger_patch_gray = cv2.cvtColor(larger_patch_rgb, cv2.COLOR_BGR2GRAY)

            num_rows_patch, num_columns_patch, num_channels_patch = np.shape(larger_patch_rgb)
            
            larger_patch_center_x = np.int(np.floor(num_rows_patch / 2.0) + 1)
            larger_patch_center_y = np.int(np.floor(num_columns_patch / 2.0) + 1)

            patch_rgb = larger_patch_rgb[larger_patch_center_x - half_patch_size : larger_patch_center_x + half_patch_size + 1, \
                                         larger_patch_center_y - half_patch_size : larger_patch_center_y + half_patch_size + 1, :]
            patch_gray = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)

            '''
            No need to rotate, we use bounding boxes

            temp_rgb_rot = imrotate(larger_patch_rgb, ori)
            rot_patch_rgb = temp_rgb_rot[larger_patch_center_x - half_patch_size : larger_patch_center_x + half_patch_size+1, \
                                         larger_patch_center_y - half_patch_size : larger_patch_center_y + half_patch_size+1, :]
            rot_patch_gray = cv2.cvtColor(rot_patch_rgb, cv2.COLOR_BGR2GRAY)
            '''
    else:
        patch_rgb = []
        patch_gray = []
        #rot_patch_rgb = []
        #rot_patch_gray = []

    return patch_rgb, patch_gray

'''
Treats annotations as the base, not frames

Adjusted to remove rotations, not needed
'''
def get_sealice_orb_descriptors(annotations_subset, half_patch_size, num_annotations):
    orb_descriptor = cv2.ORB_create()
    # haar_descriptor = # HAAR and LBP cascade classifier training requires 
    # data preparation in a special way, too time consuming for now. Will be 
    # done if required
    
    all_frames_lice_kps = []
    all_frames_nonlice_kps = []
    all_frames_lice_rgb_patches = [] 
    all_frames_nonlice_rgb_patches = [] 
    #all_frames_lice_rotated_rgb_patches = []
    #all_frames_nonlice_rotated_rgb_patches = []
    all_frames_lice_gray_patches = [] 
    all_frames_nonlice_gray_patches = [] 
    #all_frames_lice_rotated_gray_patches = []
    #all_frames_nonlice_rotated_gray_patches = []
    all_frames_lice_ORB_descr = []
    all_frames_nonlice_ORB_descr = []

    for annotation_index, annotation in enumerate(annotations_subset):
        if annotation_index % 10 == 0:
            print 'Processing annotation %i of %i' % (annotation_index, num_annotations)
        
        image_filename, x1, y1, x2, y2, label = annotation
        
        center_x = int(np.round((x1 + x2) / 2.0))
        center_y = int(np.round((y1 + y2) / 2.0))
    
        try:
            frame = np.array(Image.open(image_filename))
        except:
            print 'Image not found: %s' % (image_filename, )
            continue
            
        this_frame_lice_kps = []
        this_frame_nonlice_kps = []
        this_frame_lice_rgb_patches = [] 
        this_frame_nonlice_rgb_patches = [] 
        #this_frame_lice_rotated_rgb_patches = []
        #this_frame_nonlice_rotated_rgb_patches = []
        this_frame_lice_gray_patches = [] 
        this_frame_nonlice_gray_patches = [] 
        #this_frame_lice_rotated_gray_patches = []
        #this_frame_nonlice_rotated_gray_patches = []
    
        # TODO: figure out what this means
        # I think it referred to when we had polygonal annotations, we would want to know the number of points in the annotation
        #if anno[1] > 2 and anno[1] < 5:

        #center, ori = compute_center_ori_annotation(annotation)
        patch_rgb, patch_gray = get_patch_from_image(frame, center_x, center_y, half_patch_size)

        if len(patch_rgb) > 0:
            temp_kp = cv2.KeyPoint()
            temp_kp.pt = (center_x, center_y)
                
            if utils.is_lice(label):
                this_frame_lice_kps.append(temp_kp)
                this_frame_lice_rgb_patches.append(patch_rgb)
                #this_frame_lice_rotated_rgb_patches.append(rot_patch_rgb)
                this_frame_lice_gray_patches.append(patch_gray)
                #this_frame_lice_rotated_gray_patches.append(rot_patch_gray)
            else:
                this_frame_nonlice_kps.append(temp_kp)
                this_frame_nonlice_rgb_patches.append(patch_rgb)
                #this_frame_nonlice_rotated_rgb_patches.append(rot_patch_rgb)
                this_frame_nonlice_gray_patches.append(patch_gray)
                #this_frame_nonlice_rotated_gray_patches.append(rot_patch_gray)

        nonlice_kps, nonlice_descr = orb_descriptor.compute(frame, this_frame_nonlice_kps)
        lice_kps, lice_descr = orb_descriptor.compute(frame, this_frame_lice_kps) # ORB descriptor class drops some key points, hence getting the new key points here
        
        all_frames_lice_kps.append(this_frame_lice_kps)
        all_frames_nonlice_kps.append(this_frame_nonlice_kps)
        all_frames_lice_rgb_patches.append(this_frame_lice_rgb_patches) 
        all_frames_nonlice_rgb_patches.append(this_frame_nonlice_rgb_patches) 
        all_frames_lice_gray_patches.append(this_frame_lice_gray_patches) 
        all_frames_nonlice_gray_patches.append(this_frame_nonlice_gray_patches) 

        #all_frames_lice_rotated_rgb_patches.append(this_frame_lice_rotated_rgb_patches)
        #all_frames_nonlice_rotated_rgb_patches.append(this_frame_nonlice_rotated_rgb_patches)
        #all_frames_lice_rotated_gray_patches.append(this_frame_lice_rotated_gray_patches)
        #all_frames_nonlice_rotated_gray_patches.append(this_frame_nonlice_rotated_gray_patches)
        
        if lice_descr is not None:
            all_frames_lice_ORB_descr.append(lice_descr)
        if nonlice_descr is not None:
            all_frames_nonlice_ORB_descr.append(nonlice_descr)
           
    return {
        #sorted_licelist, 
        'all_frames_lice_kps': all_frames_lice_kps, 
        'all_frames_nonlice_kps': all_frames_nonlice_kps, 
        'all_frames_lice_rgb_patches': all_frames_lice_rgb_patches, 
        'all_frames_nonlice_rgb_patches': all_frames_nonlice_rgb_patches, 
        #all_frames_lice_rotated_rgb_patches, 
        #all_frames_nonlice_rotated_rgb_patches, 
        'all_frames_lice_gray_patches': all_frames_lice_gray_patches, 
        'all_frames_nonlice_gray_patches': all_frames_nonlice_gray_patches, 
        #all_frames_lice_rotated_gray_patches, 
        #all_frames_nonlice_rotated_gray_patches, 
        'all_frames_lice_ORB_descr': all_frames_lice_ORB_descr, 
        'all_frames_nonlice_ORB_descr': all_frames_nonlice_ORB_descr
    }
     
def extract_sealice_candidate_kps(thresh_mask, bbox_halfwidth, area_min, area_max, ecc_min, ecc_max):
    num_rows = np.size(thresh_mask, 0)
    num_columns = np.size(thresh_mask, 1) 

    if np.max(thresh_mask, axis = None) == 255:
        thresh_mask = np.where(thresh_mask == 255, 1, 0)
        
    labels = skimage.measure.label(thresh_mask, background = 0)
    regions = skimage.measure.regionprops(labels)

    sealice_candidate_kps = []

    for region in regions:
        area_condition = region.area > area_min and region.area < area_max
        eccentricity_condition = region.eccentricity > ecc_min and region.eccentricity < ecc_max
        bounding_condition_row = region.centroid[0] - bbox_halfwidth > 0 and region.centroid[0] - bbox_halfwidth < num_rows
        bounding_condition_column = region.centroid[1] - bbox_halfwidth > 1 and region.centroid[1] - bbox_halfwidth < num_columns

        if area_condition and eccentricity_condition and bounding_condition_row and bounding_condition_column:
            temp_kp = cv2.KeyPoint()
            temp_kp.pt = (region.centroid[1], region.centroid[0]) # TODO: check if this is correct - it looks opposite
            sealice_candidate_kps.append(temp_kp)
    
    return sealice_candidate_kps


