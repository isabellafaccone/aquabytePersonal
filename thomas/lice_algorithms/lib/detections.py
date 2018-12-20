'''
    Creates lice detections
'''
def create_lice_detection(x1, y1, x2, y2, confidence):
    lice_detection = {
        'site_fsid': 1,
        'pen_state_fsid': 1,
        'box_software_fsid': 1,
        'fish_detection_bqid': 1,
        'lice_annotated_fish_detection_bqid': None,
        'stereo_frame_pair_bqid': None,
        'stage': 'adult_female',
        'length': None,
        'width': None,
        'bounding_box_x1_on_fish_image': x1,
        'bounding_box_y1_on_fish_image': x2,
        'bounding_box_x2_on_fish_image': y1,
        'bounding_box_y2_on_fish_image': y2,
        'lice_centroid_x_on_fish_image': (x1 + x1) / 2,
        'lice_centroid_y_on_fish_image': (y1 + y2) / 2,
        'lice_position_on_fish_r': None,
        'lice_position_on_fish_theta': None,
        'lice_position_on_fish_z': None,
        'location_on_fish': None,
        'detection_type': 'brain', # Alok - not sure what this is
        'confidence': confidence
    }
    
    return lice_detection


'''
    Creates fish detection
    
    Fish detections will be populated somewhere else...
    Lice detections are a list of dicts { x1, x2, y1, x2, confidence }
'''
def create_fish_detection(raw_fish_detection, raw_lice_detections):
    fish_detection = {
        'site_fsid': 1,
        'pen_state_fsid': 1,
        'pen_cohort_fsid': 1,
        'box_software_fsid': 1,
        'species': 'salmon', # Alok - should this be capitalized or not

        'biomass': 1,
        'total_length': None,
        'fork_length': None,
        'standard_length': None,
        'width': None,
        'breadth': None,
        'girth': None,
        'lateral_area': None,
        'condition_factor': None,
        'volume': None,

        'num_lice': 1,
        'num_chalimus_I_lice': None,
        'num_chalimus_II_lice': None,
        'num_preadult_I_male_lice': None,
        'num_preadult_I_female_lice': None,
        'num_preadult_II_male_lice': None,
        'num_preadult_II_female_lice': None,
        'num_adult_male_lice': None,
        'num_adult_female_lice': 1,
        'num_moving_lice': 0,
        'num_fixed_lice': 0,
        'num_scottish_lice': None,

        'measurement_bqid': 1,
        'stereo_frame_pair_bqid': None,
        'epoch': 1,
        'utc_timestamp': 1,
        'timezone': 'Europe/Oslo',
        'position_r': 1,
        'position_theta': 1,
        'position_z': 1,
        'velocity_r': None,
        'velocity_theta': None,
        'velocity_z': None,
        'orientation_theta': None,
        'orientation_phi': None,
        'orientation_roll_angle': None,
        'visible_side': None,
        'visible_side_type': None,
        'bounding_box_x1': None,
        'bounding_box_y1': None,
        'bounding_box_x2': None,
        'bounding_box_y2': None,
        'fish_image_url': None,
        'fish_image_pixel_width': None,
        'fish_image_pixel_height': None,
        'fish_image_file_size': None,
        'segmentation_polygon': None,
        'segmentation_mask_url': None,
        'centroid_x': None,
        'centroid_y': None,
        'sex': None,
        'has_wounds': None,
        'has_matured': None,
        'fish_keypoints': None
    }
    
    lice_detections = []

    for raw_lice_detection in raw_lice_detections:
        lice_detection = create_lice_detection(raw_lice_detection['x1'], raw_lice_detection['y1'], raw_lice_detection['x2'], raw_lice_detection['x2'], raw_lice_detection['confidence'])
        lice_detections.append(lice_detection)
        
    return {
        'fish_detection': fish_detection,
        'lice_detections': lice_detections
    }