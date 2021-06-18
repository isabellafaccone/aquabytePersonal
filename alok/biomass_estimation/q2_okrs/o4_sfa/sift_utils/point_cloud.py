import numpy as np
from picture import Picture
from akpr import in_hull, apply_convex_hull_filter, get_homography_and_matches


def get_bbox(kps, bp_list):
    x_min = min([kps[bp][0] for bp in bp_list])
    x_max = max([kps[bp][0] for bp in bp_list])
    y_min = min([kps[bp][1] for bp in bp_list])
    y_max = max([kps[bp][1] for bp in bp_list])
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    return bbox


def generate_point_cloud(ann, left_image_arr, right_image_arr):
    left_kps = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in ann['leftCrop']}
    right_kps = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in ann['rightCrop']}
    left_kps['DORSAL_FIN_OFFSET'] = [left_kps['DORSAL_FIN'][0] - 200, left_kps['DORSAL_FIN'][1] + 100]
    right_kps['DORSAL_FIN_OFFSET'] = [right_kps['DORSAL_FIN'][0] - 200, right_kps['DORSAL_FIN'][1] + 100]
    left_kps['PECTORAL_FIN_OFFSET'] = [left_kps['PECTORAL_FIN'][0] + 400, \
        min(left_kps['PECTORAL_FIN'][1] + 200, left_image_arr.shape[0]-1)]
    right_kps['PECTORAL_FIN_OFFSET'] = [right_kps['PECTORAL_FIN'][0] + 400, \
        min(right_kps['PECTORAL_FIN'][1] + 200, right_image_arr.shape[0]-1)]
    left_kps['EYE_OFFSET'] = [left_kps['EYE'][0] + 200, left_kps['EYE'][1] - 200]
    right_kps['EYE_OFFSET'] = [right_kps['EYE'][0] + 200, right_kps['EYE'][1] - 200]

    left_crop_metadata = {
        'x_coord': ann['leftCrop'][0]['xFrame'] - ann['leftCrop'][0]['xCrop'],
        'y_coord': ann['leftCrop'][0]['yFrame'] - ann['leftCrop'][0]['yCrop']
    }
    right_crop_metadata = {
        'x_coord': ann['rightCrop'][0]['xFrame'] - ann['rightCrop'][0]['xCrop'],
        'y_coord': ann['rightCrop'][0]['yFrame'] - ann['rightCrop'][0]['yCrop']
    }

    left_fish_picture = Picture(image_arr=left_image_arr)
    right_fish_picture = Picture(image_arr=right_image_arr)
    left_fish_picture.enhance(in_place=True)
    right_fish_picture.enhance(in_place=True)

    section_bps_list_set = [
        ['TAIL_NOTCH', 'ANAL_FIN', 'PELVIC_FIN'],
        ['ADIPOSE_FIN', 'ANAL_FIN', 'PELVIC_FIN', 'DORSAL_FIN'],
        ['TAIL_NOTCH', 'ADIPOSE_FIN', 'ANAL_FIN'],
        ['PELVIC_FIN', 'DORSAL_FIN', 'PECTORAL_FIN'],
        ['PECTORAL_FIN', 'UPPER_LIP'],
        ['DORSAL_FIN', 'UPPER_LIP'],
        ['DORSAL_FIN', 'ADIPOSE_FIN'],
        ['DORSAL_FIN', 'PECTORAL_FIN'],
        ['DORSAL_FIN', 'DORSAL_FIN_OFFSET'],
        ['PELVIC_FIN', 'ANAL_FIN'],
        ['PECTORAL_FIN', 'PECTORAL_FIN_OFFSET'],
        ['EYE', 'EYE_OFFSET']
    ]

    left_corner_x = ann['leftCrop'][0]['xFrame'] - ann['leftCrop'][0]['xCrop']
    left_corner_y = ann['leftCrop'][0]['yFrame'] - ann['leftCrop'][0]['yCrop']
    right_corner_x = ann['rightCrop'][0]['xFrame'] - ann['rightCrop'][0]['xCrop']
    right_corner_y = ann['rightCrop'][0]['yFrame'] - ann['rightCrop'][0]['yCrop']
    left_corner = np.array([left_corner_x, left_corner_y])
    right_corner = np.array([right_corner_x, right_corner_y])

    left_points, right_points = [], []
    for bp_list in section_bps_list_set:
        left_bbox = get_bbox(left_kps, bp_list)
        right_bbox = get_bbox(right_kps, bp_list)
        left_patch = left_fish_picture.generate_crop(*left_bbox)
        right_patch = right_fish_picture.generate_crop(*right_bbox)
        H, kp1, kp2, good, matches_mask = get_homography_and_matches(left_patch, right_patch, left_kps, 
                                                                     right_kps, left_bbox, right_bbox)

        print(len(good), len(matches_mask))
        i = 0
        for m in good:
            if matches_mask[i] == 1:
                p1 = np.round(kp1[m.queryIdx].pt).astype(int) + np.array(left_bbox[:2]) + left_corner
                p2 = np.round(kp2[m.trainIdx].pt).astype(int) + np.array(right_bbox[:2]) + right_corner
                left_points.append(p1)
                right_points.append(p2)
            i += 1

    return np.array(left_points), np.array(right_points), left_corner, right_corner

    


    

