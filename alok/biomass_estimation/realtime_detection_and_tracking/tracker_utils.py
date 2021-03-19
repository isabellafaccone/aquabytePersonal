from dataclasses import dataclass
import os

import cv2
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


S3_DIR = '/root/data/s3'
# OUTPUT_DIR = '/root/data/alok/biomass_estimation/playground/gopro_anns'
OUTPUT_DIR = '/root/data/alok/biomass_estimation/playground/gopro_anns_ablated'


# class FishObject:
#     def __init__(self, centroid, id, frame_age=0):
#         self.centroid = centroid
#         self.id = id
#         self.frame_age = frame_age

#     def __update(new_centroid)


# def hungarian_centroid_matcher(centroids_1, centroids_2):

#     # match the bboxes. Return a list of matched bboxes
#     COST_THRESHOLD = 100.0

#     pairs = []
#     ids_1 = list(range(len(centroids_1)))
#     ids_2 = list(range(len(centroids_2)))
    
#     if centroids_1 and centroids_2:
        
#         # pairwise euclidean distance matrix
#         cost_matrix = cdist(centroids_1, centroids_2, metric='euclidean')

#         # hungarian algorithm to minimize weights in bipartite graph
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)

#         # move matched items from left_ids/right_ids to pairs
#         for (r, c) in zip(row_ind, col_ind):
#             if cost_matrix[r, c] < COST_THRESHOLD:
#                 pairs.append((ids_1[r], ids_2[c]))
#                 ids_1[r] = None
#                 ids_2[c] = None

#     # unmatched singles
#     lefts = [(key, None) for key in ids_1 if key]
#     rights = [(None, key) for key in ids_2 if key]

#     # merge all into pairs as final result
#     pairs.extend(lefts)
#     pairs.extend(rights)
#     return pairs


def process_annotated_crops(annotated_crops):
    
    processed_annotations = []
    for ann in annotated_crops:
        c1 = ann['xCrop']
        c2 = ann['yCrop']
        c3 = ann['xCrop'] + ann['width']
        c4 = ann['yCrop'] + ann['height']

        processed_annotation = { 'bbox': [c1, c2, c3, c4] }
        processed_annotations.append(processed_annotation)

    return processed_annotations



def overlay_crops_on_image(image_f, crops):

    # draw boxes on images
    im = Image.open(image_f)
    draw = ImageDraw.Draw(im)
    for ann in crops:
        c1, c2, c3, c4, c5 = ann
        draw.rectangle([(c1, c2), (c3, c4)], outline='red')
        draw.text((0.5 * (c1 + c3), 0.5 * (c2 + c4)), str(c5))

    output_f = image_f.replace(S3_DIR, OUTPUT_DIR)
    if not os.path.exists(os.path.dirname(output_f)):
        os.makedirs(os.path.dirname(output_f))
    im.save(output_f)

    return output_f



def stitch_frames_into_video(image_fs, video_f):
    im = cv2.imread(image_fs[0])
    height, width, layers = im.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter(video_f, fourcc, 20, (width, height), True)
    video = cv2.VideoWriter(video_f, fourcc, 2, (width, height), True)
    for idx, image_f in enumerate(image_fs):
        if idx % 10 == 0:
            print(idx)
        im = cv2.imread(image_f, cv2.IMREAD_COLOR)
        video.write(im)
    cv2.destroyAllWindows()
    video.release()


