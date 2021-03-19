import cv2
import json
import os
import sys
sys.path.append('../q1_o2kr2_dataset_annotations')
from PIL import Image, ImageDraw
from research_lib.utils.data_access_utils import S3AccessUtils
from thumbnail_selector import PEN_SITE_MAPPING, get_capture_keys
from crop_annotations import CropAnnotations
from crops_processor import match_annotations

THUMBNAIL_WIDTH = 512
PIXEL_COUNT_WIDTH = 4096
PIXEL_COUNT_HEIGHT = 3000
X_PADDING_FULLRES = 190
Y_PADDING_FULLRES = 140
X_PADDING = X_PADDING_FULLRES * float(THUMBNAIL_WIDTH / PIXEL_COUNT_WIDTH)
Y_PADDING = Y_PADDING_FULLRES * float(THUMBNAIL_WIDTH / PIXEL_COUNT_HEIGHT)

S3_DIR = '/root/data/s3'
OUTPUT_DIR = '/root/data/alok/biomass_estimation/playground/match_evaluation_video'
s3 = S3AccessUtils('/root/data')


def stitch_frames_into_video(image_fs, video_f):
    im = cv2.imread(image_fs[0])
    height, width, layers = im.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_f, fourcc, 1, (width, height), True)
    for idx, image_f in enumerate(image_fs):
        if idx % 1000 == 0:
            print(idx)
        im = cv2.imread(image_f, cv2.IMREAD_COLOR)
        video.write(im)
    cv2.destroyAllWindows()
    video.release()


def get_crops(left_key):
    crop_key = os.path.join(os.path.dirname(left_key), 'crops.json')
    crop_f = s3.download_from_s3('aquabyte-frames-resized-inbound', crop_key)
    crops = json.load(open(crop_f))
    return crops


def get_left_right_pair(capture_key):
    left_key = os.path.join(os.path.dirname(capture_key), 'left_frame.resize_512_512.jpg')
    right_key = os.path.join(os.path.dirname(capture_key), 'right_frame.resize_512_512.jpg')
    return left_key, right_key


def process_detected_crops(detected_crops):
    
    processed_annotations = []
    for ann in detected_crops:
        c1 = max(ann['bbox'][0] - X_PADDING, 0)
        c2 = max(ann['bbox'][1] - Y_PADDING, 0)
        c3 = min(ann['bbox'][0] + ann['bbox'][2] + X_PADDING, THUMBNAIL_WIDTH)
        c4 = min(ann['bbox'][1] + ann['bbox'][3] + Y_PADDING, THUMBNAIL_WIDTH)
        id = ann['id']
        processed_annotation = { 'bbox': [c1, c2, c3, c4] , 'id': id}
        processed_annotations.append(processed_annotation)

    return processed_annotations


def get_centroid(crop):
    c1, c2, c3, c4 = crop['bbox']
    centroid_x = 0.5 * (c1 + c3)
    centroid_y = 0.5 * (c2 + c4)
    return (centroid_x, centroid_y)


def generate_stitched_image(left_image_f, right_image_f, crops_json, matches):
    
    # open images and metadata files
    left_im = Image.open(left_image_f)
    right_im = Image.open(right_image_f)

    # stitch images
    result = Image.new('RGB', (2 * THUMBNAIL_WIDTH, THUMBNAIL_WIDTH))
    result.paste(im=left_im, box=(0, 0))
    result.paste(im=right_im, box=(THUMBNAIL_WIDTH, 0))

    
    # draw crops

    # get category_id
    model_name = 'object-detection-bati'
    model_id = [m for m in crops_json['models'] if m['name'] == model_name][0]['id']
    categories = [c for c in crops_json['categories'] if c['model_id'] == model_id]
    if len(categories) != 1:
        return None
    category_id = categories[0]['id']
    left_anns = [crop for crop in crops_json['annotations'] if crop['image_id'] == 1 and crop['category_id'] == category_id]
    right_anns = [crop for crop in crops_json['annotations'] if crop['image_id'] == 2 and crop['category_id'] == category_id]

    left_detected_crops = process_detected_crops(left_anns)
    right_detected_crops = process_detected_crops(right_anns)

    draw = ImageDraw.Draw(result)
    for ann in left_detected_crops:
        c1, c2, c3, c4 = ann['bbox']
        draw.rectangle([(c1, c2), (c3, c4)])

    for ann in right_detected_crops:
        c1, c2, c3, c4 = ann['bbox']
        draw.rectangle([(c1 + THUMBNAIL_WIDTH, c2), (c3 + THUMBNAIL_WIDTH, c4)])

    # draw connecting lines
    for match in matches:
        left_id, right_id = match
        if left_id is None or right_id is None:
            continue
        left_id, right_id = (int(left_id), int(right_id))
        left_detected_crop = [crop for crop in left_detected_crops if crop['id'] == left_id][0]
        right_detected_crop = [crop for crop in right_detected_crops if crop['id'] == right_id][0]
        left_centroid = get_centroid(left_detected_crop)
        right_centroid = get_centroid(right_detected_crop)
        adj_right_centroid = (right_centroid[0] + THUMBNAIL_WIDTH, right_centroid[1])
        draw.line(left_centroid + adj_right_centroid)
    
    output_f = left_image_f.replace(S3_DIR, OUTPUT_DIR).replace('left_', 'stereo_')
    if not os.path.exists(os.path.dirname(output_f)):
        os.makedirs(os.path.dirname(output_f))
    result.save(output_f)    
    return output_f


def generate_match_video(pen_id, date):
    print('Getting capture keys...')
    capture_keys = get_capture_keys(pen_id, date, date)
    print('Capture keys obtained!')
    stitched_image_fs = []

    print('Starting image stitching')
    count = 0
    for capture_key in capture_keys:
        left_key, right_key = get_left_right_pair(capture_key)
        try:
            crops_json = get_crops(left_key)
        except:
            continue
        cas = CropAnnotations(crops_json=crops_json)
        matches = match_annotations(cas, 'BATI')
        left_image_f = s3.download_from_s3('aquabyte-frames-resized-inbound', left_key)
        right_image_f = s3.download_from_s3('aquabyte-frames-resized-inbound', right_key)
        print(capture_key)
        stitched_image_f = generate_stitched_image(left_image_f, right_image_f, crops_json, matches)
        if stitched_image_f:
            stitched_image_fs.append(stitched_image_f)

        if count % 10 == 0:
            print('Percentage complete: {}%'.format(round(100 * count / len(capture_keys), 2)))
        count += 1
    
    stitch_frames_into_video(stitched_image_fs, os.path.join(OUTPUT_DIR, 'match_evaluation_video.avi'))
    