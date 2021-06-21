import cv2
import os
from PIL import Image, ImageDraw
from research_lib.utils.data_access_utils import S3AccessUtils

THUMBNAIL_WIDTH = 512
PIXEL_COUNT_WIDTH = 4096
PIXEL_COUNT_HEIGHT = 3000
X_PADDING_FULLRES = 190
Y_PADDING_FULLRES = 140
X_PADDING = X_PADDING_FULLRES * float(THUMBNAIL_WIDTH / PIXEL_COUNT_WIDTH)
Y_PADDING = Y_PADDING_FULLRES * float(THUMBNAIL_WIDTH / PIXEL_COUNT_HEIGHT)

S3_DIR = '/root/data/s3'
OUTPUT_DIR = '/root/data/alok/biomass_estimation/playground/crop_evaluation_video'

s3 = S3AccessUtils('/root/data')

def get_image_f(image_s3_url):
    try:
        image_url_components = image_s3_url.replace('s3://', '').split('/')
        bucket, key = image_url_components[0], os.path.join(*image_url_components[1:])
        f = s3.download_from_s3(bucket, key)
    except Exception as e:
        print(e)
        return
    
    return f


def process_detected_crops(detected_crops):
    
    processed_annotations = []
    for ann in detected_crops:
        c1 = max(ann['bbox'][0] - X_PADDING, 0)
        c2 = max(ann['bbox'][1] - Y_PADDING, 0)
        c3 = min(ann['bbox'][0] + ann['bbox'][2] + X_PADDING, THUMBNAIL_WIDTH)
        c4 = min(ann['bbox'][1] + ann['bbox'][3] + Y_PADDING, THUMBNAIL_WIDTH)
        
        processed_annotation = { 'bbox': [c1, c2, c3, c4] }
        processed_annotations.append(processed_annotation)

    return processed_annotations


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


def overlay_crops_on_image(f, detected_crops, annotated_crops, text):
    # open images and metadata files
    im = Image.open(f)

    # draw boxes on images
    draw = ImageDraw.Draw(im)
    for ann in annotated_crops:
        c1, c2, c3, c4 = ann['bbox']
        draw.rectangle([(c1, c2), (c3, c4)], outline='red')

    for ann in detected_crops:
        c1, c2, c3, c4 = ann['bbox']
        draw.rectangle([(c1, c2), (c3, c4)], outline='blue')

    draw.text((10, 10), text)
    output_f = f.replace(S3_DIR, OUTPUT_DIR)
    if not os.path.exists(os.path.dirname(output_f)):
        os.makedirs(os.path.dirname(output_f))
    im.save(output_f)

    return output_f


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
