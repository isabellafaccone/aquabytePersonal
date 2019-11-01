
from copy import copy
import json
import cv2
import json
import numpy as np
import boto3
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from collections import namedtuple
import psycopg2.extras
import psycopg2
import pandas as pd
import datetime


REGION="eu-west-1"
OUTPUT_BUCKET="aquabyte-research"


def enhance(image, clip_limit=5):
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert image from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image

def crop_images(imageL, imageR, left_crop_metadata, right_crop_metadata, left_keypoints_dict, 
    right_keypoints_dict, PADDING=100):
    
    # crop the data
    l_width = left_crop_metadata['width']
    l_height = left_crop_metadata['height']
    r_width = right_crop_metadata['width']
    r_height = right_crop_metadata['height']
    
    cropL_x_left = max(min([kp[0] for kp in left_keypoints_dict.values()]) - PADDING, 0)
    cropL_x_right = min(max([kp[0] for kp in left_keypoints_dict.values()]) + PADDING, l_width)
    cropL_y_top = max(min([kp[1] for kp in left_keypoints_dict.values()]) - PADDING, 0)
    cropL_y_bottom = min(max([kp[1] for kp in left_keypoints_dict.values()]) + PADDING, l_height)

    cropR_x_left = max(min([kp[0] for kp in right_keypoints_dict.values()]) - PADDING, 0)
    cropR_x_right = min(max([kp[0] for kp in right_keypoints_dict.values()]) + PADDING, r_width)
    cropR_y_top = max(min([kp[1] for kp in right_keypoints_dict.values()]) - PADDING, 0)
    cropR_y_bottom = min(max([kp[1] for kp in right_keypoints_dict.values()]) + PADDING, r_height)

    imageL = imageL[cropL_y_top:cropL_y_bottom, cropL_x_left:cropL_x_right]
    imageR = imageR[cropR_y_top:cropR_y_bottom, cropR_x_left:cropR_x_right]

    return imageL, imageR, cropL_x_left, cropL_y_top, cropR_x_left, cropR_y_top


def get_body_keypoints(kp1, kp2, matches, matchesMask, left_crop_metadata, right_crop_metadata): 
    kps = []
    for i, m in enumerate(matches):
        # Generate random color for RGB/BGR and grayscale images as needed.            
        if matchesMask[i] == 1:
            p1 = tuple(np.round(kp1[m.queryIdx].pt).astype(float))
            p2 = tuple(np.round(kp2[m.trainIdx].pt).astype(float))
            p1_x_frame = p1[0] + left_crop_metadata['x_coord']
            p1_y_frame = p1[1] + left_crop_metadata['y_coord']
            p2_x_frame = p2[0] + right_crop_metadata['x_coord']
            p2_y_frame = p2[1] + right_crop_metadata['y_coord']
            disp = abs(p1_x_frame - p2_x_frame)
            kp = [p1_x_frame, p1_y_frame, p2_x_frame, p2_y_frame]
            kps.append(kp)
        
    return kps

def get_modified_crop_metadata(cropL_x_left, cropL_y_top, cropR_x_left, cropR_y_top):
    modified_left_crop_metadata = {
        'x_coord': cropL_x_left,
        'y_coord': cropL_y_top
    }

    modified_right_crop_metadata = {
        'x_coord': cropR_x_left,
        'y_coord': cropR_y_top
    }

    return modified_left_crop_metadata, modified_right_crop_metadata



def find_matches_and_homography(imageL, imageR, keypoints, cm, left_crop_metadata, right_crop_metadata, 
    MIN_MATCH_COUNT=11, GOOD_PERC=0.7, FLANN_INDEX_KDTREE=0):

    if 'leftCrop' in keypoints and 'rightCrop' in keypoints:
        print('Keypoints: {}'.format(keypoints))
        left_keypoints_dict = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in keypoints['leftCrop']}
        right_keypoints_dict = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in keypoints['rightCrop']}
        
        # crop images to speed up the template matching process
        imageL, imageR, cropL_x_left, cropL_y_top, cropR_x_left, cropR_y_top = \
            crop_images(imageL, imageR, left_crop_metadata, right_crop_metadata, left_keypoints_dict, right_keypoints_dict)

        modified_left_crop_metadata, modified_right_crop_metadata = \
            get_modified_crop_metadata(cropL_x_left, cropL_y_top, cropR_x_left, cropR_y_top)

        # find template matches

        sift = cv2.KAZE_create()
        img1 = enhance(imageL)
        img2 = enhance(imageR)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2, k=2)
        good = []
        matchesMask = []

        for m,n in matches:
            if m.distance < GOOD_PERC*n.distance:
                good.append(m)
        if len(good)>=MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1 ,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            if H is None:
                return [[[]], [[]]]
            H = H.tolist()
            kps = get_body_keypoints(kp1, kp2, good, matchesMask, modified_left_crop_metadata, modified_right_crop_metadata)
            return [H, kps]

        else:
            print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            return [[[]], [[]]]
    return [[[]], [[]]]



DbParams = namedtuple("DbParams", "user password host port db_name")



def generate_matches_and_homography(left_crop_url, right_crop_url, keypoints, cm, left_crop_metadata, right_crop_metadata):

    imageL = load_image(left_crop_url)
    imageR = load_image(right_crop_url)
    left_crop_metadata = json.loads(left_crop_metadata)
    right_crop_metadata = json.loads(right_crop_metadata)
    cm = json.loads(cm)
    keypoints = json.loads(keypoints)
    homography_and_matches = find_matches_and_homography(imageL, imageR, keypoints, cm, left_crop_metadata, right_crop_metadata)
    return homography_and_matches


def convert_url_to_s3_bk(url):
    # old style https://s3.amazonaws.com/<bucket>/<key>
    # new style https://<bucket>.s3.amazonaws.com/<key>

    splitted = url.split("/")

    # eg s3.amazonaws.com
    first_part = splitted[2].split('.')
    if len(first_part) != 3:
        # new style
        bucket = first_part[0]
        key = "/".join(splitted[3:])
    else:
        bucket=splitted[3]
        key = "/".join(splitted[4:])

    return bucket, key

def load_image(url):
    s3_res = boto3.resource('s3', region_name=REGION)
    b, k = convert_url_to_s3_bk(url)
    obj = s3_res.Object(b, k).get()['Body']
    print("key", k)
    image = np.asarray(bytearray(obj.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def get_db_params(db_type):
    ssm_client = boto3.client('ssm', REGION)
    param_response = ssm_client.get_parameters(Names=[f'{db_type}_USER',f'{db_type}_PASSWORD',
                                                      f'{db_type}_HOST', f'{db_type}_PORT',f'{db_type}_NAME'], WithDecryption=True)

    params = param_response['Parameters']
    d = {i["Name"]:i["Value"] for i in params}
    dbParams = DbParams(d[f'{db_type}_USER'],d[f'{db_type}_PASSWORD'], d[f'{db_type}_HOST'], d[f'{db_type}_PORT'],d[f'{db_type}_NAME'])
    return dbParams

def get_data(sql, db_type):
    print(f"Using query: {sql}")
    db_params = get_db_params( db_type)
    print(db_params)
    db_name = db_params.db_name
    db_user = db_params.user
    db_host = db_params.host
    db_pass = db_params.password
    conn = psycopg2.connect(f"dbname={db_name} user={db_user} host={db_host} password={db_pass}")
    dat = pd.read_sql_query(sql, conn)
    return dat

if __name__ == '__main__':
    sc = SparkContext(appName="template_matching")
    sqlContext = SQLContext(sc)
    sql = query = """
        select left_image_url, right_image_url, keypoints::text, camera_metadata::text, left_crop_metadata::text, right_crop_metadata::text from research.fish_metadata a left join keypoint_annotations b
        on a.left_url = b.left_image_url 
        where b.keypoints is not null and b.is_qa = false;
    """
    # sql = "select left_crop_url, right_crop_url, keypoints, camera_metadata from prod.crop_annotation where annotation_state_id=1 and service_id=2 and captured_at between '2019-10-18' and '2019-10-19' and pen_id=61"
    pdf = get_data(sql, "RESEARCH_DB")

    udfValue = udf(generate_matches_and_homography, ArrayType(ArrayType(ArrayType(DoubleType()))))


    #df = sqlContext.createDataFrame(pdf).withColumn("plane", udfValue("left_crop_url", "right_crop_url")).collect()
    df = sqlContext.createDataFrame(pdf).withColumn("homography_and_matches", \
        udfValue("left_image_url", "right_image_url", "keypoints", "camera_metadata", "left_crop_metadata", "right_crop_metadata"))

    dt_iso = datetime.datetime.now().isoformat()
    key = f"template-matching/{dt_iso}/"
    df.repartition(1).write.parquet(f"s3://{OUTPUT_BUCKET}/{key}")


    # Write to pandas parquet
    #pdf = df.select("*").toPandas()

    #file_name = "/tmp/output.parquet"
    #pdf.to_parquet(file_name)

    #s3_res = boto3.resource('s3', region_name=REGION)
    #s3_res.Bucket(OUTPUT_BUCKET).Object(key).upload_file('/tmp/output.parquet')
