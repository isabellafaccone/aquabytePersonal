import cv2
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
# SIFT based correction - functions



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



def find_matches_and_homography(left_crop_url, right_crop_url, MIN_MATCH_COUNT=11, GOOD_PERC=0.7, FLANN_INDEX_KDTREE=0):
    imageL = load_image(left_crop_url)
    imageR = load_image(right_crop_url)

    sift = cv2.KAZE_create()
    img1 = enhance(imageL)
    img2 = enhance(imageR)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)


    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    H = []
    matchesMask = []
    for m,n in matches:
        if m.distance < GOOD_PERC*n.distance:
            good.append(m)
    if len(good)>=MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1 ,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        H = H.tolist()
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    return (good, matchesMask, H)


def adjust_keypoints(keypoints, H):
    left_keypoints_crop = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in keypoints['leftCrop']}
    right_keypoints_crop = {item['keypointType']: [item['xCrop'], item['yCrop']] for item in keypoints['rightCrop']}

    # adjust left and right keypoints
    left_keypoints_crop_adjusted, right_keypoints_crop_adjusted = [], []
    for i, bp in enumerate([item['keypointType'] for item in keypoints['leftCrop']]):
        kpL = left_keypoints_crop[bp]
        ptx = np.array([kpL[0], kpL[1], 1])
        zx = np.dot(H, ptx)
        kpL2R = [zx[0] / zx[2], zx[1] / zx[2]]

        kpR = right_keypoints_crop[bp]
        pty = np.array([kpR[0], kpR[1], 1])
        zy = np.dot(np.linalg.inv(H), pty)
        kpR2L = [zy[0] / zy[2], zy[1] / zy[2]]

        kpL_adjusted = [(kpL[0] + kpR2L[0]) / 2.0, (kpL[1] + kpR2L[1]) / 2.0]
        kpR_adjusted = [(kpR[0] + kpL2R[0]) / 2.0, (kpR[1] + kpL2R[1]) / 2.0]
        item_left = keypoints['leftCrop'][i]
        item_right = keypoints['rightCrop'][i]

        new_item_left = {
            'keypointType': bp,
            'xCrop': kpL_adjusted[0],
            'xFrame': item_left['xFrame'] - item_left['xCrop'] + kpL_adjusted[0],
            'yCrop': kpL_adjusted[1],
            'yFrame': item_left['yFrame'] - item_left['yCrop'] + kpL_adjusted[1]
        }
        left_keypoints_crop_adjusted.append(new_item_left)

        new_item_right = {
            'keypointType': bp,
            'xCrop': kpR_adjusted[0],
            'xFrame': item_right['xFrame'] - item_right['xCrop'] + kpR_adjusted[0],
            'yCrop': kpR_adjusted[1],
            'yFrame': item_right['yFrame'] - item_right['yCrop'] + kpR_adjusted[1]
        }
        right_keypoints_crop_adjusted.append(new_item_right)

    adjusted_keypoints = {
        'leftCrop': left_keypoints_crop_adjusted,
        'rightCrop': right_keypoints_crop_adjusted
    }
    return adjusted_keypoints

DbParams = namedtuple("DbParams", "user password host port db_name")


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
    s3_res = boto3.resource('s3')
    b, k = convert_url_to_s3_bk(url)
    obj = s3_res.Object(b, k).get()['Body']
    print("key", k)
    image = np.asarray(bytearray(obj.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def get_db_params(db_type):
    ssm_client = boto3.client('ssm')
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
    sql = "select left_crop_url, right_crop_url from prod.crop_annotation where annotation_state_id=1 and service_id=2 and captured_at > now() - interval '1 day' and pen_id=61"
    pdf = get_data(sql, "DW_DB_RO")

    udfValue = udf(find_matches_and_homography, ArrayType(DoubleType()))

    df = sqlContext.createDataFrame(pdf).repartition(12).withColumn("plane", udfValue("left_crop_url", "right_crop_url")).collect()
    #tdf = df.rdd.repartition(12).map(find_matches_and_homography)
    #features = tdf.map(lambda x: (Row(features=x[2])))
    #sdf = sqlContext.createDataFrame(features)
    pdf = df.select("*").toPandas()

    file_name = "/tmp/output.parquet"
    pdf.to_parquet(file_name)

    s3_res = boto3.resource('s3')
    output_path = f"s3://aquabyte-research/template-matching/"
    s3_res.Bucket("aquabyte-research").Object("template-matching/{datetime.datetime.now().isoformat()}/output.parquet").upload_file('/tmp/output.parquet')
    #df.write.parquet(output_path)
