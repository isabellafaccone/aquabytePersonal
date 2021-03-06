import os
from PIL import Image
from research_lib.utils.data_access_utils import S3AccessUtils
from rectification import rectify

s3 = S3AccessUtils('/root/data')


def download_from_s3_url(s3_url):
    url_components = s3_url.replace('s3://', '').split('/')
    bucket = url_components[0]
    key = os.path.join(*url_components[1:])
    f = s3.download_from_s3(bucket, key)
    return f


def generate_rectified_stereo_frame(left_full_res_frame_s3_url, right_full_res_frame_s3_url, stereo_parameters_url):
    
    left_full_res_frame_f = download_from_s3_url(left_full_res_frame_s3_url)
    right_full_res_frame_f = download_from_s3_url(right_full_res_frame_s3_url)
    stereo_parameters_f, _, _ = s3.download_from_url(stereo_parameters_url)

    left_image_rectified, right_image_rectified = rectify(left_full_res_frame_f, right_full_res_frame_f, stereo_parameters_f)
    return left_image_rectified, right_image_rectified


def upload_rectified_stereo_frame(left_image_rectified, right_image_rectified)