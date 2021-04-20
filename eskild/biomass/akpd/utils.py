#!/usr/bin/env python3

import cv2
import urllib.request
import time
import numpy as np

def url_to_image(url):
    retries = 10
    while True:
        try:
            resp = urllib.request.urlopen(url)
            break
        except (urllib.error.URLError, ConnectionResetError) as e:
            print(f'Exeption fetching: {url}\n {str(e)} \n Stop trying in {retries} times')
            t = 1.0 / (float(retries) / 10.0)
            retries -= 1
            if retries <= 0:
                raise e
            time.sleep(t)

    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def image_resize(image, FLAGS):
    height, width, _ = image.shape
    ratio_width = width / FLAGS.input_size[0]
    ratio_height = height / FLAGS.input_size[1]
    image = cv2.resize(image, FLAGS.input_size)
    image  = image / 255.0 - 0.5
    image = image[np.newaxis, ...]
    return image

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

def delta_kp(A, B, validate=True, ix=-1):
    if validate:
        assert A['url'] == B['url']
    kp_a = A['kp']
    kp_b = B['kp']
    assert len(kp_a) == len(kp_b)
    diff =[] 
    for i, a in enumerate(kp_a):
        b = kp_b[i]
        assert a['kp'] == b['kp']
        kpa = a['point']
        kpb = b['point']
        x = np.abs(kpa[0] - kpb[0])
        y = np.abs(kpa[1] - kpb[1])
        #diff.append(x+y)
        diff.append({
            'ix':ix,
            'url': A['url'],
            'kp': a['kp'],
            'point':x+y,
            'score': np.abs(a['score'] - b['score']),
            'avg': np.abs(a['avg'] - b['avg']),
            'max': np.abs(a['max'] - b['max']),
        })
    return diff

def delta_frame(A, B, validate=True):
    assert len(A) == len(B)
    diff = []
    for ix , a in enumerate(A):
        b = B[ix]
        diff.append(delta_kp(a['left_image'],b['left_image'],validate,ix))
        diff.append(delta_kp(a['right_image'],b['right_image'],validate,ix))
    return diff
        
