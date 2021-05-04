import urllib

import cv2
import numpy as np


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def file_to_img(fp):
    with open(fp, 'rb') as f:
        image = np.asarray(bytearray(f.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
