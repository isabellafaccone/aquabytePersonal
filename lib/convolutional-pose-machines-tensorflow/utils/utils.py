import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import keras
import os
from utils.cpm_utils import make_heatmaps_from_joints, make_heatmap_single
import random
from multiprocessing import Pool
# from OpenGL.GL import *
# from OpenGL.GLU import *
import time


class DataGeneratorAsync(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, annotations_set, FLAGS):
        'Initialization'
        start = time.time()
        self.annotations = annotations_set
        self.on_epoch_end()
        self.FLAGS = FLAGS
        pool = Pool(8)
        self.data = pool.map_async(self._load_image_keypoints_gaussian, self.annotations[:100])
        pool.close()
        pool.join()
        
        self.batches = []
        self._create_batches()
        end = time.time()
        print('Data is ready for training {}'.format(end-start))

    def _load_image_keypoints_gaussian(self, annotation):
        image, keypoints = load_image_keypoints(annotation, self.FLAGS)
        heatmap = make_heatmap_single(self.FLAGS.input_size[0], 
                                      self.FLAGS.heatmap_size, 
                                      self.FLAGS.joint_gaussian_variance,
                                      keypoints)
        return image, keypoints, heatmap
        
    def _create_batches(self):
        data = self.data.get()
        images = [t[0] for t in data]
        keypoints = [t[1] for t in data]
        heatmaps = [t[2] for t in data]
        
        batch_images = [images[k:k+self.FLAGS.batch_size] for k in range(0, len(images), self.FLAGS.batch_size)]
        batch_keypoints = [keypoints[k:k+self.FLAGS.batch_size] for k in range(0, len(keypoints), self.FLAGS.batch_size)]
        batch_heatmaps = [heatmaps[k:k+self.FLAGS.batch_size] for k in range(0, len(heatmaps), self.FLAGS.batch_size)]
        
        for (img, kp, hm) in zip(batch_images, batch_keypoints, batch_heatmaps):
            self.batches.append([np.array(img), 
                                 np.array(batch_keypoints), 
                                 np.array(batch_heatmaps)])
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.annotations) // self.FLAGS.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        return self.batches[index]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.annotations)



def data_generator(annotations, FLAGS, nsteps=1000):
    index = 0
    while True:
        xbatch = []
        ybatch = []
        start = index*FLAGS.batch_size
        for i in range(FLAGS.batch_size):
            buffer = np.random.choice(FLAGS.buffer_range)
            image, keypoints = load_image_keypoints(annotations[start + i], 
                                                    FLAGS,
                                                    reshape=True,
                                                    buffer=buffer)
            xbatch.append(image)
            ybatch.append(keypoints)
        xbatch = np.array(xbatch)
        if FLAGS.normalize:
            xbatch = xbatch / 255.0 - 0.5
        else:
            xbatch -= 128.0
        ybatch = np.array(ybatch)
        batch_gt_heatmap_np = make_heatmaps_from_joints(FLAGS.input_size[0],
                                                        FLAGS.heatmap_size,
                                                        FLAGS.joint_gaussian_variance,
                                                        ybatch)
        index += 1
        if index >= nsteps:
            index = 0
            random.shuffle(annotations)
        yield xbatch, ybatch, batch_gt_heatmap_np    


        
        
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, annotations_set, FLAGS):
        'Initialization'
        self.annotations = annotations_set
        self.on_epoch_end()
        self.FLAGS = FLAGS
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.annotations) // self.FLAGS.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        xbatch = []
        ybatch = []
        start = index*self.FLAGS.batch_size
        i = 0
        while i < self.FLAGS.batch_size:
            buffer = np.random.choice(self.FLAGS.buffer_range)
            image, keypoints = load_image_keypoints(self.annotations[start + i], 
                                                    self.FLAGS,
                                                    reshape=True,
                                                    buffer=buffer)
            xbatch.append(image)
            ybatch.append(keypoints)
            i += 1

        xbatch = np.array(xbatch)
        if self.FLAGS.normalize:
            xbatch = xbatch / 255.0 - 0.5
        else:
            xbatch -= 128.0
        ybatch = np.array(ybatch)
        batch_gt_heatmap_np = make_heatmaps_from_joints(self.FLAGS.input_size[0],
                                                        self.FLAGS.heatmap_size,
                                                        self.FLAGS.joint_gaussian_variance,
                                                        ybatch)
        return xbatch, ybatch, batch_gt_heatmap_np

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.annotations)


def load_ann_keypoints(annotation, order):
    """load keypoints coordinates to numpy array"""
    # the order is given by mapping
    keypoints = []
    for kp_name in order:
        geometry = annotation["Label"][kp_name][0]["geometry"]
        keypoints.append([geometry["x"], geometry["y"]])
    keypoints = np.array(keypoints)
    return keypoints        


def load_image_keypoints(annotation, FLAGS, reshape=True, buffer=100):
    """from annotation load image + keypoints"""
    # load image first
    if 'local_path' in annotation:
        local_path = annotation['local_path']
    else:
        local_path = os.path.join("/root/data/gtsf_phase_I/", 
                                  "/".join(annotation["Labeled Data"].split("/")[7:]))
    image = cv2.imread(local_path)
    #print(local_path)
       
    # load annotations second
    keypoints = []
    for kp_name in FLAGS.keypoints_order:
        value = annotation["Label"][kp_name]
        keypoints.append([int(value[0]["geometry"]["x"]), 
                          int(value[0]["geometry"]["y"])])
    if FLAGS.augmentation:
        transform = FLAGS.augmentation(image=image, 
                                       keypoints=keypoints)
        image = transform["image"]
        keypoints = transform["keypoints"]
    
    # crop the image min / max value
    keypoints = np.array(keypoints)
    height, width, _ = image.shape
    if FLAGS.crop:
        xs = keypoints[:, 0]
        min_x = np.max([np.min(xs) - buffer, 0])
        max_x = np.min([np.max(xs) + buffer, width])
        
        ys = keypoints[:, 1]
        min_y = np.max([np.min(ys) - buffer, 0])
        max_y = np.min([np.max(ys) + buffer, height])

        #print(min_y,max_y, min_x, max_x)
        image = image[min_y:max_y, min_x:max_x, : ]
#         print(min_y, min_x)
    else:
        min_x = 0
        min_y = 0
        
    height, width, _ = image.shape
    if not reshape:
        ratio_width = 1.0
        ratio_height = 1.0
    else:
        ratio_width = width / FLAGS.input_size[0]
        ratio_height = height / FLAGS.input_size[1]
        image = cv2.resize(image, FLAGS.input_size)
    
    # let's recalculate the keypoints
    keypoints[:, 0] = (keypoints[:, 0] - min_x) / ratio_width
    keypoints[:, 1] = (keypoints[:, 1] - min_y) / ratio_height
      
    
    return image, keypoints

def read_square_image(file, cam, boxsize, type):
    # from file
    if type == 'IMAGE':
        oriImg = cv2.imread(file)
    # from webcam
    elif type == 'WEBCAM':
        _, oriImg = cam.read()

    scale = boxsize / (oriImg.shape[0] * 1.0)
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((boxsize, boxsize, 3)) * 128

    if imageToTest.shape[1] < boxsize:
        offset = imageToTest.shape[1] % 2
        output_img[:, int(boxsize/2-math.ceil(imageToTest.shape[1]/2)):int(boxsize/2+math.ceil(imageToTest.shape[1]/2)+offset), :] = imageToTest
    else:
        output_img = imageToTest[:, int(imageToTest.shape[1]/2-boxsize/2):int(imageToTest.shape[1]/2+boxsize/2), :]
    return output_img


def resize_pad_img(img, scale, output_size):
    resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    pad_h = (output_size - resized_img.shape[0]) // 2
    pad_w = (output_size - resized_img.shape[1]) // 2
    pad_h_offset = (output_size - resized_img.shape[0]) % 2
    pad_w_offset = (output_size - resized_img.shape[1]) % 2
    resized_pad_img = np.pad(resized_img, ((pad_w, pad_w+pad_w_offset), (pad_h, pad_h+pad_h_offset), (0, 0)),
                             mode='constant', constant_values=128)

    return resized_pad_img


def img_white_balance(img, white_ratio):
    for channel in range(img.shape[2]):
        channel_max = np.percentile(img[:, :, channel], 100-white_ratio)
        channel_min = np.percentile(img[:, :, channel], white_ratio)
        img[:, :, channel] = (channel_max-channel_min) * (img[:, :, channel] / 255.0)
    return img


def img_white_balance_with_bg(img, bg, white_ratio):
    for channel in range(img.shape[2]):
        channel_max = np.percentile(bg[:, :, channel], 100-white_ratio)
        channel_min = np.percentile(bg[:, :, channel], white_ratio)
        img[:, :, channel] = (channel_max-channel_min) * (img[:, :, channel] / 255.0)
    return img


def draw_predicted_heatmap(heatmap, input_size):
    heatmap_resized = cv2.resize(heatmap, (input_size, input_size))

    output_img = None
    tmp_concat_img = None
    h_count = 0
    for joint_num in range(heatmap_resized.shape[2]):
        if h_count < 4:
            tmp_concat_img = np.concatenate((tmp_concat_img, heatmap_resized[:, :, joint_num]), axis=1) \
                if tmp_concat_img is not None else heatmap_resized[:, :, joint_num]
            h_count += 1
        else:
            output_img = np.concatenate((output_img, tmp_concat_img), axis=0) if output_img is not None else tmp_concat_img
            tmp_concat_img = None
            h_count = 0
    # last row img
    if h_count != 0:
        while h_count < 4:
            tmp_concat_img = np.concatenate((tmp_concat_img, np.zeros(shape=(input_size, input_size), dtype=np.float32)), axis=1)
            h_count += 1
        output_img = np.concatenate((output_img, tmp_concat_img), axis=0)

    # adjust heatmap color
    output_img = output_img.astype(np.uint8)
    output_img = cv2.applyColorMap(output_img, cv2.COLORMAP_JET)
    return output_img


def draw_stages_heatmaps(stage_heatmap_list, orig_img_size):

    output_img = None
    nStages = len(stage_heatmap_list)
    nJoints = stage_heatmap_list[0].shape[3]
    for stage in range(nStages):
        cur_heatmap = np.squeeze(stage_heatmap_list[0][0, :, :, 0:nJoints-1])
        cur_heatmap = cv2.resize(cur_heatmap, (orig_img_size, orig_img_size))

        channel_max = np.percentile(cur_heatmap, 99)
        channel_min = np.percentile(cur_heatmap, 1)
        cur_heatmap = 255.0 / (channel_max - channel_min) * (cur_heatmap - channel_min)
        cur_heatmap = np.clip(cur_heatmap, 0, 255)

        cur_heatmap = np.repeat(np.expand_dims(np.amax(cur_heatmap, axis=2), axis=2), 3, axis=2)
        output_img = np.concatenate((output_img, cur_heatmap), axis=1) if output_img is not None else cur_heatmap
    return output_img.astype(np.uint8)


def extract_2d_joint_from_heatmap(heatmap, input_size, joints_2d):
    heatmap_resized = cv2.resize(heatmap, (input_size, input_size))

    for joint_num in range(heatmap_resized.shape[2]):
        joint_coord = np.unravel_index(np.argmax(heatmap_resized[:, :, joint_num]), (input_size, input_size))
        joints_2d[joint_num, :] = joint_coord

    return joints_2d


def extract_3d_joints_from_heatmap(joints_2d, x_hm, y_hm, z_hm, input_size, joints_3d):

    for joint_num in range(x_hm.shape[2]):
        coord_2d_y = joints_2d[joint_num][0]
        coord_2d_x = joints_2d[joint_num][1]

        # x_hm_resized = cv2.resize(x_hm, (input_size, input_size))
        # y_hm_resized = cv2.resize(y_hm, (input_size, input_size))
        # z_hm_resized = cv2.resize(z_hm, (input_size, input_size))
        # joint_x = x_hm_resized[max(int(coord_2d_x), 1), max(int(coord_2d_y), 1), joint_num] * 100
        # joint_y = y_hm_resized[max(int(coord_2d_x), 1), max(int(coord_2d_y), 1), joint_num] * 100
        # joint_z = z_hm_resized[max(int(coord_2d_x), 1), max(int(coord_2d_y), 1), joint_num] * 100


        joint_x = x_hm[max(int(coord_2d_x/8), 1), max(int(coord_2d_y/8), 1), joint_num] * 10
        joint_y = y_hm[max(int(coord_2d_x/8), 1), max(int(coord_2d_y/8), 1), joint_num] * 10
        joint_z = z_hm[max(int(coord_2d_x/8), 1), max(int(coord_2d_y/8), 1), joint_num] * 10
        joints_3d[joint_num, 0] = joint_x
        joints_3d[joint_num, 1] = joint_y
        joints_3d[joint_num, 2] = joint_z
    joints_3d -= joints_3d[14, :]

    return joints_3d

def draw_limbs_2d(img, joints_2d, limb_parents):
    for limb_num in range(len(limb_parents)-1):
        x1 = joints_2d[limb_num, 0]
        y1 = joints_2d[limb_num, 1]
        x2 = joints_2d[limb_parents[limb_num], 0]
        y2 = joints_2d[limb_parents[limb_num], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        # if length < 10000 and length > 5:
        deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
        polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                   (int(length / 2), 3),
                                   int(deg),
                                   0, 360, 1)
        cv2.fillConvexPoly(img, polygon, color=(0,255,0))
    return img

def draw_limbs_3d(joints_3d, limb_parents, ax):

    for i in range(joints_3d.shape[0]):
        x_pair = [joints_3d[i, 0], joints_3d[limb_parents[i], 0]]
        y_pair = [joints_3d[i, 1], joints_3d[limb_parents[i], 1]]
        z_pair = [joints_3d[i, 2], joints_3d[limb_parents[i], 2]]
        ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3)


def draw_limb_3d_gl(joints_3d, limb_parents):

    glLineWidth(2)
    glBegin(GL_LINES)
    glColor3f(1,0,0)
    glVertex3fv((0,0,0))
    glVertex3fv((100,0,0))
    glColor3f(0,1,0)
    glVertex3fv((0,0,0))
    glVertex3fv((0,100,0))
    glColor3f(0,0,1)
    glVertex3fv((0,0,0))
    glVertex3fv((0,0,100))
    glEnd()

    glColor3f(1,1,1)
    glBegin(GL_LINES)
    for i in range(joints_3d.shape[0]):
        glVertex3fv((joints_3d[i, 0], joints_3d[i, 1], joints_3d[i, 2]))
        glVertex3fv((joints_3d[limb_parents[i], 0], joints_3d[limb_parents[i], 1], joints_3d[limb_parents[i], 2]))
    glEnd()

    # glBegin(GL_TRIANGLES)
    # glVertex3f(0, 100, 0)
    # glVertex3f(100, 0, 50)
    # glVertex3f(0, -100, 100)
    # glEnd()


def draw_float_range_img(img):
    tmp_min = np.min(img)
    tmp_max = np.max(img)
    img = cv2.convertScaleAbs(img, None, 255.0 / (tmp_max - tmp_min))
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img.astype(np.uint8)












