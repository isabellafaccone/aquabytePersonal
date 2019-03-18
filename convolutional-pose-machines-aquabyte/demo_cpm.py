import numpy as np
from utils import cpm_utils
import cv2
import time
import math
import sys
import os
import imageio
import tensorflow as tf
from models.nets import cpm_body
from scipy import ndimage

"""Parameters
"""
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('DEMO_TYPE',
                           default_value='test_imgs/img_44.jpg',
                           docstring='MULTI: show multiple stage,'
                                     'SINGLE: only last stage,'
                                     'HM: show last stage heatmap,'
                                     'paths to .jpg or .png image'
                                     'paths to .avi or .flv or .mp4 video')
tf.app.flags.DEFINE_string('model_path',
                           default_value='trained_models/_cpm_body_i512x512_o64x64_6s-6800',
                           docstring='Your model')
tf.app.flags.DEFINE_integer('input_size',
                            default_value=512,
                            docstring='Input image size')
tf.app.flags.DEFINE_integer('hmap_size',
                            default_value=64,
                            docstring='Output heatmap size')
tf.app.flags.DEFINE_integer('cmap_radius',
                            default_value=8,
                            docstring='Center map gaussian variance')
tf.app.flags.DEFINE_integer('joints',
                            default_value=8,
                            docstring='Number of joints')
tf.app.flags.DEFINE_integer('stages',
                            default_value=6,
                            docstring='How many CPM stages')
tf.app.flags.DEFINE_integer('cam_num',
                            default_value=0,
                            docstring='Webcam device number')
tf.app.flags.DEFINE_bool('KALMAN_ON',
                         default_value=False,
                         docstring='enalbe kalman filter')
tf.app.flags.DEFINE_float('kalman_noise',
                            default_value=3e-2,
                            docstring='Kalman filter noise value')
tf.app.flags.DEFINE_string('color_channel',
                           default_value='RGB',
                           docstring='')


box_size = 512
num_of_joints = 8
gaussian_radius = 8

# Set color for each keypoint
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]


# Not currently used
limbs = [[0, 1],
         [2, 3],
         [3, 4],
         [5, 6],
         [6, 7]]

if sys.version_info.major == 3:
    PYTHON_VERSION = 3
else:
    PYTHON_VERSION = 2


def mgray(test_img_resize, test_img):
    test_img_resize = np.dot(test_img_resize[..., :3], [0.299, 0.587, 0.114]).reshape(
                    (FLAGS.input_size, FLAGS.input_size, 1))
    cv2.imshow('color', test_img.astype(np.uint8))
    cv2.imshow('gray', test_img_resize.astype(np.uint8))
    cv2.waitKey(0)
    return test_img_resize

def transform_raw_landmarks_to_ground_truh(test_img, line):
	cur_img = test_img
	
	# Read in bbox and joints coords
	tmp = [float(x) for x in line[1:5]]
	cur_hand_bbox = [min([tmp[0], tmp[2]]),
	                         min([tmp[1], tmp[3]]),
	                         max([tmp[1], tmp[3]]),
	                         max([tmp[0], tmp[2]])
	                         ]
	if cur_hand_bbox[0] < 0: cur_hand_bbox[0] = 0
	if cur_hand_bbox[1] < 0: cur_hand_bbox[1] = 0
	if cur_hand_bbox[2] > cur_img.shape[1]: cur_hand_bbox[2] = cur_img.shape[1]
	if cur_hand_bbox[3] > cur_img.shape[0]: cur_hand_bbox[3] = cur_img.shape[0]
	
	cur_hand_joints_y = [float(i) for i in line[5:49:2]]
	cur_hand_joints_x = [float(i) for i in line[6:49:2]]
	
	        # Crop image and adjust joint coords
	cur_img = cur_img[int(float(cur_hand_bbox[1])):int(float(cur_hand_bbox[3])),
	                  int(float(cur_hand_bbox[0])):int(float(cur_hand_bbox[2])),
	                  :]
	cur_hand_joints_x = [x - cur_hand_bbox[0] for x in cur_hand_joints_x]
	cur_hand_joints_y = [x - cur_hand_bbox[1] for x in cur_hand_joints_y]

	output_image = np.ones(shape=(box_size, box_size, 3)) * 128
	
	scale = box_size / (cur_img.shape[1] * 1.0)
	
	# Relocalize points
	cur_hand_joints_x = list(map(lambda x: x * scale, cur_hand_joints_x))
	cur_hand_joints_y = list(map(lambda x: x * scale, cur_hand_joints_y))
	
	# Resize image
	image = cv2.resize(cur_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
	offset = image.shape[0] % 2
	
	output_image[int(box_size / 2 - math.floor(image.shape[0] / 2)): int(
	                box_size / 2 + math.floor(image.shape[0] / 2) + offset), :, :] = image
	cur_hand_joints_y = list(map(lambda x: x + (box_size / 2 - math.floor(image.shape[0] / 2)),
	                                    cur_hand_joints_y))
	
	cur_hand_joints_x = np.asarray(cur_hand_joints_x)
	cur_hand_joints_y = np.asarray(cur_hand_joints_y)
	
	for i in range(len(cur_hand_joints_x)):
		cv2.circle(output_image, center=(int(cur_hand_joints_x[i]), int(cur_hand_joints_y[i])),radius=3, color=(255,0,0), thickness=-1)
		cv2.imshow('ground_truth', output_image.astype(np.uint8))
	coords_set = np.concatenate((np.reshape(cur_hand_joints_y, (num_of_joints, 1)),
                                     np.reshape(cur_hand_joints_x, (num_of_joints, 1))),
                                    axis=1)
	return coords_set
	
	
def run_inference_for_single_image(test_img, test_center_map, kalman_filter_array, sess, model):
	scale = box_size / (test_img.shape[1] * 1.0)
	image = cv2.resize(test_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
	offset = image.shape[0] % 2
	output_image = np.ones(shape=(box_size, box_size, 3)) * 128
	output_image[int(box_size / 2 - math.floor(image.shape[0] / 2)): int(box_size / 2 + math.floor(image.shape[0] / 2) + offset), :, :] = image
	test_img = output_image
	test_img_resize = test_img
	
	test_img_input = test_img_resize / 256.0 - 0.5
	test_img_input = np.expand_dims(test_img_input, axis=0)
	fps_t = time.time()
	predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
		model.stage_heatmap, ],
		feed_dict={'input_image:0': test_img_input,
		'center_map:0': test_center_map})

	# Show visualized image
	[demo_img, joint_coord_set] = visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array)
	cv2.imshow('demo_img', demo_img.astype(np.uint8))
	
	return joint_coord_set
	print('fps: %.2f' % (1 / (time.time() - fps_t)))

def main(argv):
    tf_device = '/gpu:0'
    tf_device = '/cpu:0'
    with tf.device(tf_device):
        """Build graph
        """
        if FLAGS.color_channel == 'RGB':
            input_data = tf.placeholder(dtype=tf.float32,
                                        shape=[None, FLAGS.input_size, FLAGS.input_size, 3],
                                        name='input_image')
        else:
            input_data = tf.placeholder(dtype=tf.float32,
                                        shape=[None, FLAGS.input_size, FLAGS.input_size, 1],
                                        name='input_image')

        center_map = tf.placeholder(dtype=tf.float32,
                                    shape=[None, FLAGS.input_size, FLAGS.input_size, 1],
                                    name='center_map')

        model = cpm_body.CPM_Model(FLAGS.stages, FLAGS.joints + 1)
        model.build_model(input_data, center_map, 1)

    saver = tf.train.Saver()

    """Create session and restore weights
    """
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    if FLAGS.model_path.endswith('pkl'):
        model.load_weights_from_file(FLAGS.model_path, sess, False)
    else:
        saver.restore(sess, FLAGS.model_path)

    test_center_map = cpm_utils.gaussian_img(FLAGS.input_size,
                                             FLAGS.input_size,
                                             FLAGS.input_size / 2,
                                             FLAGS.input_size / 2,
                                             FLAGS.cmap_radius)
    test_center_map = np.reshape(test_center_map, [1, FLAGS.input_size,
                                                   FLAGS.input_size, 1])

    # Check weights
    for variable in tf.trainable_variables():
        with tf.variable_scope('', reuse=True):
            var = tf.get_variable(variable.name.split(':0')[0])
            print(variable.name, np.mean(sess.run(var)))

    # Create kalman filters
    if FLAGS.KALMAN_ON:
        kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.joints)]
        for _, joint_kalman_filter in enumerate(kalman_filter_array):
            joint_kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                             [0, 1, 0, 1],
                                                             [0, 0, 1, 0],
                                                             [0, 0, 0, 1]],
                                                            np.float32)
            joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                              [0, 1, 0, 0]],
                                                             np.float32)
            joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                            [0, 1, 0, 0],
                                                            [0, 0, 1, 0],
                                                            [0, 0, 0, 1]],
                                                           np.float32) * FLAGS.kalman_noise
    else:
        kalman_filter_array = None


    # image processing
    gt_file = 'utils/dataset/validation/fish/labels.txt'
    gt_content = open(gt_file, 'rb').readlines()
    with tf.device(tf_device):
        if FLAGS.DEMO_TYPE.endswith(('avi', 'flv', 'mp4')):
            pass
        else:
            for idx, line in enumerate(gt_content):
                test_img_t = time.time()
                line = line.decode("utf-8") 
                line = line.split()
                print(line)

                cur_img_path = "utils/dataset/validation/fish/" + line[0]

                if cur_img_path.endswith(('png', 'jpg')):
                    test_img = cv2.imread(cur_img_path)
                    joint_coord_set = run_inference_for_single_image(test_img, test_center_map, kalman_filter_array, sess, model)
                    gt_joint_coord_set = transform_raw_landmarks_to_ground_truh(test_img, line)
                    print("Coordinates")
                    print(joint_coord_set)
                    print(gt_joint_coord_set)
                    cv2.waitKey(0)
                    


def get_center_of_mass_from_heatmap(heatmap, joint_coord):
# 	return joint_coord
	joint_coord = [int(c) for c in joint_coord]
	center = ndimage.measurements.center_of_mass(heatmap)
	int_center = [int(c) for c in center]
	if(int_center[0] > box_size):
		int_center[0] = joint_coord[0]
	elif(int_center[0] < 0):
		int_center[0] = joint_coord[0]
	if(int_center[1] > box_size):
		int_center[1] = joint_coord[1]
	elif(int_center[1] < 0):
		int_center[1] = joint_coord[1]
	return int_center
	
	
def visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array):
    hm_t = time.time()
    demo_stage_heatmaps = []
    if FLAGS.DEMO_TYPE == 'MULTI':
        for stage in range(len(stage_heatmap_np)):
            demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.joints].reshape(
                (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
            demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0]))
            demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
            demo_stage_heatmap = np.reshape(demo_stage_heatmap, (test_img.shape[1], test_img.shape[0], 1))
            demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
            demo_stage_heatmap *= 255
            demo_stage_heatmaps.append(demo_stage_heatmap)

        # last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
        #     (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = stage_heatmap_np[-1][0, :, :, 0:FLAGS.joints].reshape(
            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    else:
        # last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
        #     (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = stage_heatmap_np[-1][0, :, :, 0:FLAGS.joints].reshape(
            (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
        last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    print('hm resize time %f' % (time.time() - hm_t))

    joint_t = time.time()
    joint_coord_set = np.zeros((FLAGS.joints, 2))

    # Plot joint colors
    if kalman_filter_array is not None:
        for joint_num in range(FLAGS.joints):
            joint_coord_max = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                           (test_img.shape[0], test_img.shape[1]))
            joint_coord = get_center_of_mass_from_heatmap(last_heatmap[:, :, joint_num], joint_coord_max)
            # add a dimension for kalman filter
            joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
            kalman_filter_array[joint_num].correct(joint_coord)
            kalman_pred = kalman_filter_array[joint_num].predict()
            joint_coord_set[joint_num, :] = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
    else:
        for joint_num in range(FLAGS.joints):
            joint_coord_max = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                           (test_img.shape[0], test_img.shape[1]))
            joint_coord = get_center_of_mass_from_heatmap(last_heatmap[:, :, joint_num], joint_coord_max)
            joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
    print('plot joint time %f' % (time.time() - joint_t))

    limb_t = time.time()
    # Plot limb colors
#     for limb_num in range(len(limbs)):
# 
#         x1 = joint_coord_set[limbs[limb_num][0], 0]
#         y1 = joint_coord_set[limbs[limb_num][0], 1]
#         x2 = joint_coord_set[limbs[limb_num][1], 0]
#         y2 = joint_coord_set[limbs[limb_num][1], 1]
#         length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
#         if length < 200 and length > 5:
#             deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
#             polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
#                                        (int(length / 2), 6),
#                                        int(deg),
#                                        0, 360, 1)
#             color_code_num = limb_num // 4
#             if PYTHON_VERSION == 3:
#                 limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
#             else:
#                 limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])
# 
#             cv2.fillConvexPoly(test_img, polygon, color=limb_color)
#    
   
    print('plot limb time %f' % (time.time() - limb_t))

    if FLAGS.DEMO_TYPE == 'MULTI':
        upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]), axis=1)
        lower_img = np.concatenate((demo_stage_heatmaps[3], demo_stage_heatmaps[len(stage_heatmap_np) - 1], test_img),
                                   axis=1)
        demo_img = np.concatenate((upper_img, lower_img), axis=0)
        return [demo_img, joint_coord_set]
    else:
        return [test_img, joint_coord_set]


def get_euclidean_distance_between_coord_sets(coord_set_a, coord_set_b):
	return np.linalg.norm(coord_set_a-coord_set_b)

if __name__ == '__main__':
    tf.app.run()
