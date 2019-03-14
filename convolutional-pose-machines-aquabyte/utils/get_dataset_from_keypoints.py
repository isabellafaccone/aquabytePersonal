import urllib.request
import json
import numpy as np
import cv2
import random

save_string_training = ""
save_string_validation = ""
training_directory = "utils/dataset/training/fish/"
validation_directory = "utils/dataset/validation/fish/"

validation_percentage = 0.1;

import os
if not os.path.exists(training_directory + "img"):
	os.makedirs(training_directory + "img")
	
if not os.path.exists(validation_directory + "img"):
	os.makedirs(validation_directory + "img")
    
def get_joints_string(labels):
	objects = []
	final_string = ""
	print(labels)
	num_points = 0;
	for key, value in labels.items():
		for key2, value2 in value[0].items():
			final_string += " " + str(value2["y"]) + " " + str(value2["x"])
			num_points = num_points + 1
	if(num_points != 8):
		return None
	return final_string
	
with open('keypoints.json') as json_data:
    d = json.load(json_data)
    on_image = 0;
    
    for obj in d:
    	print(obj)
    	url = obj["Labeled Data"]
    	
    	req = urllib.request.urlopen(url)
    	arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    	img = cv2.imdecode(arr, -1) # 'Load it as it is'

    	print(img.shape)
    	labels = obj["Label"]
    	bounding_box = " " + str(img.shape[0]) + " 0 0 " + str(img.shape[1])
    	joints_string = get_joints_string(labels)
    	if(joints_string == None):
    		continue
    	print(bounding_box)
    	print(joints_string)
    	for i in range(20):
    		image_save_name = "img/img_" + str(on_image) + ".jpg"
    		on_image = on_image + 1;
    		this_line = image_save_name + bounding_box + joints_string
    		if(random.uniform(0, 1) < validation_percentage):
    			cv2.imwrite(validation_directory + image_save_name, img)
    			save_string_validation += this_line + "\n"
    		else:
    			cv2.imwrite(training_directory + image_save_name, img)
    			save_string_training += this_line + "\n"
    with open(training_directory + "labels.txt", "w") as text_file:
    	text_file.write(save_string_training)
    with open(validation_directory + "labels.txt", "w") as text_file:
    	text_file.write(save_string_validation)
