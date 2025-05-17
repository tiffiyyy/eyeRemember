# By Tiffany Le 
# Last Updated: May 16, 2025 
# This program contains the code to process images using Intel Tiber's
# facial reidentification model (face-reidentification-retail-0095) and 
# draws boxes around each identified face and more. 

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

ie = Core()

model = ie.read_model(model = 'models/face-detection-adas-0001.xml')
compiled_model = ie.compile_model(model = model, device_name = "CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)

image = cv2.imread("data/1-face.jpg")
height, width = image.shape[0:2]

# n = num of images, c = channels (colors detected)
# h = image height, w = image width 
N, C, H, W = input_layer_ir.shape 
resized_image = cv2.resize(image, (W, H))

# image is resized for model to process images in format: (1, W, C, H)
input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# threshold value; used to filter out faces with confidence level < 0.3 
THRESH = 0.3 

# generates a matrix of vectors containing facial information 
# format: image_id, label, confidence level, coordinates of box 
boxes = compiled_model([input_image])[output_layer_ir]

# removes image_id and label values from each vector in boxes 
boxes = boxes.squeeze()[:, -5:]
# for each x in boxes, the last four values will be kept 
# if the confidence level is greater than the threshold value 
boxes = np.array([x[-4:] for x in boxes if x[0] > THRESH])

# restore original image dimensions to be returned with box on detected face 
boxes = boxes * np.array([width, height, width, height])

# function to draw rectangle around detected face 
def draw(image, boxes): 
    new_image = image.copy()
    color = (0, 200, 0)
    for box in boxes: 
        x1, y1, x2, y2 = box 
        cv2.rectangle(img = new_image, pt1 = (x1, y1), pt2 = (x2, y2), color = color, thickness = 10)
    return new_image 

# draw boxes 
plt.figure(figsize = (10, 6))
final_image = draw(image, boxes)
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))