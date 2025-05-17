# By Tiffany Le 
# Last Updated: May 17, 2025 
# This program contains the code to process images using Intel Tiber's
# facial reidentification model (face-reidentification-retail-0095) and 
# draws boxes around each identified face and more. 

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core
from face_detector import detect_faces

ie = Core()

model = ie.read_model(model = 'models/face-detection-adas-0001.xml')
compiled_model = ie.compile_model(model = model, device_name = "CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)

image = cv2.imread("data/1-face.jpg")

# generates a matrix of vectors containing facial information and 
# filters out entries with low confidence level 
boxes = detect_faces(image, 0.3)


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