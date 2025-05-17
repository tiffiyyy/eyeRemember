# By Tiffany Le and Linus Wong
# Last Updated: May 17, 2025 

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
N, C, H, W = input_layer_ir.shape 

def detect_faces(image: np.ndarray, threshold: float): 
    # n = num of images, c = channels (colors detected)
    # h = image height, w = image width 
    height, width = image.shape[0:2]
    # initialize resized image 
    resized_image = cv2.resize(image, (W, H))
    # image is resized for model to process images in format: (1, W, C, H)
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

    # generates a matrix of vectors containing facial information 
    # format: image_id, label, confidence level, coordinates of box 
    boxes = compiled_model([input_image])[output_layer_ir]
    # removes image_id and label values from each vector in boxes 
    boxes = boxes.squeeze()[:, -5:]

    # for each box in boxes, the last four values will be kept 
    # if the confidence level is greater than the threshold value 
    boxes = np.array([box[-4:] for box in boxes if box[0] > threshold])
    # restore original image dimensions to be returned with box on detected face 
    boxes = boxes * np.array([width, height, width, height])

    return boxes.astype(int)

