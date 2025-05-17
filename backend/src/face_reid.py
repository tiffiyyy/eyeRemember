# By Tiffany Le and Linus Wong
# Last Updated: May 17, 2025 
# This program processes detected faces and returns embeddings. 

from pathlib import Path

import cv2
import numpy as np
import pickle
from openvino.runtime import Core

from face_detector import detect_faces

ie = Core()

model = ie.read_model(model = 'models/face-reidentification-retail-0095.xml')
compiled_model = ie.compile_model(model = model, device_name = "CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)


N, C, H, W = input_layer_ir.shape 
    #extract from image upload
image = cv2.imread("data/IMG_2875.jpeg")

def getW():
    return W
def getH():
    return H

def preprocess(image:np.ndarray,threshold:float,width:int,height:int):
    # generates boxes for each face detected 
    boxes = detect_faces(image,threshold)
    embeddings = [] # embeddings for all boxes
    # for each face detected, crop image to just be the face 
    # and in a format in which the model can process it in 
    # and returns all results into an array embeddings 
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        face_crop = image[y_min:y_max, x_min:x_max]
        resizedReid = cv2.resize(face_crop,(W,H))
        # resize image so that model can process it
        input_blob = np.expand_dims(resizedReid.transpose(2, 0, 1), axis=0).astype(np.float32)
        # passes image through model; returns model ouput into an array (embeddings)
        embeddings.append(compiled_model([input_blob])[output_layer_ir])
    return embeddings

