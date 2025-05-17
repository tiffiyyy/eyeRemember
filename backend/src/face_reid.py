# By Tiffany Le 
# Last Updated: May 17, 2025 

from pathlib import Path

import cv2
import numpy as np
from openvino.runtime import Core

from face_detector import detect_faces

ie = Core()

model = ie.read_model(model = 'models/face-reidentification-retail-0095.xml')
compiled_model = ie.compile_model(model = model, device_name = "CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)


N, C, H, W = input_layer_ir.shape 
image = cv2.imread("data/1-face.jpg")

def preprocess(image:np.ndarray,threshold:float,width:int,height:int):
    
    boxes = detect_faces(image,threshold)
    embeddings = [] # embeddings for all all boxes
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        face_crop = image[y_min:y_max, x_min:x_max]
        resizedReid = cv2.resize(face_crop,(W,H))
        input_blob = np.expand_dims(resizedReid.transpose(2, 0, 1), axis=0).astype(np.float32)
        embeddings.append(compiled_model([input_blob])[output_layer_ir])
    return embeddings

# make fucntion to process the face
