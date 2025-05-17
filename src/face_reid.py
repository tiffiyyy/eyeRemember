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


# make fucntion to process the face
image = cv2.imread("data/1-face.jpg")

preprocessed_face = detect_faces(image,0.3)


embedding = compiled_model([preprocessed_face])[output_layer_ir]