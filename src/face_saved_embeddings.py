import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

from face_detector import detect_faces

ie = Core()

model = ie.read_model(model = 'models/face-reidentification-retail-0095.xml')
compiled_model = ie.compile_model(model = model, device_name = "CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)

#from frontend take photos/name/desc and embed the photos into face_db.pkl

