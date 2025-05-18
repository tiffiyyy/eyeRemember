# By Tiffany Le and Linus Wong
# Last Updated: May 17, 2025 
# This program contains the code to process images using Intel Tiber's
# facial reidentification model (face-reidentification-retail-0095) and 
# draws boxes around each identified face and more. 

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino import Core
import pickle 
from pathlib import Path

from face_detector import detect_faces
from face_reid import preprocess, getH, getW
from utils import find_best_match

# function to draw rectangle around detected face 
def draw(image, boxes): 
    new_image = image.copy()
    color = (0, 200, 0)
    for box in boxes: 
        x1, y1, x2, y2 = box 
        cv2.rectangle(img = new_image, pt1 = (x1, y1), pt2 = (x2, y2), color = color, thickness = 10)
    return new_image 

def main(): 
    ie = Core()

    file_path = Path("face-detection-adas-0001.xml")
    model = ie.read_model(model = file_path)
    compiled_model = ie.compile_model(model = model, device_name = "CPU")

    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output(0)

    print("test")

    # populates faces with profile dictionary generated via function in face_reid.py
    #with open("../data/face_db.pkl", "rb") as f:
    #    faces = pickle.load(f)

    pickle_path = Path("face_db.pkl")
    if pickle_path.exists() and pickle_path.stat().st_size > 0:
        with open(pickle_path, "rb") as f:
            faces = pickle.load(f)
    else:
        faces = {}
        with open(file_path, "wb") as f:
            pickle.dump(faces, f)

    # set running to when camera is on 
    running = 0

    while(running < 1):
        # at every frame, return an image that is the current frame 
            # extract current frame from front end and store into variable to pass through imread 
        frameImage = cv2.imread("../data/IMG_2875.jpeg")
        if frameImage is None:
            raise FileNotFoundError("Image not found or failed to load.")
        boxes = detect_faces(frameImage, 0.3)

        embeddings = preprocess(frameImage,0.3,getW(),getH())
        for e in embeddings:
            personIdentified = find_best_match(e,faces, 0.3)
            
        # generates a matrix of vectors containing facial information and 
        # filters out entries with low confidence level 
        plt.figure(figsize = (10, 6))
        final_image = draw(frameImage, boxes)
        plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        running = running + 1

    print("done!")
