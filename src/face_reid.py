# By Tiffany Le 
# Last Updated: May 17, 2025 

from pathlib import Path

import cv2
import numpy as np
from openvino.runtime import Core

ie = Core()

model = ie.read_model(model = 'face-reidentification-retail-0095.xml')
compiled_model = ie.compile_model(model = model, device_name = "CPU")