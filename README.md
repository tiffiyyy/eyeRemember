# Intel Tiber Model 
This repository contains the programs and models to implement Intel OpenVINO's face-reidentification-retail-0095 for facial identification for Santa Clara University's Intel Hackathon on May 17th, 2025.  

## Setup 

### Model Installation  
Please run the following commands to download necessary models. 
Note: These commands are Mac specific. Window commands may differ. 

To download the models for face_detector.py 
```
mkdir -p models/face-detection-adas-0001/FP32
cd models/face-detection-adas-0001/FP32

wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-adas-0001/FP32/face-detection-adas-0001.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-adas-0001/FP32/face-detection-adas-0001.bin
```

To download the models for face_reid.py 
```
mkdir -p models/face-reidentification-retail-0095/FP32
cd models/face-reidentification-retail-0095/FP32

wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin
```

In the case of using a Mac, run `brew install wget` in your command line in order to utilize the above commands. 

### Requirements
- Python 3.x  
- OpenVINO  
- OpenCV  
- Other dependencies (list from `requirements.txt`)

### Cloning the Repo 
To clone this repository, run the following commands: 
```
git clone https://github.com/tiffiyyy/itm
cd itm 
```