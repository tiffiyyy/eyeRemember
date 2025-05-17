# Intel Tiber Model 
This repository contains the programs and models to implement Intel OpenVINO's face-reidentification-retail-0095 for facial identification for Santa Clara University's Intel Hackathon on May 17th, 2025.  

## Setup 

### Model Installation  
Please run the following commands to download necessary models. 
```
mkdir -p models/face-reidentification-retail-0095/FP32
cd models/face-reidentification-retail-0095/FP32

wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml
wget https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.bin
```

In the case of using a Mac, run 'brew install wget' in your command line in order to utilize the above commands. 

### Requirements
- Python 3.x  
- OpenVINO  
- OpenCV  
- Other dependencies (list from `requirements.txt`)

### Cloning the Repo 
git clone https://github.com/tiffiyyy/itm
cd itm 