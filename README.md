# wze-uav classifying species specific crown condition by means of deep learning
Open-source repository to classify images of trees and predict their species and health status using EfficientNet B7. 
The images were previously extracted using findatree (https://github.com/FlorianStehr/findatree) using drone-based orthomosaics, digital surface models (DSMs), digital terrain models (DTMs), and shapefiles of the tree crowns.

A link to the respective paper gets updated here after the review process. 

# Installation 
This package was developed using PyTorch and tested on Windows Server 2016 running python=3.9.

We use conda to create a new environment:

user@userpc: /wze-uav_dl$ conda create -n py3.9
user@userpc: /wze-uav_dl$ conda activate py3.9
(py3.9) user@userpc: /wze-uav_dl$ conda install python=3.9 

Then we use pip to install all other packages specified in the requirements.txt:

(py3.9) user@userpc: /wze-uav_dl$ pip install -r requirements.txt
