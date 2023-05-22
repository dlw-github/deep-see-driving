 Deep See Driving
====================

## Overview

This repository contains code for a graduate student project for Harvard Extension School's
DGMD E-17 Robotics, Autonomous Vehicles, Drones, and Artificial Intelligence class.

This project builds a data capture platform consisting of a stereo camera pair
and Velodyne Puck Hi-Res LiDAR sensor. Using the captured data, it trains a computer vision
model to predict depth maps, relying solely on the stereo camera
sensor data as an input.

The final project report, which explains the project in more detail, is 
available [here](https://docs.google.com/document/d/1SI0otW8thLhnat8kOqcOyT9Yd1F_Rv1EcEzfGSpaOZ0/edit?usp=share_link).

We also have a video overview of the project and results available [here](https://drive.google.com/file/d/1VeXNqEOcZzDiOvtsKE_OKAsWMVVRiA0m/view?usp=share_link).

## Robot setup on StereoPi v2 with Raspberry Pi Compute Module 4

* Download OS image from https://www.mediafire.com/file/bf7wwnvb6vv9tgb/opencv-4.5.3.56.zip/file
* Install OS image on SD card using Raspberry Pi installer
    Settings:
        - Hostname: testpi.local
        - Enable SSH
        - user: pi
        - password: deepsee
        - Wireless LAN config
* ssh into testpi.local
* sudo raspi-config
    - set max resolution
    - enable VNC
    - enable camera
    - expand filesystem
    - enable predictable network interface names
    - reboot
* sudo apt update
* sudo apt install git
* git clone https://github.com/andrewtratz/deep-see-driving
* sudo apt-get install tcpdump
* pip install --pre scapy[basic]
* pip install picamera
* pip install -r requirements.txt

# Capturing data on the Robot

To capture stereo camera data on the robot, run the capture.py
script from the ./Robot subfolder.

While camera data is being recorded, the user should run tcpdump command line utility
simultaneously to write .pcap files to disk.

# Calibrating the sensors

Default sensor calibration settings are specified in **calibration_settings.py**.

If calibration adjustments are required, the file **manual_calibration.py** can be
used to overlay LiDAR and camera images and test out modifications to the
predefined calibration values.

# Preprocessing

Two scripts are provided for preprocessing the camera images and LiDAR data, respectively.
To use, first update the data paths in config.py, then run:
* **process_raw_images.py**
* **process_raw_pcap.py**

# Training

To train a new model, check settings in **config.py** and run the **train.py** script.

# Inference

The python program **inference.py** can take a directory as input (update path
in config.py), run inference given a set of pretrained model weights, and output depth map visualizations to a
specified directory.

Pre-trained weights are provided in the "trained_weights" folder. 
* kitti_weights.pth for model trained on KITTI dataset
* deepsee_weights.pth for model trained on Deep See dataset

The python program **makevideo.py** can be run on the output of the prior step to convert these into a video.