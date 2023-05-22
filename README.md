 Deep See Driving
====================

# Overview
This project builds a data capture platform consisting of a stereo camera pair
and Velodyne Puck Hi-Res LiDAR sensor. Using the captured data, it trains a computer vision model to predict depth maps, relying solely on the stereo camera sensor data as an input.

[A video overview of the project and results available here](https://youtu.be/wh4yxYBXdNc).

[The final project report, which explains the project in more detail, is available here](https://docs.google.com/document/d/1SI0otW8thLhnat8kOqcOyT9Yd1F_Rv1EcEzfGSpaOZ0/edit?usp=share_link).

*Originally developed by Andrew Tratz, Daniel Waldroop, and Eric Margolis for a graduate student project for Harvard University's DGMD E-17 Robotics, Autonomous Vehicles, Drones, and Artificial Intelligence class.*

## Robot setup on StereoPi v2 with Raspberry Pi Compute Module 4

1. Download OS image from https://www.mediafire.com/file/bf7wwnvb6vv9tgb/opencv-4.5.3.56.zip/file
2. Install OS image on SD card using Raspberry Pi installer
    Settings:
    - Hostname: testpi.local
    - Enable SSH
    - user: pi
    - password: deepsee
    - Wireless LAN config
3.  ssh into testpi.local
4. ```sudo raspi-config```
    - set max resolution
    - enable VNC
    - enable camera
    - expand filesystem
    - enable predictable network interface names
    - reboot
5. ```sudo apt update```
6. ```sudo apt install git```
7. ```git clone https://github.com/dlw-github/deep-see-driving```
8. ```sudo apt-get install tcpdump```
9. ```pip install --pre scapy[basic]```
10. ```pip install picamera```
11. ```pip install -r requirements.txt```

## Capturing data on the Robot
To capture stereo camera data on the robot, run [capture.py](Robot/capture.py).

While camera data is being recorded, the user should run tcpdump command line utility
simultaneously to write .pcap files to disk.

## Calibrating the sensors
Default sensor calibration settings are specified in [calibration_settings.py](calibration_settings.py).

If calibration adjustments are required, the file [manual_calibration.py](manual_calibration.py) can be used to overlay LiDAR and camera images and test out modifications to the predefined calibration values.

## Preprocessing
Two scripts are provided for preprocessing the camera images and LiDAR data, respectively.
To use, first update the data paths in [config.py](config.py), then run:

- [process_raw_images.py](process_raw_images.py)
- [process_raw_pcap.py](process_raw_pcap.py)

## Training
To train a new model, check settings in [config.py](config.py) and run [train.py](train.py).

## Inference
[inference.py](inference.py) can take a directory as input (update path in [config.py](config.py)), run inference given a set of pretrained model weights, and output depth map visualizations to a specified directory.

Pre-trained weights are provided in [trained_weights](trained_weights). 

- [kitti_weights.pth](trained_weights/kitti_weights.pth) for model trained on KITTI dataset
- [deepsee_weights.pth](trained_weights/deepsee_weights.pth) for model trained on Deep See dataset

[makevideo.py](makevideo.py) can be run on the output of the prior step to convert these into a video.