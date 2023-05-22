import os
import time
from datetime import datetime
import picamera
from picamera import PiCamera
import cv2
import numpy as np

epoch = datetime.utcfromtimestamp(0)

def unix_time_millis(dt):
    return int(round((dt - epoch).total_seconds() * 1000.0,0))

# Camera settings
cam_width = 1280  # Cam sensor width settings
cam_height = 480  # Cam sensor height settings

# Final image capture settings
scale_ratio = 1

# Camera resolution height must be dividable by 16, and width by 32
cam_width = int((cam_width + 31) / 32) * 32
cam_height = int((cam_height + 15) / 16) * 16
print("Camera resolution: " + str(cam_width) + " x " + str(cam_height))

# Buffer for captured image settings
img_width = int(cam_width * scale_ratio)
img_height = int(cam_height * scale_ratio)
capture = np.zeros((img_height, img_width, 4), dtype=np.uint8)
print("Scaled image resolution: " + str(img_width) + " x " + str(img_height))

# Initialize the camera
camera = PiCamera(stereo_mode='side-by-side', stereo_decimate=False)
camera.resolution = (cam_width, cam_height)
camera.framerate = 20
# camera.hflip = True

# Lets start taking photos!
t2 = datetime.now()
print("Starting photo sequence")
counter = 0
if (os.path.isdir("../photos") == False):
    os.makedirs("../photos")
for frame in camera.capture_continuous(capture, format="rgba", use_video_port=True, resize=(img_width, img_height)):
    counter += 1
    filename = '../photos/photo_' + str(img_width) + 'x' + str(img_height) + '_' + \
               str(unix_time_millis(datetime.now())) + '.jpg'
    cv2.imwrite(filename, frame)
    time.sleep(0.05)
