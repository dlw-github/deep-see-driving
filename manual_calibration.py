
import velodyne_decoder as vd
import numpy as np
import math
import cv2
from calibration_settings import *

#############
#
# calibrate.py
#
# Helper script to perform manual calibration of the sensors given camera and LiDAR data inputs
#
#############

# Load camera image and make corrections
source = cv2.imread("./calibration_inputs/cups_photo.jpg")
source = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
source = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
source = source // 4
source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)

# Extract both images
width = source.shape[1]//2
cam = source[:, :width, :]
cam_right = source[:, width:, :]

# Crop bottom of image
cam = cam[crop_bottom:, :, :]
cam_right = cam_right[crop_bottom:, :, :]

h, w, c = cam.shape

# Data input
pcap_file = r'.\calibration_inputs\velo.pcap'

# Read first frame of the pcap
config = vd.Config(model=model_type, rpm=rpm, calibration_file=calibration_file)
config.min_angle = 360 - fov/2
config.max_angle = fov/2
pcap_file = pcap_file
cloud_arrays = []

i = 0
for stamp, points in vd.read_pcap(pcap_file, config):
    i += 1
    cloud_arrays.append(points)
    if i == 800:
        break # Only get first set of data points

# Add distance as a column to the dataset
data = np.array(cloud_arrays[i-1])
dist = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
data = np.hstack((data, np.expand_dims(dist, 1)))

# Filter all points behind the view
data = data[data[:,0] >= 0.0]

# Reorder and reorient axes
newdata = np.hstack([np.expand_dims(dim, 1) for dim in (-data[:,2], -data[:,1], data[:,0], data[:,6])])


translation_matrix = np.array([[1, 0, 0, x_translation],
                               [0, 1, 0, y_translation],
                               [0, 0, 1, z_translation],
                               [0, 0, 0, 1]])

y_axis_rotation_matrix = np.array([[1, 0, 0, 0],
                                   [0, math.cos(y_axis_rotation), -math.sin(y_axis_rotation), 0],
                                   [0, math.sin(y_axis_rotation), math.cos(y_axis_rotation), 0],
                                   [0, 0, 0, 1]])

x_axis_rotation_matrix = np.array([[math.cos(x_axis_rotation), 0, math.sin(x_axis_rotation), 0],
                                   [0, 1, 0, 0],
                                   [0, -math.sin(x_axis_rotation), math.cos(x_axis_rotation), 0],
                                   [0, 0, 0, 1]])

# Camera matrix
C = np.array([[fl,0,py,0],
              [0,fl,px,0],
              [0,0,1,0]])

# Format data into Nx4 matrix of [X, Y, Z, 1]
vec = newdata[:,0:3].T
t = np.ones((1, vec.shape[1]))
vec = np.vstack((vec, t))

# Rotation
vec = np.matmul(x_axis_rotation_matrix, vec)
vec = np.matmul(y_axis_rotation_matrix, vec)

# Translation
vec = np.matmul(translation_matrix, vec)

# Project into camera space
projected = np.matmul(C, vec)

# Transpose
projected = projected.T

# Add the distance information back to the projected points
projected = np.hstack((projected, np.expand_dims(newdata[:,3], 1)))

# Resize to output size
projected[:,0] = np.round((projected[:,0] * (h / 2)) + (h/2), 0)
projected[:,1] = np.round((projected[:,1] * (w / 2)) + (w/2), 0)

# Create a test image
img = cam.copy()

# Populate the image with color-coded pixel values
def color_code(img, projected):
    for i in range(projected.shape[0]):
        row = projected[i]
        dist = row[3]

        max_clip = 1.25
        min_clip = 0.75

        if dist >= max_clip:
            color = 1
        elif dist <= min_clip:
            color = 0
        else:
            color = (dist - min_clip) / (max_clip - min_clip)

        if row[0] > 0 and row[0] < h and row[1] > 0 and row[1] < w:
            img[int(row[0]), int(row[1]), 0] = (1 - color) * 255
            img[int(row[0]), int(row[1]), 2] = (color) * 255
    return img

img = color_code(img, projected)
right = color_code(cam_right, projected)

# Display the test image
cv2.imshow("Calibration Test", img)
cv2.imwrite('output.jpg', img)
cv2.waitKey(0)

print(f'File: {str(pcap_file)} | Number of point cloud frames: {len(cloud_arrays)}')