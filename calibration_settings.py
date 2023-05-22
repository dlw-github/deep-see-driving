import math

##########
#
# calibration_settings.py
#
# Multi-sensor calibration configuration settings
# These are defined through trial-and-error process using manual_calibration.py
#
##########

# Focal length
fl = 2.3

# Define field of view
fov = 60

# Rotation transformation to apply
y_axis_rotation = math.radians(-5)
x_axis_rotation = math.radians(-12)

# Translation transformation to apply
x_translation = 0.23
y_translation = 0.01
z_translation = 0

# Define coordinates of principal point
px = 0
py = 0

# Should we auto-crop the bottom?
crop_bottom = 0

# Velodyne decoder settings
model_type = 'VLP-16' # VLP default settings will get overwritten by Puck Hi-Res configuration file
rpm = 600
calibration_file = r'Puck Hi-Res.yml'

# Fixed image dimensions
image_height = 480
image_width = 640

# Minimum data points for valid LiDAR scan
minimum_data = 100
