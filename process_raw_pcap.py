import os
from tqdm import tqdm
import velodyne_decoder as vd
import numpy as np
from calibration_settings import *
from config import PROC_DATA_DIR, PROC_OUTPUT_DIR

#############
#
# process_raw_pcap.py
#
# LiDAR data preprocessing script
#
#############

data_dir = PROC_DATA_DIR
output_dir = PROC_OUTPUT_DIR

# Create Velodyne decoder config object
config = vd.Config(model=model_type, rpm=rpm, calibration_file=calibration_file)
config.min_angle = 360 - fov/2
config.max_angle = fov/2

# Define transformations based on calibration_settings.py inputs
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
C = np.array([[fl, 0, py, 0],
              [0, fl, px, 0],
              [0, 0, 1, 0]])


# Walk though all of the input files
walk = os.walk(data_dir)
for entry in tqdm(walk):
    dir, subdir, files = entry

    for file in files:
        skipped_frames = 0
        if '.pcap' in file:

            pcap_file = os.path.join(dir, file)
            print("Processing " + pcap_file)

            # Make output directory if needed
            subdirectory_path = dir[len(data_dir):]
            if not os.path.exists(output_dir + subdirectory_path):
                os.makedirs(output_dir + subdirectory_path)

            # Skip any pcap file that the velodyne decoder thinks is corrupted
            try:
                # Test to make sure we can actually open the file and read first line
                for stamp, points in vd.read_pcap(pcap_file, config):
                    break
            except:
                continue

            # Iterate through each frame in the velodyne decoder output
            for stamp, points in vd.read_pcap(pcap_file, config):

                # Skip frames with very little data
                if len(points) < minimum_data:
                    skipped_frames += 1
                    continue

                millisecond_stamp = str(int(stamp * 1000))  # Get timestamp in milliseconds
                data = np.array(points)

                # Add distance as a column to the dataset
                dist = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2 + data[:, 2] ** 2)  # Euclidean distance
                data = np.hstack((data, np.expand_dims(dist, 1)))

                # Filter all points behind the view
                data = data[data[:, 0] >= 0.0]

                # Reorder and reorient axes
                newdata = np.hstack(
                    [np.expand_dims(dim, 1) for dim in (-data[:, 2], -data[:, 1], data[:, 0], data[:, 6])])

                # Format data into Nx4 matrix of [X, Y, Z, 1]
                vec = newdata[:, 0:3].T
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
                projected = np.hstack((projected, np.expand_dims(newdata[:, 3], 1)))

                # Scale up - seems to work better when height and width reversed, for some strange reason
                projected[:, 0] = np.round((projected[:, 0] * (image_height / 2)) + (image_height / 2), 0)
                projected[:, 1] = np.round((projected[:, 1] * (image_width / 2)) + (image_width / 2), 0)

                # Clip points to viewable area
                projected = projected[np.amin(projected[:,0:2], axis=1) > 0] # Must be greater than zero
                projected = projected[projected[:, 0] < image_width] # Can't exceed image height
                projected = projected[projected[:, 1] < image_height] # Can't exceed image width

                # Make sure we have enough data points available to use as a valid train case
                if projected.shape[0] < minimum_data:
                    skipped_frames += 1
                    continue

                # Create output depth map
                d_map = np.zeros((image_height, image_width), dtype=float)

                # Function to fill in depth map values, keeping lowest non-zero entries
                def fill_vals(a, **kwargs):
                    for d_map in kwargs.values():
                        x = int(a[1])
                        y = int(a[0])
                        d = a[3]
                        if d_map[y, x] == 0.0 or d_map[y, x] > d:
                            d_map[y, x] = d

                # Fill in values of depth map, keeping lowest non-zero entries
                np.apply_along_axis(fill_vals, axis=1, arr=projected, kwargs=d_map)

                # Drop frames which have fewer than minimum_data data points
                if np.count_nonzero(d_map) < minimum_data:
                    skipped_frames += 1
                    continue

                # Save output as .npz
                outfile = os.path.join(output_dir + subdirectory_path, 'depth_' + str(millisecond_stamp) + '.npz')
                np.savez_compressed(outfile, d_map)
            print('Skipped ' + str(skipped_frames) + ' frames\n')
