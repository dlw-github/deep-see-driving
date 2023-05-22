from config import *
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

##########################
# makevideo.py
#
# Stitch output images together into a video
#
#########################

base_path = SOURCE_IMAGE_DIRECTORY
depth_path = DEPTH_IMAGE_DIRECTORY

img_list = []

print('Combining files')

# Traverse the list of source images
walk = os.walk(base_path)
for entry in walk:
    dir, subdir, files = entry
    for file in files:
        if '.png' in file or '_left':

            # Open the file
            base = np.asarray(Image.open(os.path.join(base_path, file)))

            # Check if the depth map has been created for this file
            if os.path.exists(os.path.join(depth_path, file)):
                # Apply a light Gaussian blur
                depth = cv2.GaussianBlur(np.asarray(Image.open(os.path.join(depth_path, file))),
                                         (5, 5), cv2.BORDER_DEFAULT)
                # Create an overlay of the two images
                mixed = ((base / 2) + (depth / 3)).astype(np.uint8)
                size = (base.shape[1] * 3 // 4, base.shape[0] * 9 // 4)

                # Stack the three images together
                out = cv2.resize(cv2.cvtColor(np.concatenate((base, mixed, depth), axis=0), cv2.COLOR_BGR2RGB),
                                 size, cv2.INTER_AREA)

                img_list.append(out)

# Write out the video output
print('Writing video...')
vid = cv2.VideoWriter(OUTPUT_FILE_NAME, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
for i in range(len(img_list)):
    vid.write(img_list[i])
vid.release()
