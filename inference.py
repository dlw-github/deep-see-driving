from config import *
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from backbones import ResNetLike
from PIL import Image
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from dataset import KITTIDataset


##########################
# inference.py
#
# Perform inference using a trained model
# Does batch inference across images in a defined directory
# Output images can then be stitched together into video using makevideo.py
#
#########################

# Do we have a GPU available?
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# Flag to indicate the source of the file to perform inference on
inference_type = TYPE

if inference_type == 'KITTI':
    crop_pattern = (0, 120, 1242, 375)

model_path = MODEL_PATH

# Create overlapping patchwork of different crops of the image
# These will be averaged together to create the final output
def create_patchwork(left_image, right_image):
    w, h = left_image.size
    crop_ul = [] # List of upper-left points of each patch

    # Create set of overlapping patch upper-left points
    for y in range(0, h - CROP_SIZE, 20):
        for x in range(0, w - CROP_SIZE, 20):
            crop_ul.append((y, x))
        crop_ul.append((y, w - CROP_SIZE))
    for x in range(0, w - CROP_SIZE, 20):
        crop_ul.append((h - CROP_SIZE, x))
    crop_ul.append((h - CROP_SIZE, w - CROP_SIZE))
    crop_ul = list(set(crop_ul)) # Make unique

    # Set up return data structures
    patch_count = len(crop_ul)
    patches = torch.from_numpy(np.ndarray((patch_count, 6, CROP_SIZE, CROP_SIZE), dtype=np.float32))
    overlaps = np.zeros((h, w), dtype=np.uint8)

    # Create a transform function to convert into pytorch tensors
    tensor_conv = transforms.ToTensor()

    # Create the individual patches
    for patch_id, ul in zip(range(len(patches)), crop_ul):
        y, x = ul
        crop = (x, y, x+CROP_SIZE, y+CROP_SIZE)
        patches[patch_id, :3] = tensor_conv(left_image.crop(crop))
        patches[patch_id, 3:] = tensor_conv(right_image.crop(crop))
        overlaps[y:y+CROP_SIZE, x:x+CROP_SIZE] = np.add(overlaps[y:y+CROP_SIZE, x:x+CROP_SIZE], 1)

    return patches, crop_ul, overlaps

# Instantiate our model
model = ResNetLike().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

walk = os.walk(INFERENCE_INPUT)

# Traverse the input directory
p = None # Percentile data
for entry in walk:
    dir, subdir, files = entry
    for file in files:
        skipped_frames = 0
        if '.png' in file or '_left' in file:
            left_image_path = os.path.join(dir, file)

            # Open the right and left images
            if inference_type == 'KITTI':
                right_image_path = left_image_path.replace('image_02', 'image_03')
                left_image = Image.open(left_image_path).crop(crop_pattern)
                right_image = Image.open(right_image_path).crop(crop_pattern)
            else:
                right_image_path = left_image_path.replace('left', 'right')
                left_image = Image.open(left_image_path)
                right_image = Image.open(right_image_path)

            # Create a patchwork of overlapping patches of the images
            patches, crop_ul, overlaps = create_patchwork(left_image, right_image)

            # Set up our output array
            w, h = left_image.size
            output_data = np.zeros((h, w), dtype=np.float32)

            # Do individual patch inference
            # Could make faster by doing batch inference, if needed (not implemented)
            for patch_id, ul in tqdm(zip(range(patches.shape[0]), crop_ul)):
                y, x = ul
                with torch.no_grad():
                    out = model(patches[patch_id].unsqueeze(0).to(device))
                output_data[y:y+CROP_SIZE, x:x+CROP_SIZE] += out.squeeze(0).cpu().numpy()

            # Divide by the number of overlapping patches per pixel to get average output
            output_data = np.divide(output_data, overlaps)

            # Convert the output from normalized form into actual output
            if inference_type == 'KITTI':
                # De-normalize
                output_data *= 3186
                output_data += 4582

                # For visualization, posterize the depth map by percentile value
                percs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
                percs.sort(reverse=True)
                if p is None:  # Use first image to calibrate the percentiles - fixed going forward.
                    p = np.percentile(output_data.flatten(), percs)

                # Array of flags to see which values have already been changed to percentiles
                changed = np.zeros(output_data.shape, dtype=bool)

                # Apply posterization
                for val, perc in zip(list(p), percs):
                    mask = output_data > val
                    output_data[np.logical_and(output_data > val, ~changed)] = perc * (255 / 90)
                    changed[mask] = True
                output_data[~changed] = 0.0
                output_data = output_data.astype(np.uint8)

            if inference_type == 'DeepSee':
                # De-normalize
                output_data *= 0.318
                output_data += 1.05
                output_data[output_data < 0] = 0 # Threshold negative values are black

                # For visualization, posterize the depth map by percentile value
                percs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
                percs.sort(reverse=True)
                if p is None: # Use first image to calibrate the percentiles - fixed going forward.
                    p = np.percentile(output_data.flatten(), percs)

                # Array of flags to see which values have already been changed to percentiles
                changed = np.zeros(output_data.shape, dtype=bool)

                # Apply posterization
                for val, perc in zip(list(p), percs):
                    mask = output_data > val
                    output_data[np.logical_and(output_data > val, ~changed)] = perc * (255 / 90)
                    changed[mask] = True
                output_data[~changed] = 0.0
                output_data = output_data.astype(np.uint8)

            # Write the output image
            cv2.imwrite(os.path.join(INFERENCE_OUT, file), cv2.applyColorMap(output_data, cv2.COLORMAP_RAINBOW))




