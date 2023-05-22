import os
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import random
from tqdm import tqdm


##########
#
# dataset.py
#
# PyTorch Datasets for the Deep See project data
# Two Dataset classes are provided: one for KITTI, and the other for the custom DeepSee data set
#
##########

# Class for our custom data
class DeepSeeDataset(Dataset):

    # Initialization with directories and transforms specified
    def __init__(self, img_root_dir, depth_root_dir, transform=None, source_additional_transform=None, random_flip=True):
        self.img_root_dir = img_root_dir
        self.depth_root_dir = depth_root_dir
        self.transform = transform
        self.source_additional_transform = source_additional_transform
        self.random_flip = random_flip

        # Populate directory of file paths and depth paths, indexed by timestamp
        self.file_paths = {}
        self.depth_paths = {}
        self.data_map = {} # Map valid image files to valid depth maps

        self.valid_paths = []

        # Traverse files and build up list of file paths
        walk = os.walk(self.img_root_dir)
        for entry in walk:
            dir, subdir, files = entry
            for file in files:
                if file[-9:] == '_left.jpg':
                    # Extract timestamp from filename
                    ts = int(file[-22:-9])
                    self.file_paths[ts] = os.path.join(dir, file)

        # Populate the list of depth paths (ground truth labels)
        walk = os.walk(self.depth_root_dir)
        for entry in walk:
            dir, subdir, files = entry
            for file in files:
                if file[-4:] == '.npz':
                    # Extract timestamp from filename
                    ts = int(file[-17:-4])
                    self.depth_paths[ts] = os.path.join(dir, file)

        # Map corresponding timestamps
        ts_images = np.sort(np.asarray(list(self.file_paths.keys())))
        ts_depths = np.sort(np.asarray(list(self.depth_paths.keys())))
        ts_depth_min = ts_depths[0]
        ts_depth_max = ts_depths[-1]
        ts_image_min = ts_images[0]
        ts_image_max = ts_images[-1]

        # Ignore data outside of other sensor's time range
        ts_images = ts_images[np.logical_and(ts_images > ts_depth_min, ts_images < ts_depth_max)]
        ts_depths = ts_depths[np.logical_and(ts_depths > ts_image_min, ts_depths < ts_image_max)]

        # Sync the timestamps to find depth and camera images which are within 200ms of each other
        last_found = 0
        print('Syncing timestamps...')
        for ts_i in tqdm(ts_images):
            best_delta = 201 # Only accept matches within 200 milliseconds of each other
            best_match = None
            for i, ts_d in zip(range(len(ts_depths[last_found:])), ts_depths[last_found:]):
                delta = abs(ts_i - ts_d)
                if delta < best_delta:
                    best_delta = delta
                    best_match = ts_d
                elif ts_d > ts_i and delta > best_delta: # Short-circuit if we've passed our mark
                    if best_match is not None:
                        last_found = max(0, i - 2)  # Update last_found so we can short circuit the beginning search
                        self.data_map[self.file_paths[ts_i]] = self.depth_paths[best_match] # Create 1:1 file correspondence
                        self.valid_paths.append(self.file_paths[ts_i])
                    break

    # Return the total length of the Dataset
    def __len__(self):
        return len(self.valid_paths)

    # Retrieve a single item (defined as img*2 and depth label) from the Dataset
    def __getitem__(self, idx):

        # Will we flip?
        if self.random_flip:
            self.flip = random.choice(('none', 'vertical', 'horizontal'))
        else:
            self.flip = 'none'

        # Treat the original path as left, get revised path for right image
        left_image_path = self.valid_paths[idx]
        right_image_path = left_image_path.replace('_left', '_right')
        depth_path = self.data_map[left_image_path]

        # Load the images
        left_image = Image.open(left_image_path)
        right_image = Image.open(right_image_path)
        with np.load(depth_path) as npz_file:
            depth_image = npz_file['arr_0']

        # Apply transform augmentations to the data
        # Convert images into pyTorch tensor data format, which will be used for analysis
        tensor_conv = transforms.ToTensor() # Also normalizes at the same time
        left_image = tensor_conv(left_image).to(torch.float32)
        right_image = tensor_conv(right_image).to(torch.float32)
        depth_image = tensor_conv(depth_image).squeeze(0)

        # If we're flipping, pretend we're also swapping eyes
        if self.flip != 'none':
            temp = left_image
            left_image = right_image
            right_image = temp

        # Create a Boolean mask of locations where depth information is provided
        valid_mask = depth_image > 0

        # Normalize the depth image
        depth_image = torch.div(torch.subtract(depth_image, 1.05), 0.318).to(torch.float32) # Subtract mean and divide by std

        # Stack the data into a single tensor so we apply the same random augmentations to everything
        full_data = torch.vstack((left_image, right_image, depth_image.unsqueeze(0),
                                  valid_mask.to(dtype=torch.float32).unsqueeze(0)))

        # Apply the data augmentations
        if self.transform:
            full_data = self.transform(full_data)

        # Apply flipping transform
        if self.flip != 'none':
            if self.flip == 'vertical':
                flip = transforms.RandomVerticalFlip(1.0)
            else:
                flip = transforms.RandomHorizontalFlip(1.0)
            full_data = flip(full_data)

        # Split the data back into our source data and ground truth
        source_data = full_data[0:full_data.shape[0]-2]
        ground_truth = full_data[-2:-1].squeeze(0)
        valid_mask = full_data[-1:].squeeze(0).to(dtype=torch.bool)

        # Apply additional augmentations only to the source data (things we don't want to affect ground truth)
        if self.source_additional_transform:
            source_data = self.source_additional_transform(source_data)

        return source_data, ground_truth, valid_mask


# pyTorch Dataset for the KITTI autonomous driving data
# Will recursively search img_dir and depth_dir to build up the dataset and apply specified transformations
class KITTIDataset(Dataset):

    # Initialization with directories and transforms specified
    def __init__(self, img_root_dir, depth_root_dir, img_dirs, transform=None, source_additional_transform=None):
        self.img_root_dir = img_root_dir
        self.depth_root_dir = depth_root_dir
        self.img_dirs = img_dirs
        self.transform = transform
        self.source_additional_transform = source_additional_transform

        # Populate the list of file paths and depth paths
        self.file_paths = []
        self.depth_paths = []

        for img_dir in img_dirs:
            walk = os.walk(os.path.join(self.img_root_dir, img_dir + '_2'))
            for entry in walk:
                dir, subdir, files = entry
                if 'image_02' in dir:
                    for file in files:
                        if file[-4:] == '.png':
                            # First several frames of each video have no ground truth data, skip them
                            if file.split('.')[0] in ['0000000000', '0000000001', '0000000002', '0000000003', '0000000004']:
                                continue
                            self.file_paths.append(dir + '\\' + file)

            # Populate the list of depth paths (ground truth labels)
            walk = os.walk(os.path.join(self.depth_root_dir, img_dir))
            for entry in walk:
                dir, subdir, files = entry
                if 'image_02' in dir:
                    for file in files:
                        if file[-4:] == '.png':
                            self.depth_paths.append(dir + '\\' + file)
                            # Get rid of file paths which don't exist in the depth ground truth
                            while self.depth_paths[-1].split('\\')[-1] != self.file_paths[len(self.depth_paths)-1].split('\\')[-1]:
                                print('Discarding ' + self.file_paths[len(self.depth_paths)-1])
                                self.file_paths.remove(self.file_paths[len(self.depth_paths)-1])

            # Truncate file paths, since some of the final depths may not exist
            self.file_paths = self.file_paths[0:len(self.depth_paths)]

    # Return the total length of the Dataset
    def __len__(self):
        return len(self.file_paths)

    # Retrieve a single item (defined as img*2 and depth label) from the Dataset
    def __getitem__(self, idx):

        # Treat the original path as left, get revised path for right image
        left_image_path = self.file_paths[idx]
        right_image_path = left_image_path.replace('image_02', 'image_03')
        depth_path = self.depth_paths[idx]

        # Apply a crop to the data since the LiDAR scans don't provide info for the top third of the image
        crop_pattern = (0, 120, 1242, 375)
        left_image = Image.open(left_image_path).crop(crop_pattern)
        right_image = Image.open(right_image_path).crop(crop_pattern)
        depth_image = Image.open(depth_path).crop(crop_pattern)

        # Debugging code to show comparative frames
        # dst = Image.new('RGB', (left_image.width, left_image.height + left_image.height))
        # dst.paste(left_image, (0, 0))
        # dst.paste(depth_image, (0, left_image.height))
        # dst.show()

        # Apply transform augmentations to the data
        # Convert images into pyTorch tensor data format, which will be used for analysis
        tensor_conv = transforms.ToTensor()
        left_image = tensor_conv(left_image)
        right_image = tensor_conv(right_image)
        depth_image = tensor_conv(depth_image).to(torch.float32).squeeze(0)

        # Create a Boolean mask of locations where depth information is provided
        valid_mask = depth_image > 0

        # Normalize the depth image
        depth_image = torch.div(torch.subtract(depth_image, 4582.0), 3186) # Subtract mean and divide by std

        # Stack the data into a single tensor so we apply the same random augmentations to everything
        full_data = torch.vstack((left_image, right_image, depth_image.unsqueeze(0),
                                  valid_mask.to(dtype=torch.uint8).unsqueeze(0)))

        # Apply the data augmentations
        if self.transform:
            full_data = self.transform(full_data)

        # Split the data back into our source data and ground truth
        source_data = full_data[0:full_data.shape[0]-2]
        ground_truth = full_data[-2:-1].squeeze(0)
        valid_mask = full_data[-1:].squeeze(0).to(dtype=torch.bool)

        # Apply additional augmentations only to the source data (things we don't want to affect ground truth)
        if self.source_additional_transform:
            source_data = self.source_additional_transform(source_data)

        return source_data, ground_truth, valid_mask



