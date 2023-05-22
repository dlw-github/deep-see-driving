##########
#
# config.py
#
# Global configuration settings
#
##########

# Parameters to swap between models
TYPE = 'DeepSee' # Set to 'KITTI' for KITTI training or inference
MODEL_PATH = r'./trained_weights/deepsee_weights.pth'

# Training data locations
KITTI_DEPTH_PATH_TRAIN = r'C:\Users\andre\Dropbox\~DGMD E-17\Project\Datasets\KITTI\data_depth_annotated\train'
KITTI_DEPTH_PATH_CV = r'C:\Users\andre\Dropbox\~DGMD E-17\Project\Datasets\KITTI\data_depth_annotated\val'
KITTI_IMAGE_PATH_TRAIN = r'D:\KITTI'
KITTI_IMAGE_PATH_CV = r'D:\KITTI'

DEEPSEE_PATH_TRAIN = r'../DeepSeeData/Processed'
DEEPSEE_PATH_CV = r'../DeepSeeData/CV'

# Preprocessing parameters
CROP_SIZE = 120
PROC_DATA_DIR = r'../DeepSeeData/Raw'
PROC_OUTPUT_DIR = r'../DeepSeeData/Processed'

# Training parameters
FREEZE = False
BATCH_SIZE = 24
EPOCHS = 150
LR = 1e-2
GRAD_CLIP = 1.0

# Inference parameters
INFERENCE_INPUT = r'D:\run3\camera\run3'
INFERENCE_OUTPUT = r'D:\out'

# Video parameters
SOURCE_DIRECTORY = r'D:\KITTI\2011_09_26_drive_0005_sync_2\2011_09_26\2011_09_26_drive_0005_sync\image_02\data'
DEPTH_IMAGE_DIRECTORY = r'D:\out'
OUTPUT_FILE_NAME = r'D:\project0005_v2.avi'

# Mean testing parameters
DATA_DIR = r'../DeepSeeData/Processed/lidar/run7'