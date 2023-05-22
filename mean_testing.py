'''
Samples Pcap files and finds mean and standard deviation of non-zero elements.
Returns average mean and standard deviation of sample.
'''
import numpy as np
import os
import random
from tqdm import tqdm
from config import DATA_DIR

#############
#
# mean_testing.py
#
# Helper script to calculate normalization parameters for the dataset
#
#############

# Calculate depth mean and std
def depth_mean(npz):
	npz = np.load(npz)
	depth_map = npz['arr_0']
	depth_map_nonzero = depth_map[np.nonzero(depth_map)] # Get non-zero elements
	std = np.std(depth_map_nonzero)
	mean = np.mean(depth_map_nonzero)
	return std, mean

data_dir = DATA_DIR

stdevs = []
means = []
sample = 1 # Sampling portion out of 1.0

# Calc mean and std of the files
random.seed(1023)
for file in tqdm(os.listdir(data_dir)):
	if file[-4:] == '.npz':
		if random.random() >= (1-sample):
			npz_path = os.path.join(data_dir, file)
			std, mean = depth_mean(npz_path)
			stdevs.append(std)
			means.append(mean)
print(f'Average mean: {sum(means)/len(means)} | Average std: {sum(stdevs)/len(stdevs)}')
