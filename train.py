from config import *
import os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from backbones import ResNetLike

from dataset import KITTIDataset, DeepSeeDataset

##################################
#
# train.py
#
# Stereo vision model training routine
#
##################################

# Define GPU to use
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# Can train on KITTI data or DeepSee data
train_type = TYPE # Defined in config.py

# Load model from a checkpoint
model_path = MODEL_PATH

# Freeze model?
freeze_model = FREEZE

# Instantiate our model and optimizer
model = ResNetLike()
if model_path != '':
    print('Loading model from checkpoint')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

# Push the model onto the GPU
model = model.to(device)

# If we're freezing all but the last layer
if freeze_model:
    for param in model.parameters():
        param.requires_grad = False

    # Last layer of the model should still be trainable
    model.model.collapse.weight.requires_grad = True
    model.model.collapse.bias.requires_grad = True

# Put model in training mode
model.train()

# Set up our Adam optimizer object, to progressively learn the model weights
non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(non_frozen_parameters, lr=LR)

# A list of transformations to apply to both the source data and ground truth
basic_trans = transforms.Compose([
    transforms.RandomCrop(CROP_SIZE),
])

# A list of transformations to apply ONLY to the source data, not the depth maps
source_trans = transforms.Compose([
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(5, sigma=(0.05, 0.15)),
                                                transforms.ColorJitter()]))
])

# Define the Dataset objects by the type of training data we're using
if train_type == 'KITTI':
    # List of train and validation data (this split is provided by the KITTI dataset creators)
    train_folders = os.listdir(KITTI_DEPTH_PATH_TRAIN)
    val_folders = os.listdir(KITTI_DEPTH_PATH_CV)
    # Load the KITTI dataset
    data = KITTIDataset(KITTI_IMAGE_PATH_TRAIN, KITTI_DEPTH_PATH_TRAIN,
                        train_folders,
                        basic_trans, source_trans)

    data_cv = KITTIDataset(KITTI_IMAGE_PATH_CV,
                           KITTI_DEPTH_PATH_CV,
                           val_folders,
                           basic_trans, source_trans)
if train_type == 'DeepSee':
    train_folder = DEEPSEE_PATH_TRAIN
    val_folder = DEEPSEE_PATH_CV
    data = DeepSeeDataset(train_folder, train_folder, basic_trans, source_trans, True)
    data_cv = DeepSeeDataset(val_folder, val_folder, basic_trans, None, False)

# DataLoader objects for the two datasets
train_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
cv_loader = DataLoader(data_cv, batch_size=BATCH_SIZE, shuffle=False)

# Create a TensorBoard logging writer
writer = SummaryWriter()

# Our loss function - using mean squared error over all pixels where we have LiDAR depth data
def compute_loss(output, truth, valid_mask):
    loss = nn.MSELoss()
    q = output[valid_mask]
    r = truth[valid_mask]
    return loss(q, r)

# Run a big loop multiple times throughout the entire dataset
iter = 0
for epoch in range(EPOCHS):

    # Set the model in training mode
    model.train()

    # Training loop
    for i, (source, truth, valid_mask) in enumerate(train_loader):
        iter += 1

        # Run the model on the data and get its outputs
        outputs = model(source.to(device))

        # Compute the loss function for the output data
        loss = compute_loss(outputs, truth.to(device), valid_mask.to(device))

        # Print out loss term every 10 batches
        if iter % 10 == 0:
            loss_copy = loss.item()
            print('Epoch: %d\tBatch: %d\tLoss: %.4f' % (epoch, i+1, loss_copy))
            writer.add_scalar('loss/train', loss_copy, iter)

        # Backward gradient propagation to learn and improve the models
        optimizer.zero_grad()
        loss.backward()

        # Clip the gradient to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # Take a step down in the direction of steepest gradient
        optimizer.step()

    # Cross validation loop - the same, but we are testing, not learning
    losses = []
    model.eval() # Put the model in inference mode
    for i, (source, truth, valid_mask) in enumerate(cv_loader):
        # torch.no_grad makes it so the model won't learn anything new
        with torch.no_grad():
            outputs = model(source.to(device))
            loss = compute_loss(outputs, truth.to(device), valid_mask.to(device)).item()

            # Keep a running list of the calculated losses
            if not math.isnan(loss):
                losses.append(loss)

    # Compute the average loss over the validation set and print it
    avg_loss = sum(losses) / len(losses)

    print('-------------------------------')
    print('Epoch: %d\tLoss: %.2f' % (epoch, avg_loss))
    print('-------------------------------\n')
    writer.add_scalar('loss/val', avg_loss, iter)

    # Save an intermediate copy of the model weights
    torch.save(model.state_dict(), 'model' + str(epoch) + '.pth')
    torch.save(optimizer.state_dict(), 'optimizer' + str(epoch) + '.pth')



