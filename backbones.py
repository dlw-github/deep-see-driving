from blocks import ResidualBlock, ResNet
import torch
import torch.nn as nn

##################
# backbones.py
#
# Define the ResNet backbone for the neural network, specifying number of blocks to use
# All convolutions are 3x3 with stride 1
# Configuration defined as 3x64 block, 2x128 block, 1x256 block, 1x512 block
#
##################

# Define a ResNet-like neural network model composed of residual blocks inside a modified ResNet architecture
class ResNetLike(nn.Module):
    def __init__(self):
        super(ResNetLike, self).__init__()

        # Six channel ResNet-like model with block configuration
        self.model = ResNet(6, ResidualBlock, [3, 2, 1, 1])

    # Pass the tensor to the model for processing
    def forward(self, x):
        x = self.model(x)
        return x

