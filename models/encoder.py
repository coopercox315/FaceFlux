import torch
import torch.nn as nn

class ContentEncoder(nn.Module):
    """
    Content Encoder which extracts the facial structure, pose, and expression from the source image (the image/face to be manipulated).
    """
    def __init__(self, in_channels=3, latent_dim=256):
        """
        Args:
        - in_channels (int): number of channels in the input image (default: 3 for RGB)
        - latent_dim (int): dimension of the latent content vector (default: 256)
        """
        super(ContentEncoder, self).__init__()
        #A series of convolutional layers to progresively extract features and downsample the image.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), #spacial size reduction from 128x128 -> 64x64, outputs 64 channels
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #64x64 -> 32x32, outputs 128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), #32x32 -> 16x16, outputs 256 channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        #Fully connected layer to project the flattened feature map into the latent space
        #For a 128x128 input image, the feature map will be 16x16 (after 3 conv layers with stride 2)
        self.fc = nn.Linear(256*16*16, latent_dim)

    def forward(self, x):
        """
        Forward pass:
        - x (tensor): (batch_size, 3, H, W) input image
        Returns:
        - tensor: (batch_size, latent_dim) latent content vector of the input image
        """
        x = self.conv(x) #extract features
        x = x.view(x.size(0), -1) #flatten the feature map
        latent = self.fc(x) #project to latent space
        return latent
        