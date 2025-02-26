import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    """
    Attention Block module which applies a self-attention mechanism to the input feature maps,
    allowing the network to focus on important regions.
    This refines the feature maps during the upsampling process in the decoder by attending to the most relevant regions,
    e.g the eyes, nose, and mouth in a face image.
    """
    def __init__(self, in_channels):
        """
        Args:
        - in_channels (int): number of channels in the input features
        """
        super(AttentionBlock, self).__init__()
        #A simple attention block using two 1x1 convolutional layers
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1), #reduce the number of channels by 8, when using 128 channels, this outputs 16 channels
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1), #restore the number of channels to the original value
            nn.Sigmoid() #sigmoid activation to produce attention weights in the range [0, 1] for each channel at each spatial location
        )

    def forward(self, x):
        """
        Forward pass: apply self-attention mechanism to the input features
        - x (tensor): input feature maps of shape (batch_size, in_channels, H, W)
        Returns:
        - tensor: refined feature maps weighted by the attention mask
        """
        attn_weights = self.attn(x)
        return x * attn_weights
    
class Decoder(nn.Module):
    """
    Decoder reconstructs a high-resolution face image from a fused latent vector.
    It upsamples the latent vector using transposed convolutions and applies attention blocks
    to enhance the quality of the reconstructed features. 
    """
    def __init__(self, latent_dim=768, out_channels=3): #768 represents the fused latent vector size
        """
        Args:
        - latent_dim (int): dimension of the fused latent vector (default: 768)
        - out_channels (int): number of channels in the output image (default: 3 for RGB)
        """
        super(Decoder, self).__init__()
        #Fully connected layer to map the fused latent vector to a feature map
        self.fc = nn.Linear(latent_dim, 256*16*16) #16x16 feature map with 256 channels
        #A series of transposed convolutional layers to upsample the feature map back to the original image size
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), #16x16 -> 32x32, outputs 128 channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            AttentionBlock(128), #apply attention to refine the feature maps
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), #32x32 -> 64x64, outputs 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            AttentionBlock(64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1), #64x64 -> 128x128, outputs 3 channels
            nn.Tanh() #tanh activation to scale the output pixels to the range [-1, 1], matching normalization used in training
        )

    def forward(self, z):
        """
        Forward pass:
        - z (tensor): (batch_size, latent_dim) fused latent vector
        Returns:
        - tensor: (batch_size, out_channels, 128, 128) reconstructed face image
        """
        x = self.fc(z) #map the fused latent vector to a flat feature map of shape (batch_size, 256*16*16)
        x = x.view(x.size(0), 256, 16, 16) #reshape to 4D tensor of shape (batch_size, 256, 16, 16)
        x = self.deconv(x) #upsample the feature map to reconstruct the full image of shape (batch_size, out_channels, 128, 128)
        return x