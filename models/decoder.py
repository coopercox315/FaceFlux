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
    ...

        