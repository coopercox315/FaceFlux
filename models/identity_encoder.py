import torch
import torch.nn as nn
from dependencies.insightface.recognition.arcface_torch.backbones import get_model as arcface_get_model

class IdentityEncoder(nn.Module):
    """
    Identity Encoder which identifies features from a face in an image using a pre-trained ArcFace model.
    """
    def __init__(self, pretrained=True, model_path=None):
        """
        Args:
        - pretrained (bool): whether to load pre-trained weights
        - model_path (str): path to the pre-trained weights
        """
        super(IdentityEncoder, self).__init__()
        #load Arcface backbone using the ResNet100 architecture (r100)
        self.model = arcface_get_model('r100', fp16=False) #using full precision for now
        if pretrained:
            self.model.load_state_dict(torch.load(model_path))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass:
        - x (tensor): (batch_size, 3, H, W) input image
        Returns:
        - tensor: (batch_size, 512) features of the face in the image (Identity embedding)
        """
        embedding = self.model(x)
        return embedding
        