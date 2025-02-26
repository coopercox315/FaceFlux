import torch
import torch.nn as nn
from models.encoder import ContentEncoder
from models.identity_encoder import IdentityEncoder
from models.decoder import Decoder

class FaceSwapModel(nn.Module):
    """
    FaceSwapModel combines the ContentEncoder, IdentityEncoder, and Decoder to perform face swapping.

    - The ContentEncoder extracts non-identity features (e.g pose, expression) from the source face
    - The IdentityEncoder (using ArcFace) extracts identity features from the target face
    - Their outputs are fused (by concatenation) to form a latent vector (e.g., 256 + 512 = 768 dimensions).
    - The Decoder then reconstructs a full-resolution face image from the fused latent vector.
    """
    def __init__(self, content_latent_dim=256):
        """
        Args:
        - content_latent_dim (int): dimensionality of the content encoder's output
        """
        super(FaceSwapModel, self).__init__()
        self.content_encoder = ContentEncoder(latent_dim=content_latent_dim) #outputs a vector of size content_latent_dim
        self.identity_encoder = IdentityEncoder(pretrained=True, model_path='ms1mv3_arcface_r100_fp16.pth') #outputs a 512-dimensional identity embedding
        self.identity_latent_dim = 512

        fused_dim = content_latent_dim + self.identity_latent_dim #calculates the size of the fused latent vector (sum of the other two)

        #Initialize the decoder which takes a fused latent vector and reconstructs the face image
        self.decoder = Decoder(latent_dim=fused_dim, out_channels=3) #outputs a 128x128 RGB image

    def fuse_latents(self, content_latent, identity_latent):
        """
        Fuse the content and identity latents via concatenation.
        Args:
        - content_latent (tensor): (batch_size, content_latent_dim) latent vector from the content encoder
        - identity_latent (tensor): (batch_size, identity_latent_dim) identity embedding from the identity encoder
        Returns:
        - tensor: (batch_size, content_latent_dim + identity_latent_dim) fused latent vector
        """
        return torch.cat([content_latent, identity_latent], dim=1)
    
    def forward(self, input_face, target_identity_face = None):
        """
        Forward pass:
        - input_face (tensor): (batch_size, 3, 128, 128) source face image tensor from which content features are extracted
        - target_identity_face (tensor, optional): (batch_size, 3, 128, 128) target identity face image tensor from which identity features are extracted
        Returns:
        - tensor: (batch_size, 3, 128, 128) reconstructed (swapped) face image.
        """
        #Extract the content features from the source/input face
        content_latent = self.content_encoder(input_face)
        #Decide which image to use for identity extraction, if not provided, use the input face (useful for self-swapping or debugging)
        identity_input = target_identity_face if target_identity_face is not None else input_face
        #Extract the identity features from the target identity
        identity_latent = self.identity_encoder(identity_input)
        #Fuse the content and identity latents into one latent vector
        fused_latent = self.fuse_latents(content_latent, identity_latent)
        #Decode the fused latent vector to reconstruct the swapped face image
        reconstructed_face = self.decoder(fused_latent)
        return reconstructed_face