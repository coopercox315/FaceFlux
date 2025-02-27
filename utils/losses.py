import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=[3, 8, 15, 22], weights=[1/32, 1/16, 1/8, 1/4]): #layers as conv1_2, conv2_2, conv3_3, conv4_2 in VGG19
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.weights = weights
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0
        for i, layer_idx in enumerate(self.layers): #iterate over the specified layers
            x_features = self.vgg[:layer_idx+1](x) #extracts features from various layers of VGG19 up to each layer_idx
            y_features = self.vgg[:layer_idx+1](y)
            loss += self.weights[i] * torch.nn.functional.l1_loss(x_features, y_features) #L1 loss between feature maps
        return loss