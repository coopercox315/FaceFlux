import torchvision.transforms as T

#Transform for the generator branch (ContentEncoder, Decoder)
generator_transform = T.Compose([
    T.Resize((128, 128), T.InterpolationMode.BILINEAR), #hardcoded for now as our model expects only 128x128 images
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #our network output uses tanh activation, so we normalize to [-1, 1]
])

#Transform for the identity branch (IdentityEncoder)
identity_transform = T.Compose([
    T.Resize((112, 112), T.InterpolationMode.BICUBIC), #ArcFace model expects 112x112 images 
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #mean and std for ImageNet (which pretrained ArcFace model was trained on)
])