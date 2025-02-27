import os
from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    """
    FaceDataset loads images from a given directory and applies transformations to them.
    It assumes the given directory contains image files in .jpg, .jpeg or .png format.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
        - root_dir (str): path to the directory containing the images
        - transform (callable, optional): optional transform to be applied to the images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.fnames = [
            f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.fnames[idx])
        img = Image.open(img_path).convert('RGB') #convert to RGB to ensure 3 channels
        if self.transform:
            img = self.transform(img)

        return img