import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
from models.face_swap_model import FaceSwapModel
from models.discriminator import Discriminator
from utils.dataset import FaceDataset
from torchvision import transforms
from utils.transforms import generator_transform, identity_transform
from utils.losses import VGGPerceptualLoss

#Load configuration file for training
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Create datasets and dataloaders (images in these folders must be aligned and cropped already)
dataset_A = FaceDataset(root_dir='data/processed/subjectA', transform=generator_transform) #for source (content) images [128x128, normalized to [-1,1]]
dataset_B = FaceDataset(root_dir='data/processed/subjectB', transform=identity_transform) #for target (identity) images [112x112, ImageNet normalization for ArcFace]

dataloader_A = DataLoader(dataset_A, batch_size=config['batch_size'], shuffle=True)
dataloader_B = DataLoader(dataset_B, batch_size=config['batch_size'], shuffle=True)

#Initialize the generator (FaceSwapModel) and discriminator
generator = FaceSwapModel(content_latent_dim=config['content_latent_dim']).to(device)
discriminator = Discriminator(in_channels=3).to(device)

#Define optimizers for generator and discriminator
optimizer_G = optim.Adam(generator.parameters(), lr=config['lr'])
optimizer_D = optim.Adam(discriminator.parameters(), lr=config['lr'])

#Define loss functions
criterion_recon = nn.L1Loss() #L1 loss for reconstruction
criterion_adv = nn.BCEWithLogitsLoss() #Binary cross-entropy loss for adversarial loss
criterion_id = nn.CosineEmbeddingLoss() #Cosine embedding loss for comparing identity embeddings

if config['use_perceptual_loss'] == True: #Define perceptual loss if enabled
    criterion_perc = VGGPerceptualLoss().to(device) #VGG perceptual loss for comparing generated and target images
    normalize_for_vgg = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
else:
    criterion_perc = None

timestamp = datetime.datetime.now().strftime('%d%m%Y_%H%M%S')
checkpoint_dir = os.path.join('checkpoints', timestamp)
os.makedirs(checkpoint_dir, exist_ok=True)

#Training loop
def train():
    epochs = config.get('epochs', 50)
    for epoch in range(epochs):
        for batchA, batchB in zip(dataloader_A, dataloader_B):
            batchA = batchA.to(device)
            batchB = batchB.to(device)

            #Train Generator
            # -----------------
            optimizer_G.zero_grad()
            fake_face = generator(batchA, target_identity_face=batchB) #generate swapped face (content from A, identity from B)

            loss_recon = criterion_recon(fake_face, batchA) #reconstruction loss (compares generated face content to input face content)

            with torch.inference_mode():
                fake_identity = generator.identity_encoder(fake_face)
                target_identity = generator.identity_encoder(batchB)
            target_label = torch.ones(batchA.size(0)).to(device) #in cosine embedding loss, a label of 1 indicates similar embeddings
            loss_id = criterion_id(fake_identity, target_identity, target_label) #identity loss (compares generated identity to target identity)

            pred_fake = discriminator(fake_face) #predict if generated face is real or fake (generator tries to fool discriminator)
            valid_label = torch.ones_like(pred_fake).to(device)
            loss_adv = criterion_adv(pred_fake, valid_label) #adversarial loss (compares discriminator prediction to true label)

            if criterion_perc is not None: #if perceptual loss is enabled
                #convert fake_face and source image from [-1,1] to [0,1] and normalize for VGG
                fake_face_norm = (fake_face + 1) / 2
                batchA_norm = (batchA + 1) / 2
                fake_face_vgg = normalize_for_vgg(fake_face_norm)
                batchA_vgg = normalize_for_vgg(batchA_norm)
                loss_perc = criterion_perc(fake_face_vgg, batchA_vgg) #perceptual loss (compares high-level features of generated and input images)
            else:
                loss_perc = 0

            #Calculate total generator loss with lambda weights for each loss component
            lambda_recon = config.get('lambda_recon', 1.0) 
            lambda_id = config.get('lambda_id', 0.5)
            lambda_adv = config.get('lambda_adv', 0.1)
            lambda_perc = config.get('lambda_perc', 0.1)
            loss_G_total = (lambda_recon * loss_recon) + (lambda_id * loss_id) + (lambda_adv * loss_adv) + (lambda_perc * loss_perc)

            loss_G_total.backward()
            optimizer_G.step()

            #Train Discriminator
            # -----------------
            optimizer_D.zero_grad()

            #Discriminator on real faces (batchB)
            pred_real = discriminator(batchB)
            real_label = torch.ones_like(pred_real).to(device)
            loss_D_real = criterion_adv(pred_real, real_label)

            #Discriminator on fake faces (generated by generator)
            pred_fake = discriminator(fake_face.detach()) #detach to prevent backpropagation into generator
            fake_label = torch.zeros_like(pred_fake).to(device)
            loss_D_fake = criterion_adv(pred_fake, fake_label)

            loss_D = (loss_D_real + loss_D_fake) * 0.5 #discriminator loss is average of real and fake losses
            loss_D.backward()
            optimizer_D.step()

        print(f'Epoch {epoch+1}/{epochs}: Generator Loss: {loss_G_total.item():.4f}, Discriminator Loss: {loss_D.item():.4f}')
        #Save checkpoints every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(generator.state_dict(), f'{checkpoint_dir}/generator_epoch{epoch+1}_{timestamp}.pt')
            torch.save(discriminator.state_dict(), f'{checkpoint_dir}/discriminator_epoch{epoch+1}_{timestamp}.pt')

    #Save final model
    torch.save(generator.state_dict(), f'{checkpoint_dir}/generator_final_{timestamp}.pt')
    torch.save(discriminator.state_dict(), f'{checkpoint_dir}/discriminator_final_{timestamp}.pt')
    print(f'Training complete. Models saved in {checkpoint_dir}')

if __name__ == '__main__':
    train()