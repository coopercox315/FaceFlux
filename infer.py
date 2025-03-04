import torch
import torchvision.transforms as T
from PIL import Image
import argparse
from models.face_swap_model import FaceSwapModel
from utils.transforms import generator_transform, identity_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#function to load and apply transforms to an image
def load_img(img_path, transform): 
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    return img

def infer(source_path, target_path, output_path, model_path):
    #Load the pre-trained model
    model = FaceSwapModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #Load and transform the source and target images
    source_img = load_img(source_path, generator_transform).to(device)
    target_img = load_img(target_path, identity_transform).to(device)

    #Perform face swapping
    with torch.inference_mode():
        output = model(source_img, target_identity_face=target_img)

    #Denormalize the output image from [-1, 1] to [0, 1] for saving
    output = (output + 1) / 2
    output = output.squeeze(0).cpu()
    output_img = T.ToPILImage()(output)
    output_img.save(output_path)
    print(f"Saved swapped face image to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Swap Inference')
    parser.add_argument('--source', type=str, required=True, help='Path to source content face image')
    parser.add_argument('--target', type=str, required=True, help='Path to target identity face image')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output image')
    parser.add_argument('--model', type=str, required=True, help='Path to pre-trained model/generator checkpoint')
    args = parser.parse_args()

    infer(args.source, args.target, args.output, args.model) #run inference with given arguments