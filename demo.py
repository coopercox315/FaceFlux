import os
import argparse
from infer import infer

#Demo function to perform face swapping on all images in a directory
def demo(input_dir, target_path, output_dir, model_path):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all images in the input directory
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            source_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, f"swapped_{img_name}")
            infer(source_path, target_path, output_path, model_path) #perform face swapping on the image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Swap Demo')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing source face images')
    parser.add_argument('--target', type=str, required=True, help='Path to the target identity face image')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the face swapped images')
    parser.add_argument('--model', type=str, required=True, help='Path to pre-trained model/generator checkpoint')
    args = parser.parse_args()

    demo(args.input_dir, args.target, args.output_dir, args.model) #run face swap demo with given arguments