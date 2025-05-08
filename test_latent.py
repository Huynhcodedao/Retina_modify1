import argparse
import os
import torch
import numpy as np
from model.model import RetinaFace
from model.config import LATENT_INPUT_SHAPE

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test RetinaFace with latent input')
    parser.add_argument('--latent_path', type=str, required=True, help='Path to directory containing latent files')
    parser.add_argument('--image_dir', type=str, required=True, help='Subdirectory name for the specific image')
    parser.add_argument('--latent_suffix', type=str, default='latent_75.npy', help='Specific latent file to use')
    parser.add_argument('--model_name', type=str, default='resnet50', help='Backbone model name')
    parser.add_argument('--weight', type=str, default=None, help='Path to model weights')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (e.g., cuda:0, cpu)')
    
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    
    # Initialize model
    print(f"Initializing RetinaFace model with {args.model_name} backbone for latent input...")
    model = RetinaFace(model_name=args.model_name, use_latent=True, is_train=False)
    
    # Load weights if provided
    if args.weight and os.path.isfile(args.weight):
        print(f"Loading weights from {args.weight}")
        checkpoint = torch.load(args.weight, map_location=device)
        model.load_state_dict(checkpoint)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Construct the full path to the latent file
    latent_file = os.path.join(args.latent_path, args.image_dir, args.latent_suffix)
    
    if not os.path.isfile(latent_file):
        print(f"Error: Latent file not found at {latent_file}")
        return
    
    # Load latent representation
    print(f"Loading latent representation from {latent_file}")
    latent = np.load(latent_file)
    
    # Check if latent has batch dimension
    if len(latent.shape) == 3:
        # Add batch dimension if not present
        latent = np.expand_dims(latent, axis=0)
    
    # Assert latent has correct shape
    expected_shape = (1, LATENT_INPUT_SHAPE[0], LATENT_INPUT_SHAPE[1], LATENT_INPUT_SHAPE[2])
    if latent.shape != expected_shape:
        print(f"Warning: Latent shape {latent.shape} doesn't match expected shape {expected_shape}")
    
    # Convert to tensor and move to device
    latent_tensor = torch.from_numpy(latent).float().to(device)
    
    # Forward pass
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(latent_tensor)
    
    # Print output shapes
    print("\nModel outputs:")
    print(f"Bounding boxes shape: {outputs[0].shape}")
    print(f"Classification shape: {outputs[1].shape}")
    print(f"Landmarks shape: {outputs[2].shape}")
    
    print("\nTest successful! Model can process latent representations.")

if __name__ == "__main__":
    main() 