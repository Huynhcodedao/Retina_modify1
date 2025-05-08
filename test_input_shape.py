import torch
import numpy as np
from model.model import RetinaFace, BridgeModule

def test_model_input_shapes():
    print("Testing RetinaFace model with different input shapes...")
    
    # Create model with latent mode enabled and debugging
    model = RetinaFace(model_name='resnet50', use_latent=True, debug=True)
    model.eval()
    
    # Test case 1: 5D tensor [B, 1, 256, 40, 40]
    print("\nTest case 1: 5D tensor [B, 1, 256, 40, 40]")
    input_5d = torch.randn(2, 1, 256, 40, 40)
    print(f"Input shape: {input_5d.shape}")
    
    try:
        output = model(input_5d)
        print(f"Success! Output shapes:")
        for i, o in enumerate(output):
            print(f"  Output {i}: {o.shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 2: 4D tensor with single channel [B, 1, 40, 40]
    print("\nTest case 2: 4D tensor with single channel [B, 1, 40, 40]")
    input_4d_single = torch.randn(2, 1, 40, 40)
    print(f"Input shape: {input_4d_single.shape}")
    
    try:
        output = model(input_4d_single)
        print(f"Success! Output shapes:")
        for i, o in enumerate(output):
            print(f"  Output {i}: {o.shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 3: 4D tensor with correct channels [B, 256, 40, 40]
    print("\nTest case 3: 4D tensor with correct channels [B, 256, 40, 40]")
    input_4d = torch.randn(2, 256, 40, 40)
    print(f"Input shape: {input_4d.shape}")
    
    try:
        output = model(input_4d)
        print(f"Success! Output shapes:")
        for i, o in enumerate(output):
            print(f"  Output {i}: {o.shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 4: 3D tensor [256, 40, 40]
    print("\nTest case 4: 3D tensor [256, 40, 40]")
    input_3d = torch.randn(256, 40, 40)
    print(f"Input shape: {input_3d.shape}")
    
    try:
        output = model(input_3d)
        print(f"Success! Output shapes:")
        for i, o in enumerate(output):
            print(f"  Output {i}: {o.shape}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model_input_shapes() 