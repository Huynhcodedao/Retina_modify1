# Modified RetinaFace for Latent Representation Input

This repository contains a modified version of the RetinaFace face detector that has been adapted to use latent representations as input instead of RGB images.

## Modifications Overview

The original RetinaFace model has been modified to work with latent feature maps of size `[1, 256, 40, 40]` instead of RGB images. The main modifications include:

1. **Bridge Module**: A new module that transforms the latent representation `[1, 256, 40, 40]` to the expected input size `[256, 160, 160]` for stage 2 of ResNet50.

2. **Pruned ResNet-50**: Removed the first 11 convolutional layers (stages 0 and 1) from ResNet50, using only stages 2, 3, and 4 to generate feature maps.

3. **Custom Dataset Class**: Created a new dataset class `LatentWiderFaceDataset` to handle loading latent representations from `.npy` files.

4. **Modified Training Script**: Updated the training script to support both traditional RGB inputs and latent representation inputs.

## Feature Maps Generated

The model produces feature maps with the following specifications:

- **C2**: [512, 80, 80] (stride 8)
- **C3**: [1024, 40, 40] (stride 16)
- **C4**: [2048, 20, 20] (stride 32)

The Feature Pyramid Network (FPN) then produces:

- **P2**: [256, 160, 160] (stride 4)
- **P3**: [256, 80, 80] (stride 8)
- **P4**: [256, 40, 40] (stride 16)
- **P5**: [256, 20, 20] (stride 32)
- **P6**: [256, 10, 10] (stride 64)

## Dataset Structure

The dataset has the following structure:

```
data/
├──train/
│   ├── latent/
│   │   ├── 0_Parade_Parade_0_939/
│   │   │   ├── latent_100.npy
│   │   │   ├── latent_12.npy
│   │   │   ├── latent_25.npy
│   │   │   ├── latent_50.npy
│   │   │   ├── latent_75.npy
│   │   ├── [other_image_dirs]/
│   ├── labelstxt/
│       ├── 0_Parade_Parade_0_939.txt
│       ├── [other_label_files].txt
├──val/
    ├── latent/
    │   ├── [image_dirs]/
    │       ├── latent_75.npy
    │       ├── [other_latent_files].npy
    ├── labelstxt/
        ├── [label_files].txt
```

For training, we specifically use the `latent_75.npy` files from each image directory.

## Usage

### Training with Latent Representations

To train the model with latent representations:

```bash
python train.py --model resnet50 --use_latent --latent_dir latent --label_dir labelstxt --latent_suffix latent_75.npy
```

Parameters:
- `--use_latent`: Flag to use latent representations instead of RGB images
- `--latent_dir`: Directory containing latent representation files (default: 'latent')
- `--label_dir`: Directory containing annotation files (default: 'labelstxt')
- `--latent_suffix`: Specific latent file to use from each image directory (default: 'latent_75.npy')

### Testing the Model

To test if the model can process latent representations:

```bash
python test_latent.py --latent_path data/val/latent --image_dir 0_Parade_Parade_0_939 --latent_suffix latent_75.npy --model_name resnet50 --weight path/to/weights.pth
```

Parameters:
- `--latent_path`: Path to the directory containing latent subdirectories
- `--image_dir`: Name of the specific image subdirectory
- `--latent_suffix`: Specific latent file to use (default: 'latent_75.npy')

## Key Files Modified

1. `model/model.py`: Added `BridgeModule` and modified `RetinaFace` to handle latent inputs
2. `model/config.py`: Added configuration parameters for latent input mode
3. `utils/dataset.py`: Added `LatentWiderFaceDataset` class for latent input data
4. `train.py`: Updated to support both RGB and latent input modes
5. `test_latent.py`: New script to test the model with latent inputs

## Implementation Details

### Bridge Module

The Bridge Module uses transposed convolutions to upsample the latent representation from 40x40 to 160x160:

1. 3x3 Convolution + Batch Normalization + ReLU: [256, 40, 40] → [512, 40, 40]
2. Transposed Convolution (stride 2) + Batch Normalization + ReLU: [512, 40, 40] → [512, 80, 80]
3. Transposed Convolution (stride 2) + Batch Normalization + ReLU: [512, 80, 80] → [256, 160, 160]

### Backbone Modification

For the ResNet50 backbone:
- Removed stages 0 and 1 (the first 11 convolutional layers)
- The latent representation is processed by the Bridge Module and then fed directly to stage 2
- Stages 2, 3, and 4 are kept intact to produce feature maps C2, C3, and C4 