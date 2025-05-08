import os
import torch
import wandb
import numpy as np
from utils.data_augment import WiderFacePreprocess
from model.config import INPUT_SIZE, TRAIN_PATH, VALID_PATH, LATENT_INPUT_SHAPE
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class WiderFaceDataset(Dataset):
    """
    Wider Face custom dataset.
    Args:
        root_path (string): Path to dataset directory
        is_train (bool): Train dataset or test dataset
        transform (function): whether to apply the data augmentation scheme
                mentioned in the paper. Only applied on the train split.
    """

    def __init__(self, root_path, input_size=INPUT_SIZE, is_train=True):
        self.ids       = []
        self.transform = WiderFacePreprocess(image_size=input_size)
        self.is_train  = is_train

        if is_train: 
            self.path = os.path.join(root_path, TRAIN_PATH)
        else: 
            self.path = os.path.join(root_path, VALID_PATH)
        
        for dirname in os.listdir(os.path.join(self.path, 'images')):
            for file in os.listdir(os.path.join(self.path, 'images', dirname)):
                self.ids.append(os.path.join(dirname, file)[:-4])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, 'images', self.ids[index]+'.jpg'))
        img = np.array(img)

        f = open(os.path.join(self.path, 'labels', self.ids[index]+'.txt'), 'r')
        lines = f.readlines()

        annotations = np.zeros((len(lines), 15)) 

        if len(lines) == 0:
            return annotations
        
        for idx, line in enumerate(lines):
            line = line.strip().split()
            line = [float(x) for x in line]

            # bbox
            annotations[idx, 0] = line[0]               # x1
            annotations[idx, 1] = line[1]               # y1
            annotations[idx, 2] = line[0] + line[2]     # x2
            annotations[idx, 3] = line[1] + line[3]     # y2

            if self.is_train:
                # landmarks
                annotations[idx, 4] = line[4]               # l0_x
                annotations[idx, 5] = line[5]               # l0_y
                annotations[idx, 6] = line[7]               # l1_x
                annotations[idx, 7] = line[8]               # l1_y
                annotations[idx, 8] = line[10]              # l2_x
                annotations[idx, 9] = line[11]              # l2_y
                annotations[idx, 10] = line[13]             # l3_x
                annotations[idx, 11] = line[14]             # l3_y
                annotations[idx, 12] = line[16]             # l4_x
                annotations[idx, 13] = line[17]             # l4_y

                if (annotations[idx, 4]<0):
                    annotations[idx, 14] = -1
                else:
                    annotations[idx, 14] = 1
            
            else:
                annotations[idx, 14] = 1

        if self.transform is not None:
            img, annotations = self.transform(image=img, targets=annotations)

        return img, annotations

class LatentWiderFaceDataset(Dataset):
    """
    Dataset class for using latent feature maps as input instead of RGB images.
    Using a single labels.txt file for all images.
    
    Args:
        root_path (string): Path to dataset directory containing latent features
        latent_dir (string): Subdirectory containing the latent .npy files
        label_file (string): Name of the labels file (default: 'labels.txt')
        latent_suffix (string): Suffix for the specific latent file to use (e.g., 'latent_75.npy')
        is_train (bool): Train dataset or test dataset
    """
    def __init__(self, root_path, latent_dir='latent', label_file='labels.txt', latent_suffix='latent_75.npy', is_train=True):
        self.ids = []
        self.latent_paths = []
        self.annotations = []
        self.is_train = is_train
        self.latent_shape = LATENT_INPUT_SHAPE
        self.latent_suffix = latent_suffix
        
        # Xác định đường dẫn đến train hoặc val
        if is_train: 
            self.path = os.path.join(root_path, 'train')
        else: 
            self.path = os.path.join(root_path, 'val')
        
        self.latent_path = os.path.join(self.path, latent_dir)
        self.label_file_path = os.path.join(self.path, label_file)
        
        print(f"Đường dẫn thư mục latent: {self.latent_path}")
        print(f"Đường dẫn file nhãn: {self.label_file_path}")
        
        # Kiểm tra thư mục và file có tồn tại không
        if not os.path.exists(self.latent_path):
            print(f"CẢNH BÁO: Thư mục latent {self.latent_path} không tồn tại!")
            return
            
        if not os.path.exists(self.label_file_path):
            print(f"CẢNH BÁO: File nhãn {self.label_file_path} không tồn tại!")
            return
            
        # Đọc file nhãn
        print("Đang đọc file nhãn...")
        labels_data = {}
        try:
            with open(self.label_file_path, 'r') as f:
                lines = f.readlines()
                
            i = 0
            while i < len(lines):
                # Dòng đầu tiên chứa tên ảnh
                img_path = lines[i].strip()
                # Trích xuất tên thư mục từ đường dẫn ảnh
                img_name = img_path.split('/')[-1].replace('.jpg', '')
                
                # Handle case when path doesn't contain directories
                if '/' in img_path:
                    folder_name = img_path.split('/')[0]
                    image_dir = f"{folder_name}_{img_name}"
                else:
                    # If no directory in path, use only the filename
                    image_dir = img_name
                    
                i += 1
                if i >= len(lines):
                    break
                    
                # Dòng thứ hai chứa số lượng khuôn mặt
                num_faces = int(lines[i].strip())
                i += 1
                
                # Đọc thông tin về các khuôn mặt
                face_annotations = []
                for j in range(num_faces):
                    if i >= len(lines):
                        break
                    face_data = lines[i].strip().split()
                    face_data = [float(x) for x in face_data]
                    
                    # Validate the face data to ensure it's valid
                    if len(face_data) >= 4:  # Ensure we have at least x, y, w, h
                        # Ensure width and height are positive
                        if face_data[2] <= 0:
                            face_data[2] = 1.0  # Default width
                        if face_data[3] <= 0:
                            face_data[3] = 1.0  # Default height
                        
                        # Add this face to annotations
                        face_annotations.append(face_data)
                    else:
                        print(f"Skipping invalid face data: {face_data}")
                    
                    i += 1
                
                if len(face_annotations) > 0:
                    labels_data[image_dir] = face_annotations
                else:
                    print(f"Warning: No valid faces found for {image_dir}")
            
            print(f"Đã đọc thông tin nhãn cho {len(labels_data)} hình ảnh.")
        except Exception as e:
            print(f"Lỗi khi đọc file nhãn: {e}")
            return
        
        # Tìm các file latent
        print("Đang tìm các file latent...")
        for image_dir in os.listdir(self.latent_path):
            image_dir_path = os.path.join(self.latent_path, image_dir)
            if os.path.isdir(image_dir_path):
                latent_file_path = os.path.join(image_dir_path, self.latent_suffix)
                if os.path.isfile(latent_file_path):
                    if image_dir in labels_data:
                        self.ids.append(image_dir)
                        self.latent_paths.append(latent_file_path)
                        self.annotations.append(labels_data[image_dir])
                    else:
                        # Thử tìm các định dạng tên khác có thể phù hợp
                        matched = False
                        for label_key in labels_data.keys():
                            if image_dir in label_key or label_key in image_dir:
                                self.ids.append(image_dir)
                                self.latent_paths.append(latent_file_path)
                                self.annotations.append(labels_data[label_key])
                                matched = True
                                print(f"Đã khớp {image_dir} với nhãn {label_key}")
                                break
                        if not matched:
                            print(f"Cảnh báo: Không tìm thấy nhãn cho {image_dir}")
        
        print(f"Đã tìm thấy {len(self.ids)} cặp file latent và nhãn hợp lệ.")
        
        if len(self.ids) == 0:
            print("CẢNH BÁO: Không tìm thấy dữ liệu hợp lệ!")
            # In một số ví dụ để giúp chẩn đoán
            try:
                print(f"Nội dung của {self.latent_path}: {os.listdir(self.latent_path)[:5]}")
                first_dir = os.listdir(self.latent_path)[0]
                first_dir_path = os.path.join(self.latent_path, first_dir)
                if os.path.isdir(first_dir_path):
                    print(f"Nội dung của {first_dir_path}: {os.listdir(first_dir_path)[:5]}")
            except Exception as e:
                print(f"Lỗi khi in thông tin chẩn đoán: {e}")
        else:
            # Validate a sample of the annotations to ensure they're in the right format
            sample_idx = min(len(self.annotations) - 1, 5)  # Check the 5th item or last if fewer
            sample_annot = self.annotations[sample_idx]
            print(f"Example annotation for item {sample_idx}: {sample_annot[0] if len(sample_annot) > 0 else 'empty'}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        try:
            # Load the latent representation
            latent_file = self.latent_paths[index]
            latent_data = np.load(latent_file)
            
            # Convert latent data to tensor and ensure it's the right shape
            latent_tensor = torch.from_numpy(latent_data).float()
            
            # Debug the shape of the loaded tensor
            if index == 0:
                print(f"DEBUG: Original latent tensor shape from file: {latent_tensor.shape}")
            
            # Ensure the tensor has the correct dimensions - we want [C, H, W] for stacking in collate
            if len(latent_tensor.shape) == 4 and latent_tensor.shape[0] == 1:  # [1, C, H, W]
                # If it has a batch dimension of 1, remove it
                latent_tensor = latent_tensor.squeeze(0)
                if index == 0:
                    print(f"DEBUG: Squeezed batch dimension, new shape: {latent_tensor.shape}")
            elif len(latent_tensor.shape) == 2:  # [H, W]
                # If it's just a 2D tensor, add channel dimension
                latent_tensor = latent_tensor.unsqueeze(0)
                if index == 0:
                    print(f"DEBUG: Added channel dimension, new shape: {latent_tensor.shape}")
            
            # Ensure we have 256 channels
            if latent_tensor.shape[0] != 256 and len(latent_tensor.shape) == 3:
                if index == 0:
                    print(f"DEBUG: Adjusting channel count from {latent_tensor.shape[0]} to 256")
                
                if latent_tensor.shape[0] < 256:
                    # If we have fewer channels than needed, duplicate them
                    repeat_factor = int(np.ceil(256 / latent_tensor.shape[0]))
                    expanded = torch.repeat_interleave(latent_tensor, repeat_factor, dim=0)
                    latent_tensor = expanded[:256]
                else:
                    # If we have more channels than needed, truncate
                    latent_tensor = latent_tensor[:256]
                
                if index == 0:
                    print(f"DEBUG: Final channel count: {latent_tensor.shape[0]}")
                
            # Process annotations
            face_annotations = self.annotations[index]
            num_faces = len(face_annotations)
            
            # Create an array to hold the data for all faces
            # Format: [x1, y1, x2, y2, landmark_x1, landmark_y1, ..., landmark_x5, landmark_y5, label]
            target_array = np.zeros((num_faces, 15))
            
            # Get the expected image size from either the tensor or a fixed size
            # For latent tensors, we need to work backwards to get original image size
            image_width = 640  # Assuming the original image was 640x640
            image_height = 640
            
            # Process each face annotation
            for i, face in enumerate(face_annotations):
                if len(face) >= 4:  # Ensure this face annotation has at least bbox
                    # Extract bbox coordinates and normalize to [0,1]
                    x = face[0] / image_width
                    y = face[1] / image_height
                    w = face[2] / image_width
                    h = face[3] / image_height
                    
                    # Ensure width and height are positive and within bounds
                    w = max(0.001, min(w, 1.0))
                    h = max(0.001, min(h, 1.0))
                    
                    # Convert to [x1, y1, x2, y2] format
                    x1 = max(0.0, min(x, 1.0))
                    y1 = max(0.0, min(y, 1.0))
                    x2 = max(0.0, min(x + w, 1.0))
                    y2 = max(0.0, min(y + h, 1.0))
                    
                    # Set bbox coordinates
                    target_array[i, 0] = x1
                    target_array[i, 1] = y1
                    target_array[i, 2] = x2
                    target_array[i, 3] = y2
                    
                    # Process landmarks if available
                    if len(face) >= 14:  # 4 bbox + 10 landmarks
                        has_landmarks = True
                        for j in range(5):
                            lm_x = face[4 + j*2] / image_width
                            lm_y = face[5 + j*2] / image_height
                            
                            # Check if landmarks are valid (some datasets mark -1 for unavailable)
                            if lm_x < 0 or lm_y < 0:
                                has_landmarks = False
                                break
                            
                            target_array[i, 4 + j*2] = lm_x
                            target_array[i, 5 + j*2] = lm_y
                        
                        # Set landmark flag
                        target_array[i, 14] = 1.0 if has_landmarks else -1.0
                    else:
                        # No landmarks
                        target_array[i, 14] = -1.0
                else:
                    # Invalid face data
                    target_array[i, 14] = -1.0
            
            # Validate that we don't have any NaN or Inf values
            if np.isnan(target_array).any() or np.isinf(target_array).any():
                print(f"WARNING: NaN or Inf values found in annotations for item {index}. Fixing...")
                target_array = np.nan_to_num(target_array, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Convert annotations to tensor
            target_tensor = torch.from_numpy(target_array).float()
            
            if index == 0:
                print(f"DEBUG: Final latent tensor shape: {latent_tensor.shape}")
                print(f"DEBUG: Target tensor shape: {target_tensor.shape}")
            
            return latent_tensor, target_tensor
            
        except Exception as e:
            print(f"Error processing item {index} ({self.ids[index]}): {e}")
            # Return empty tensors as a fallback
            empty_latent = torch.zeros((256, 40, 40)).float()  # Changed to [C, H, W] format
            empty_target = torch.zeros((0, 15)).float()
            return empty_latent, empty_target

def log_dataset(use_artifact, 
        artifact_name, 
        artifact_path, dataset_name, 
        job_type='preprocess dataset', 
        project_name='Content-based RS'):

    run = wandb.init(project=project_name, job_type=job_type)
    run.use_artifact(use_artifact)
    artifact = wandb.Artifact(artifact_name, dataset_name)

    if os.path.isdir(artifact_path):
        artifact.add_dir(artifact_path)
    else:
        artifact.add_file(artifact_path)
    run.log_artifact(artifact)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []

    for _, (image, target) in enumerate(batch):
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).to(dtype=torch.float)

        # Ensure images have the correct shape before stacking
        # For latent features: [1, 256, 40, 40] - Remove the batch dimension so we can stack properly
        if image.dim() == 4 and image.size(0) == 1:
            if image.size(1) == 256:  # This is a latent tensor with shape [1, 256, 40, 40]
                image = image.squeeze(0)  # Convert to [256, 40, 40]
            else:
                # Don't squeeze if it's already the correct shape (like RGB images)
                pass

        imgs.append(image)
        targets.append(target)
    
    # Note: Stack adds a batch dimension, so result is [batch_size, channels, height, width]
    return (torch.stack(imgs, dim=0), targets)