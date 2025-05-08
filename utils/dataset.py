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
            
            # Ensure the tensor has the correct dimensions
            if len(latent_tensor.shape) == 3:  # Missing batch dimension
                latent_tensor = latent_tensor.unsqueeze(0)
            
            if latent_tensor.shape[0] != 1:
                print(f"Warning: Unexpected batch dimension in latent tensor: {latent_tensor.shape}")
            
            # Extract the latent features (remove batch dimension) if needed
            if latent_tensor.shape[0] == 1:
                latent_features = latent_tensor.squeeze(0)
            else:
                latent_features = latent_tensor
                
            # Ensure the channel dimension is correct (should be 256)
            if latent_features.shape[0] != 256:
                print(f"Warning: Unexpected channel dimension: {latent_features.shape}. Reshaping to match 256 channels.")
                
                # If tensor is [C, H, W] with C != 256
                if len(latent_features.shape) == 3:
                    # Handle the case where channel dimension is wrong
                    if latent_features.shape[0] < 256:
                        # Duplicate existing channels to get to 256
                        repeat_factor = int(np.ceil(256 / latent_features.shape[0]))
                        expanded = latent_features.repeat(repeat_factor, 1, 1)
                        latent_features = expanded[:256]
                    else:
                        # Truncate to 256 channels
                        latent_features = latent_features[:256]
                
                # If tensor is [H, W] (missing channel dim)
                elif len(latent_features.shape) == 2:
                    # Treat the single channel as 256 by duplicating it
                    h, w = latent_features.shape
                    latent_features = latent_features.unsqueeze(0).repeat(256, 1, 1)
                    print(f"Expanded 2D tensor of shape [{h}, {w}] to [256, {h}, {w}]")
            
            # Ensure the tensor has spatial dimensions of 40x40
            if latent_features.shape[1] != 40 or latent_features.shape[2] != 40:
                print(f"Resizing latent spatial dimensions from {latent_features.shape[1]}x{latent_features.shape[2]} to 40x40")
                # Resize to match expected spatial dimensions
                latent_features = torch.nn.functional.interpolate(
                    latent_features.unsqueeze(0),  # Add batch dim for interpolation
                    size=(40, 40),
                    mode='bilinear'
                ).squeeze(0)  # Remove batch dim
            
            # Verify final shape
            if latent_features.shape != (256, 40, 40):
                print(f"ERROR: Final tensor shape {latent_features.shape} doesn't match required (256, 40, 40)")
                # Final fallback: create a zero tensor with correct shape
                latent_features = torch.zeros((256, 40, 40))
            
            # Process annotations
            annots = self.annotations[index]
            annotations = np.zeros((len(annots), 15))
            
            if len(annots) == 0:
                # For empty annotations, return a dummy target
                dummy_target = np.zeros((1, 15))
                dummy_target[0, 14] = -1  # Mark as invalid
                return latent_features, dummy_target
            
            # Lưu lại kích thước ảnh gốc để chuẩn hóa tọa độ
            # Nếu không có kích thước thực tế, mặc định là 640x640 (như INPUT_SIZE)
            img_width, img_height = 640, 640
            
            for idx, face_data in enumerate(annots):
                # Format the data to match the expected output format
                
                # bbox: Convert from x,y,w,h to x1,y1,x2,y2
                if len(face_data) >= 4:
                    # TRỌNG YẾU: Chuẩn hóa tọa độ về khoảng [0,1]
                    annotations[idx, 0] = face_data[0] / img_width                 # x1
                    annotations[idx, 1] = face_data[1] / img_height                # y1
                    annotations[idx, 2] = (face_data[0] + face_data[2]) / img_width    # x2
                    annotations[idx, 3] = (face_data[1] + face_data[3]) / img_height    # y2
                else:
                    print(f"Warning: Invalid bbox data for {self.ids[index]}, face {idx}")
                
                # landmarks: 5 points with x,y coordinates (10 values)
                # If there are landmarks in the data (expected len >= 14)
                if len(face_data) >= 14:
                    # Process the 5 landmarks (10 values) - cũng chuẩn hóa về [0,1]
                    annotations[idx, 4] = face_data[4] / img_width    # l0_x
                    annotations[idx, 5] = face_data[5] / img_height   # l0_y
                    annotations[idx, 6] = face_data[6] / img_width    # l1_x
                    annotations[idx, 7] = face_data[7] / img_height   # l1_y
                    annotations[idx, 8] = face_data[8] / img_width    # l2_x
                    annotations[idx, 9] = face_data[9] / img_height   # l2_y
                    annotations[idx, 10] = face_data[10] / img_width  # l3_x
                    annotations[idx, 11] = face_data[11] / img_height # l3_y
                    annotations[idx, 12] = face_data[12] / img_width  # l4_x
                    annotations[idx, 13] = face_data[13] / img_height # l4_y
                    
                    # Set landmark valid flag (1 = valid)
                    annotations[idx, 14] = 1
                else:
                    # If we don't have landmark data, mark landmarks as invalid
                    annotations[idx, 14] = -1
            
            # Validate boxes have proper dimensions
            for i in range(len(annotations)):
                # Ensure width and height are at least 1 pixel (đã chuẩn hóa nên sẽ là 1/width và 1/height)
                min_width = 1 / img_width
                min_height = 1 / img_height
                
                if annotations[i, 2] <= annotations[i, 0] + min_width:
                    annotations[i, 2] = annotations[i, 0] + min_width
                if annotations[i, 3] <= annotations[i, 1] + min_height:
                    annotations[i, 3] = annotations[i, 1] + min_height
                    
                # Debug in ra một số box để kiểm tra
                if i < 5:  # Chỉ in 5 box đầu tiên
                    width = annotations[i, 2] - annotations[i, 0]
                    height = annotations[i, 3] - annotations[i, 1]
                    print(f"DEBUG: Normalized box {i}: ({annotations[i, 0]:.4f}, {annotations[i, 1]:.4f}, {annotations[i, 2]:.4f}, {annotations[i, 3]:.4f}), size={width:.4f}×{height:.4f}")
            
            return latent_features, annotations
            
        except Exception as e:
            print(f"Error loading item {index} ({self.ids[index]}): {e}")
            # Return a dummy tensor and target as fallback
            dummy_tensor = torch.zeros((256, 40, 40))
            dummy_target = np.zeros((1, 15))
            dummy_target[0, 14] = -1  # Mark as invalid
            return dummy_tensor, dummy_target

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

        imgs.append(image)
        targets.append(target)

    return (torch.stack(imgs, dim=0), targets)