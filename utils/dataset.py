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
                folder_name = img_path.split('/')[0]
                image_dir = f"{folder_name}_{img_name}"
                
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
                    face_annotations.append(face_data)
                    i += 1
                
                labels_data[image_dir] = face_annotations
            
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
                print(f"Nội dung của {os.path.join(self.latent_path, first_dir)}: {os.listdir(os.path.join(self.latent_path, first_dir))[:5]}")
                print(f"Các khóa nhãn: {list(labels_data.keys())[:5]}")
            except Exception as e:
                print(f"Lỗi khi liệt kê file: {e}")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        try:
            # Tải biểu diễn latent
            latent_file = self.latent_paths[index]
            latent = np.load(latent_file)
            
            # Chuyển đổi sang tensor và đảm bảo kích thước chính xác
            latent = torch.from_numpy(latent).float()
            
            # Nếu latent có shape [1, C, H, W], loại bỏ chiều batch
            if len(latent.shape) == 4 and latent.shape[0] == 1:
                latent = latent.squeeze(0)
            
            # Lấy annotations từ danh sách đã đọc trước
            face_annotations = self.annotations[index]
            
            # Tạo tensor annotations theo định dạng mong muốn
            annotations = np.zeros((len(face_annotations), 15))
            
            for idx, line in enumerate(face_annotations):
                if len(line) < 4:  # Kiểm tra xem có đủ thông tin không
                    continue
                
                # bbox
                annotations[idx, 0] = line[0]               # x1
                annotations[idx, 1] = line[1]               # y1
                annotations[idx, 2] = line[0] + line[2]     # x2
                annotations[idx, 3] = line[1] + line[3]     # y2
                
                if len(line) > 4:  # Nếu có thông tin về landmarks
                    # landmarks - chuyển đổi từ định dạng [x, y, vis] * 5 sang định dạng mong muốn
                    lm_idx = 4
                    for i in range(5):  # 5 điểm landmark
                        if lm_idx + i*3 + 1 < len(line):
                            annotations[idx, 4 + i*2] = line[lm_idx + i*3]     # landmark x
                            annotations[idx, 5 + i*2] = line[lm_idx + i*3 + 1]  # landmark y
                
                # Set landmark visibility flag
                if annotations[idx, 4] < 0:
                    annotations[idx, 14] = -1
                else:
                    annotations[idx, 14] = 1
            
            # Chuyển đổi annotations sang tensor
            annotations_tensor = torch.from_numpy(annotations).float()
            return latent, annotations_tensor
            
        except Exception as e:
            print(f"Lỗi khi xử lý mẫu {self.ids[index]}: {e}")
            # Trả về mẫu trống trong trường hợp lỗi
            empty_latent = torch.zeros(self.latent_shape)
            empty_annotations = torch.zeros((0, 15))
            return empty_latent, empty_annotations

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