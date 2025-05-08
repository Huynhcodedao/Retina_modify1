import torch
import torch.nn as nn
from math import sqrt, pow
import torch.nn.functional as F

from model.config import *
from model._utils import IntermediateLayerGetter
from model.common import FPN, SSH, MobileNetV1

class BridgeModule(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        """
        Bridge module to transform latent representation [1, 256, 40, 40] to [256, 160, 160]
        to match the expected input size for stage 2 of ResNet50
        """
        super(BridgeModule, self).__init__()
        
        # Upsampling pathway using transposed convolutions
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.upconv2 = nn.ConvTranspose2d(512, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Print để debug kích thước đầu vào
        # print(f"Bridge input shape: {x.shape}")
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.upconv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.upconv2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Print để debug kích thước đầu ra
        # print(f"Bridge output shape: {x.shape}")
        
        return x

class ClassHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        """
        Face classification 
        """
        super(ClassHead, self).__init__()
        self.conv        = nn.Conv2d(in_channels, num_anchors*2, 1)

    def forward(self, input):
        out = self.conv(input)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        """
        Face bounding box
        """
        super(BboxHead, self).__init__()
        self.conv        = nn.Conv2d(in_channels, num_anchors*4, 1)

    def forward(self, input):
        out = self.conv(input)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        """
        Facial landmark
        """
        super(LandmarkHead, self).__init__()
        # 5 (x, y) refer to coordinate of 5 landmarks
        self.conv = nn.Conv2d(in_channels, num_anchors*10, 1)

    def forward(self, input):
        out = self.conv(input)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, model_name='resnet50', freeze_backbone=False, pretrain_path=None, is_train=True, use_latent=False, debug=False):
        """
        Model RetinaFace for face recognition based on:
        `"RetinaFace: Single-stage Dense Face Localisation in the Wild" <https://arxiv.org/abs/1905.00641>`_.
        
        Args:
            model_name (str): Name of the backbone model
            freeze_backbone (bool): Whether to freeze the backbone
            pretrain_path (str): Path to pretrained weights
            is_train (bool): Whether the model is in training mode
            use_latent (bool): Whether to use latent representations instead of RGB images
            debug (bool): Whether to print debug information
        """
        super(RetinaFace, self).__init__()
        self.is_train = is_train
        self.use_latent = use_latent
        self.debug = debug
        
        if self.debug:
            print(f"Initializing RetinaFace with model_name={model_name}, use_latent={use_latent}")
        
        # load backbone
        backbone = None
        if model_name == 'mobilenet0.25':
            backbone            = MobileNetV1(start_frame=START_FRAME)
            return_feature      = RETURN_MAP_MOBN1
            self.feature_map    = FEATURE_MAP_MOBN1
            
            if not pretrain_path is None:
                pretrain_weight = torch.load(pretrain_path)
                backbone.load_state_dict(pretrain_weight)

        elif model_name == 'mobilenetv2':
            return_feature = FEATURE_MAP_MOBN2
            backbone = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        
        elif 'resnet' in model_name:
            import torchvision.models as models
            if '18' in  model_name:
                backbone = models.resnet18(pretrained=True)

            elif '34' in  model_name:
                backbone = models.resnet34(pretrained=True)

            elif '50' in model_name:
                backbone = models.resnet50(pretrained=True)
            
            return_feature      = RETURN_MAP
            self.feature_map    = FEATURE_MAP

        else:
            print(f'Unable to select {model_name}.')

        num_fpn             = len(self.feature_map)
        
        if use_latent:
            # For latent input, we'll use a custom Bridge module and modified backbone
            self.bridge = BridgeModule(in_channels=256, out_channels=256)
            
            # We need to modify the return_feature to skip the first two stages (stage 0 and stage 1)
            # of ResNet50 and only use stages 2, 3, and 4
            modified_return_feature = {'layer2': 'out_feature2', 
                                       'layer3': 'out_feature3',
                                       'layer4': 'out_feature4'}
            
            # Create a modified backbone without the first two stages
            # For ResNet50, we'll only use the last 3 stages (layer2, layer3, layer4)
            if 'resnet' in model_name:
                # We'll create a new sequential model with only the needed layers
                new_backbone = nn.Module()
                new_backbone.layer2 = backbone.layer2
                new_backbone.layer3 = backbone.layer3
                new_backbone.layer4 = backbone.layer4
                backbone = new_backbone
                
            self.body = IntermediateLayerGetter(backbone, modified_return_feature)
            
            # Các kênh đầu vào khác nhau cho mô hình latent
            # ResNet50 output channels: layer2=512, layer3=1024, layer4=2048
            in_channels_list = [512, 512, 1024, 2048]
        else:
            # Original implementation for RGB image input
            self.body = IntermediateLayerGetter(backbone, return_feature)
            in_channels_list = [IN_CHANNELS*2, IN_CHANNELS*4, IN_CHANNELS*8, IN_CHANNELS*16]

        if freeze_backbone:
            for param in self.body.parameters():
                param.requires_grad = False
            print('\tBackbone freezed')

        self.fpn = FPN(in_channels_list=in_channels_list, out_channels=OUT_CHANNELS)
        self.ssh = SSH(in_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS)

        # class head + bbox head + landmark head
        self.ClassHead      = self._make_class_head(inchannels=OUT_CHANNELS, anchor_num=6, fpn_num=num_fpn)
        self.BboxHead       = self._make_bbox_head(inchannels=OUT_CHANNELS, anchor_num=6, fpn_num=num_fpn)
        self.LandmarkHead   = self._make_landmark_head(inchannels=OUT_CHANNELS, anchor_num=6, fpn_num=num_fpn)

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, input):
        """
        The input to the RetinaFace is expected to be a Tensor
        
        Args:
            input (Tensor): For regular input - RGB image(s) [B, 3, H, W]
                            For latent input - latent representation [B, 256, 40, 40]
        """
        if self.debug:
            print(f"Input shape: {input.shape}")
            
        if self.use_latent:
            # For latent input, first pass through bridge module to match expected size
            x = self.bridge(input)
            
            if self.debug:
                print(f"After bridge shape: {x.shape}")
            
            # Feed directly to layer2 of backbone
            out = self.body(x)
            
            if self.debug:
                print(f"Backbone output keys: {out.keys()}")
                for k, v in out.items():
                    print(f"  {k} shape: {v.shape}")
            
            # Feature Pyramid Net - với latent inputs, chúng ta không cần out_feature1
            fpn = self.fpn(out)
        else:
            # Original implementation for RGB input
            out = self.body(input)
            
            if self.debug:
                print(f"Backbone output keys: {out.keys()}")
                for k, v in out.items():
                    print(f"  {k} shape: {v.shape}")
            
            # Feature Pyramid Net - RGB inputs có đủ 4 feature maps
            fpn = self.fpn(out)

        if self.debug:
            for i, feature in enumerate(fpn):
                print(f"FPN output {i} shape: {feature.shape}")

        # Single-stage headless
        feature_1 = self.ssh(fpn[0])
        feature_2 = self.ssh(fpn[1])
        feature_3 = self.ssh(fpn[2])
        feature_4 = self.ssh(fpn[3])
        feature_5 = self.ssh(fpn[4])
        features = [feature_1, feature_2, feature_3, feature_4, feature_5]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications  = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions  = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.debug:
            print(f"bbox_regressions shape: {bbox_regressions.shape}")
            print(f"classifications shape: {classifications.shape}")
            print(f"ldm_regressions shape: {ldm_regressions.shape}")

        if self.is_train:
            output = (bbox_regressions, classifications, ldm_regressions)
        else: 
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        return output

def forward(model, input, targets, anchors, loss_function, optimizer):
    """Due to the probability of OOM problem can happen, which might
    cause the "CUDA out of memory". I've passed all require grad into
    a function to free it while there is nothing refer to it.
    """
    predict = model(input)
    loss_l, loss_c, loss_landm = loss_function(predict, anchors, targets)
    loss = 1.3*loss_l + loss_c + loss_landm

    loss_l      = loss_l.item()
    loss_c      = loss_c.item()
    loss_landm  = loss_landm.item()

    # zero the gradient + backprpagation + step
    optimizer.zero_grad()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

    loss.backward()
    optimizer.step()

    del predict
    del loss

    return loss_l, loss_c, loss_landm
