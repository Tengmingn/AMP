import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
third_party_dir = os.path.join(current_dir, '../../third_party')
if third_party_dir not in sys.path:
    sys.path.insert(0, third_party_dir)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from torchvision.transforms import v2

class DINOV2(nn.Module):
    def __init__(self,selected_layers, fileroot, model_size):
        super(DINOV2, self).__init__()
        self.model =  torch.hub.load(fileroot, model_size, source = 'local').cuda()
        self.outfeat_layer = selected_layers
        for param in self.model.parameters():
            param.requires_grad = False
    def base_forward(self, x):
        # frozen parameters
        with torch.no_grad():
            features = self.model.user_forward_features(x, self.outfeat_layer)
            if isinstance(features[0], torch.Tensor):
               features[0] = F.interpolate(
                   features[0], scale_factor = 4, mode = 'bilinear', align_corners = False
               )
               features[1] = F.interpolate(
                   features[1], scale_factor = 2, mode = 'bilinear', align_corners = False
              )
            else:
               features[0][0] = F.interpolate(
                   features[0][0], scale_factor = 4, mode = 'bilinear', align_corners = False
               )
               features[0][1] = F.interpolate(
                   features[0][1], scale_factor = 2, mode = 'bilinear', align_corners = False
               )
        return features
    
def DINOV2s():
    name = 'DINOV2s'
    selected_layers = [3, 5, 7, 11]
    model = DINOV2(selected_layers, 'third_party/dinov2', 'dinov2_vits14')
    return name, model

def DINOV2b():
    name = 'DINOV2b'
    selected_layers = [3, 5, 7, 11]
    model = DINOV2(selected_layers,'third_party/dinov2', 'dinov2_vitb14')
    return name, model

def DINOV2l():
    name = 'DINOV2l'
    selected_layers = [7, 11, 15, 23]
    model = DINOV2(selected_layers, 'third_party/dinov2', 'dinov2_vitl14')
    return name, model

def DINOV2g():
    name = 'DINOV2g'
    selected_layers = [11, 19, 23, 39]
    model = DINOV2(selected_layers,'third_party/dinov2', 'dinov2_vitg14')
    return name, model

class DINOV3(nn.Module):
    def __init__(self,selected_layers, fileroot, model_size):
        super(DINOV3, self).__init__()
        self.model =  torch.hub.load(fileroot, model_size, source = 'local').cuda()
        # frozen parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.outfeat_layer = selected_layers
    def base_forward(self, x):
        token_f, features = self.model.user_forward_features(x, self.outfeat_layer)
        if isinstance(features[0], torch.Tensor):
            features[0] = F.interpolate(
                features[0], scale_factor = 4, mode = 'bilinear', align_corners = False
            )
            features[1] = F.interpolate(
                features[1], scale_factor = 2, mode = 'bilinear', align_corners = False
            )
        else:
            features[0][0] = F.interpolate(
                features[0][0], scale_factor = 4, mode = 'bilinear', align_corners = False
            )
            features[0][1] = F.interpolate(
                features[0][1], scale_factor = 2, mode = 'bilinear', align_corners = False
            )

        return features

def DINOV3_vits16():
    name = 'DINOV3_vits16'
    selected_layers = [3, 5, 7, 11]
    model = DINOV3(selected_layers, 'third_party/dinov3', 'dinov3_vits16')
    return name, model

def DINOV3_vits16plus():
    name = 'DINOV3_vits16plus'
    selected_layers = [3, 5, 7, 11]
    model = DINOV3(selected_layers,'third_party/dinov3', 'dinov3_vits16plus')
    return name, model

def DINOV3_vitb16():
    name = 'DINOV3_vitb16'
    selected_layers = [3, 5, 7, 11]
    model = DINOV3(selected_layers, 'third_party/dinov3', 'dinov3_vitb16')
    return name, model

def DINOV3_vitl16():
    name = 'DINOV3_vitl16'
    selected_layers = [7, 11, 15, 23]
    model = DINOV3(selected_layers,'third_party/dinov3', 'dinov3_vitl16')
    return name, model

def DINOV3_vitl16plus():
    name = 'DINOV3_vitl16plus'
    selected_layers = [7, 11, 15, 23]
    model = DINOV3(selected_layers,'third_party/dinov3', 'dinov3_vitl16plus')
    return name, model

def DINOV3_vith16plus():
    name = 'DINOV3_vith16plus'
    selected_layers = [8, 15, 23, 31]
    model = DINOV3(selected_layers,'third_party/dinov3', 'dinov3_vith16plus')
    return name, model

def DINOV3_vit7b16():
    name = 'DINOV3_vit7b16'
    selected_layers = [10, 19, 29, 39]
    model = DINOV3(selected_layers,'third_party/dinov3', 'dinov3_vit7b16')
    return name, model

class SAM3_backbone(nn.Module):
    def __init__(self, resolution=1008, fileroot=None, model_size=None):
        super(SAM3_backbone, self).__init__()
        self.model = build_sam3_image_model(checkpoint_path=fileroot)
        # frozen parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    def base_forward(self, x):
        x = self.transform(x)
        backbone_out = self.model.backbone.forward_image(x)
        # backbone_out_plus = backbone_out[2]
        feature = backbone_out['vision_features'] # [B, 256, 72, 72]
        fpn_out = backbone_out['backbone_fpn'] # [B, 256, 288, 288] [B, 256, 144, 144] [B, 256, 72, 72]
        out = [fpn_out[0], fpn_out[1], fpn_out[2], feature]
        return out

def SAM3():
    name = 'SAM3'
    checkpoint_path = './third_party/sam3/sam3/pretrained/sam3.pt'
    model = SAM3_backbone(fileroot=checkpoint_path)
    return name, model
