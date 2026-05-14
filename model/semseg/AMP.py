import torch
from torch import nn
import torch.nn.functional as F
import model.backbone.foundation_models as backbone_model
import logging
from model.wavelet.WT import WT
from model.muti_branch_cnns import ASPP, Decoder, CBAM, SpatialAttention, ChannelAttention, Decoder_Attention

logger = logging.getLogger()

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)
    
def resize_for_tensors(tensors, size, mode='bilinear', align_corners=False):
    return F.interpolate(tensors, size, mode=mode, align_corners=align_corners)

class AMP(nn.Module):
    def __init__(self, cfg, aux=True):
        super(AMP, self).__init__()
        self.backbone_name, self.backbone  =  \
             backbone_model.__dict__[cfg['backbone']]()
        backbone_channels = {
            'DINOV2s': 384,
            'DINOV3_vits16': 384,
            'DINOV3_vits16plus': 384,
            'DINOV2b': 768,
            'DINOV3_vitb16': 768,
            'DINOV2l': 1024,
            'DINOV3_vitl16': 1024,
            'DINOV3_vitl16plus': 1024,
            'DINOV3_vith16plus': 1280,
            'DINOV2g': 1536,
            'DINOV3_vit7b16': 4096,
            'SAM3': 256
        }

        if self.backbone_name in backbone_channels:
            self.channels = backbone_channels[self.backbone_name]
        else:
            raise ValueError(f"Invalid backbone name: {self.backbone_name}")
        self.aux = aux
        self.conclude = nn.Sequential(nn.Conv2d(self.channels, self.channels, 1, bias=False),
                                     nn.BatchNorm2d(self.channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

        self.fuse = nn.Sequential(nn.Conv2d(self.channels//4 + self.channels //2 + self.channels + self.channels, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
        
        self.produce = nn.Sequential(nn.Conv2d(3, self.channels, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(self.channels),
                                 nn.ReLU(True))
        
        self.layer4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, 1, bias = False),
                                    nn.BatchNorm2d(self.channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels, self.channels, 3, padding=1, bias = False),
                                    nn.BatchNorm2d(self.channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels, self.channels, 1, bias = False),
                                    nn.BatchNorm2d(self.channels),
                                    nn.ReLU(True)
                                    )
        self.layer3 = nn.Sequential(nn.Conv2d(self.channels, self.channels, 1, bias = False),
                                    nn.BatchNorm2d(self.channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels, self.channels, 3, padding=1, bias = False),
                                    nn.BatchNorm2d(self.channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels, self.channels, 1, bias = False),
                                    nn.BatchNorm2d(self.channels),
                                    nn.ReLU(True)
                                    )
        self.layer2 = nn.Sequential(nn.Conv2d(self.channels, self.channels, 1, bias=False),
                                    nn.BatchNorm2d(self.channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels, self.channels//2, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.channels//2),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels // 2, self.channels//2, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.channels//2),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels // 2, self.channels//2, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.channels//2),
                                    nn.ReLU(True)
                                    )
        self.layer1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, 1, bias=False),
                                    nn.BatchNorm2d(self.channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels, self.channels//4, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.channels//4),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels // 4, self.channels//4, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.channels//4),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels // 4, self.channels//4, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.channels//4),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.channels // 4, self.channels//4, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(self.channels//4),
                                    nn.ReLU(True)
                                    )
        # Muti_Branch_Pseudo-Labels
        norm_fn_for_extra_modules = FixedBatchNorm
        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules,inplanes=256)
        self.attn1 = cfg['attn1']
        self.attn2 = cfg['attn2']
        self.decoder1 = Decoder_Attention(cfg['nclass'],3 * self.channels //4, norm_fn_for_extra_modules, attention_mode=self.attn1)
        self.decoder2 = Decoder_Attention(cfg['nclass'],3 * self.channels //4, norm_fn_for_extra_modules, attention_mode=self.attn2)

        # wavelet part 
        self.wavelet_H = WT(3, self.channels)

    def forward(self, x):
        b, _, h, w = x.shape
        ori_f = self.backbone.base_forward(x) # dinov2 & dinov3
        f1, f2, f3, f4 = ori_f[:4]
        _, fre_feat_H = self.wavelet_H(x)
        f1 = F.interpolate(f1,size=fre_feat_H.shape[-2:], mode='bicubic', align_corners=False)
        f2 = F.interpolate(f2,size=fre_feat_H.shape[-2:], mode='bicubic', align_corners=False)

        c1 = 0.99 * f1 + 0.01*fre_feat_H
        c2 = 0.99 * f2 + 0.01*fre_feat_H
        c1 = self.layer1(c1)
        c2 = self.layer2(c2)
        c3 = self.layer3(f3)
        c3 = F.interpolate(c3, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c4 = self.layer4(f4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        
        after_adapter =  [c1, c2, c3, c4]
        feat = torch.cat([c1, c2, c3, c4], dim = 1)
        x_low_level = torch.cat([c1,c2], dim = 1)


        feat = self.fuse(feat)
        out = self.classifier(feat)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        pseudo_out = self.aspp(feat)
        
        x_main = self.decoder1(pseudo_out, x_low_level)
        x_1 = resize_for_tensors(x_main, x.size()[2:], align_corners=True) #cnn output1

        x_aux= self.decoder2(pseudo_out, x_low_level)
        x_2 = resize_for_tensors(x_aux, x.size()[2:], align_corners=True) #cnn output2 

        if self.training:
            return ori_f, after_adapter, feat, x_1, x_2, out
        else:
            return out.softmax(1)
