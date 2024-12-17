import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops.misc import Conv3dNormActivation
from functools import partial
from typing import Callable, Dict

class TemporalSeparableConv(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size_s: int,
        stride: int,
        padding_s: int,
        norm_layer: Callable[..., nn.Module],
        kernel_size_t: int = None,
        padding_t: int = None,
    ):
        if kernel_size_t is None:
            kernel_size_t = kernel_size_s
            padding_t = padding_s
        super().__init__(
            Conv3dNormActivation(
                in_planes,
                out_planes,
                kernel_size=(1, kernel_size_s, kernel_size_s),
                stride=(1, stride, stride),
                padding=(0, padding_s, padding_s),
                bias=False,
                norm_layer=norm_layer,
            ),
            Conv3dNormActivation(
                out_planes,
                out_planes,
                kernel_size=(kernel_size_t, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding_t, 0, 0),
                bias=False,
                norm_layer=norm_layer,
            ),
        )

class SepInceptionBlock3D(nn.Module):
    def __init__(
        self,
        in_planes: int,
        b0_out: int,
        b1_mid: int,
        b1_out: int,
        b2_mid: int,
        b2_out: int,
        b3_out: int,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()

        self.branch0 = Conv3dNormActivation(in_planes, b0_out, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.branch1 = nn.Sequential(
            Conv3dNormActivation(in_planes, b1_mid, kernel_size=1, stride=1, norm_layer=norm_layer),
            TemporalSeparableConv(b1_mid, b1_out, kernel_size_s=3, stride=1, padding_s=1, norm_layer=norm_layer),
        )
        self.branch2 = nn.Sequential(
            Conv3dNormActivation(in_planes, b2_mid, kernel_size=1, stride=1, norm_layer=norm_layer),
            TemporalSeparableConv(b2_mid, b2_out, kernel_size_s=3, stride=1, padding_s=1, norm_layer=norm_layer),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            Conv3dNormActivation(in_planes, b3_out, kernel_size=1, stride=1, norm_layer=norm_layer),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out

class embedNet_light(nn.Module):
    def __init__(self,
                 input_channel:int = 3,
                 input_time:int = 32,
                 embed_dim:int = 96,
                 out_resolution:int = 56, # or 56
                 facial_project:bool = False,
                 ):
        super().__init__()
        stride_s = 224//out_resolution
        stride_t = 2 if input_time > 8 else 1
        self.facial_project = facial_project
        self.conv_proj = nn.Conv3d(
            in_channels=input_channel,
            out_channels=embed_dim,
            kernel_size=(3,7,7),
            stride=(stride_t,stride_s,stride_s),
            padding=(1,3,3),
        )
        if input_time == 32:
            self.temporal_pool = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        elif input_time == 16 or input_time == 8:
            self.temporal_pool = nn.Identity()
        else:
            raise NotImplementedError()
        if self.facial_project:
            from core.prompt.mobilefacenet import MobileFaceFeature
            self.facial_crop_size= (112,112)
            self.facial_backbone = MobileFaceFeature([112, 112],136)
            # self.facial_temporal_pool = nn.ModuleList()
            self.facial_temporal_pool = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if self.facial_project:
            b,c,t,h,w = x.shape
            xf = x.clone()
            xf = xf.transpose(1,2).reshape(-1,c,h,w) # b*t,c,h,w
            bt = xf.shape[0]
            if (h,w) != self.facial_crop_size:
                xf = F.interpolate(xf,size=self.facial_crop_size)
            # the last stage is removed for MFViT
            xf:Dict[int, torch.Tensor] = self.facial_backbone(xf,return_keys=[0,1,2]) # [ n*t,c,h,w ]
            for i in xf.keys():
                _,c,h,w = xf[i].shape
                # DEBUG: time-pool     AvgPool      3D  ？
                xf[i] = xf[i].reshape(b,t,c,h,w).transpose(1,2) # .permute(0,1,3,4,2).reshape(b,-1,c)
                xf[i] = self.facial_temporal_pool(xf[i])
                xf[i] = xf[i].permute(0,2,3,4,1).reshape(b,-1,c)
        x = self.conv_proj(x)
        x = self.temporal_pool(x) # UCF-101 32->16->8; DFEW 16->8->8
        if self.facial_project:
            return x, xf
        else:
            return x # b,c=embed_dim,t=8,h,w
    
    def load_pretrain_face(self,path:str='pretrain_weights/mobilefacenet_model_best.pth.tar'):
        if self.facial_project:
            pretrained = torch.load(path)
            self.load_state_dict(pretrained['state_dict'])
        return 0
    
    def load_pretrain(self,path:str='pretrain_weights/mvit_v2_s.pth'):
        pretrained = torch.load(path)
        target = self.state_dict()      
        for key,value in pretrained.items():
            if 'conv_proj' in key:
                target[key] = value
        self.load_state_dict(target,strict=True)
        return 0
        
    
class embedNet(nn.Module):
    """
        Following the implementation of PyTorch
        for:
        S3D at http://arxiv.org/abs/1712.04851 
    """
    def __init__(self,
                 input_channel:int = 3,
                 embed_dim:int = 128,
                 out_resolution:int = 28, # or 56
                 ):
        super().__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=0.001, momentum=0.001)
        self.out_hw = out_resolution # or 56
        self.features = nn.Sequential(
            TemporalSeparableConv(
                in_planes=input_channel, 
                out_planes=64, 
                kernel_size_s=7, 
                stride=2, 
                padding_s=3, 
                norm_layer=norm_layer
            ),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Conv3dNormActivation(
                64,
                64,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
            ),
            TemporalSeparableConv(64, 192, 3, 1, 1, norm_layer),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            SepInceptionBlock3D(192, 64, 96, 128, 16, 32, 32, norm_layer),
        )
        if self.out_hw == 56:
            del self.features[4:6]
        input_ch = 256 if self.out_hw == 28 else 192
        self.fuse = nn.Sequential(
            Conv3dNormActivation(
                input_ch, # 480
                embed_dim,
                kernel_size=(3,1,1),
                stride=(2,1,1),
                padding = (1,0,0),
                norm_layer=norm_layer,
            ),)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.features(x)
        x = self.fuse(x)
        return x # b,c=96,t=8,h=56,w=56
    
    def load_pretrain(self, path:str='pretrain_weights/s3d-pytorch.pth'):
        pretrained = torch.load(path)
        target = self.state_dict()
        load_length = 6 if self.out_hw == 28 else 4
        for key,value in pretrained.items():
            if 'features' in key:
                if int(key.split('.')[1]) < load_length:
                    target[key] = value
        self.load_state_dict(target,strict=True)
        return 0
    
class S3D(nn.Module):
    """S3D main class.

    Args:
        num_class (int): number of classes for the classification task.
        dropout (float): dropout probability.
        norm_layer (Optional[Callable]): Module specifying the normalization layer to use.

    Inputs:
        x (Tensor): batch of videos with dimensions (batch, channel, time, height, width)
    """

    def __init__(
        self,
        num_classes: int = 101,
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        norm_layer = partial(nn.BatchNorm3d, eps=0.001, momentum=0.001)

        self.features = nn.Sequential(
            TemporalSeparableConv(3, 64, 7, 2, 3, norm_layer),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Conv3dNormActivation(
                64,
                64,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
            ),
            TemporalSeparableConv(64, 192, 3, 1, 1, norm_layer),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            SepInceptionBlock3D(192, 64, 96, 128, 16, 32, 32, norm_layer),
            SepInceptionBlock3D(256, 128, 128, 192, 32, 96, 64, norm_layer),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            SepInceptionBlock3D(480, 192, 96, 208, 16, 48, 64, norm_layer),
            SepInceptionBlock3D(512, 160, 112, 224, 24, 64, 64, norm_layer),
            SepInceptionBlock3D(512, 128, 128, 256, 24, 64, 64, norm_layer),
            SepInceptionBlock3D(512, 112, 144, 288, 32, 64, 64, norm_layer),
            SepInceptionBlock3D(528, 256, 160, 320, 32, 128, 128, norm_layer),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            SepInceptionBlock3D(832, 256, 160, 320, 32, 128, 128, norm_layer),
            SepInceptionBlock3D(832, 384, 192, 384, 48, 128, 128, norm_layer),
        )
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = torch.mean(x, dim=(2, 3, 4))
        return x
    
    def load_pretrain(self, path:str='pretrain_weights/s3d-pytorch.pth'):
        pretrained = torch.load(path)
        target = self.state_dict()
        for key,value in pretrained.items():
            if 'classifier' not in key:
                    target[key] = value
        self.load_state_dict(target,strict=True)
        return 0