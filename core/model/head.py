from math import ceil, sqrt
from typing import List
import torch
from torch import nn
from torch.nn import functional as F
class headNet(nn.Module):
    def __init__(self,num_classes:int,embed_dim:int=768,dropout:float=0.5,sigmoid:bool=False,enable_cosHead:bool=False):
        super().__init__()
        self.cos_head = enable_cosHead
        if not enable_cosHead:
            self.head = nn.Sequential(
                nn.Dropout(dropout, inplace=True),
                nn.Linear(768, num_classes),
            )
        else:
            self.head = nn.Linear(embed_dim,num_classes, bias=False)
            # it's hard for cos similarity to dropout
            # self.dropout = nn.Dropout(dropout, inplace=True)
            self.cos_scale = 20
        self.out_norm = None
            
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        return None
    
    def forward(self, x:torch.Tensor):
        if not self.cos_head:
            if self.out_norm:
                return self.out_norm(self.head(x))
            else:
                x = self.head(x)
                return x
        else:
            x_norm = torch.norm(x,p=2,dim=1).unsqueeze(1).expand_as(x)
            x_normalized = x.div(x_norm+1e-5)
            temp_norm = (torch.norm(self.head.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.head.weight.data))
            self.head.weight.data = self.head.weight.data.div(temp_norm + 1e-5)
            cos_clsScore = self.head(x_normalized)
            x = self.cos_scale*cos_clsScore
            return x

class headNet_vit(nn.Module):
    def __init__(self,embed_dim:int,num_classes:int,dropout:float=0.5):
        super().__init__()
        self.head=nn.Identity()
        # self.embed_dim = embed_dim
        # self.head = nn.Sequential(
        #     nn.Dropout(dropout, inplace=True),
        #     nn.Linear(embed_dim, num_classes),
        # )
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.trunc_normal_(m.weight, std=0.01)
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             nn.init.constant_(m.bias, 0.0)

    def forward(self, x:torch.Tensor):
        return self.head(x)

class headNet_SSL(nn.Module):
    # when given cls_num, then SSL will become CLS status
    # wave_pool is corresponding to WaveLift_pool
    def __init__(self,embed_dim:int,cls_num:int=None,dropout:float = 0.0,enable_cosHead:bool=False,wave_pool:bool=False) -> None:
        super().__init__()
        self.embed_dim = embed_dim # embedded feature
        self.x_len = 3
        self.in_channels = [96,192,384]
        self.t = 4
        self.biggest_hw = 56
        self.stride_list = [1,2,4]
        self.kernel_stride_list = [4,2,1] if not wave_pool else [1,1,1]
        self.hw_list = [self.biggest_hw//i for i in self.stride_list]
        self.hwt_project = nn.ModuleList([])
        self.cosHead = enable_cosHead
        self.wave_pool = wave_pool
        
        for i in range(self.x_len):
            stride = self.kernel_stride_list[i]
            if not self.wave_pool:
                self.hwt_project.append(
                    nn.Conv3d(
                        in_channels=self.in_channels[i],
                        out_channels=self.embed_dim,
                        kernel_size=(self.t,stride,stride),
                        stride=(1,stride,stride),
                    )
                )
            else:
                self.hwt_project.append(
                    nn.Conv3d(
                        in_channels=self.in_channels[i],
                        out_channels=self.embed_dim,
                        kernel_size=(self.t,stride,stride),
                        stride=(1,1,1),
                    )
                )                
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_dim*3,out_channels=self.embed_dim,kernel_size=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=7,stride=7)
        )
        if cls_num:
            if not self.cosHead:
                self.output_project = nn.Linear(self.embed_dim*4,cls_num)
            else:
                self.output_project = nn.Linear(self.embed_dim*4,cls_num, bias=False)
        else:
            self.output_project = nn.Linear(self.embed_dim*4,self.embed_dim,bias=False)
            self.cos_scale = 20
        if dropout>0:
            self.dropout = nn.Dropout(p=dropout,inplace=True)
        else:
            self.dropout = nn.Identity()
        self.active = nn.ReLU(inplace=True)
        self.cls = True if cls_num else False
        
        # init
        nn.init.normal_(self.output_project.weight, std=0.01)
        nn.init.constant_(self.output_project.bias, val=0.0)
        for name, param in self.fusion_layer.named_parameters():
            if 'weight' in name: 
                nn.init.normal_(param,std=0.01)
            if 'bias' in name:
                nn.init.constant_(param, val=0.0)
        for l in self.hwt_project:
            nn.init.normal_(l.weight,std=0.01)
            nn.init.constant_(l.bias, val=0.0)
            

    def forward(self, x:List[torch.Tensor])->torch.Tensor:
        # x:list(y),y ~ b,c,t,hw, h=w
        # t,h,w: 4,56,56->2,28.28->1,14,14
        # x_skip = x # DEBUG for nan probelm
        x_num = len(x)
        for _id in range(x_num):
            b,c,t,hw = x[_id].shape
            h = ceil(sqrt(hw))
            w = hw//h
            x[_id] = x[_id].view(b,c,t,h,w)
        
        b,c,t,h,w = x[-1].shape
        for _id in range(x_num):
            x[_id] = self.hwt_project[_id](x[_id])
        x = torch.cat(x,dim=1).squeeze(dim=2) # b,c*3,t=1,14,14
        x = self.fusion_layer(x)
        x = x.flatten(start_dim = 1)
        if not self.cosHead:
            x = self.dropout(x)
            x = self.output_project(x)
        else:
            x_norm = torch.norm(x,p=2,dim=1).unsqueeze(1).expand_as(x)
            x_normalized = x.div(x_norm+1e-5)
            temp_norm = (torch.norm(self.output_project.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.output_project.weight.data))
            self.output_project.weight.data = self.output_project.weight.data.div(temp_norm + 1e-5)
            cos_clsScore = self.output_project(x_normalized)
            x = self.cos_scale*cos_clsScore

        if not self.cls:
            x = F.normalize(x,p=2,dim=1) # FIXME F.normalize(x, dim=1) or nn.BatchNorm1d(self.embed_dim) ?
        else:
            pass

        return x
    
class headNet_SSL_FERV39k(nn.Module):
    # when given cls_num, then SSL will become CLS status
    def __init__(self,embed_dim:int,cls_num:int=None,dropout:float = 0.0,use_lowFreq:bool=False) -> None:
        super().__init__()
        self.embed_dim = embed_dim # embedded feature
        self.use_lowFreq = use_lowFreq
        self.x_len = 3
        self.in_channels = [96,192,384]
        self.t = 4
        self.biggest_hw = 56
        self.stride_list = [1,2,4]
        self.kernel_stride_list = [4,2,1]
        self.hw_list = [self.biggest_hw//i for i in self.stride_list]
        self.hwt_project = self.construct_hwt_project()
        if self.use_lowFreq:
            self.hwt_project2 = self.construct_hwt_project()
        self.pramid_pool = nn.AvgPool2d(kernel_size=7,stride=7)
        if cls_num:
            self.output_project = nn.Linear(self.embed_dim*12,cls_num)
        else:
            self.output_project = nn.Linear(self.embed_dim*12,self.embed_dim)
        if dropout>0:
            self.dropout = nn.Dropout(p=dropout,inplace=True)
        else:
            self.dropout = nn.Identity()
        self.active = nn.ReLU(inplace=True)
        self.cls = True if cls_num else False
            
    def construct_hwt_project(self):
        hwt_project = nn.ModuleList([])
        for i in range(self.x_len):
            stride = self.kernel_stride_list[i]
            hwt_project.append(
                nn.Conv3d(
                    in_channels=self.in_channels[i],
                    out_channels=self.embed_dim,
                    kernel_size=(self.t,stride,stride),
                    stride=(1,stride,stride),
                )
            )
        return hwt_project
    
    def forward(self, x:List[torch.Tensor], y:List[torch.Tensor]=None)->torch.Tensor:
        # x:list(y),y ~ b,c,t,hw, h=w
        # t,h,w: 4,56,56->2,28.28->1,14,14
        # x_skip = x # DEBUG for nan probelm
        x_num = len(x)
        for _id in range(x_num):
            b,c,t,hw = x[_id].shape
            h = ceil(sqrt(hw))
            w = hw//h
            x[_id] = x[_id].view(b,c,t,h,w)
            if self.use_lowFreq:
                y[_id] = y[_id].view(b,c,t,h,w)
        
        b,c,t,h,w = x[-1].shape
        for _id in range(x_num):
            x[_id] = self.hwt_project[_id](x[_id])
            if self.use_lowFreq:
                y[_id] = self.hwt_project2[_id](y[_id])
        x = torch.cat(x,dim=1).squeeze(dim=2)
        
        x = F.relu_(x) # inplace=True
        x = self.pramid_pool(x)
        x = x.flatten(start_dim = 1)
        x = self.dropout(x)
        x = self.active(self.output_project(x))

        if not self.cls: # FIXME should be if not self.cls ?
            x = F.normalize(x,p=2,dim=1) # FIXME F.normalize(x, dim=1) or nn.BatchNorm1d(self.embed_dim) ?

        return x
    
class headNet_SSL_FERV39k_lowFreq(nn.Module):
    # when given cls_num, then SSL will become CLS status
    def __init__(self,embed_dim:int,cls_num:int=None,dropout:float = 0.0,use_lowFreq:bool=False,neutral_decouple:bool=False) -> None:
        super().__init__()
        self.embed_dim = embed_dim # embedded feature
        self.use_lowFreq = use_lowFreq
        self.x_len = 3
        self.in_channels = [96,192,384]
        self.t = 4
        self.biggest_hw = 56
        self.stride_list = [1,2,4]
        self.kernel_stride_list = [4,2,1]
        self.hw_list = [self.biggest_hw//i for i in self.stride_list]
        self.hwt_project = self.construct_hwt_project()
        if self.use_lowFreq:
            self.hwt_project2 = self.construct_hwt_project()
        self.pramid_pool = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        self.dropout = nn.Dropout(p=dropout)
        self.cls_num = cls_num if cls_num else None
        self.is_neutral_decouple = neutral_decouple
        if cls_num and not neutral_decouple:
            self.output_project = nn.Sequential(
                nn.Conv2d(self.embed_dim*2*3,self.embed_dim,kernel_size=3),
                nn.GELU(),
                nn.BatchNorm2d(self.embed_dim),
                nn.Dropout(p=dropout),
                nn.Flatten(start_dim=1),
                nn.Linear(self.embed_dim*5*5,self.embed_dim),
                nn.GELU(),
                nn.BatchNorm1d(self.embed_dim),
                nn.Dropout(p=dropout),
                nn.Linear(self.embed_dim,cls_num),
                nn.Sigmoid(),
            )
        elif cls_num and neutral_decouple:
            self.output_project = nn.Sequential(
                nn.Conv2d(self.embed_dim*2*3,self.embed_dim,kernel_size=3),
                nn.GELU(),
                nn.BatchNorm2d(self.embed_dim),
                nn.Flatten(start_dim=1),
                nn.Dropout(p=dropout),
            )
            self.output_project_emotion = nn.Sequential(
                nn.Linear(self.embed_dim*5*5,self.embed_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(self.embed_dim,cls_num-1),
                nn.Sigmoid()
            )
            self.output_project_neutral = nn.Sequential(
                nn.Linear(self.embed_dim*5*5,self.embed_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(self.embed_dim,2),
                nn.Sigmoid()
            )
        else:
            self.output_project = nn.Sequential(
                nn.Conv2d(self.embed_dim*2*3,self.embed_dim,kernel_size=3),
                nn.GELU(),
                nn.BatchNorm2d(self.embed_dim),
                nn.Dropout(p=dropout),
                nn.Flatten(start_dim=1),
                nn.Linear(self.embed_dim*5*5,self.embed_dim),
                nn.GELU(),
                nn.BatchNorm1d(self.embed_dim),
                nn.Dropout(p=dropout),
                nn.Linear(self.embed_dim,self.embed_dim),
                nn.Softmax(dim=1),
            )
            
        self.cls = True if cls_num else False
            
    def construct_hwt_project(self):
        hwt_project = nn.ModuleList([])
        for i in range(self.x_len):
            stride = self.kernel_stride_list[i]
            hwt_project.append(
                nn.Conv3d(
                    in_channels=self.in_channels[i],
                    out_channels=self.embed_dim,
                    kernel_size=(self.t,stride,stride),
                    stride=(1,stride,stride),
                )
            )
        return hwt_project
    
    def forward(self, x:List[torch.Tensor], y:List[torch.Tensor]=None)->torch.Tensor:
        # x:list(y),y ~ b,c,t,hw, h=w
        # t,h,w: 4,56,56->2,28.28->1,14,14
        # x_skip = x # DEBUG for nan probelm
        x_num = len(x)
        for _id in range(x_num):
            b,c,t,hw = x[_id].shape
            h = ceil(sqrt(hw))
            w = hw//h
            x[_id] = x[_id].view(b,c,t,h,w)
            if self.use_lowFreq:
                y[_id] = y[_id].view(b,c,t,h,w)
        
        b,c,t,h,w = x[-1].shape
        for _id in range(x_num):
            x[_id] = self.hwt_project[_id](x[_id])
            if self.use_lowFreq:
                y[_id] = self.hwt_project2[_id](y[_id])
        x = torch.cat(x,dim=1).squeeze(dim=2)
        if self.use_lowFreq:
            y = torch.cat(y,dim=1).squeeze(dim=2)
            x = torch.cat([x,y],dim=1) # b,2c,14,14
        x = self.dropout(x)
        x = F.relu_(x) # inplace=True
        x = self.pramid_pool(x)

        if not self.is_neutral_decouple:
            x = self.output_project(x)
            return x
        else:
            x = self.output_project(x)
            x_neutral = self.output_project_neutral(x)
            x_emotion = self.output_project_emotion(x)
            return (x_neutral, x_emotion)
            
            
        