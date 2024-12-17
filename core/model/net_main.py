# -*- encoding: utf-8 -*-
from typing import List
import torch
import torch.nn as nn
from .embedNet import S3D
from loguru import logger

class MFViT(nn.Module):
    def __init__(self, embedNet:nn.Module, backbone:nn.Module, head:nn.Module,use_WaveSF:bool,SSL:bool):
        super().__init__()
        self.embed = embedNet
        self.backbone = backbone
        self.head = head
        self.use_WaveSF = use_WaveSF
        self.SSL = SSL
        self.print_flag = True
    def forward(self, x:torch.Tensor):
        """
            X: video cubes/patches
        """
        x = self.embed(x) # b*96*8*28*28 or b*96*8*56*56
        if self.use_WaveSF and self.SSL: # train or test both return x
            pool_a, x_d = self.backbone(x)
            x = self.head(x_d)
            if self.training:
                return x,pool_a,x_d
            else:
                return x
        elif self.use_WaveSF and not self.SSL:
            # original network with change_x
            x, pool_a, x_d = self.backbone(x)
            x = self.head(x)
            if self.training:
                return x,pool_a,x_d
            else:
                return x
        elif not self.use_WaveSF and not self.SSL: # not used WaveSF and not use SSL
            x = self.backbone(x)
            x = self.head(x)
            return x
        else:
            raise NotImplementedError('SSL without WaveLift is not implemented.')
        
    def _frozen(self,freeze_list:List):
        # CLS_train 
        # for example: ['embed','backbone','head','norm','wave']
        #    LayerNorm BatchNorm，             
        logger.info(f'Frozen layers:{freeze_list}')
        if 'embed' in freeze_list:
            for name,module in self.embed._modules.items():
                if (isinstance(module,nn.LayerNorm) or isinstance(module,nn.BatchNorm2d)):
                    if not 'norm' in freeze_list:
                        pass
                    else:
                        for param in module.parameters():
                            param.requires_grad = False
                        module.eval()
                else:
                    for param in module.parameters():
                        param.requires_grad = False
        if 'backbone' in freeze_list:
            for name,module in self.backbone.backbone._modules.items():
                if (isinstance(module,nn.LayerNorm) or isinstance(module,nn.BatchNorm2d)):
                    if not 'norm' in freeze_list:
                        pass
                    else:
                        for param in module.parameters():
                            param.requires_grad = False
                        module.eval()
                elif name == 'WaveLifts' and not 'wave' in freeze_list:
                    pass
                elif name == 'prompt_embed' or name == 'prompt_projection':
                    # for child class MFViT_prompt
                    pass
                else:
                    for param in module.parameters():
                        param.requires_grad = False
        if 'head' in freeze_list:
            for name,module in self.head._modules.items():
                # DEBUG it is meanless to freeze LayerNorm for head
                for param in module.parameters():
                    param.requires_grad = False
                    
        if self.print_flag:
            trainable_list = []
            for name, param in self.named_parameters():
                if param.requires_grad:
                    trainable_list.append(name)
            logger.info(f'Trainable Params:{trainable_list}')
            self.print_flag = False

class MFViT_Main(nn.Module):
    def __init__(self, embedNet:nn.Module, backbone:nn.Module, head:nn.Module,use_WaveSF:bool,SSL:bool,facial:bool):
        super().__init__()
        self.embed = embedNet
        self.backbone = backbone
        self.head = head
        self.use_WaveSF = use_WaveSF
        self.SSL = SSL
        self.print_flag = True
        self.facial = facial
    def forward(self, x:torch.Tensor):
        """
            X: video cubes/patches
        """
        if not self.facial:
            x = self.embed(x) # b*96*8*28*28 or b*96*8*56*56
        else:
            x, xf = self.embed(x)
            
        if self.use_WaveSF: # train or test both return x
            if not self.facial:
                pool_a, x_d = self.backbone(x)
            else:
                pool_a, x_d = self.backbone(x,xf)
            x = self.head(x_d)
            if self.training:
                return x,pool_a,x_d
            else:
                return x
        else:
            x = self.backbone(x)
            x = self.head(x)
            return x

        
    def _frozen(self,freeze_list:List):
        # CLS_train 
        # for example: ['embed','backbone','head','norm','wave']
        #    LayerNorm BatchNorm，             
        logger.info(f'Frozen layers:{freeze_list}')
        if 'facial' in freeze_list:
            for name,module in self.embed.facial_backbone._modules.items():
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()
                    
        if 'embed' in freeze_list:
            for name,module in self.embed._modules.items():
                if (isinstance(module,nn.LayerNorm) or isinstance(module,nn.BatchNorm2d)):
                    if not 'norm' in freeze_list:
                        pass
                    else:
                        for param in module.parameters():
                            param.requires_grad = False
                        module.eval()
                else:
                    for param in module.parameters():
                        param.requires_grad = False
                        
        if 'backbone' in freeze_list:
            # self.backbone has a child backbone, thus using self.backbone.backbone
            for name,module in self.backbone._modules.items():
                if name == 'WaveLifts' and not 'wave' in freeze_list:
                    pass
                elif (name == 'PANLayer' or name == 'CatConv') and not 'wave' in freeze_list:
                    pass
                else:
                    for param in module.parameters():
                        param.requires_grad = False
            for name,module in self.backbone.backbone._modules.items():
                if (isinstance(module,nn.LayerNorm) or isinstance(module,nn.BatchNorm2d)):
                    if not 'norm' in freeze_list:
                        pass
                    else:
                        for param in module.parameters():
                            param.requires_grad = False
                        module.eval()
                else:
                    for param in module.parameters():
                        param.requires_grad = False
                        
        if 'head' in freeze_list:
            for name,module in self.head._modules.items():
                for param in module.parameters():
                    param.requires_grad = False
                    
        if self.print_flag:
            trainable_list = []
            for name, param in self.named_parameters():
                if param.requires_grad:
                    trainable_list.append(name)
            logger.info(f'Trainable Params:{trainable_list}')
            self.print_flag = False

class MFCNN(nn.Module):
    def __init__(self,num_class:int) -> None:
        super().__init__()
        self.backbone = S3D(num_classes=num_class)
    def forward(self, x:torch.Tensor):
        """
            X: video cubes/patches
        """
        x = self.backbone(x)
        return x