# -*- encoding: utf-8 -*-
from typing import List
import torch
import torch.nn as nn
from loguru import logger

class ViT(nn.Module):
    def __init__(self, 
                 embed:nn.Module, 
                 backbone:nn.Module, 
                 head:nn.Module,
                 use_WaveSF:bool=False,
                 prompt:bool=False
                 ):
        super().__init__()
        self.embed = embed
        self.backbone = backbone
        self.head = head
        self.use_WaveSF = use_WaveSF
        self.prompt = prompt
        self.print_flag = True
    def forward(self, x:torch.Tensor, prompt:torch.Tensor=None):
        x = self.embed(x)
        if self.use_WaveSF:
            if self.prompt:
                x_d,pool_a = self.backbone(x,prompt=prompt)
            else:
                x_d,pool_a = self.backbone(x)
            x = self.head(x_d)
            if self.training:
                return x,pool_a,x_d
            else:
                return x
        else:
            x = self.backbone(x)
            x = self.head(x)    
            return x
    
    def _frozen(self,freeze_list:List[str]):
        logger.info(f'Frozen layers:{freeze_list}')
        if 'embed' in freeze_list:
            for name,module in self.embed._modules.items():
                for param in module.parameters():
                    param.requires_grad = False
        if 'backbone' in freeze_list:
            self.backbone._frozen(freeze_list)
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
        return None
    
        
