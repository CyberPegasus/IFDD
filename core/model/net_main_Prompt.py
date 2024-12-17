# -*- encoding: utf-8 -*-
from typing import List
import torch
import torch.nn as nn
from .embedNet import S3D
from loguru import logger
from .net_main import MFViT
from torch.cuda.amp import autocast

class MFViT_prompt(MFViT):
    def __init__(self, embedNet:nn.Module, backbone:nn.Module, head:nn.Module,use_WaveSF:bool,SSL:bool,prompt:bool=False,use_lowFreq:bool=False):
        super().__init__(embedNet, backbone, head,use_WaveSF,SSL)
        self.prompt = prompt
        self.use_lowFreq = use_lowFreq
        self.print_flag = True

    def forward(self, x:torch.Tensor, prompt:torch.Tensor=None):
        """
            X: video cubes/patches
            prompt: b, 1, d
        """
        x = self.embed(x) # b*96*8*28*28 or b*96*8*56*56
        if self.use_WaveSF and self.SSL and not self.prompt: 
            # support no prompt input
            # train or test both return x
            if self.use_lowFreq:
                pool_a, x_a, x_d = self.backbone(x)
                x = self.head(x_d,x_a)
            else:
                pool_a, x_d = self.backbone(x)
                x = self.head(x_d)
            if self.training:
                return x,pool_a,x_d
            else:
                return x
        elif self.use_WaveSF and self.SSL and self.prompt:
            if self.use_lowFreq:
                pool_a, x_a, x_d = self.backbone(x,prompt=prompt)
                x = self.head(x_d,x_a)
            else:
                pool_a, x_d = self.backbone(x,prompt=prompt)
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
            # self.backbone has a child backbone, thus using self.backbone.backbone
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
            
        return None
                    
                    
class MFViT_prompt_decouleHead(MFViT):
    def __init__(self, embedNet:nn.Module, backbone:nn.Module, head:nn.Module,use_WaveSF:bool,SSL:bool,prompt:bool=False,use_lowFreq:bool=False,decoupleHead:bool=False):
        super().__init__(embedNet, backbone, head,use_WaveSF,SSL)
        self.prompt = prompt
        self.use_lowFreq = use_lowFreq
        self.decoupleHead = decoupleHead
        
    def forward(self, x:torch.Tensor, prompt:torch.Tensor=None):
        """
            X: video cubes/patches
            prompt: b, 1, d
        """
        x = self.embed(x) # b*96*8*28*28 or b*96*8*56*56
        if self.use_WaveSF and self.SSL and not self.prompt: 
            # support no prompt input
            # train or test both return x
            if self.use_lowFreq:
                pool_a, x_a, x_d = self.backbone(x)
                if self.decoupleHead:
                    xn, xe = self.head(x_d,x_a) # neutral, emotion
                else:
                    x = self.head(x_d,x_a)
            else:
                pool_a, x_d = self.backbone(x)
                if self.decoupleHead:
                    xn, xe = self.head(x_d,x_a) # neutral, emotion
                else:
                    x = self.head(x_d,x_a)
            if self.training:
                if self.decoupleHead:
                    return xn, xe, pool_a, x_d
                else:
                    return x, pool_a, x_d
            else:
                if self.decoupleHead:
                    return xn, xe
                else:
                    return x
                
        elif self.use_WaveSF and self.SSL and self.prompt:
            if self.use_lowFreq:
                pool_a, x_a, x_d = self.backbone(x,prompt=prompt)
                if self.decoupleHead:
                    xn, xe = self.head(x_d,x_a) # neutral, emotion
                else:
                    x = self.head(x_d,x_a)
            else:
                pool_a, x_d = self.backbone(x,prompt=prompt)
                if self.decoupleHead:
                    xn, xe = self.head(x_d,x_a) # neutral, emotion
                else:
                    x = self.head(x_d,x_a)
            if self.training:
                if self.decoupleHead:
                    return xn, xe, pool_a, x_d
                else:
                    return x, pool_a, x_d
            else:
                if self.decoupleHead:
                    return xn, xe
                else:
                    return x
                
        elif self.use_WaveSF and not self.SSL:
            # original network with change_x
            x, pool_a, x_d = self.backbone(x)
            if self.decoupleHead:
                xn, xe = self.head(x) # neutral, emotion
            else:
                x = self.head(x)
            if self.training:
                if self.decoupleHead:
                    return xn, xe, pool_a, x_d
                else:
                    return x, pool_a, x_d
            else:
                if self.decoupleHead:
                    return xn, xe
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
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad = False
                else:
                    for param in module.parameters():
                        param.requires_grad = False
        if 'backbone' in freeze_list:
            # self.backbone has a child backbone, thus using self.backbone.backbone
            for name,module in self.backbone.backbone._modules.items():
                if (isinstance(module,nn.LayerNorm) or isinstance(module,nn.BatchNorm2d)):
                    if not 'norm' in freeze_list:
                        pass
                    else:
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad = False
                elif name == 'WaveLifts' and not 'wave' in freeze_list:
                    pass
                elif name == 'prompt_embed' or name == 'prompt_projection':
                    pass
                else:
                    for param in module.parameters():
                        param.requires_grad = False
        if 'head' in freeze_list:
            for name,module in self.head._modules.items():
                # DEBUG it is meanless to freeze LayerNorm for head
                for param in module.parameters():
                    param.requires_grad = False