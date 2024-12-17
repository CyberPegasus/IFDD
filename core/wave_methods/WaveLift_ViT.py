import torch
from torch import nn
from functools import partial
from typing import List
import math
from .AdaptSplit import adaptSplit_dotsim, adaptSplit_dotsim_weight, adaptIndexSplit_dotsim, adaptIndexSplit_dotsim_1D, adaptIndexSplit_dimLearn
from .blocks import TemporalSeparableConv
from einops import rearrange
class WaveSF_ViT_region(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 Non_CLSToken:bool = True,
                 ):
        super().__init__()
        self.relaxed_constrain = relaxed_constrain
        self.change_x = change_x
        self.SSL = SSL
        self.WaveSF_preUpdate = WaveSF_preUpdate
        self.use_lowFreq = lowFreq
        self.Non_CLSToken = Non_CLSToken
    
        self.embed_dim = embed_dim
        self.T,self.H,self.W = input_size
        self.s = split_stride
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.unfold = torch.nn.PixelUnshuffle(self.s)
        self.fold = torch.nn.PixelShuffle(self.s)
        self.q_d = nn.Linear(embed_dim,embed_dim)
        self.kv_a = nn.Linear(embed_dim,2*embed_dim)
        self.q_a = nn.Linear(embed_dim,embed_dim)
        self.kv_d = nn.Linear(embed_dim,2*embed_dim)
        
        layers_d: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        layers_d.append(nn.GELU())
        if dropout > 0.0:
            layers_d.append(nn.Dropout(dropout, inplace=True))
        self.project_d = nn.Sequential(*layers_d)
        
        layers_a: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        layers_a.append(nn.GELU())
        if dropout > 0.0:
            layers_a.append(nn.Dropout(dropout, inplace=True))
        self.project_a = nn.Sequential(*layers_a)
        self.norm_x = nn.LayerNorm(embed_dim,eps=1e-6)
        self.norm_d = nn.LayerNorm(embed_dim,eps=1e-6)
        self.norm_a = nn.LayerNorm(embed_dim,eps=1e-6)
        if self.change_x:
            self.out_project = nn.Linear(embed_dim, embed_dim)
            
        self.use_adaptSplit = AdaptSplit
        if AdaptSplit:
            self.adapt_split = adaptIndexSplit_dotsim(input_size,down_size=7,embed_dim=embed_dim)
        
    def forward(self, x_in:torch.Tensor, add_return:bool=False):
        # input shape: B,THW,C
        # ----- reshape and ignore cls token in waveLift------
        b,_,_ = x_in.shape
        if self.Non_CLSToken:
            x = x_in
        else:
            cls,x = x_in[:,0:1,:],x_in[:,1:,:] # filter out cls_embed

        x_skip = x
        x = x.view(b,self.T,self.H*self.W,self.embed_dim)
        x = x.permute(0,3,1,2) # B,T,H*W,C -> B,C,T,H*W
        
        # ----- split ------
        if not self.use_adaptSplit:
            x_a:torch.Tensor = x[:,:,0::2,:] # Approxiamation
            x_d:torch.Tensor = x[:,:,1::2,:] # Details
        else:
            x_a, x_d = self.adapt_split(x)
            
        # # unfold for local cross-attention
        # x_a = x_a.view(b,self.embed_dim,self.T//2,self.H,self.W)
        # x_a = self.unfold(x_a).flatten(-2) # B,C,T*s2,H/s2,W/s2 - >B,C,T*s2,H*W/s2 # FIXME    T   T        ，   C 
        x_a = x_a.permute(0,2,3,1) # B,T*s2,H*W/s2,C
        # x_d = x_d.view(b,self.embed_dim,self.T//2,self.H,self.W)
        # x_d = self.unfold(x_d).flatten(-2) # B,C,T*s2,H*W/s2
        x_d = x_d.permute(0,2,3,1) # B,T*s2,H*W/s2,C
        
        b,t,hw,c = x_d.shape
        if not self.WaveSF_preUpdate:
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,hw,head_dim
            q = self.q_d(x_d).reshape(b,t,hw,self.num_head,self.head_dim).transpose(2,3)
            # b,t,head,hw,head_dim
            k, v = self.kv_a(x_a).reshape(b,t,hw,2,self.num_head,self.head_dim).transpose(2,4).unbind(3)
            _d = torch.matmul(self.scaler*q, k.transpose(3,4))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v)
            _d = _d.transpose(2,3).reshape(b,t,hw,-1)
            _d = self.project_d(_d)
            x_d = x_d - _d        
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            q = self.q_a(x_a).reshape(b,t,hw,self.num_head,self.head_dim).transpose(2,3)
            # b,t,head,hw,head_dim
            k, v = self.kv_d(x_d).reshape(b,t,hw,2,self.num_head,self.head_dim).transpose(2,4).unbind(3)
            _a = torch.matmul(self.scaler*q, k.transpose(3,4))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v)
            _a = _a.transpose(2,3).reshape(b,t,hw,-1)
            _a = self.project_a(_a)
            x_a = x_a + _a
        else:
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            q = self.q_a(x_a).reshape(b,t,hw,self.num_head,self.head_dim).transpose(2,3)
            # b,t,head,hw,head_dim
            k, v = self.kv_d(x_d).reshape(b,t,hw,2,self.num_head,self.head_dim).transpose(2,4).unbind(3)
            _a = torch.matmul(self.scaler*q, k.transpose(3,4))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v)
            _a = _a.transpose(2,3).reshape(b,t,hw,-1)
            _a = self.project_a(_a)
            x_a = x_a + _a            
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,hw,head_dim
            q = self.q_d(x_d).reshape(b,t,hw,self.num_head,self.head_dim).transpose(2,3)
            # b,t,head,hw,head_dim
            k, v = self.kv_a(x_a).reshape(b,t,hw,2,self.num_head,self.head_dim).transpose(2,4).unbind(3)
            _d = torch.matmul(self.scaler*q, k.transpose(3,4))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v)
            _d = _d.transpose(2,3).reshape(b,t,hw,-1)
            _d = self.project_d(_d)
            x_d = x_d - _d 
            
        # unshuffle recover
        # b,t,hw,c -> b,c,t,hw
        x_d = x_d.permute(0,3,1,2)
        # x_d = x_d.view(b,self.embed_dim,self.T//2*self.s**2,self.H//self.s,self.W//self.s)
        # x_d = self.fold(x_d).flatten(-2)
        x_a = x_a.permute(0,3,1,2)
        # x_a = x_a.view(b,self.embed_dim,self.T//2*self.s**2,self.H//self.s,self.W//self.s)
        # x_a = self.fold(x_a).flatten(-2)
        
        if self.relaxed_constrain:
            # Relax constrain boundaries to restrain the mean of x_a channel to be the same, and restrain the mean of x_d channel to be zero
            b,c,t,hw = x_a.shape
            x_skip = x_skip.view(b,t*2,hw,c).permute(0,3,1,2)
            # so far only SSL network will use lowFreq
            pool_a = torch.mean(nn.functional.avg_pool2d(x_skip,(2,1),stride=(2,1)),dim=1) - torch.mean(x_a,dim=1)

            x_skip = x_skip.permute(0,2,3,1).view(b,-1,c)
            if not self.SSL: # for SSL network, x_d should not be decreased dimension
                x_d = torch.mean(x_d,dim=1,keepdim=True)
        else:
            pool_a = nn.functional.avg_pool2d(x_skip,(2,1),stride=(2,1)) - x_a
        
        if not self.use_lowFreq:
            return pool_a, x_d
        else:
            return pool_a, x_a, x_d

class WaveSF_ViT_region_1D(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 Non_CLSToken:bool = True,
                 ):
        super().__init__()
        self.relaxed_constrain = relaxed_constrain
        self.change_x = change_x
        self.SSL = SSL
        self.WaveSF_preUpdate = WaveSF_preUpdate
        self.use_lowFreq = lowFreq
        self.Non_CLSToken = Non_CLSToken
    
        self.embed_dim = embed_dim
        self.T,self.H,self.W = input_size
        self.s = split_stride
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.q_d = nn.Linear(embed_dim,embed_dim)
        self.kv_a = nn.Linear(embed_dim,2*embed_dim)
        self.q_a = nn.Linear(embed_dim,embed_dim)
        self.kv_d = nn.Linear(embed_dim,2*embed_dim)
        
        layers_d: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        layers_d.append(nn.GELU())
        if dropout > 0.0:
            layers_d.append(nn.Dropout(dropout, inplace=True))
        self.project_d = nn.Sequential(*layers_d)
        
        layers_a: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        layers_a.append(nn.GELU())
        if dropout > 0.0:
            layers_a.append(nn.Dropout(dropout, inplace=True))
        self.project_a = nn.Sequential(*layers_a)
        # self.norm_x = nn.LayerNorm(embed_dim,eps=1e-6)
        # self.norm_d = nn.LayerNorm(embed_dim,eps=1e-6)
        # self.norm_a = nn.LayerNorm(embed_dim,eps=1e-6)
        if self.change_x:
            self.out_project = nn.Linear(embed_dim, embed_dim)
            
        self.use_adaptSplit = AdaptSplit
        if AdaptSplit:
            self.adapt_split = adaptIndexSplit_dotsim_1D(input_size,embed_dim=embed_dim)
        
    def forward(self, x_in:torch.Tensor, add_return:bool=False):
        # input shape: B,THW,C
        # ----- reshape and ignore cls token in waveLift------
        b,_,_ = x_in.shape
        if self.Non_CLSToken:
            x = x_in
        else:
            cls,x = x_in[:,0:1,:],x_in[:,1:,:] # filter out cls_embed

        x_skip = x # B,T,C
        
        # ----- split ------
        if not self.use_adaptSplit:
            x_a:torch.Tensor = x[:,0::2,:] # Approxiamation
            x_d:torch.Tensor = x[:,1::2,:] # Details
        else:
            x_a, x_d = self.adapt_split(x)
        
        b,t,c = x_d.shape
        if not self.WaveSF_preUpdate:
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,head_dim
            q = self.q_d(x_d).reshape(b,t,self.num_head,self.head_dim).transpose(1,2)
            # b,t,head,head_dim
            k, v = self.kv_a(x_a).reshape(b,t,2,self.num_head,self.head_dim).transpose(1,3).unbind(2) # b,num_head,t,head_dim
            _d = torch.matmul(self.scaler*q, k.transpose(2,3))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v) # b,num_head,t,head_dim
            _d = _d.transpose(1,2).reshape(b,t,-1)
            _d = self.project_d(_d)
            x_d = x_d - _d  
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            q = self.q_a(x_a).reshape(b,t,self.num_head,self.head_dim).transpose(1,2) # b,num_head,t,head_dim
            k, v = self.kv_d(x_d).reshape(b,t,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            _a = torch.matmul(self.scaler*q, k.transpose(2,3))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v)
            _a = _a.transpose(1,2).reshape(b,t,-1)
            _a = self.project_a(_a)
            x_a = x_a + _a
        else:
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            q = self.q_a(x_a).reshape(b,t,self.num_head,self.head_dim).transpose(1,2) # b,num_head,t,head_dim
            k, v = self.kv_d(x_d).reshape(b,t,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            _a = torch.matmul(self.scaler*q, k.transpose(2,3))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v)
            _a = _a.transpose(1,2).reshape(b,t,-1)
            _a = self.project_a(_a)
            x_a = x_a + _a 
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,head_dim
            q = self.q_d(x_d).reshape(b,t,self.num_head,self.head_dim).transpose(1,2)
            # b,t,head,head_dim
            k, v = self.kv_a(x_a).reshape(b,t,2,self.num_head,self.head_dim).transpose(1,3).unbind(2) # b,num_head,t,head_dim
            _d = torch.matmul(self.scaler*q, k.transpose(2,3))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v) # b,num_head,t,head_dim
            _d = _d.transpose(1,2).reshape(b,t,-1)
            _d = self.project_d(_d)
            x_d = x_d - _d          
            
        # unshuffle recover
        #  x_a: b,t,c -> b,c,t for avg_pool2d
        x_a = x_a.permute(0,2,1)
        
        if self.relaxed_constrain:
            # Relax constrain boundaries to restrain the mean of x_a channel to be the same, and restrain the mean of x_d channel to be zero
            b,c,t = x_a.shape
            x_skip = x_skip.permute(0,2,1) # b,c,2t
            # so far only SSL network will use lowFreq
            pool_a = torch.mean(nn.functional.avg_pool1d(x_skip,2,stride=2),dim=1) - torch.mean(x_a,dim=1)

            x_skip = x_skip.permute(0,2,1) # b,t,c
            if not self.SSL: # for SSL network, x_d should not be decreased dimension
                x_d = torch.mean(x_d,dim=1,keepdim=True)
        else:
            x_skip = x_skip.permute(0,2,1) # b,c,2t
            x_skip = nn.functional.avg_pool1d(x_skip,2,stride=2)
            pool_a = x_skip - x_a
        
        if not self.use_lowFreq:
            return pool_a, x_d
        else:
            return pool_a, x_a, x_d
        
class WaveSF_ViT_org(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 use_norm:bool = False,
                 ):
        super().__init__()
        self.relaxed_constrain = relaxed_constrain
        self.change_x = change_x
        self.SSL = SSL
        self.WaveSF_preUpdate = WaveSF_preUpdate
        self.use_lowFreq = lowFreq
        self.use_norm = use_norm
        
        self.embed_dim = embed_dim
        self.nT,self.nH,self.nW,self.T,self.H,self.W = input_size
        self.s = split_stride
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.unfold = torch.nn.PixelUnshuffle(self.s)
        self.fold = torch.nn.PixelShuffle(self.s)
        self.q_d = nn.Linear(embed_dim,embed_dim)
        self.kv_a = nn.Linear(embed_dim,2*embed_dim)
        self.q_a = nn.Linear(embed_dim,embed_dim)
        self.kv_d = nn.Linear(embed_dim,2*embed_dim)
        
        layers_d: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        layers_d.append(nn.GELU())
        if dropout > 0.0:
            layers_d.append(nn.Dropout(dropout, inplace=True))
        self.project_d = nn.Sequential(*layers_d)
        
        layers_a: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        layers_a.append(nn.GELU())
        if dropout > 0.0:
            layers_a.append(nn.Dropout(dropout, inplace=True))
        self.project_a = nn.Sequential(*layers_a)
        self.norm_d = nn.LayerNorm(embed_dim,eps=1e-6)
        self.norm_a = nn.LayerNorm(embed_dim,eps=1e-6)
        if self.change_x:
            self.out_project = nn.Linear(embed_dim, embed_dim)
            
        self.use_adaptSplit = AdaptSplit
        if AdaptSplit:
            self.adapt_split = adaptIndexSplit_dotsim(input_size,down_size=7,embed_dim=embed_dim)
        
    def forward(self, x_in:torch.Tensor, add_return:bool=False):
        # input shape: B,THW,C
        # ----- reshape and ignore cls token in waveLift------
        b,_,_ = x_in.shape
        x = rearrange(x_in, 'b (nt nh nw t h w) c -> b (nt t) (nh h) (nw w) c', nt=self.nT,nh=self.nH,nw=self.nW,t=self.T,h=self.H,w=self.W,c=self.embed_dim)
        b,t,h,w,c = x.shape
        
        x_skip = x
        x = x.view(b,t,h*w,self.embed_dim)
        x = x.permute(0,3,1,2) # B,T,H*W,C -> B,C,T,H*W
        
        # ----- split ------
        if not self.use_adaptSplit:
            x_a:torch.Tensor = x[:,:,0::2,:] # Approxiamation
            x_d:torch.Tensor = x[:,:,1::2,:] # Details
        else:
            x_a, x_d = self.adapt_split(x)
            
        # # unfold for local cross-attention
        # x_a = x_a.view(b,self.embed_dim,self.T//2,self.H,self.W)
        # x_a = self.unfold(x_a).flatten(-2) # B,C,T*s2,H/s2,W/s2 - >B,C,T*s2,H*W/s2 # FIXME    T   T        ，   C 
        x_a = x_a.permute(0,2,3,1) # B,T*s2,H*W/s2,C
        # x_d = x_d.view(b,self.embed_dim,self.T//2,self.H,self.W)
        # x_d = self.unfold(x_d).flatten(-2) # B,C,T*s2,H*W/s2
        x_d = x_d.permute(0,2,3,1) # B,T*s2,H*W/s2,C
        if self.use_norm:
            x_a = self.norm_a(x_a)
            x_d = self.norm_d(x_d)
        
        b,t,hw,c = x_d.shape
        if not self.WaveSF_preUpdate:
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,hw,head_dim
            q = self.q_d(x_d).reshape(b,t,hw,self.num_head,self.head_dim).transpose(2,3)
            # b,t,head,hw,head_dim
            k, v = self.kv_a(x_a).reshape(b,t,hw,2,self.num_head,self.head_dim).transpose(2,4).unbind(3)
            _d = torch.matmul(self.scaler*q, k.transpose(3,4))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v)
            _d = _d.transpose(2,3).reshape(b,t,hw,-1)
            _d = self.project_d(_d)
            x_d = x_d - _d        
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            q = self.q_a(x_a).reshape(b,t,hw,self.num_head,self.head_dim).transpose(2,3)
            # b,t,head,hw,head_dim
            k, v = self.kv_d(x_d).reshape(b,t,hw,2,self.num_head,self.head_dim).transpose(2,4).unbind(3)
            _a = torch.matmul(self.scaler*q, k.transpose(3,4))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v)
            _a = _a.transpose(2,3).reshape(b,t,hw,-1)
            _a = self.project_a(_a)
            x_a = x_a + _a
        else:
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            q = self.q_a(x_a).reshape(b,t,hw,self.num_head,self.head_dim).transpose(2,3)
            # b,t,head,hw,head_dim
            k, v = self.kv_d(x_d).reshape(b,t,hw,2,self.num_head,self.head_dim).transpose(2,4).unbind(3)
            _a = torch.matmul(self.scaler*q, k.transpose(3,4))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v)
            _a = _a.transpose(2,3).reshape(b,t,hw,-1)
            _a = self.project_a(_a)
            x_a = x_a + _a            
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,hw,head_dim
            q = self.q_d(x_d).reshape(b,t,hw,self.num_head,self.head_dim).transpose(2,3)
            # b,t,head,hw,head_dim
            k, v = self.kv_a(x_a).reshape(b,t,hw,2,self.num_head,self.head_dim).transpose(2,4).unbind(3)
            _d = torch.matmul(self.scaler*q, k.transpose(3,4))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v)
            _d = _d.transpose(2,3).reshape(b,t,hw,-1)
            _d = self.project_d(_d)
            x_d = x_d - _d 
            
        # unshuffle recover
        # b,t,hw,c -> b,c,t,hw
        x_d = x_d.permute(0,3,1,2)
        # x_d = x_d.view(b,self.embed_dim,self.T//2*self.s**2,self.H//self.s,self.W//self.s)
        # x_d = self.fold(x_d).flatten(-2)
        x_a = x_a.permute(0,3,1,2)
        # x_a = x_a.view(b,self.embed_dim,self.T//2*self.s**2,self.H//self.s,self.W//self.s)
        # x_a = self.fold(x_a).flatten(-2)
        
        if self.relaxed_constrain:
            # Relax constrain boundaries to restrain the mean of x_a channel to be the same, and restrain the mean of x_d channel to be zero
            b,c,t,hw = x_a.shape
            x_skip = x_skip.view(b,t*2,hw,c).permute(0,3,1,2)
            # so far only SSL network will use lowFreq
            pool_a = torch.mean(nn.functional.avg_pool2d(x_skip,(2,1),stride=(2,1)),dim=1) - torch.mean(x_a,dim=1)

            x_skip = x_skip.permute(0,2,3,1).view(b,-1,c)
            if not self.SSL: # for SSL network, x_d should not be decreased dimension
                x_d = torch.mean(x_d,dim=1,keepdim=True)
        else:
            pool_a = nn.functional.avg_pool2d(x_skip,(2,1),stride=(2,1)) - x_a
        
        if not self.use_lowFreq:
            return pool_a, x_d
        else:
            return pool_a, x_a, x_d