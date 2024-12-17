import torch
from torch import nn
from functools import partial
from typing import List, Optional
import math
from .AdaptSplit import adaptSplit_dotsim, adaptSplit_dotsim_weight, adaptIndexSplit_dotsim, adaptIndexSplit_dimLearn
from .blocks import TemporalSeparableConv

# ------------- CNN-based Video Lifting Scheme ------------- 
# minimize realization
class WaveSF_CNN(nn.Module):
    def __init__(self,
                input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 prompt:bool = False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 WaveNorm:bool = True,
                 residual:bool = True,
                 use_CLS:bool = False,
                 output_dim:Optional[int]=None,
                 ) -> None:
        super().__init__()
        channel = embed_dim
        self.T,self.H,self.W = input_size
        self.embed_dim = embed_dim
        norm_layer = partial(nn.BatchNorm3d, eps=0.001, momentum=0.001)
        self.P = TemporalSeparableConv(in_planes=channel,
                                       out_planes=channel,
                                       kernel_size_s=3,
                                       stride=1,
                                       padding_s=1,
                                       norm_layer=norm_layer
                                       )
        self.U = TemporalSeparableConv(in_planes=channel,
                                       out_planes=channel,
                                       kernel_size_s=3,
                                       stride=1,
                                       padding_s=1,
                                       norm_layer=norm_layer
                                       )
        self.project_compress = nn.Linear(self.T//2*self.H*self.W,1,bias=False)
        
    def forward(self, x_in:torch.Tensor, input_CLS:bool = True, add_return:bool=False):
        b,cthw,c = x_in.shape
        if input_CLS:
            cls,x = x_in[:,0:1,:],x_in[:,1:,:] # filter out cls_embed
        else:
            x = x_in
        x = x.view(b,self.T,self.H,self.W,self.embed_dim).permute(0,4,1,2,3)
        b,c,t,h,w = x.shape
        x_skip = x
        if t%2!=0:
            raise ValueError()
        # time split
        x_pre = x[:,:,0::2,:,:] 
        x_next = x[:,:,1::2,:,:]
        # predict the detailed information
        # predict the approxiated information
        x_a = x_pre + self.U(x_next)
        x_d = x_next - self.P(x_a)
        
        x_d = x_d.flatten(start_dim=2)
        x_d = self.project_compress(x_d)
        x_d = x_d.squeeze(-1)
        
        x_skip = nn.functional.adaptive_avg_pool3d(x_skip,output_size=(self.T//2,h,w))
        pool_a = torch.mean(x_skip-x_a,dim=(1,2))
        
        return pool_a, x_d

# --------- Transformer-based Video Lifting Scheme --------- 
# TODO: Linear Attention
# TODO: Relative Position Encoding
class WaveSF_trans(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 prompt:bool = False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 WaveNorm:bool = False,
                 residual:bool = True,
                 ):
        super().__init__()
        self.prompt = prompt
        self.relaxed_constrain = relaxed_constrain
        self.change_x = change_x
        self.SSL = SSL
        self.WaveSF_preUpdate = WaveSF_preUpdate
        self.use_lowFreq = lowFreq
    
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
        # layers_d.append(nn.GELU())
        if dropout > 0.0:
            layers_d.append(nn.Dropout(dropout, inplace=True))
        self.project_d = nn.Sequential(*layers_d)
        
        layers_a: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        # layers_a.append(nn.GELU())
        if dropout > 0.0:
            layers_a.append(nn.Dropout(dropout, inplace=True))
        self.project_a = nn.Sequential(*layers_a)
        if self.change_x:
            self.out_project = nn.Linear(embed_dim, embed_dim)
            
        self.use_adaptSplit = AdaptSplit
        if AdaptSplit:
            self.adapt_split = adaptIndexSplit_dotsim(input_size,down_size=7,embed_dim=embed_dim)
        self.WaveNorm = WaveNorm
        if WaveNorm:
            self.xa_norm = nn.LayerNorm(self.embed_dim)
            self.xd_norm = nn.LayerNorm(self.embed_dim)
        self.residual = residual
        # adaptIndexSplit_dimLearn(input_size,down_size=7,embed_dim=embed_dim)
        # adaptIndexSplit_dotsim(input_size,down_size=7,embed_dim=embed_dim)
        # adaptIndexSplit_dimLearn(input_size,down_size=7,embed_dim=embed_dim)
        # adaptIndexSplit_dotsim(input_size,down_size=7,embed_dim=embed_dim)
        
        # split_args = [7,True]
        # self.adapt_split = adaptSplit_dotsim(input_size,down_size=split_args[0],embed_dim=embed_dim,prior_order=split_args[1])
        
        # split_args = [7,True,True]
        # adaptSplit_dotsim_weight(input_size,down_size=split_args[0],embed_dim=embed_dim,position_encode=split_args[1],T_project=split_args[2])
    def forward(self, x_in:torch.Tensor, prompt:torch.Tensor=None,  add_return:bool=False):
        # input shape: B,THW+1,C
        # ----- reshape and ignore cls token in waveLift------
        b,_,_ = x_in.shape
        cls,x = x_in[:,0:1,:],x_in[:,1:,:] # filter out cls_embed
        # x = self.norm_x(x)
        x_skip = x
        x = x.view(b,self.T,self.H*self.W,self.embed_dim)
        x = x.permute(0,3,1,2) # B,T,H*W,C -> B,C,T,H*W
        
        # ----- split ------
        if not self.use_adaptSplit:
            x_a:torch.Tensor = x[:,:,0::2,:] # Approxiamation
            x_d:torch.Tensor = x[:,:,1::2,:] # Details
        else:
            x_a, x_d = self.adapt_split(x)
            
        # unfold for local cross-attention
        x_a = x_a.view(b,self.embed_dim,self.T//2,self.H,self.W)
        x_a = self.unfold(x_a).flatten(-2) # B,C,T*s2,H/s2,W/s2 - >B,C,T*s2,H*W/s2 # FIXME    T   T        ，   C 
        x_a = x_a.permute(0,2,3,1) # B,T*s2,H*W/s2,C
        x_d = x_d.view(b,self.embed_dim,self.T//2,self.H,self.W)
        x_d = self.unfold(x_d).flatten(-2) # B,C,T*s2,H*W/s2
        x_d = x_d.permute(0,2,3,1) # B,T*s2,H*W/s2,C
        
        # norm
        if self.WaveNorm:
            x_a = self.xa_norm(x_a)
            x_d = self.xd_norm(x_d)

        if self.prompt:
            # ------ prompt ------
            # b,1,C
            _pb, _pd = prompt.shape
            prompt = prompt.view(_pb,1,1,_pd).repeat(1,1,x_a.shape[2],1)
            x_a = torch.cat([prompt,x_a],dim=1)
            x_d = torch.cat([prompt,x_d],dim=1)

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
            if self.residual:
                v:torch.tensor = v.transpose(2,3).flatten(start_dim=3)
                _d += v
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
            if self.residual:
                v:torch.tensor = v.transpose(2,3).flatten(start_dim=3)
                _a += v
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
            if self.residual:
                v:torch.tensor = v.transpose(2,3).flatten(start_dim=3)
                _a += v
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
            if self.residual:
                v:torch.tensor = v.transpose(2,3).flatten(start_dim=3)
                _d += v
            _d = self.project_d(_d)
            x_d = x_d - _d 
            
        # unshuffle recover
        # b,t,hw,c -> b,c,t,hw
        x_d = x_d.permute(0,3,1,2).view(b,self.embed_dim,self.T//2*self.s**2,self.H//self.s,self.W//self.s)
        if self.prompt:
            x_d = x_d[:,:,1:,:,:]
        x_d = self.fold(x_d).flatten(-2)
        x_a = x_a.permute(0,3,1,2).view(b,self.embed_dim,self.T//2*self.s**2,self.H//self.s,self.W//self.s)
        if self.prompt:
            x_a = x_a[:,:,1:,:,:]
        x_a = self.fold(x_a).flatten(-2)
        
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

                
        
        if self.change_x:    
            # b,c,t,2,hw -> b,c,t,hw,  t 2  x_a x_d      
            x = torch.stack((x_a,x_d),dim=3).view(b,c,self.T,self.H*self.W)
            # b,c,t,hw -> b,t,hw,c -> b,thw,c
            x = x.permute(0,2,3,1).view(b,-1,c)
            x = self.out_project(x)
            x += x_skip
            x = torch.cat((cls,x),dim=1)
            
        out = x if self.change_x else x_in
        if not self.use_lowFreq:
            return out, pool_a, x_d
        else:
            return out, pool_a, x_a, x_d


class WaveSF_trans_prompt(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 prompt:bool = False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 WaveNorm:bool=False
                 ):
        super().__init__()
        self.relaxed_constrain = relaxed_constrain
        self.change_x = change_x
        self.SSL = SSL
        self.use_prompt = prompt
        self.WaveSF_preUpdate = WaveSF_preUpdate
        self.use_lowFreq = lowFreq
        
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
        if self.change_x:
            self.out_project = nn.Linear(embed_dim, embed_dim)

        self.use_adaptSplit = AdaptSplit
        if AdaptSplit:
            self.adapt_split =  adaptIndexSplit_dotsim(input_size,down_size=7,embed_dim=embed_dim)
        self.WaveNorm = WaveNorm
        if WaveNorm:
            self.xa_norm = nn.LayerNorm(self.embed_dim)
            self.xd_norm = nn.LayerNorm(self.embed_dim)
        
    def forward(self, x_in:torch.Tensor, prompt:torch.Tensor=None, add_return:bool=None):
        # input shape: B,THW+1,C
        # ----- reshape and ignore cls token in waveLift------
        assert self.use_prompt, 'This class should use use_prompt = True'
        b,_,_ = x_in.shape
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

        
        # ----- unfold ------
        # unfold for local cross-attention
        x_a = x_a.view(b,self.embed_dim,self.T//2,self.H,self.W)
        x_a = self.unfold(x_a).flatten(-2) # B,C,T*s2,H/s2,W/s2 - >B,C,T*s2,H*W/s2 # FIXME    T   T        ，   C 
        x_a = x_a.permute(0,2,3,1) # B,T*s2,H*W/s2,C
        x_d = x_d.view(b,self.embed_dim,self.T//2,self.H,self.W)
        x_d = self.unfold(x_d).flatten(-2) # B,C,T*s2,H*W/s2
        x_d = x_d.permute(0,2,3,1) # B,T*s2,H*W/s2,C
        
        # ------ prompt ------
        # b,1,C
        _pb, _pd = prompt.shape
        prompt = prompt.view(_pb,1,1,_pd).repeat(1,1,x_a.shape[2],1)
        x_a = torch.cat([prompt,x_a],dim=1)
        x_d = torch.cat([prompt,x_d],dim=1)
        
        b,t,hw,c = x_d.shape
        if not self.WaveSF_preUpdate:
            # ------ Predictor -----
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
            # ------ Predictor -----
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
            
        # ------ unshuffle recover ------
        # b,t,hw,c -> b,c,t,hw
        x_d = x_d.permute(0,3,1,2).view(b,self.embed_dim,(self.T//2)*self.s**2+1,self.H//self.s,self.W//self.s)[:,:,1:,:,:]
        x_d = self.fold(x_d).flatten(-2)
        x_a = x_a.permute(0,3,1,2).view(b,self.embed_dim,(self.T//2)*self.s**2+1,self.H//self.s,self.W//self.s)[:,:,1:,:,:]
        x_a = self.fold(x_a).flatten(-2)
        
        if self.relaxed_constrain:
            # Relax constrain boundaries to restrain the mean of x_a channel to be the same, and restrain the mean of x_d channel to be zero
            b,c,t,hw = x_a.shape
            x_skip = x_skip.view(b,t*2,hw,c).permute(0,3,1,2)
            pool_a = torch.mean(nn.functional.avg_pool2d(x_skip,(2,1),stride=(2,1)),dim=1) - torch.mean(x_a,dim=1)
        else:
            pool_a = nn.functional.avg_pool2d(x_skip,(2,1),stride=(2,1)) - x_a
        if not self.use_lowFreq:
            return x_in, pool_a, x_d
        else:
            return x_in, pool_a, x_a, x_d
        
# FIXME new version that need to be added
class WaveSF_trans_pool(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 prompt:bool = False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 WaveNorm:bool = False,
                 residual:bool = True,
                 attn_size:int = 14
                 ):
        super().__init__()
        self.prompt = prompt
        self.relaxed_constrain = relaxed_constrain
        self.change_x = False
        self.SSL = SSL
        self.WaveSF_preUpdate = WaveSF_preUpdate
        self.use_lowFreq = lowFreq
    
        self.embed_dim = embed_dim
        self.T,self.H,self.W = input_size
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.q_d = nn.Linear(embed_dim,embed_dim)
        self.kv_a = nn.Linear(embed_dim,2*embed_dim)
        self.q_a = nn.Linear(embed_dim,embed_dim)
        self.kv_d = nn.Linear(embed_dim,2*embed_dim)
        
        layers_d: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        # layers_d.append(nn.GELU())
        if dropout > 0.0:
            layers_d.append(nn.Dropout(dropout, inplace=True))
        self.project_d = nn.Sequential(*layers_d)
        
        layers_a: List[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        # layers_a.append(nn.GELU())
        if dropout > 0.0:
            layers_a.append(nn.Dropout(dropout, inplace=True))
        self.project_a = nn.Sequential(*layers_a)
        if self.change_x:
            self.out_project = nn.Linear(embed_dim, embed_dim)
            
        self.use_adaptSplit = AdaptSplit
        if AdaptSplit:
            self.adapt_split = adaptIndexSplit_dotsim(input_size,down_size=7,embed_dim=embed_dim)
        # norm for detail and approximation branch
        self.WaveNorm = WaveNorm
        if WaveNorm:
            self.xa_norm = nn.LayerNorm(self.embed_dim)
            self.xd_norm = nn.LayerNorm(self.embed_dim)
        # residual connection
        self.residual = residual
        # rel position # TODO
        from core.model.net_blocks import Pool
        self.attn_s = attn_size
        _s = self.H//self.attn_s
        kernel = [3,3,3]
        stride_q = [1,_s,_s]
        _s = self.H//self.attn_s
        stride_kv = [1,_s,_s]
        padding_q = [int(q // 2) for q in kernel]
        # q,kv pool for x_a and x_d
        self.xa_xd_q_pool = nn.ModuleList([
            Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_size=kernel,
                    stride=stride_q,
                    padding=padding_q,
                    groups=self.head_dim,
                    bias=False,
                ),
                nn.LayerNorm(self.head_dim),
            )
            for i in range(2)
        ])
        self.xa_xd_kv_pool = nn.ModuleList([
            Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_size=kernel,
                    stride=stride_kv,
                    padding=padding_q,
                    groups=self.head_dim,
                    bias=False,
                ),
                nn.LayerNorm(self.head_dim),
            )
            for i in range(4)
        ])

    def forward(self, x_in:torch.Tensor, add_return:bool=False):
        # input shape: B,THW+1,C
        # ----- reshape and ignore cls token in waveLift------
        b,cthw,c = x_in.shape
        
        cls,x = x_in[:,0:1,:],x_in[:,1:,:] # filter out cls_embed
        # x = self.norm_x(x)
        x_skip = x
        x = x.view(b,self.T,self.H*self.W,self.embed_dim)
        x = x.permute(0,3,1,2) # B,T,H*W,C -> B,C,T,H*W
        
        # ----- split ------
        if not self.use_adaptSplit:
            x_a:torch.Tensor = x[:,:,0::2,:] # Approxiamation
            x_d:torch.Tensor = x[:,:,1::2,:] # Details
        else:
            x_a, x_d = self.adapt_split(x)
            
        # unfold for nhead splitting of spatial dimension for local cross-attention
        x_a = x_a.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        x_d = x_d.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        
        # norm
        if self.WaveNorm:
            x_a = self.xa_norm(x_a)
            x_d = self.xd_norm(x_d)
        
        b,thw,c = x_d.shape
        t, h, w = self.T//2, self.H, self.W
        if not self.WaveSF_preUpdate:
            assert False
        else:
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            q = self.q_a(x_a).reshape(b,thw,self.num_head,self.head_dim).transpose(1,2)
            # b,t,head,hw,head_dim
            k, v = self.kv_d(x_d).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            q,q_thw = self.xa_xd_q_pool[0](q,(t,h,w),no_CLS=True)
            k,k_thw = self.xa_xd_kv_pool[1](k,(t,h,w),no_CLS=True)
            v,_ = self.xa_xd_kv_pool[2](v,(t,h,w),no_CLS=True)
            
            _a = torch.matmul(self.scaler*q, k.transpose(2,3))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v) # b,head,thw,head_dim
            _a = _a.transpose(1,2).reshape(b,t*(self.attn_s**2),-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).flatten(start_dim=2)
                _a += v
            _a = self.project_a(_a)
            # reshape and pool x
            x_a = x_a.reshape(b,self.T//2,self.H,self.W,c).permute(0,1,4,2,3)
            x_a = x_a.reshape(b,self.T//2*c,self.H,self.W)
            x_a = nn.functional.adaptive_avg_pool2d(x_a, output_size=(self.attn_s,self.attn_s))
            x_a = x_a.reshape(b,self.T//2,c,self.attn_s,self.attn_s).permute(0,1,3,4,2).reshape(b,-1,c)
            # Lift Scheme
            x_a = x_a + _a
            
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,hw,head_dim
            q = self.q_d(x_d).reshape(b,thw,self.num_head,self.head_dim).transpose(1,2)
            # b,t,head,hw,head_dim
            thw = t*self.attn_s*self.attn_s
            k, v = self.kv_a(x_a).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            q, q_thw = self.xa_xd_q_pool[1](q,(t,h,w),no_CLS=True)
            # k,k_thw = self.xa_xd_kv_pool[0](k,(t,h,w),no_CLS=True)
            # v,_ = self.xa_xd_kv_pool[1](v,(t,h,w),no_CLS=True)
            _d = torch.matmul(self.scaler*q, k.transpose(2,3))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v) # b,thw,head,head_dim
            _d = _d.transpose(1,2).reshape(b,thw,-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).flatten(start_dim=2)
                _d += v
            _d = self.project_d(_d)
            # reshape and pool x
            x_d = x_d.reshape(b,self.T//2,self.H,self.W,c).permute(0,1,4,2,3)
            x_d = x_d.reshape(b,self.T//2*c,self.H,self.W)
            x_d = nn.functional.adaptive_avg_pool2d(x_d, output_size=(self.attn_s,self.attn_s))
            x_d = x_d.reshape(b,self.T//2,c,self.attn_s,self.attn_s).permute(0,1,3,4,2).reshape(b,-1,c)
            # Lift Scheme
            x_d = x_d - _d
        
            # Relax constrain boundaries to restrain the mean of x_a channel to be the same, and restrain the mean of x_d channel to be zero
            x_a = x_a.view(b,self.T//2,self.attn_s,self.attn_s,c).permute(0,4,1,2,3) # b,c,t,h,w
            b,thw,c = x_skip.shape
            x_skip = x_skip.view(b,self.T,self.H,self.W,c).permute(0,4,1,2,3) # b,c,t,h,w
            x_skip = nn.functional.adaptive_avg_pool3d(x_skip,output_size=(self.T//2,self.attn_s,self.attn_s)) # b,c,t,h,w
            # so far only SSL network will use lowFreq
            if self.relaxed_constrain:
                pool_a = torch.mean(x_skip-x_a,dim=1)
            else:
                pool_a = x_skip-x_a
        
            # b,c,t,hw for head input
            x_d = x_d.view(b,self.T//2,self.attn_s,self.attn_s,c).permute(0,4,1,2,3).reshape(b,c,self.T//2,self.attn_s**2)
            if not self.SSL: # for SSL network, x_d should not be decreased dimension
                x_d = torch.mean(x_d,dim=1,keepdim=True)

            
        if not self.use_lowFreq:
            return x_in, pool_a, x_d
        else:
            return x_in, pool_a, x_a, x_d

class WaveSF_trans_CLS(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 prompt:bool = False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 WaveNorm:bool = True,
                 residual:bool = True,
                 use_CLS:bool = False,
                 output_dim:Optional[int]=None,
                 ):
        super().__init__()
        self.use_CLS = False
        self.prompt = prompt
        self.relaxed_constrain = relaxed_constrain
        self.change_x = False
        self.SSL = SSL
        self.WaveSF_preUpdate = WaveSF_preUpdate
        self.use_lowFreq = lowFreq
    
        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim
        self.T,self.H,self.W = input_size
        self.num_head = num_head
        self.head_dim = self.hidden_dim // num_head
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.q_d = nn.Linear(embed_dim,self.hidden_dim)
        self.kv_a = nn.Linear(embed_dim,2*self.hidden_dim)
        self.q_a = nn.Linear(embed_dim,self.hidden_dim)
        self.kv_d = nn.Linear(embed_dim,2*self.hidden_dim)
        
        self.spatial_pos = nn.Parameter(torch.zeros(self.H * self.W, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(self.T, embed_dim))
        self.class_pos = nn.Parameter(torch.zeros(embed_dim))
        
        layers_d: List[nn.Module] = [nn.Linear(self.hidden_dim, embed_dim)]
        # layers_d.append(nn.GELU())
        if dropout > 0.0:
            layers_d.append(nn.Dropout(dropout, inplace=True))
        self.project_d = nn.Sequential(*layers_d)
        
        layers_a: List[nn.Module] = [nn.Linear(self.hidden_dim, embed_dim)]
        # layers_a.append(nn.GELU())
        if dropout > 0.0:
            layers_a.append(nn.Dropout(dropout, inplace=True))
        self.project_a = nn.Sequential(*layers_a)
        if self.change_x:
            self.out_project = nn.Linear(embed_dim, embed_dim)
            
        self.use_adaptSplit = AdaptSplit
        if AdaptSplit:
            self.adapt_split = adaptIndexSplit_dotsim(input_size,down_size=7,embed_dim=embed_dim)
        # norm for detail and approximation branch
        self.WaveNorm = WaveNorm
        if WaveNorm:
            self.xa_norm = nn.LayerNorm(self.embed_dim)
            self.xd_norm = nn.LayerNorm(self.embed_dim)
        self.out_norm1 = nn.LayerNorm(self.embed_dim)
        self.out_norm2 = nn.LayerNorm(self.embed_dim)
        # residual connection
        self.residual = residual
        
        # output self attention
        out_dim = output_dim if output_dim else embed_dim
        self.out_dim = out_dim
        self.num_head_out = num_head
        self.head_dim_out = self.out_dim // self.num_head_out
        self.qkv = nn.Linear(embed_dim, 3 * out_dim)
        layers: List[nn.Module] = [nn.Linear(out_dim, out_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.project = nn.Sequential(*layers)
        from core.model.net_blocks import StochasticDepth, MLP
        self.stochastic_depth = StochasticDepth(0.2, "row")
        self.mlp = MLP(
            out_dim,
            [4 * out_dim, out_dim],
            activation_layer=nn.GELU,
            dropout=dropout,
            inplace=None,
        )
        # compression dim
        self.project_compress = nn.Linear(self.T//2*self.H*self.W,1,bias=False)
        
        # For hook
        self.hook_xa = nn.Identity()
        self.hook_xd = nn.Identity()

    def forward(self, x_in:torch.Tensor, input_CLS:bool = True, add_return:bool=False):
        # input shape: B,THW+1,C
        # ----- reshape and ignore cls token in waveLift------
        b,cthw,c = x_in.shape
        if input_CLS:
            cls,x = x_in[:,0:1,:],x_in[:,1:,:] # filter out cls_embed
        else:
            x = x_in
        # x = self.norm_x(x)
        x_skip = x
        # pos_embedding：FIXME      ，       pos  
        pos = False
        if pos:
            hw_size, embed_size = self.spatial_pos.shape
            pos_embedding = torch.repeat_interleave(self.temporal_pos, hw_size, dim=0)
            pos_embedding.add_(self.spatial_pos.unsqueeze(0).expand(self.T, -1, -1).reshape(-1, embed_size))
            x.add_(pos_embedding)
        # reshape
        x = x.view(b,self.T,self.H*self.W,self.embed_dim)
        x = x.permute(0,3,1,2) # B,T,H*W,C -> B,C,T,H*W
        
        # ----- split ------
        if not self.use_adaptSplit:
            x_a:torch.Tensor = x[:,:,0::2,:] # Approxiamation
            x_d:torch.Tensor = x[:,:,1::2,:] # Details
        else:
            x_a, x_d = self.adapt_split(x)
            
        # unfold for nhead splitting of spatial dimension for local cross-attention
        x_a = x_a.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        x_d = x_d.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        
        # norm
        if self.WaveNorm:
            x_a = self.xa_norm(x_a)
            x_d = self.xd_norm(x_d)
        
        b,thw,c = x_d.shape
        t, h, w = self.T//2, self.H, self.W
        if not self.WaveSF_preUpdate:
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,hw,head_dim
            # cls_token
            if self.use_CLS:
                cls_token, x_a = x_a[:,0,:], x[:,1:,:]
                x_d = torch.cat((cls_token, x_d ), dim=1)
                thw1 = thw + 1
            else:
                thw1 = thw
            # Lifting
            q = self.q_d(x_d).reshape(b,thw1,self.num_head,self.head_dim).transpose(1,2)
            k, v = self.kv_a(x_a).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            _d = torch.matmul(self.scaler*q, k.transpose(2,3))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v) # b,thw+1,head,head_dim
            _d = _d.transpose(1,2).reshape(b,thw1,-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).reshape(b,thw,-1)
                if self.use_CLS:
                    _d[:,1:,:] += v
                else:
                    _d += v
            _d = self.project_d(_d)
            
            # Lift Scheme
            if self.use_CLS:
                x_d[:,1:,:] = x_d[:,1:,:] - _d[:,1:,:]
            else:
                x_d = x_d - _d
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            if self.use_CLS:
                cls_token = self.class_token.expand(x_d.size(0), -1).unsqueeze(1)
                x_a = torch.cat((cls_token, x_a ), dim=1)
                thw1 = thw + 1
            else:
                thw1 = thw
            q = self.q_a(x_a).reshape(b,thw1,self.num_head,self.head_dim).transpose(1,2)
            # b,t,head,hw,head_dim
            k, v = self.kv_d(x_d).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            
            _a = torch.matmul(self.scaler*q, k.transpose(2,3))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v) # b,head,thw,head_dim
            _a = _a.transpose(1,2).reshape(b,thw1,-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).reshape(b,thw,-1)
                if self.use_CLS:
                    _a[:,1:,:] += v
                else:
                    _a += v
            _a = self.project_a(_a)
            # Lift Scheme
            if self.use_CLS:
                x_a[:,1:,:] = x_a[:,1:,:] + _a[:,1:,:]
            else:
                x_a = x_a + _a
                
        else:
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            if self.use_CLS:
                cls_token = self.class_token.expand(x_d.size(0), -1).unsqueeze(1)
                x_a = torch.cat((cls_token, x_a ), dim=1)
                thw1 = thw + 1
            else:
                thw1 = thw
            q = self.q_a(x_a).reshape(b,thw1,self.num_head,self.head_dim).transpose(1,2)
            # b,t,head,hw,head_dim
            k, v = self.kv_d(x_d).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            
            _a = torch.matmul(self.scaler*q, k.transpose(2,3))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v) # b,head,thw,head_dim
            _a = _a.transpose(1,2).reshape(b,thw1,-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).reshape(b,thw,-1)
                if self.use_CLS:
                    _a[:,1:,:] += v
                else:
                    _a += v
            _a = self.project_a(_a)
            # Lift Scheme
            if self.use_CLS:
                x_a[:,1:,:] = x_a[:,1:,:] + _a[:,1:,:]
            else:
                x_a = x_a + _a
            
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,hw,head_dim
            # cls_token
            if self.use_CLS:
                cls_token, x_a = x_a[:,0,:], x[:,1:,:]
                x_d = torch.cat((cls_token, x_d ), dim=1)
                thw1 = thw + 1
            else:
                thw1 = thw
            # Lifting
            q = self.q_d(x_d).reshape(b,thw1,self.num_head,self.head_dim).transpose(1,2)
            k, v = self.kv_a(x_a).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            _d = torch.matmul(self.scaler*q, k.transpose(2,3))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v) # b,thw+1,head,head_dim
            _d = _d.transpose(1,2).reshape(b,thw1,-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).reshape(b,thw,-1)
                if self.use_CLS:
                    _d[:,1:,:] += v
                else:
                    _d += v
            _d = self.project_d(_d)
            # Lift Scheme
            if self.use_CLS:
                x_d[:,1:,:] = x_d[:,1:,:] - _d[:,1:,:]
            else:
                x_d = x_d - _d
        
        x_a = self.hook_xa(x_a) # b,thw,c
        
        # t-dimensional self.attn for x_t
        x = self.out_norm1(x_d)
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_head_out, self.head_dim).transpose(1, 3).unbind(dim=2)
        attn = torch.matmul(self.scaler * q, k.transpose(2, 3))
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.project(x)
        x_d = x_d + self.stochastic_depth(x)
        
        # further mlp        
        x_d = self.out_norm2(x_d)
        x_d = self.stochastic_depth(self.mlp(x_d)) # b,thw,c
        
        x_d = self.hook_xd(x_d)

        #    
        if self.use_CLS:
            x_d = x_d[:,0,:]
        else:
            # Prior Code
            x_d = x_d.transpose(1,2)
            x_d = self.project_compress(x_d)
            x_d = x_d.squeeze(-1)
            # DEBUG
            # x_d = x_d.transpose(1,2) # b,c,thw
            # x_d = torch.nn.functional.adaptive_avg_pool1d(x_d,1)
            # x_d = x_d.squeeze(-1)
            
        
        # Relax constrain boundaries to restrain the mean of x_a channel to be the same, and restrain the mean of x_d channel to be zero
        x_a = x_a.view(b,self.T//2,h,w,c).permute(0,4,1,2,3) # b,c,t,h,w
        b,thw,c = x_skip.shape
        x_skip = x_skip.view(b,self.T,self.H,self.W,c).permute(0,4,1,2,3) # b,c,t,h,w
        
        # so far only SSL network will use lowFreq
        if self.relaxed_constrain:
            # downsampling H and W to get more flexible result
            x_a = nn.functional.adaptive_avg_pool3d(x_a,output_size=(self.T//2,h,w))
            x_skip = nn.functional.adaptive_avg_pool3d(x_skip,output_size=(self.T//2,h,w))
            pool_a = torch.mean(x_skip-x_a,dim=(1,2)) # NOTE: calculate the mean along Temporal dimension
        else:
            x_skip = nn.functional.adaptive_avg_pool3d(x_skip,output_size=(self.T//2,h,w)) # b,c,t,h,w
            pool_a = x_skip-x_a
            
        return pool_a, x_d

class WaveSF_trans_CLS_Self(WaveSF_trans_CLS):
    def __init__(self, input_size: List[int], embed_dim: int, num_head: int, split_stride: int = 2, dropout: float = 0.2, relaxed_constrain: bool = True, change_x: bool = False, SSL: bool = False, prompt: bool = False, WaveSF_preUpdate: bool = False, AdaptSplit: bool = False, lowFreq: bool = False, WaveNorm: bool = True, residual: bool = True, use_CLS: bool = False, output_dim:  Optional[int] = None):
        super().__init__(input_size, embed_dim, num_head, split_stride, dropout, relaxed_constrain, change_x, SSL, prompt, WaveSF_preUpdate, AdaptSplit, lowFreq, WaveNorm, residual, use_CLS, output_dim)
    
    def forward(self, x_in:torch.Tensor, input_CLS:bool = True, add_return:bool=False):
        # input shape: B,THW+1,C
        # ----- reshape and ignore cls token in waveLift------
        b,cthw,c = x_in.shape
        if input_CLS:
            cls,x = x_in[:,0:1,:],x_in[:,1:,:] # filter out cls_embed
        else:
            x = x_in
        # x = self.norm_x(x)
        x_skip = x
        # pos_embedding：FIXME      ，       pos  
        pos = False
        if pos:
            hw_size, embed_size = self.spatial_pos.shape
            pos_embedding = torch.repeat_interleave(self.temporal_pos, hw_size, dim=0)
            pos_embedding.add_(self.spatial_pos.unsqueeze(0).expand(self.T, -1, -1).reshape(-1, embed_size))
            x.add_(pos_embedding)
        # reshape
        x = x.view(b,self.T,self.H*self.W,self.embed_dim)
        x = x.permute(0,3,1,2) # B,T,H*W,C -> B,C,T,H*W
        
        # ----- split ------
        if not self.use_adaptSplit:
            x_a:torch.Tensor = x[:,:,0::2,:] # Approxiamation
            x_d:torch.Tensor = x[:,:,1::2,:] # Details
        else:
            x_a, x_d = self.adapt_split(x)
            
        # unfold for nhead splitting of spatial dimension for local cross-attention
        x_a = x_a.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        x_d = x_d.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        
        # norm
        if self.WaveNorm:
            x_a = self.xa_norm(x_a)
            x_d = self.xd_norm(x_d)
        
        b,thw,c = x_d.shape
        t, h, w = self.T//2, self.H, self.W
        if not self.WaveSF_preUpdate:
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,hw,head_dim
            # cls_token
            if self.use_CLS:
                cls_token, x_a = x_a[:,0,:], x[:,1:,:]
                x_d = torch.cat((cls_token, x_d ), dim=1)
                thw1 = thw + 1
            else:
                thw1 = thw
            # Lifting
            q = self.q_d(x_a).reshape(b,thw1,self.num_head,self.head_dim).transpose(1,2)
            k, v = self.kv_a(x_a).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            _d = torch.matmul(self.scaler*q, k.transpose(2,3))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v) # b,thw+1,head,head_dim
            _d = _d.transpose(1,2).reshape(b,thw1,-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).reshape(b,thw,-1)
                if self.use_CLS:
                    _d[:,1:,:] += v
                else:
                    _d += v
            _d = self.project_d(_d)
            
            # Lift Scheme
            if self.use_CLS:
                x_d[:,1:,:] = x_d[:,1:,:] - _d[:,1:,:]
            else:
                x_d = x_d - _d
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            if self.use_CLS:
                cls_token = self.class_token.expand(x_d.size(0), -1).unsqueeze(1)
                x_a = torch.cat((cls_token, x_a ), dim=1)
                thw1 = thw + 1
            else:
                thw1 = thw
            q = self.q_a(x_d).reshape(b,thw1,self.num_head,self.head_dim).transpose(1,2)
            # b,t,head,hw,head_dim
            k, v = self.kv_d(x_d).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            
            _a = torch.matmul(self.scaler*q, k.transpose(2,3))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v) # b,head,thw,head_dim
            _a = _a.transpose(1,2).reshape(b,thw1,-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).reshape(b,thw,-1)
                if self.use_CLS:
                    _a[:,1:,:] += v
                else:
                    _a += v
            _a = self.project_a(_a)
            # Lift Scheme
            if self.use_CLS:
                x_a[:,1:,:] = x_a[:,1:,:] + _a[:,1:,:]
            else:
                x_a = x_a + _a
                
        else:
            # ----- Updater ----- 
            # for approxiamation: x_a + MLP[Cross(x_a,x_d)*x_d]
            if self.use_CLS:
                cls_token = self.class_token.expand(x_d.size(0), -1).unsqueeze(1)
                x_a = torch.cat((cls_token, x_a ), dim=1)
                thw1 = thw + 1
            else:
                thw1 = thw
            q = self.q_a(x_d).reshape(b,thw1,self.num_head,self.head_dim).transpose(1,2)
            # b,t,head,hw,head_dim
            k, v = self.kv_d(x_d).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            
            _a = torch.matmul(self.scaler*q, k.transpose(2,3))
            _a = _a.softmax(dim=-1)
            _a = torch.matmul(_a, v) # b,head,thw,head_dim
            _a = _a.transpose(1,2).reshape(b,thw1,-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).reshape(b,thw,-1)
                if self.use_CLS:
                    _a[:,1:,:] += v
                else:
                    _a += v
            _a = self.project_a(_a)
            # Lift Scheme
            if self.use_CLS:
                x_a[:,1:,:] = x_a[:,1:,:] + _a[:,1:,:]
            else:
                x_a = x_a + _a
            
            # ----- Predictor ----- 
            # for details: x_d - MLP[Cross(x_d,x_a)*x_a]
            # b,t,head,hw,head_dim
            # cls_token
            if self.use_CLS:
                cls_token, x_a = x_a[:,0,:], x[:,1:,:]
                x_d = torch.cat((cls_token, x_d ), dim=1)
                thw1 = thw + 1
            else:
                thw1 = thw
            # Lifting
            q = self.q_d(x_a).reshape(b,thw1,self.num_head,self.head_dim).transpose(1,2)
            k, v = self.kv_a(x_a).reshape(b,thw,2,self.num_head,self.head_dim).transpose(1,3).unbind(2)
            _d = torch.matmul(self.scaler*q, k.transpose(2,3))
            _d = _d.softmax(dim=-1)
            _d = torch.matmul(_d, v) # b,thw+1,head,head_dim
            _d = _d.transpose(1,2).reshape(b,thw1,-1)
            if self.residual:
                v:torch.tensor = v.transpose(1,2).reshape(b,thw,-1)
                if self.use_CLS:
                    _d[:,1:,:] += v
                else:
                    _d += v
            _d = self.project_d(_d)
            # Lift Scheme
            if self.use_CLS:
                x_d[:,1:,:] = x_d[:,1:,:] - _d[:,1:,:]
            else:
                x_d = x_d - _d
        
        x_a = self.hook_xa(x_a) # b,thw,c
        
        # t-dimensional self.attn for x_t
        x = self.out_norm1(x_d)
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_head_out, self.head_dim).transpose(1, 3).unbind(dim=2)
        attn = torch.matmul(self.scaler * q, k.transpose(2, 3))
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.project(x)
        x_d = x_d + self.stochastic_depth(x)
        
        # further mlp        
        x_d = self.out_norm2(x_d)
        x_d = self.stochastic_depth(self.mlp(x_d)) # b,thw,c
        
        x_d = self.hook_xd(x_d)

        #    
        if self.use_CLS:
            x_d = x_d[:,0,:]
        else:
            # Prior Code
            x_d = x_d.transpose(1,2)
            x_d = self.project_compress(x_d)
            x_d = x_d.squeeze(-1)
            # DEBUG
            # x_d = x_d.transpose(1,2) # b,c,thw
            # x_d = torch.nn.functional.adaptive_avg_pool1d(x_d,1)
            # x_d = x_d.squeeze(-1)
            
        
        # Relax constrain boundaries to restrain the mean of x_a channel to be the same, and restrain the mean of x_d channel to be zero
        x_a = x_a.view(b,self.T//2,h,w,c).permute(0,4,1,2,3) # b,c,t,h,w
        b,thw,c = x_skip.shape
        x_skip = x_skip.view(b,self.T,self.H,self.W,c).permute(0,4,1,2,3) # b,c,t,h,w
        
        # so far only SSL network will use lowFreq
        if self.relaxed_constrain:
            # downsampling H and W to get more flexible result
            x_a = nn.functional.adaptive_avg_pool3d(x_a,output_size=(self.T//2,h//2,w//2))
            x_skip = nn.functional.adaptive_avg_pool3d(x_skip,output_size=(self.T//2,h//2,w//2))
            pool_a = torch.mean(x_skip-x_a,dim=(1,2)) # NOTE: calculate the mean along Temporal and Embedding dimension for H,W Spatial Dimension
        else:
            x_skip = nn.functional.adaptive_avg_pool3d(x_skip,output_size=(self.T//2,h,w)) # b,c,t,h,w
            pool_a = x_skip-x_a
            
        return pool_a, x_d

    
class WaveSF_trans_CLS_onlySplit(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 prompt:bool = False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 WaveNorm:bool = True,
                 residual:bool = True,
                 use_CLS:bool = False,
                 ):
        super().__init__()
        self.use_CLS = False
        self.prompt = prompt
        self.relaxed_constrain = relaxed_constrain
        self.change_x = False
        self.SSL = SSL
        self.WaveSF_preUpdate = WaveSF_preUpdate
        self.use_lowFreq = lowFreq
    
        self.embed_dim = embed_dim
        self.T,self.H,self.W = input_size
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scaler = 1.0 / math.sqrt(self.head_dim)
            
        self.use_adaptSplit = AdaptSplit
        if AdaptSplit:
            from .AdaptSplit import adaptIndexSplit_dotsim,adaptIndexSplit_dotsim_weightedScore
            self.adapt_split = adaptIndexSplit_dotsim(input_size,down_size=7,embed_dim=embed_dim)
        self.WaveNorm = WaveNorm
        # residual connection
        self.residual = residual
        
        # compression dim
        # FIXME linear bias True or False
        # or use max_pool for the timeHW?
        self.project_compress = nn.Linear(self.T//2*self.H*self.W,1)

    def forward(self, x_in:torch.Tensor, input_CLS:bool = True, add_return:bool=False):
        # input shape: B,THW+1,C
        # ----- reshape and ignore cls token in waveLift------
        b,cthw,c = x_in.shape
        if input_CLS:
            cls,x = x_in[:,0:1,:],x_in[:,1:,:] # filter out cls_embed
        else:
            x = x_in
        x_skip = x
        
        # reshape
        x = x.view(b,self.T,self.H*self.W,self.embed_dim)
        x = x.permute(0,3,1,2) # B,T,H*W,C -> B,C,T,H*W
        
        # ----- split ------
        if not self.use_adaptSplit:
            x_a:torch.Tensor = x[:,:,0::2,:] # Approxiamation
            x_d:torch.Tensor = x[:,:,1::2,:] # Details
        else:
            x_a, x_d = self.adapt_split(x)
            
        # unfold for nhead splitting of spatial dimension for local cross-attention
        x_a = x_a.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        x_d = x_d.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        
        # Prior Code
        x_d = x_d.transpose(1,2) # B,T//2*H*W,C -> B,T
        x_d = self.project_compress(x_d)
        x_d = x_d.squeeze(-1)
        # DEBUG
        
        
        t, h, w = self.T//2, self.H, self.W
        # Relax constrain boundaries to restrain the mean of x_a channel to be the same, and restrain the mean of x_d channel to be zero
        x_a = x_a.view(b,t,h,w,c).permute(0,4,1,2,3) # b,c,t,h,w
        b,thw,c = x_skip.shape
        x_skip = x_skip.view(b,self.T,h,w,c).permute(0,4,1,2,3) # b,c,t,h,w
        x_skip = nn.functional.adaptive_avg_pool3d(x_skip,output_size=(self.T//2,h,w)) # b,c,t,h,w
        # so far only SSL network will use lowFreq
        if self.relaxed_constrain:
            channel_mean = False
            if channel_mean:
                pool_a = torch.mean(x_skip-x_a,dim=1)
            else:
                # h,w mean
                x_a = x_a.reshape(b,-1,h,w)
                x_skip = x_skip.reshape(b,-1,h,w)
                pool_a = torch.mean(x_skip-x_a,dim=1)
        else:
            pool_a = x_skip-x_a
            
        return pool_a, x_d
    
class WaveSF_trans_CLS_onlySplit_v2(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int,
                 num_head:int,
                 split_stride:int=2,
                 dropout:float=0.2,
                 relaxed_constrain:bool=True,
                 change_x:bool=False,
                 SSL:bool=False,
                 prompt:bool = False,
                 WaveSF_preUpdate:bool = False,
                 AdaptSplit:bool = False,
                 lowFreq:bool = False,
                 WaveNorm:bool = True,
                 residual:bool = True,
                 use_CLS:bool = False,
                 ):
        super().__init__()
        self.use_CLS = False
        self.prompt = prompt
        self.relaxed_constrain = relaxed_constrain
        self.change_x = False
        self.SSL = SSL
        self.WaveSF_preUpdate = WaveSF_preUpdate
        self.use_lowFreq = lowFreq
    
        self.embed_dim = embed_dim
        self.T,self.H,self.W = input_size
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        self.scaler = 1.0 / math.sqrt(self.head_dim)
            
        self.use_adaptSplit = AdaptSplit
        if AdaptSplit:
            from .AdaptSplit import adaptIndexSplit_dotsim, adaptIndexSplit_dotsim_v1,adaptSplit_dotsim_v2
            self.adapt_split = adaptIndexSplit_dotsim_v1(input_size,down_size=7,embed_dim=embed_dim)
        self.WaveNorm = WaveNorm
        # residual connection
        self.residual = residual
        
        # compression dim
        # FIXME linear bias True or False
        # or use max_pool for the timeHW?
        self.project_compress = nn.Linear(self.T//2*self.H*self.W,1)

    def forward(self, x_in:torch.Tensor, input_CLS:bool = True, add_return:bool=False):
        # input shape: B,THW+1,C
        # ----- reshape and ignore cls token in waveLift------
        b,cthw,c = x_in.shape
        if input_CLS:
            cls,x = x_in[:,0:1,:],x_in[:,1:,:] # filter out cls_embed
        else:
            x = x_in
        x_skip = x
        
        # reshape
        x = x.view(b,self.T,self.H*self.W,self.embed_dim)
        x = x.permute(0,3,1,2) # B,T,H*W,C -> B,C,T,H*W
        
        # ----- split ------
        if not self.use_adaptSplit:
            x_a:torch.Tensor = x[:,:,0::2,:] # Approxiamation
            x_d:torch.Tensor = x[:,:,1::2,:] # Details
        else:
            x_a, x_d = self.adapt_split(x)
            
        # unfold for nhead splitting of spatial dimension for local cross-attention
        x_a = x_a.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        x_d = x_d.permute(0,2,3,1).reshape(b,self.T//2*self.H*self.W,self.embed_dim) # B,T//2*H*W,C
        
        x_d = x_d.transpose(1,2) # B,T//2*H*W,C -> B,T
        x_d = self.project_compress(x_d)
        x_d = x_d.squeeze(-1)
        
        t, h, w = self.T//2, self.H, self.W
        # Relax constrain boundaries to restrain the mean of x_a channel to be the same, and restrain the mean of x_d channel to be zero
        x_a = x_a.view(b,t,h,w,c).permute(0,4,1,2,3) # b,c,t,h,w
        b,thw,c = x_skip.shape
        x_skip = x_skip.view(b,self.T,h,w,c).permute(0,4,1,2,3) # b,c,t,h,w
        x_skip = nn.functional.adaptive_avg_pool3d(x_skip,output_size=(self.T//2,h,w)) # b,c,t,h,w
        # so far only SSL network will use lowFreq
        if self.relaxed_constrain:
            channel_mean = False
            if channel_mean:
                pool_a = torch.mean(x_skip-x_a,dim=1)
            else:
                # h,w mean
                x_a = x_a.reshape(b,-1,h,w)
                x_skip = x_skip.reshape(b,-1,h,w)
                pool_a = torch.mean(x_skip-x_a,dim=1)
        else:
            pool_a = x_skip-x_a
            
        return pool_a, x_d