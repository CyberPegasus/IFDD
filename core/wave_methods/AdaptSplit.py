import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class adaptSplit_dotsim_weight(nn.Module):
    # Dot Mutiplication-based Similarity -> Learnable Weighting
    def __init__(self,
                 input_size:List[int],
                 down_size:int=7,
                 embed_dim:int=512,
                 position_encode:bool=False,
                 T_project:bool=False
                 ) -> None:
        super().__init__()
        self.down_size = down_size # 7 for 56,28,14 feature pyramid
        self.T,self.H,self.W = input_size
        self.embed_dim = embed_dim
        self.top_k = self.T//2
        
        self.dim_project = nn.Sequential(
            nn.Linear(self.down_size*self.down_size*embed_dim,embed_dim),
        )
        self.scaler = 1.0 / math.sqrt(self.embed_dim)
        
        self.position = position_encode
        self.temporal_pos: Optional[nn.Parameter] = None
        if position_encode:
            self.temporal_pos = nn.Parameter(torch.zeros(self.T, self.embed_dim))
            self.dim_project2 = nn.Linear(embed_dim,embed_dim)
            
        self.use_T_project = T_project
        if T_project:
            self.T_project = nn.Sequential(
            nn.Linear(self.T,1),
            nn.GELU(),
        )
        
    def forward(self, x_in:torch.Tensor):
        # input shape: B,C,T,H*W
        b,c,t,hw = x_in.shape
        xi:torch.Tensor = x_in.view(b,self.embed_dim,self.T,self.H,self.W).permute(0,2,1,3,4).contiguous() # b,t,c,h,w
        xi = xi.view(b,self.T*self.embed_dim,self.H,self.W) # b,t*c,h,w
        # spatial downsample
        xi = F.adaptive_avg_pool2d(xi,output_size=(self.down_size,self.down_size)) # downsample HW for 56,28,14 to 7*7
        xi = xi.view(b,self.T,self.embed_dim,self.down_size,self.down_size).permute(0,1,3,4,2) # b,t,h,w,c
        # project
        xi = xi.flatten(start_dim=2,end_dim=-1)
        xi = self.dim_project(xi) # b,t,d
        if self.position:
            pos_embedding = self.temporal_pos.unsqueeze(0)
            xi.add_(pos_embedding)
            xi = self.dim_project2(xi)
        xi = torch.matmul(self.scaler*xi, xi.transpose(1,2))
        
        if self.use_T_project:
            xi = self.T_project(xi).squeeze(2) # b,t,t -> b,t
        else:
            xi = torch.sum(xi,dim=-1,keepdim=False) # (b,t), output scores imply the correlation
        xi = F.softmax(xi,dim=-1)
        xi = xi[...,None,None] # b,t -> b,t,1,1
        x_in = x_in.transpose(1,2) # b,c,t,hw -> b,t,c,hw
        x_in = x_in*xi
        x_in = x_in.transpose(1,2)
        x_a:torch.Tensor = x_in[:,:,0::2,:] # Approxiamation
        x_d:torch.Tensor = x_in[:,:,1::2,:] # Details
        
        return x_a,x_d
    
class adaptSplit_dotsim(nn.Module):
    # Dot Mutiplication-based Similarity -> Unlearnable Top-K Selection 
    def __init__(self,
                 input_size:List[int],
                 down_size:int=7,
                 embed_dim:int=512,
                 prior_order:bool=True
                 ) -> None:
        super().__init__()
        self.down_size = down_size # 7 for 56,28,14 feature pyramid
        self.T,self.H,self.W = input_size
        self.embed_dim = embed_dim
        self.top_k = self.T//2
        self.prior_order = prior_order
        self.scaler = 1.0 / self.embed_dim
        
        
    def forward(self, x_in:torch.Tensor):
        # input shape: B,C,T,H*W
        b,c,t,hw = x_in.shape
        xi:torch.Tensor = x_in.view(b,self.embed_dim,self.T,self.H,self.W).permute(0,2,1,3,4).contiguous() # b,t,c,h,w
        xi = xi.view(b,self.T*self.embed_dim,self.H,self.W) # b,t*c,h,w
        # spatial downsample
        xi = F.adaptive_avg_pool2d(xi,output_size=(self.down_size,self.down_size)) # downsample HW for 56,28,14 to 7*7
        xi = xi.view(b,self.T,self.embed_dim,self.down_size,self.down_size).permute(0,1,3,4,2) # b,t,h,w,c
        # project
        xi = xi.flatten(start_dim=2,end_dim=-1)
        xi = torch.matmul(self.scaler*xi, xi.transpose(1,2))
        xi = torch.mean(xi,dim=-1,keepdim=False) # (b,t), output scores imply the correlation
        # select top-k and the other, k = T//2
        if self.prior_order:
            xi0 = torch.Tensor([1.0,0.0]).to(xi.device).repeat(b,t//2)
            xi.add_(xi0)
        _,approx_index = torch.topk(xi,k=self.top_k,dim=-1,largest=True,sorted=False) # b,k
        _,detail_index = torch.topk(xi,k=self.top_k,dim=-1,largest=False,sorted=False)
        # NOTE: when sorted=False, the returned time index are not ordered, so need additional sort
        approx_index,_ = torch.sort(approx_index,dim=-1)
        detail_index,_ = torch.sort(detail_index,dim=-1)
        
        x_in = x_in.transpose(1,2) # B,T,C,H*W
        x_a:torch.Tensor = torch.gather(x_in, 1, approx_index[..., None, None].expand(-1,-1,c,hw)) # index become b,t,1,1 while t < T, then gather at dim=2
        x_a = x_a.transpose(1,2) # Approxiamation: B,C,T,H*W
        x_d:torch.Tensor = torch.gather(x_in, 1, detail_index[..., None, None].expand(-1,-1,c,hw))
        x_d = x_d.transpose(1,2) # Details: B,C,T,H*W
        
        return x_a,x_d


class adaptIndexSplit_dotsim(nn.Module):
    # Dot Mutiplication-based Similarity -> Learnable Index interpolation-based Selection
    # NOTE: Current best method with self.receptive_filed = 2
    def __init__(self,
                 input_size:List[int],
                 down_size:int=7,
                 embed_dim:int=512,
                 # receptive_filed:int=2,
                 ) -> None:
        super().__init__()
        self.down_size = down_size # 7 for 56,28,14 feature pyramid
        self.T,self.H,self.W = input_size
        self.embed_dim = embed_dim
        self.top_k = self.T//2
        self.scaler = 1.0 / self.embed_dim
        
        self.dim_project = nn.Sequential(
            nn.Linear(self.down_size*self.down_size*embed_dim,embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim//8),
            # DEBUG: need a layernorm
        )
        self.time_project = nn.Sequential(
            nn.Linear(self.T,self.T),
            nn.GELU(),
            nn.Linear(self.T, 1),
        )
        self.offset_project_approx = nn.Sequential(
            nn.Linear(self.T**2, self.T//2, bias=False),
            nn.Tanh(), # scale to -1 and 1
        )
        self.offset_project_detail = nn.Sequential(
            nn.Linear(self.T**2, self.T//2, bias=False),
            nn.Tanh(), # scale to -1 and 1
        )
        self.receptive_filed = 2 # 2, or 1, or self.T//2 or self.T//4
        # scale 2x up because 2x downsampling for time dimension
        self.hook_attn = nn.Identity()
        self.hook_approx = nn.Identity()
        self.hook_detail = nn.Identity()
        index_even = torch.arange(start=0,end=self.T,step=2)
        index_odd = torch.arange(start=1,end=self.T+1,step=2)
        self.register_buffer("index_even",index_even)
        self.register_buffer("index_odd",index_odd)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)        
    
    def normalize_grid(self,index_list):
        n = index_list.shape[-1]*2
        return 2.0 * index_list / max(n - 1, 1) - 1.0
    
    # @staticmethod
    # def index_sample(x_in:torch.Tensor, index:torch.Tensor):
    #     pass

    def forward(self, x_in:torch.Tensor):
        # input shape: B,C,T,H*W
        b,c,t,hw = x_in.shape
        xi:torch.Tensor = x_in.view(b,self.embed_dim,self.T,self.H,self.W).permute(0,2,1,3,4)# b,t,c,h,w
        xi = xi.reshape(b,self.T*self.embed_dim,self.H,self.W) # b,t*c,h,w
        
        # spatial downsample
        xi = F.adaptive_avg_pool2d(xi,output_size=(self.down_size,self.down_size)) # downsample HW for 56,28,14 to 7*7
        xi = xi.view(b,self.T,self.embed_dim,self.down_size,self.down_size).permute(0,1,3,4,2) # b,t,h,w,c
        
        # project
        xi = xi.flatten(start_dim=2,end_dim=-1)
        xi = self.dim_project(xi) # b,t,c//8
        
        # correlation calculation
        xi = torch.matmul(self.scaler*xi, xi.transpose(1,2)) # (b,t,t), output scores imply the correlation
        xi = torch.softmax(xi,dim=-1) # (b,t,t), output scores imply the correlation
        xi = self.hook_attn(xi) # for hook register
        # offset obtain
        # xi = self.time_project(xi) # b,t,t -> b,t,1
        # xi = xi.squeeze(2) # b,t,1 -> b,t
        xi = xi.flatten(start_dim=1)
        offset_approx = self.offset_project_approx(xi)*self.receptive_filed # b,t-> b,t/2, output of offset_project belong to (-1,1)
        offset_detail = self.offset_project_detail(xi)*self.receptive_filed
        
        # offset index
        index_even = self.index_even.repeat(b,1)
        index_odd = self.index_odd.repeat(b,1)
        index_approx = index_even + offset_approx
        index_detail = index_odd + offset_detail
        # for hook register
        index_approx = self.hook_approx(index_approx)
        index_detail = self.hook_detail(index_detail)
        index_approx = self.normalize_grid(index_approx)
        index_detail = self.normalize_grid(index_detail)
        
        # rearrange for grid_sample, input N,C,H,W, grid N,H1,W1,2, output N,C,H1,W1
        index_approx = index_approx[...,None,None] # b,n -> b,n,1,1
        index_detail = index_detail[...,None,None] # b,n -> b,n,1,1
        # since coordinate for grid is x,y for w,h
        # our 1d dimension is for y, then padding zero for x
        index_approx = F.pad(index_approx,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        index_detail = F.pad(index_detail,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        
        x_in = x_in.permute(0,3,1,2).reshape(b,hw*c,t).unsqueeze(-1) # B,C,T,H*W -> B,H*W,C,T -> B,H*W*C,T,1
        _mode = 'border' # 'zeros'
        x_a = F.grid_sample(x_in,index_approx,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,H*W*C,T,1
        x_d = F.grid_sample(x_in,index_detail,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,H*W*C,T,1
        x_a = x_a.view(b,hw,c,t//2).permute(0,2,3,1).contiguous() # B,C,T,H*W
        x_d = x_d.view(b,hw,c,t//2).permute(0,2,3,1).contiguous() # B,C,T,H*W
        
        return x_a,x_d

class adaptIndexSplit_dotsim(nn.Module):
    # Dot Mutiplication-based Similarity -> Learnable Index interpolation-based Selection
    # NOTE: Current best method with self.receptive_filed = 2
    def __init__(self,
                 input_size:List[int],
                 down_size:int=7,
                 embed_dim:int=512,
                 # receptive_filed:int=2,
                 ) -> None:
        super().__init__()
        self.down_size = down_size # 7 for 56,28,14 feature pyramid
        self.T,self.H,self.W = input_size
        self.embed_dim = embed_dim
        self.top_k = self.T//2
        self.scaler = 1.0 / self.embed_dim
        
        self.dim_project = nn.Sequential(
            nn.Linear(self.down_size*self.down_size*embed_dim,embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim//8),
            # DEBUG: need a layernorm
        )
        self.time_project = nn.Sequential(
            nn.Linear(self.T,self.T),
            nn.GELU(),
            nn.Linear(self.T, 1),
        )
        self.offset_project_approx = nn.Sequential(
            nn.Linear(self.T**2, self.T//2, bias=False),
            nn.Tanh(), # scale to -1 and 1
        )
        self.offset_project_detail = nn.Sequential(
            nn.Linear(self.T**2, self.T//2, bias=False),
            nn.Tanh(), # scale to -1 and 1
        )
        self.receptive_filed = 2 # 2, or 1, or self.T//2 or self.T//4
        # scale 2x up because 2x downsampling for time dimension
        self.hook_attn = nn.Identity()
        self.hook_approx = nn.Identity()
        self.hook_detail = nn.Identity()
        index_even = torch.arange(start=0,end=self.T,step=2)
        index_odd = torch.arange(start=1,end=self.T,step=2)
        self.register_buffer("index_even",index_even)
        self.register_buffer("index_odd",index_odd)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)        
    
    def normalize_grid(self,index_list):
        n = index_list.shape[-1]*2
        return 2.0 * index_list / max(n - 1, 1) - 1.0
    
    # @staticmethod
    # def index_sample(x_in:torch.Tensor, index:torch.Tensor):
    #     pass

    def forward(self, x_in:torch.Tensor):
        # input shape: B,C,T,H*W
        b,c,t,hw = x_in.shape
        xi:torch.Tensor = x_in.view(b,self.embed_dim,self.T,self.H,self.W).permute(0,2,1,3,4)# b,t,c,h,w
        xi = xi.reshape(b,self.T*self.embed_dim,self.H,self.W) # b,t*c,h,w
        
        # spatial downsample
        xi = F.adaptive_avg_pool2d(xi,output_size=(self.down_size,self.down_size)) # downsample HW for 56,28,14 to 7*7
        xi = xi.view(b,self.T,self.embed_dim,self.down_size,self.down_size).permute(0,1,3,4,2) # b,t,h,w,c
        
        # project
        xi = xi.flatten(start_dim=2,end_dim=-1)
        xi = self.dim_project(xi) # b,t,c//8
        
        # correlation calculation
        xi = torch.matmul(self.scaler*xi, xi.transpose(1,2)) # (b,t,t), output scores imply the correlation
        xi = torch.softmax(xi,dim=-1) # (b,t,t), output scores imply the correlation
        xi = self.hook_attn(xi) # for hook register
        # offset obtain
        # xi = self.time_project(xi) # b,t,t -> b,t,1
        # xi = xi.squeeze(2) # b,t,1 -> b,t
        xi = xi.flatten(start_dim=1)
        offset_approx = self.offset_project_approx(xi)*self.receptive_filed # b,t-> b,t/2, output of offset_project belong to (-1,1)
        offset_detail = self.offset_project_detail(xi)*self.receptive_filed
        
        # offset index # DEBUG
        index_even = self.index_even.repeat(b,1)
        index_odd = self.index_odd.repeat(b,1)
        index_approx = index_even + offset_approx
        index_detail = index_odd + offset_detail
        # for hook register
        index_approx = self.hook_approx(index_approx)
        index_detail = self.hook_detail(index_detail)
        index_approx = self.normalize_grid(index_approx)
        index_detail = self.normalize_grid(index_detail)
        
        # rearrange for grid_sample, input N,C,H,W, grid N,H1,W1,2, output N,C,H1,W1
        index_approx = index_approx[...,None,None] # b,n -> b,n,1,1
        index_detail = index_detail[...,None,None] # b,n -> b,n,1,1
        # since coordinate for grid is x,y for w,h
        # our 1d dimension is for y, then padding zero for x
        index_approx = F.pad(index_approx,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        index_detail = F.pad(index_detail,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        
        x_in = x_in.permute(0,3,1,2).reshape(b,hw*c,t).unsqueeze(-1) # B,C,T,H*W -> B,H*W,C,T -> B,H*W*C,T,1
        _mode = 'border' # 'zeros'
        x_a = F.grid_sample(x_in,index_approx,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,H*W*C,T,1
        x_d = F.grid_sample(x_in,index_detail,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,H*W*C,T,1
        x_a = x_a.view(b,hw,c,t//2).permute(0,2,3,1).contiguous() # B,C,T,H*W
        x_d = x_d.view(b,hw,c,t//2).permute(0,2,3,1).contiguous() # B,C,T,H*W
        
        return x_a,x_d


class adaptIndexSplit_dotsim_1D(nn.Module):
    # Dot Mutiplication-based Similarity -> Learnable Index interpolation-based Selection
    # NOTE: Current best method with self.receptive_filed = 2
    def __init__(self,
                 input_size:List[int],
                 embed_dim:int=512,
                 ) -> None:
        super().__init__()
        self.T = input_size[0]*input_size[1]*input_size[2]
        self.embed_dim = embed_dim
        self.top_k = self.T//2
        self.scaler = 1.0 / self.embed_dim
        
        self.dim_project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//8),
        )
        self.time_project = nn.Sequential(
            nn.Linear(self.T*self.T,self.T),
        )
        self.offset_project_approx = nn.Sequential(
            nn.Linear(self.T, self.T//2, bias=False),
            nn.Tanh(), # scale to -1 and 1
        )
        self.offset_project_detail = nn.Sequential(
            nn.Linear(self.T, self.T//2, bias=False),
            nn.Tanh(), # scale to -1 and 1
        )
        self.receptive_filed = 2 # 2, or 1, or self.T//2 or self.T//4
        # scale 2x up because 2x downsampling for time dimension
        
        index_even = torch.arange(start=0,end=self.T,step=2)
        index_odd = torch.arange(start=1,end=self.T+1,step=2)
        self.register_buffer("index_even",index_even)
        self.register_buffer("index_odd",index_odd)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)        
    
    def normalize_grid(self,index_list):
        n = index_list.shape[-1]*2
        return 2.0 * index_list / max(n - 1, 1) - 1.0

    def forward(self, x_in:torch.Tensor):
        # input shape: B,C,T
        b,t,c = x_in.shape
        xi:torch.Tensor = x_in.clone()
        
        # project
        xi = self.dim_project(xi)
        # correlation calculation
        xi = torch.matmul(self.scaler*xi, xi.transpose(1,2))
        xi = torch.softmax(xi,dim=-1) # (b,t,t), output scores imply the correlation
        
        # offset obtain
        xi = xi.flatten(start_dim=1)
        xi = self.time_project(xi) # b,t*t -> b,t
        offset_approx = self.offset_project_approx(xi)*self.receptive_filed # b,t-> b,t/2, output of offset_project belong to (-1,1)
        offset_detail = self.offset_project_detail(xi)*self.receptive_filed
        
        # offset index
        index_even = self.index_even.repeat(b,1)
        index_odd = self.index_odd.repeat(b,1)
        index_approx = index_even + offset_approx
        index_detail = index_odd + offset_detail
        index_approx = self.normalize_grid(index_approx)
        index_detail = self.normalize_grid(index_detail)
        
        # rearrange for grid_sample, input N,C,H,W, grid N,H1,W1,2, output N,C,H1,W1
        index_approx = index_approx[...,None,None] # b,n -> b,n,1,1
        index_detail = index_detail[...,None,None] # b,n -> b,n,1,1
        # since coordinate for grid is x,y for w,h
        # our 1d dimension is for y, then padding zero for x
        index_approx = F.pad(index_approx,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        index_detail = F.pad(index_detail,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        
        x_in = x_in.unsqueeze(-1) # B,C,T -> B,C,T,1
        _mode = 'border' # 'zeros'
        x_a = F.grid_sample(x_in,index_approx,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,C,T,1 -> B,C,T
        x_d = F.grid_sample(x_in,index_detail,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,C,T,1 -> B,C,T
        
        return x_a,x_d

class adaptIndexSplit_dimLearn(nn.Module):
    def __init__(self,
                 input_size:List[int],
                 down_size:int=7,
                 embed_dim:int=512,
                 # receptive_filed:int=2,
                 ) -> None:
        super().__init__()
        self.down_size = down_size # 7 for 56,28,14 feature pyramid
        self.T,self.H,self.W = input_size
        self.embed_dim = embed_dim
        self.top_k = self.T//2
        self.scaler = 1.0 / self.embed_dim
        
        # downsample h,w, and dim -> Conv2D
        self.hwd_pool = nn.AdaptiveAvgPool2d(output_size=(self.down_size,self.down_size))
        _d = embed_dim//8
        self._d = _d
        self.dim_pool = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim,out_channels=_d,kernel_size=1,), 
            nn.GELU(),
        )
        _ksize=3
        _pad = 1
        self.hwd_project = nn.Sequential(
            nn.Conv2d(in_channels=_d,out_channels=_d,kernel_size=_ksize,stride=2,padding=_pad), # 7*7->4*4
            nn.GELU(),
        )
        self.t_dim = int((down_size+2*_pad-_ksize)//2+1)**2*_d
        self.time_project = nn.Sequential(
            nn.Conv1d(self.t_dim,1,kernel_size=3,stride=1,padding=1),
            nn.GELU(),
        )
        self.offset_project_approx = nn.Sequential(
            nn.Linear(self.T, self.T//2, bias=False),
            nn.Tanh(), # scale to -1 and 1
        )
        self.offset_project_detail = nn.Sequential(
            nn.Linear(self.T, self.T//2, bias=False),
            nn.Tanh(), # scale to -1 and 1
        )
        
        self.receptive_filed = 2 # 2, or 1, or self.T//2 or self.T//4
        # scale 2x up because 2x downsampling for time dimension
        index_even = torch.arange(start=0,end=self.T,step=2)
        index_odd = torch.arange(start=1,end=self.T+1,step=2)
        self.register_buffer("index_even",index_even)
        self.register_buffer("index_odd",index_odd)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  
                    
    def normalize_grid(self,index_list):
        n = index_list.shape[-1]*2
        return 2.0 * index_list / max(n - 1, 1) - 1.0

    def forward(self, x_in:torch.Tensor):
        # input shape: B,C,T,H*W
        b,c,t,hw = x_in.shape
        xi:torch.Tensor = x_in.view(b,self.embed_dim,self.T,self.H,self.W).permute(0,2,1,3,4)# b,t,c,H,W
        xi = xi.reshape(b*self.T,self.embed_dim,self.H,self.W) # b*t,c,H,W
        
        xi = self.hwd_pool(xi) # downsample HW for 56,28,14 to h=7, w=7; # b*t,c,H,W -> b*t,c,h,w
        xi = self.dim_pool(xi) # b*t,_d,h,w
        xi = self.hwd_project(xi) # b*t,_d,l,l
        xi = xi.flatten(start_dim=1) # b*t,_d*l*l
        xi = xi.reshape(b,self.T,-1).permute(0,2,1) # b*t,_d*l*l -> b,_d*l*l,t
        xi = self.time_project(xi) # b,1,t
        
        # offset obtain
        xi = xi.squeeze(1) # b,1,t -> b,t
        offset_approx = self.offset_project_approx(xi)*self.receptive_filed # b,t-> b,t/2, output of offset_project belong to (-1,1)
        offset_detail = self.offset_project_detail(xi)*self.receptive_filed
        
        # offset index
        index_even = self.index_even.repeat(b,1)
        index_odd = self.index_odd.repeat(b,1)
        index_approx = index_even + offset_approx
        index_detail = index_odd + offset_detail
        index_approx = self.normalize_grid(index_approx)
        index_detail = self.normalize_grid(index_detail)
        
        # rearrange for grid_sample, input N,C,H,W, grid N,H1,W1,2, output N,C,H1,W1
        index_approx = index_approx[...,None,None] # b,n -> b,n,1,1
        index_detail = index_detail[...,None,None] # b,n -> b,n,1,1
        # since coordinate for grid is x,y for w,h
        # our 1d dimension is for y, then padding zero for x
        index_approx = F.pad(index_approx,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        index_detail = F.pad(index_detail,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        
        x_in = x_in.permute(0,3,1,2).reshape(b,hw*c,t).unsqueeze(-1) # B,C,T,H*W -> B,H*W,C,T -> B,H*W*C,T,1
        _mode = 'border' # 'zeros'
        x_a = F.grid_sample(x_in,index_approx,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,H*W*C,T,1
        x_d = F.grid_sample(x_in,index_detail,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,H*W*C,T,1
        x_a = x_a.view(b,hw,c,t//2).permute(0,2,3,1).contiguous() # B,C,T,H*W
        x_d = x_d.view(b,hw,c,t//2).permute(0,2,3,1).contiguous() # B,C,T,H*W
        
        return x_a,x_d
    
    

class adaptIndexSplit_dotsim_v1(nn.Module):
    # Dot Mutiplication-based Similarity -> Learnable Index interpolation-based Selection
    # NOTE: Current best method with self.receptive_filed = 2
    def __init__(self,
                 input_size:List[int],
                 down_size:int=7,
                 embed_dim:int=512,
                 # receptive_filed:int=2,
                 ) -> None:
        super().__init__()
        self.down_size = down_size # 7 for 56,28,14 feature pyramid
        self.T,self.H,self.W = input_size
        self.embed_dim = embed_dim
        self.top_k = self.T//2
        self.scaler = 1.0 / self.embed_dim
        
        # downsample h,w, and dim -> Conv2D
        self.hwd_pool = nn.AdaptiveAvgPool2d(output_size=(self.down_size,self.down_size))
        _d = embed_dim//8
        self._d = _d
        self.dim_pool = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim,out_channels=_d,kernel_size=1,), # 7*7->4*4
            nn.GELU(),
        )
        _ksize=3
        _pad = 1
        self.hwd_project = nn.Sequential(
            nn.Conv2d(in_channels=_d,out_channels=_d,kernel_size=_ksize,stride=2,padding=_pad), # 7*7->4*4
            nn.GELU(),
        )
        self.dim_compress = nn.Sequential(
            nn.Linear(in_features=self._d*16,out_features=_d),
            nn.LayerNorm(_d)
        )
        self.offset_project_approx = nn.Sequential(
            nn.Linear(self.T**2, self.T//2),
            nn.Tanh(), # scale to -1 and 1
        )
        self.offset_project_detail = nn.Sequential(
            nn.Linear(self.T**2, self.T//2),
            nn.Tanh(), # scale to -1 and 1
        )
        self.receptive_filed = 2 # 2, or 1, or self.T//2 or self.T//4
        # scale 2x up because 2x downsampling for time dimension
        self.hook_attn = nn.Identity()
        self.hook_approx = nn.Identity()
        self.hook_detail = nn.Identity()
        index_even = torch.arange(start=0,end=self.T,step=2)
        index_odd = torch.arange(start=1,end=self.T+1,step=2)
        self.register_buffer("index_even",index_even)
        self.register_buffer("index_odd",index_odd)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)        
    
    def normalize_grid(self,index_list):
        n = index_list.shape[-1]*2
        return 2.0 * index_list / max(n - 1, 1) - 1.0
    
    # @staticmethod
    # def index_sample(x_in:torch.Tensor, index:torch.Tensor):
    #     pass

    def forward(self, x_in:torch.Tensor):
        # input shape: B,C,T,H*W
        b,c,t,hw = x_in.shape
        xi:torch.Tensor = x_in.view(b,self.embed_dim,self.T,self.H,self.W).permute(0,2,1,3,4)# b,t,c,h,w
        xi = xi.reshape(b*self.T,self.embed_dim,self.H,self.W) # b*t,c,H,W
        
        xi = self.hwd_pool(xi) # downsample HW for 56,28,14 to h=7, w=7; # b*t,c,H,W -> b*t,c,h,w
        xi = self.dim_pool(xi) # b*t,_d,h,w
        xi = self.hwd_project(xi) # b*t,_d,l,l
        xi = xi.flatten(start_dim=1) # b*t,_d*l*l
        xi = xi.reshape(b,self.T,-1)# b*t,_d*l*l -> b,t,c
        xi = self.dim_compress(xi)
        
        # correlation calculation
        xi = torch.matmul(self.scaler*xi, xi.transpose(1,2)) # (b,t,t), output scores imply the correlation
        xi = 1 - F.softmax(xi,dim=-1) # min for max
        xi = self.hook_attn(xi) # for hook register
        # offset obtain
        xi = xi.flatten(start_dim=1)
        offset_approx = self.offset_project_approx(xi)*self.receptive_filed # b,t-> b,t/2, output of offset_project belong to (-1,1)
        offset_detail = self.offset_project_detail(xi)*self.receptive_filed
        
        # offset index
        index_even = self.index_even.repeat(b,1)
        index_odd = self.index_odd.repeat(b,1)
        index_approx = index_even + offset_approx
        index_detail = index_odd + offset_detail
        # for hook register
        index_approx = self.hook_approx(index_approx)
        index_detail = self.hook_detail(index_detail)
        index_approx = self.normalize_grid(index_approx)
        index_detail = self.normalize_grid(index_detail)
        
        # rearrange for grid_sample, input N,C,H,W, grid N,H1,W1,2, output N,C,H1,W1
        index_approx = index_approx[...,None,None] # b,n -> b,n,1,1
        index_detail = index_detail[...,None,None] # b,n -> b,n,1,1
        # since coordinate for grid is x,y for w,h
        # our 1d dimension is for y, then padding zero for x
        index_approx = F.pad(index_approx,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        index_detail = F.pad(index_detail,(1,0),value=0.) # b,n,1,1 -> b,n,1,2
        
        x_in = x_in.permute(0,3,1,2).reshape(b,hw*c,t).unsqueeze(-1) # B,C,T,H*W -> B,H*W,C,T -> B,H*W*C,T,1
        _mode = 'border' # 'zeros'
        x_a = F.grid_sample(x_in,index_approx,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,H*W*C,T,1
        x_d = F.grid_sample(x_in,index_detail,mode='bilinear',padding_mode=_mode,align_corners=False).squeeze(3) # B,H*W*C,T,1
        x_a = x_a.view(b,hw,c,t//2).permute(0,2,3,1).contiguous() # B,C,T,H*W
        x_d = x_d.view(b,hw,c,t//2).permute(0,2,3,1).contiguous() # B,C,T,H*W
        
        return x_a,x_d
    
import einops
from einops import rearrange, einsum
class adaptIndexSplit_dotsim_weightedScore(nn.Module):
    # Dot Mutiplication-based Similarity -> Learnable Index interpolation-based Selection
    # NOTE: Current best method with self.receptive_filed = 2
    def __init__(self,
                 input_size:List[int],
                 down_size:int=7,
                 embed_dim:int=512,
                 dropout:float=0.0
                 ) -> None:
        super().__init__()
        self.down_size = down_size # 7 for 56,28,14 feature pyramid
        self.T,self.H,self.W = input_size
        self.embed_dim = embed_dim
        self.top_k = self.T//2
        self.scaler = 1.0 / self.embed_dim
        out_dim = embed_dim
        self.out_dim = out_dim
        self.dim_project = nn.Sequential(
            nn.Linear(in_features=self.embed_dim*2,out_features=out_dim),
            nn.LayerNorm(out_dim),
        )
        
        self.hook_attn = nn.Identity()
        self.hook_approx = nn.Identity()
        self.hook_detail = nn.Identity()
        
        self.num_head_out = 4
        self.head_dim_out = self.out_dim // self.num_head_out
        self.qkv1 = nn.Linear(out_dim, 3 * out_dim)
        layers: List[nn.Module] = [nn.Linear(out_dim, out_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.project1 = nn.Sequential(*layers)
        from core.model.net_blocks import StochasticDepth, MLP
        self.stochastic_depth = StochasticDepth(0.2, "row")
        self.out_norm1 = nn.LayerNorm(self.out_dim)
        self.mlp1 = MLP(
            out_dim,
            [4 * out_dim, out_dim],
            activation_layer=nn.GELU,
            dropout=dropout,
            inplace=None,)
        
        self.q2 = nn.Conv1d(in_channels=out_dim,out_channels=out_dim*2,kernel_size=3,padding=1,stride=2)
        self.k2 = nn.Linear(out_dim, out_dim)
        
        layers: List[nn.Module] = [nn.Linear(out_dim, out_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.project_a = nn.Sequential(*layers)
        layers: List[nn.Module] = [nn.Linear(out_dim, out_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.project_d = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)        

    def forward(self, x_in:torch.Tensor):
        # input shape: B,C,T,H*W
        b,c,t,hw = x_in.shape
        
        x_in = rearrange(x_in,"b c t L->b (t L) c")
        B,N,C = x_in.shape
        q, k, v = self.qkv1(x_in).reshape(B, N, 3, self.num_head_out, self.head_dim_out).transpose(1, 3).unbind(dim=2)
        attn = torch.matmul(self.scaler * q, k.transpose(2, 3))
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v) # b,head,N,dim
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.project1(x)
        x = self.stochastic_depth(x) + x_in
        x = self.out_norm1(x)
        x = self.stochastic_depth(self.mlp1(x)) # b,N,c
        
        xi:torch.Tensor = x_in.view(b,self.T,self.H,self.W,self.embed_dim).permute(0,1,4,2,3)# b,t,c,h,w
        xi = xi.reshape(b*self.T,self.embed_dim,self.H,self.W) # b*t,c,H,W
        xi = torch.cat((F.adaptive_avg_pool2d(xi,output_size=(1,1)), F.adaptive_max_pool2d(xi,output_size=(1,1))),dim=2)
        xi = rearrange(xi,'(b t) c h w -> b t (c h w)', b=b,t=self.T)
        xi = self.dim_project(xi) # b,t,c
        xi = xi.transpose(1,2) # b,c,t
        b,c,t = xi.shape
        # b,c,t -> b,2c,t/2 -> b,2,c,t/2 -> 2* b,t/2,c
        q1, q2 = self.q2(xi).reshape(b,2,self.out_dim,t//2).transpose(2,3).unbind(dim=1)
        xi = xi.transpose(1,2) # b,t,c
        k2 = self.k2(xi).reshape(b, t, self.out_dim).transpose(1, 2) # b,t,dim -> b,dim,t
        attn1 = torch.matmul(self.scaler * q1, k2) # b,t/2,t
        attn1 = attn1.softmax(dim=-1)
        attn2 = torch.matmul(self.scaler * q2, k2) # b,t/2,t
        attn2 = attn2.softmax(dim=-1)
        
        x_v = x.reshape(b,t,hw,c)
        x_a = torch.einsum('bij,bjkl->bikl',attn1,x_v)
        x_a = self.project_a(x_a).permute(0,3,1,2) # b,t/2,hw,c -> B,C,T,H*W
        x_d = torch.einsum('bij,bjkl->bikl',attn2,x_v)
        x_d = self.project_d(x_d).permute(0,3,1,2) # b,t/2,hw,c -> B,C,T,H*W
        
        return x_a,x_d