import torch
from torch import nn
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from functools import partial
from .transformer import MSBlockConfig, MultiscaleBlock, PositionalEncoding
from core.wave_methods.WaveLift import WaveSF_trans,WaveSF_trans_prompt,WaveSF_trans_pool
from loguru import logger
class backboneNet(nn.Module):
    def __init__(self, 
                 type:str='MFViT_XS',
                 attention_dropout:float=0.0,
                 use_WaveSF:bool=False,
                 SSL:bool=False,
                 use_prompt:bool=False,
                 WaveSF_preUpdate:bool=False,
                 AdaptSplit:bool = False,
                 use_lowFreq:bool = False,
                 WaveNorm:bool = False,
                 WavePool:bool = False
                 ):
        super().__init__()
        self.use_prompt = use_prompt
        self.backbone:MFViT = MFViT_build(type,attention_dropout,use_WaveSF,SSL,use_prompt,WaveSF_preUpdate,AdaptSplit,use_lowFreq,WaveNorm,WavePool)

    def forward(self, x:torch.Tensor,prompt:torch.Tensor=None):
        if not self.use_prompt:
            return self.backbone(x)
        else:
            return self.backbone(x,prompt=prompt)
    
class MFViT(nn.Module):
    """
        Following the implementation of PyTorch
        for:
        1. Multiscale Vision Transformer
        at http://arxiv.org/abs/2104.11227
        2. MViTv2: Improved Multiscale Vision Transformers for Classification and Detection
        at http://arxiv.org/abs/2112.01526
    """
    def __init__(self,
                spatial_size: Tuple[int, int], #  The spacial size of the input as (H, W).
                temporal_size: int, # The temporal size T of the input.
                block_setting: Sequence[MSBlockConfig], # sequence of MSBlockConfig: The Network structure.
                residual_pool: bool, # If True, use MViTv2 pooling residual connection.
                residual_with_cls_embed: bool, # If True, the addition on the residual connection will include the class embedding.
                rel_pos_embed: bool, # If True, use MViTv2's relative positional embeddings.
                proj_after_attn: bool, # If True, apply the projection after the attention.
                attention_dropout: float = 0.0, # Attention dropout rate. Default: 0.0.
                stochastic_depth_prob: float = 0.0, # Stochastic depth rate. Default: 0.0.
                block: Optional[Callable[..., nn.Module]] = None, # Module specifying the layer which consists of the attention and mlp.
                norm_layer: Optional[Callable[..., nn.Module]] = None, # Module specifying the normalization layer to use.
                use_WaveSF:bool = True,
                SSL:bool=False,
                WaveSF_preUpdate:bool = False,
                AdaptSplit:bool = False,
                use_lowFreq:bool = False,
                WaveNorm:bool = False,
                WavePool:bool = False
                ):
        super().__init__()
        self.use_WaveSF = use_WaveSF
        self.SSL = SSL
        self.use_lowFreq = use_lowFreq
        change_x = not SSL # when SSL, not change x for WaveSF module
        self.block_setting = block_setting
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")

        if block is None:
            block = MultiscaleBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        input_size = [8, 56, 56]
        self.pos_encoding = PositionalEncoding(
            embed_size=block_setting[0].input_channels,
            spatial_size=(input_size[1], input_size[2]),
            temporal_size=input_size[0],
            rel_pos_embed=rel_pos_embed,
        )

        # Encoder module
        self.blocks = nn.ModuleList()
        self.WaveLifts = nn.ModuleList()
        # discard the last two stage, since can not be processed by WaveLifting
        self.stop_id = len(block_setting)-2 if SSL else len(block_setting)
        # set to save different stage
        block_set = set()
        for stage_block_id, cnf in enumerate(block_setting):
            if stage_block_id >= self.stop_id:
                break
            if use_WaveSF and (cnf.num_heads not in block_set) and (cnf.num_heads<8):
                block_set.add(cnf.num_heads)
                # only two WaveLift Module in Shallow Layers
                _input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
                if not WavePool:
                    self.WaveLifts.append(WaveSF_trans(
                        input_size=_input_size,
                        embed_dim=cnf.output_channels,
                        num_head=cnf.num_heads,
                        relaxed_constrain=True,
                        change_x=change_x,
                        SSL = self.SSL,
                        WaveSF_preUpdate=WaveSF_preUpdate,
                        AdaptSplit=AdaptSplit,
                        lowFreq = use_lowFreq,
                        WaveNorm = WaveNorm,
                        ))
                else:
                    self.WaveLifts.append(WaveSF_trans_pool(
                        input_size=_input_size,
                        embed_dim=cnf.output_channels,
                        num_head=cnf.num_heads,
                        relaxed_constrain=True,
                        change_x=change_x,
                        SSL = self.SSL,
                        WaveSF_preUpdate=WaveSF_preUpdate,
                        AdaptSplit=AdaptSplit,
                        lowFreq = use_lowFreq,
                        WaveNorm = WaveNorm,
                        ))
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
            self.blocks.append(
                block(
                    input_size=input_size,
                    cnf=cnf,
                    residual_pool=residual_pool,
                    residual_with_cls_embed=residual_with_cls_embed,
                    rel_pos_embed=rel_pos_embed,
                    proj_after_attn=proj_after_attn,
                    dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                )
            )
            if len(cnf.stride_q) > 0:
                input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
                    
        self.norm = norm_layer(block_setting[-1].output_channels)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape: (B, C, T, H, W) -> (B, THW, C)
        # suppose H=W=56
        # so: 56,28,14,7
        # C = 96 and T = 8
        x = x.flatten(2).transpose(1, 2) # B, THW, C

        # add positional encoding if not rel_pos
        x = self.pos_encoding(x) # add a cls token

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        block_num = len(self.blocks)
        wavelift_count = 0
        pool_a = []
        x_a = []
        x_d = []
        block_set = set()
        for _id in range(block_num):
            # x_pre = x # DEBUG
            x, thw = self.blocks[_id](x, thw)
            if self.use_WaveSF and (self.block_setting[_id].num_heads not in block_set) and (self.block_setting[_id].num_heads<8):
                block_set.add(self.block_setting[_id].num_heads)
                if self.SSL:
                    if self.use_lowFreq:
                        x, _pool_a, _x_a, _x_d = self.WaveLifts[wavelift_count](x,add_return=True)
                        x_a.append(_x_a)
                    else:
                        x, _pool_a, _x_d = self.WaveLifts[wavelift_count](x,add_return=True)
                    if self.training:
                        pool_a.append(_pool_a)
                    x_d.append(_x_d)
                else:
                    x = self.WaveLifts[wavelift_count](x,add_return=False)
                wavelift_count += 1
                
        if not self.SSL: # SSL will remove the last two block so will not be normed
            x = self.norm(x)
            # classifier "token" as used by standard language architectures
            x = x[:, 0]

        if self.use_WaveSF: # DEBUG self.training, since we now use WaveSF to get targets
            if self.SSL:
                if self.use_lowFreq:
                    return pool_a, x_a, x_d # SSL use x_d to classify
                else:
                    return pool_a, x_d
            else:
                return x, pool_a, x_d
        else:
            return x

    def load_pretrain(self,path:str='pretrain_weights/mvit_v2_s.pth'):
        pretrained = torch.load(path)
        target = self.state_dict()
        for key,value in pretrained.items():
            if 'conv_proj' not in key \
                and 'head' not in key \
                and 'blocks.8' not in key \
                and 'blocks.9' not in key:
                # model revised for block 8~15
                # in&out channel for 8 and 9 changed
                target[key] = value
            if not self.SSL: # SSL mode has not blocks 8 or 9
                if 'blocks.14' in key:
                    _key = key.replace('blocks.14','blocks.8')
                    target[_key] = value
                elif 'blocks.15' in key:
                    _key = key.replace('blocks.15','blocks.9')
                    target[_key] = value
        missing, unexpected = self.load_state_dict(target,strict=False)
        logger.info(f"Missing keys:{missing}\nUnexpected Keys:{unexpected}")
        return 0
    
    def load_pretrain_direct(self,path:str='pretrain_weights/mvit_v2_s.pth'):
        pretrained = torch.load(path)
        missing, unexpected = self.load_state_dict(pretrained,strict=False)
        logger.info(f"Missing keys:{missing}\nUnexpected Keys:{unexpected}")
        return 0  
         
def MFViT_S()->Dict:
    # Following MViTv2_S
    # resolution: [56,28,14,7] or [28,14,7]
    # blocks: [1, 2, 11, 2]
    # heads: [1, 2, 4, 8]
    # channels: [96, 192, 384, 768]
    config: Dict[str, List] = {
    "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
    "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
    "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
    "kernel_q": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    "kernel_kv": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    "stride_q": [
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
    ],
    "stride_kv": [
        [1, 8, 8],
        [1, 4, 4],
        [1, 4, 4],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
    ],
    }
    return config

def MFViT_XS()->Dict:
    # Following MViTv2_S
    # resolution: [56,28,14,7] or [28,14,7]
    # blocks: [1, 2, 5, 2]
    # heads: [1, 2, 4, 8]
    # channels: [96, 192, 384, 768]
    config: Dict[str, List] = {
    "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 8, 8],
    "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 768],
    "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 768, 768],
    "kernel_q": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    "kernel_kv": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    "stride_q": [
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, 2],
        [1, 1, 1],
    ],
    "stride_kv": [
        [1, 8, 8],
        [1, 4, 4],
        [1, 4, 4],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
    ],
    }
    return config

def MFViT_build(type:str,attention_dropout:float,use_WaveSF:bool,SSL:bool,use_prompt:bool,WaveSF_preUpdate:bool,AdaptSplit:bool,use_lowFreq:bool,WaveNorm:bool,WavePool:bool)->MFViT:
    config = eval(type)()
    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )
    if not use_prompt:
        model = MFViT(
                    spatial_size=(224,224),
                    temporal_size=16,
                    block_setting=block_setting,
                    residual_pool=True,
                    residual_with_cls_embed=False,
                    rel_pos_embed=True,
                    proj_after_attn=True,
                    attention_dropout=attention_dropout,
                    stochastic_depth_prob=0.2,
                    use_WaveSF=use_WaveSF,
                    SSL=SSL,
                    WaveSF_preUpdate = WaveSF_preUpdate,
                    AdaptSplit=AdaptSplit,
                    use_lowFreq=use_lowFreq,
                    WaveNorm=WaveNorm,
                    WavePool = WavePool
                    )
    else:
        model = MFViT_prompt(
            spatial_size=(224,224),
            temporal_size=16,
            block_setting=block_setting,
            residual_pool=True,
            residual_with_cls_embed=False,
            rel_pos_embed=True,
            proj_after_attn=True,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=0.2,
            use_WaveSF=use_WaveSF,
            SSL=SSL,
            use_prompt=use_prompt,
            WaveSF_preUpdate = WaveSF_preUpdate,
            AdaptSplit=AdaptSplit,
            use_lowFreq=use_lowFreq,
            WaveNorm=WaveNorm
            )
    return model

def MFViT_build_backbone(type:str,attention_dropout:float,use_WaveSF:bool,SSL:bool,use_prompt:bool,WaveSF_preUpdate:bool,AdaptSplit:bool,use_lowFreq:bool,WaveNorm:bool,WavePool:bool,facial:bool)->MFViT:
    config = eval(type)()
    block_setting = []
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
        )
    model = MFViT_backbone(
        spatial_size=(224,224),
        temporal_size=16,
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        attention_dropout=attention_dropout,
        stochastic_depth_prob=0.2,
        use_WaveSF=use_WaveSF,
        SSL=SSL,
        WaveSF_preUpdate = WaveSF_preUpdate,
        AdaptSplit=AdaptSplit,
        use_lowFreq=use_lowFreq,
        WaveNorm=WaveNorm,
        WavePool = WavePool,
        facial=facial
        )

    return model

class MFViT_prompt(MFViT):
    def __init__(self,
                spatial_size: Tuple[int, int], #  The spacial size of the input as (H, W).
                temporal_size: int, # The temporal size T of the input.
                block_setting: Sequence[MSBlockConfig], # sequence of MSBlockConfig: The Network structure.
                residual_pool: bool, # If True, use MViTv2 pooling residual connection.
                residual_with_cls_embed: bool, # If True, the addition on the residual connection will include the class embedding.
                rel_pos_embed: bool, # If True, use MViTv2's relative positional embeddings.
                proj_after_attn: bool, # If True, apply the projection after the attention.
                attention_dropout: float = 0.0, # Attention dropout rate. Default: 0.0.
                stochastic_depth_prob: float = 0.0, # Stochastic depth rate. Default: 0.0.
                block: Optional[Callable[..., nn.Module]] = None, # Module specifying the layer which consists of the attention and mlp.
                norm_layer: Optional[Callable[..., nn.Module]] = None, # Module specifying the normalization layer to use.
                use_WaveSF:bool = True,
                SSL:bool=False,
                use_prompt:bool = False,
                WaveSF_preUpdate:bool = False,
                AdaptSplit:bool=False,
                use_lowFreq:bool = False,
                WaveNorm:bool = False,
                WavePool:bool = False
                ):
        super().__init__(spatial_size,temporal_size,block_setting,residual_pool,residual_with_cls_embed,rel_pos_embed,proj_after_attn,attention_dropout,stochastic_depth_prob,block,norm_layer,use_WaveSF,SSL,WaveSF_preUpdate,AdaptSplit,use_lowFreq)
        
        self.use_prompt = use_prompt
        self.use_lowFreq = use_lowFreq
        assert self.use_prompt, f'This class is a prompt-used class'
        change_x = not SSL # when SSL, not change x for WaveSF module
        input_size = [8,56,56]
        if self.use_prompt:
            from core.prompt import facePoint_net
            self.prompt_embed = facePoint_net(dim=block_setting[0].input_channels,output_dim=block_setting[0].input_channels)
            self.prompt_projection = nn.ModuleList()

        self.WaveLifts = nn.ModuleList()
        # discard the last two stage, since can not be processed by WaveLifting
        self.stop_id = len(block_setting)-2 if SSL else len(block_setting)
        # set to save different stage
        block_set = set()
        for stage_block_id, cnf in enumerate(block_setting):
            if stage_block_id >= self.stop_id:
                break
            if use_WaveSF and (cnf.num_heads not in block_set) and (cnf.num_heads<8):
                block_set.add(cnf.num_heads)
                # only two WaveLift Module in Shallow Layers
                _input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
                self.WaveLifts.append(WaveSF_trans_prompt(
                    input_size=_input_size,
                    embed_dim=cnf.output_channels,
                    num_head=cnf.num_heads,
                    relaxed_constrain=True,
                    change_x=change_x,
                    SSL = self.SSL,
                    prompt = self.use_prompt,
                    WaveSF_preUpdate = WaveSF_preUpdate,
                    AdaptSplit=AdaptSplit,
                    lowFreq = use_lowFreq,
                    WaveNorm=WaveNorm,
                    ))
                if self.use_prompt:
                    self.prompt_projection.append(nn.Linear(block_setting[0].input_channels,cnf.output_channels))
            if len(cnf.stride_q) > 0:
                input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
         
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)
                    
    def forward(self, x: torch.Tensor, prompt:torch.Tensor=None) -> torch.Tensor:
        # reshape: (B, C, T, H, W) -> (B, THW, C)
        # suppose H=W=56
        # so: 56,28,14,7
        # C = 96 and T = 8
        x = x.flatten(2).transpose(1, 2) # B, THW, C

        # add positional encoding if not rel_pos
        x = self.pos_encoding(x) # add a cls token
        if self.use_prompt:
            prompt = self.prompt_embed(prompt)
            
        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        block_num = len(self.blocks)
        wavelift_count = 0
        pool_a = []
        x_a = []
        x_d = []
        block_set = set()
        for _id in range(block_num):
            # x_pre = x # DEBUG
            x, thw = self.blocks[_id](x, thw)
            if self.use_WaveSF and (self.block_setting[_id].num_heads not in block_set) and (self.block_setting[_id].num_heads<8):
                block_set.add(self.block_setting[_id].num_heads)
                prompt_input = self.prompt_projection[wavelift_count](prompt)
                if self.use_lowFreq:
                    x, _pool_a, _x_a, _x_d = self.WaveLifts[wavelift_count](x,prompt_input,add_return=True)
                    x_a.append(_x_a)
                else:
                    x, _pool_a, _x_d = self.WaveLifts[wavelift_count](x,prompt_input,add_return=True)
                if self.training:
                    pool_a.append(_pool_a)
                x_d.append(_x_d)
                wavelift_count += 1
        if self.use_lowFreq:      
            return pool_a, x_a, x_d # SSL use x_d to classify
        else:
            return pool_a, x_d

class MFViT_FPN(nn.Module):
    def __init__(self, 
                 type:str, 
                 use_stage:List[int],
                 attention_dropout:float=0.0,
                 use_WaveSF:bool=False,
                 SSL:bool=False,
                 use_prompt:bool=False,
                 WaveSF_preUpdate:bool=False,
                 AdaptSplit:bool = False,
                 use_lowFreq:bool = False,
                 WaveNorm:bool = False,
                 WavePool:bool = False,
                 facial:bool = False
                 ) -> None:
        super().__init__()
        self.T = 8
        self.backbone = MFViT_build_backbone(type,attention_dropout,use_WaveSF,SSL,use_prompt,WaveSF_preUpdate,AdaptSplit,use_lowFreq,WaveNorm,WavePool,facial)
        self.facial = facial
        # stage define
        self.Lift_stage = [0,1,2,3]
        # FIXME
        self.use_stage = use_stage # 0,1,2,3
        last_stage = self.use_stage[-1]
        # dim and size
        self.dim = [96,192,384,768]
        self.output_dim = self.dim[last_stage]
        self.size = [56//2**i for i in self.Lift_stage]
        # layer 
        self.PANLayer = None
        self.CatConv = None
        self.multi_stage = False
        if len(self.use_stage) > 1:
            self.multi_stage = True
            self.PANLayer = nn.ModuleList()
            _inchannel_sum = 0
            for i in self.use_stage[:-1]: # skip the last element
                dilate_s = self.size[i]//self.size[last_stage]
                # padding = dilation*(k-1)/2
                CrossConv = nn.Conv2d(in_channels=self.dim[i],out_channels=self.dim[i],kernel_size=3,dilation=dilate_s,padding=dilate_s,stride=dilate_s,bias=False)
                self.PANLayer.append(CrossConv)
                _inchannel_sum += self.dim[i]
            self.CatConv = nn.Conv2d(in_channels=_inchannel_sum+self.dim[last_stage],out_channels=self.output_dim,kernel_size=1,bias=False)
        from core.wave_methods.WaveLift import WaveSF_trans_CLS,WaveSF_trans_CLS_Self,WaveSF_trans_CLS_onlySplit,WaveSF_CNN
        h,w = self.size[last_stage],self.size[last_stage]
        assert use_WaveSF or AdaptSplit, f'Without WaveSF is not implemented !!'
        if use_WaveSF:
            #DEBUG
            self.WaveLifts = WaveSF_trans_CLS(input_size=[self.T,h,w],embed_dim=self.output_dim,num_head=8,AdaptSplit=AdaptSplit,WaveSF_preUpdate=WaveSF_preUpdate,WaveNorm=WaveNorm,dropout=attention_dropout)
        elif not use_WaveSF and AdaptSplit:
            self.WaveLifts = WaveSF_trans_CLS_onlySplit(input_size=[self.T,h,w],embed_dim=self.output_dim,num_head=8,AdaptSplit=AdaptSplit,WaveSF_preUpdate=WaveSF_preUpdate,WaveNorm=WaveNorm,dropout=attention_dropout)
        else:
            return None
        self.hook_stage2 = nn.Identity()
        
        
    def forward(self,x:torch.Tensor,xf:Dict[int, torch.Tensor]=None):
        # reshape: (B, C, T, H, W) -> (B, THW, C)
        b,_,t,_,_ = x.shape # b,c,t,h,w
        if not self.facial:
            x = self.backbone(x)
        else:
            x = self.backbone(x, xf)
        y = [x[i] for i in self.use_stage] # [B, THW+1, C]
        CLS = y[-1][:,0,:] # b,c
        if self.multi_stage:
            for index,i in enumerate(self.use_stage):
                # remove CLS Token
                _y:torch.Tensor = y[index][:,1:,:] # B, THW, C
                h,w,c = self.size[i],self.size[i],self.dim[i]
                _y = _y.reshape(b,t,h,w,c).permute(0,1,4,2,3).reshape(b*t,c,h,w)
                if i == 2:
                    _y = self.hook_stage2(_y)
                if index < len(self.use_stage) - 1:
                    y[index] = self.PANLayer[index](_y) # b*t,c,h',w'
                else:
                    y[index] = _y
            y = torch.cat(y,dim=1)
            y = self.CatConv(y) # b*t,c,h,w
        else:
            y = y[0][:,1:,:]
            i = self.use_stage[0]
            h,w,c = self.size[i],self.size[i],self.dim[i]
            y = y.reshape(b,t,h,w,c).permute(0,1,4,2,3).reshape(b*t,c,h,w)
        _,c,h,w = y.shape
        y = y.reshape(b,t,c,h,w).permute(0,1,3,4,2).reshape(b,-1,c) # b,thw,c
        pool_a, y = self.WaveLifts(y,input_CLS=False) # b,c,t,h,w -> b,c
        res = False
        if res: 
            y = y + CLS
        
        return pool_a, y
    
class MFViT_backbone(nn.Module):
    """
        used for 
        Following the implementation of PyTorch
        for:
        1. Multiscale Vision Transformer
        at http://arxiv.org/abs/2104.11227
        2. MViTv2: Improved Multiscale Vision Transformers for Classification and Detection
        at http://arxiv.org/abs/2112.01526
    """
    def __init__(self,
                spatial_size: Tuple[int, int], #  The spacial size of the input as (H, W).
                temporal_size: int, # The temporal size T of the input.
                block_setting: Sequence[MSBlockConfig], # sequence of MSBlockConfig: The Network structure.
                residual_pool: bool, # If True, use MViTv2 pooling residual connection.
                residual_with_cls_embed: bool, # If True, the addition on the residual connection will include the class embedding.
                rel_pos_embed: bool, # If True, use MViTv2's relative positional embeddings.
                proj_after_attn: bool, # If True, apply the projection after the attention.
                attention_dropout: float = 0.0, # Attention dropout rate. Default: 0.0.
                stochastic_depth_prob: float = 0.0, # Stochastic depth rate. Default: 0.0.
                block: Optional[Callable[..., nn.Module]] = None, # Module specifying the layer which consists of the attention and mlp.
                norm_layer: Optional[Callable[..., nn.Module]] = None, # Module specifying the normalization layer to use.
                use_WaveSF:bool = True,
                SSL:bool=False,
                WaveSF_preUpdate:bool = False,
                AdaptSplit:bool = False,
                use_lowFreq:bool = False,
                WaveNorm:bool = False,
                WavePool:bool = False,
                facial:bool = False
                ):
        super().__init__()
        self.use_WaveSF = use_WaveSF
        self.SSL = SSL
        self.facial = facial
        self.use_lowFreq = use_lowFreq
        self.block_setting = block_setting
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")

        if block is None:
            block = MultiscaleBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        input_size = [8, 56, 56] # fixed input size
        self.pos_encoding = PositionalEncoding(
            embed_size=block_setting[0].input_channels,
            spatial_size=(input_size[1], input_size[2]),
            temporal_size=input_size[0],
            rel_pos_embed=rel_pos_embed,
        )

        # Encoder module
        self.blocks = nn.ModuleList()
        for stage_block_id, cnf in enumerate(block_setting):
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
            self.blocks.append(
                block(
                    input_size=input_size,
                    cnf=cnf,
                    residual_pool=residual_pool,
                    residual_with_cls_embed=residual_with_cls_embed,
                    rel_pos_embed=rel_pos_embed,
                    proj_after_attn=proj_after_attn,
                    dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                )
            )
            if len(cnf.stride_q) > 0:
                input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
                    
        self.norm = norm_layer(block_setting[-1].output_channels)
        
        self.Lift_stage = [0,1,2,3]
        self.dim = [96,192,384,768]
        self.face_dim = [64,64,128,128]
        if self.facial:
            self.fuse_layer = nn.ModuleList()
            self.face_dim = [64,64,128,128]
            from core.model.net_blocks import MLP
            # except for the last stage
            for i in self.Lift_stage[:-1]:
                _in_dim = self.face_dim[i] + self.dim[i]
                _out_dim = self.dim[i]
                Layer_fuseDim = MLP(
                    _in_dim,
                    hidden_channels=[_out_dim,_out_dim],
                    activation_layer=nn.GELU,
                    dropout=attention_dropout,
                    inplace=None,
                )
                self.fuse_layer.append(Layer_fuseDim)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)
                    
    def forward(self, x: torch.Tensor, xf:Dict[int, torch.Tensor]=None) -> torch.Tensor:
        # reshape: (B, C, T, H, W) -> (B, THW, C)
        # suppose H=W=56
        # so: 56,28,14,7
        # C = 96 and T = 8
        x = x.flatten(2).transpose(1, 2) # B, THW, C

        # add positional encoding if not rel_pos
        x = self.pos_encoding(x) # add a cls token

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        block_num = len(self.blocks)
        wavelift_count = 0
        outs = {}
        stage_count = 0 # 0,1,2
        block_set = set()
        for _id in range(block_num):
            if (self.block_setting[_id].num_heads not in block_set) and (_id!=0):
                _num_heads = self.block_setting[_id].num_heads
                block_set.add(_num_heads)
                outs[stage_count] = x # c,thw+1,d
                # facial project
                if self.facial:
                    cls, x = x[:, 0:1, :], x[:, 1:, :]
                    x_skip = x
                    x = torch.cat([x,xf[stage_count]],dim=2)
                    x = self.fuse_layer[stage_count](x)
                    x.add_(x_skip)
                    x = torch.cat([cls,x],dim=1)
                stage_count += 1
                
            x, thw = self.blocks[_id](x, thw)
        # last stage
        x = self.norm(x)
        outs[stage_count] = x

        return outs

    def load_pretrain(self,path:str='pretrain_weights/mvit_v2_s.pth'):
        pretrained = torch.load(path)
        target = self.state_dict()
        if len(self.block_setting) == 16:
            for key,value in pretrained.items():
                if 'conv_proj' not in key and 'head' not in key:
                    target[key] = value
                    
        elif len(self.block_setting) == 10:
            for key,value in pretrained.items():
                if 'conv_proj' not in key \
                    and 'head' not in key \
                    and 'blocks.8' not in key \
                    and 'blocks.9' not in key:
                    # model revised for block 8~15
                    # in&out channel for 8 and 9 changed
                    target[key] = value
                # move 'blocks.14/15' of mvit_s to 'blocks.8/9' of mvit_xs
                if 'blocks.14' in key:
                    _key = key.replace('blocks.14','blocks.8')
                    target[_key] = value
                elif 'blocks.15' in key:
                    _key = key.replace('blocks.15','blocks.9')
                    target[_key] = value
        else:
            raise NotImplementedError(f'block_setting {len(self.block_setting)} is not supported.')

        missing, unexpected = self.load_state_dict(target,strict=False)
        logger.info(f"Missing keys:{missing}\nUnexpected Keys:{unexpected}")
        return 0
    
    # def load_pretrain_direct(self,path:str='pretrain_weights/mvit_v2_s.pth'):
    #     pretrained = torch.load(path)
    #     missing, unexpected = self.load_state_dict(pretrained,strict=False)
    #     logger.info(f"Missing keys:{missing}\nUnexpected Keys:{unexpected}")
    #     return 0  