import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.fx
from torch import nn

from .net_blocks import _prod,Pool,_add_rel_pos,_add_shortcut,MLP,StochasticDepth

# dataclass descriptor will create a more concrete class for only attributes
@dataclass 
class MSBlockConfig:
    num_heads: int
    input_channels: int
    output_channels: int
    kernel_q: List[int]
    kernel_kv: List[int]
    stride_q: List[int]
    stride_kv: List[int]

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, spatial_size: Tuple[int, int], temporal_size: int, rel_pos_embed: bool) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size

        self.class_token = nn.Parameter(torch.zeros(embed_size))
        self.spatial_pos: Optional[nn.Parameter] = None
        self.temporal_pos: Optional[nn.Parameter] = None
        self.class_pos: Optional[nn.Parameter] = None
        if not rel_pos_embed:
            self.spatial_pos = nn.Parameter(torch.zeros(self.spatial_size[0] * self.spatial_size[1], embed_size))
            self.temporal_pos = nn.Parameter(torch.zeros(self.temporal_size, embed_size))
            self.class_pos = nn.Parameter(torch.zeros(embed_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_token = self.class_token.expand(x.size(0), -1).unsqueeze(1)
        x = torch.cat((class_token, x), dim=1)

        if self.spatial_pos is not None and self.temporal_pos is not None and self.class_pos is not None:
            hw_size, embed_size = self.spatial_pos.shape
            pos_embedding = torch.repeat_interleave(self.temporal_pos, hw_size, dim=0)
            pos_embedding.add_(self.spatial_pos.unsqueeze(0).expand(self.temporal_size, -1, -1).reshape(-1, embed_size))
            pos_embedding = torch.cat((self.class_pos.unsqueeze(0), pos_embedding), dim=0).unsqueeze(0)
            x.add_(pos_embedding)

        return x
    
class PositionalEncoding_noCLS(nn.Module):
    def __init__(self, embed_size: int, spatial_size: Tuple[int, int], temporal_size: int, rel_pos_embed: bool) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size

        self.class_token = nn.Parameter(torch.zeros(embed_size))
        self.spatial_pos: Optional[nn.Parameter] = None
        self.temporal_pos: Optional[nn.Parameter] = None
        self.class_pos: Optional[nn.Parameter] = None
        if not rel_pos_embed:
            self.spatial_pos = nn.Parameter(torch.zeros(self.spatial_size[0] * self.spatial_size[1], embed_size))
            self.temporal_pos = nn.Parameter(torch.zeros(self.temporal_size, embed_size))
            self.class_pos = nn.Parameter(torch.zeros(embed_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_token = self.class_token.expand(x.size(0), -1).unsqueeze(1)
        x = torch.cat((class_token, x), dim=1)

        if self.spatial_pos is not None and self.temporal_pos is not None and self.class_pos is not None:
            hw_size, embed_size = self.spatial_pos.shape
            pos_embedding = torch.repeat_interleave(self.temporal_pos, hw_size, dim=0)
            pos_embedding.add_(self.spatial_pos.unsqueeze(0).expand(self.temporal_size, -1, -1).reshape(-1, embed_size))
            pos_embedding = torch.cat((self.class_pos.unsqueeze(0), pos_embedding), dim=0).unsqueeze(0)
            x.add_(pos_embedding)

        return x
    
class MultiscaleAttention(nn.Module):
    def __init__(
        self,
        input_size: List[int],
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        kernel_q: List[int],
        kernel_kv: List[int],
        stride_q: List[int],
        stride_kv: List[int],
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.residual_pool = residual_pool
        self.residual_with_cls_embed = residual_with_cls_embed

        self.qkv = nn.Linear(embed_dim, 3 * output_dim)
        layers: List[nn.Module] = [nn.Linear(output_dim, output_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.project = nn.Sequential(*layers)

        self.pool_q: Optional[nn.Module] = None
        if _prod(kernel_q) > 1 or _prod(stride_q) > 1:
            padding_q = [int(q // 2) for q in kernel_q]
            self.pool_q = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_q,  # type: ignore[arg-type]
                    stride=stride_q,  # type: ignore[arg-type]
                    padding=padding_q,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )

        self.pool_k: Optional[nn.Module] = None
        self.pool_v: Optional[nn.Module] = None
        if _prod(kernel_kv) > 1 or _prod(stride_kv) > 1:
            padding_kv = [int(kv // 2) for kv in kernel_kv]
            self.pool_k = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,  # type: ignore[arg-type]
                    stride=stride_kv,  # type: ignore[arg-type]
                    padding=padding_kv,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )
            self.pool_v = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,  # type: ignore[arg-type]
                    stride=stride_kv,  # type: ignore[arg-type]
                    padding=padding_kv,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )

        self.rel_pos_h: Optional[nn.Parameter] = None
        self.rel_pos_w: Optional[nn.Parameter] = None
        self.rel_pos_t: Optional[nn.Parameter] = None
        if rel_pos_embed:
            size = max(input_size[1:])
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            spatial_dim = 2 * max(q_size, kv_size) - 1
            temporal_dim = 2 * input_size[0] - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
            self.rel_pos_t = nn.Parameter(torch.zeros(temporal_dim, self.head_dim))
            nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_t, std=0.02)

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3).unbind(dim=2)

        if self.pool_k is not None:
            k, k_thw = self.pool_k(k, thw)
        else:
            k_thw = thw
        if self.pool_v is not None:
            v = self.pool_v(v, thw)[0]
        if self.pool_q is not None:
            q, thw = self.pool_q(q, thw)

        attn = torch.matmul(self.scaler * q, k.transpose(2, 3))
        if self.rel_pos_h is not None and self.rel_pos_w is not None and self.rel_pos_t is not None:
            attn = _add_rel_pos(
                attn,
                q,
                thw,
                k_thw,
                self.rel_pos_h,
                self.rel_pos_w,
                self.rel_pos_t,
            )
        attn = attn.softmax(dim=-1)

        x = torch.matmul(attn, v)
        if self.residual_pool:
            _add_shortcut(x, q, self.residual_with_cls_embed)
        x = x.transpose(1, 2).reshape(B, -1, self.output_dim)
        x = self.project(x)

        return x, thw

class MultiscaleBlock(nn.Module):
    def __init__(
        self,
        input_size: List[int],
        cnf: MSBlockConfig,
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        proj_after_attn: bool,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.proj_after_attn = proj_after_attn

        self.pool_skip: Optional[nn.Module] = None
        if _prod(cnf.stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in cnf.stride_q]
            padding_skip = [int(k // 2) for k in kernel_skip]
            self.pool_skip = Pool(
                nn.MaxPool3d(kernel_skip, stride=cnf.stride_q, padding=padding_skip), None  # type: ignore[arg-type]
            )

        attn_dim = cnf.output_channels if proj_after_attn else cnf.input_channels

        self.norm1 = norm_layer(cnf.input_channels)
        self.norm2 = norm_layer(attn_dim)
        self.needs_transposal = isinstance(self.norm1, nn.BatchNorm1d)

        self.attn = MultiscaleAttention(
            input_size,
            cnf.input_channels,
            attn_dim,
            cnf.num_heads,
            kernel_q=cnf.kernel_q,
            kernel_kv=cnf.kernel_kv,
            stride_q=cnf.stride_q,
            stride_kv=cnf.stride_kv,
            rel_pos_embed=rel_pos_embed,
            residual_pool=residual_pool,
            residual_with_cls_embed=residual_with_cls_embed,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        self.mlp = MLP(
            attn_dim,
            [4 * attn_dim, cnf.output_channels],
            activation_layer=nn.GELU,
            dropout=dropout,
            inplace=None,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        self.project: Optional[nn.Module] = None
        if cnf.input_channels != cnf.output_channels:
            self.project = nn.Linear(cnf.input_channels, cnf.output_channels)

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x_norm1 = self.norm1(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm1(x)
        x_attn, thw_new = self.attn(x_norm1, thw)
        x = x if self.project is None or not self.proj_after_attn else self.project(x_norm1)
        x_skip = x if self.pool_skip is None else self.pool_skip(x, thw)[0]
        x = x_skip + self.stochastic_depth(x_attn)

        x_norm2 = self.norm2(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm2(x)
        x_proj = x if self.project is None or self.proj_after_attn else self.project(x_norm2)

        return x_proj + self.stochastic_depth(self.mlp(x_norm2)), thw_new