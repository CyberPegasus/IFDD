import torch
from torch import nn
from torchvision.ops.misc import Conv3dNormActivation
from typing import List,Callable

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