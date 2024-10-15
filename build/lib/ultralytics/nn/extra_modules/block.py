import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
from collections import OrderedDict
from typing import Optional, Dict

from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad, LightConv
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .attention import *
from .ops_dcnv3.modules import DCNv3
from .transformer import LocalWindowAttention

from timm.layers import DropPath

__all__ = ['Ghost_HGBlock', 'Rep_HGBlock', 'DWRC3', 'C3_DWR', 'C2f_DWR', 'BasicBlock_Ortho', 'BottleNeck_Ortho',
           'C3_DCNv2_Dynamic', 'C2f_DCNv2_Dynamic', 'BasicBlock_DCNv2_Dynamic', 'BottleNeck_DCNv2_Dynamic',
           'C3_DCNv2', 'C2f_DCNv2', 'BasicBlock_DCNv2', 'BottleNeck_DCNv2',
           'C3_DCNv3', 'C2f_DCNv3', 'BasicBlock_DCNv3', 'BottleNeck_DCNv3',
           'C3_iRMB', 'C2f_iRMB', 'BasicBlock_iRMB', 'BottleNeck_iRMB',
           'C3_iRMB_Cascaded', 'C2f_iRMB_Cascaded', 'BasicBlock_iRMB_Cascaded', 'BottleNeck_iRMB_Cascaded',
           'BasicBlock_Attention', 'BottleNeck_Attention', 'HGBlock_Attention', 'C3_Attention', 'C2f_Attention',
           'C3_Ortho', 'C2f_Ortho'
           ]

######################################## HGBlock with RepConv and GhostConv start ########################################

class Ghost_HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = GhostConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

class RepLightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = RepConv(c2, c2, k, g=math.gcd(c1, c2), act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))

class Rep_HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = RepLightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

######################################## HGBlock with RepConv and GhostConv end ########################################

######################################## Dilation-wise Residual start ########################################

class DWR(nn.Module):
    def __init__(self, dim, act=True) -> None:
        super().__init__()

        self.conv_3x3 = Conv(dim, dim // 2, 3, act=act)
        
        self.conv_3x3_d1 = Conv(dim // 2, dim, 3, d=1, act=act)
        self.conv_3x3_d3 = Conv(dim // 2, dim // 2, 3, d=3, act=act)
        self.conv_3x3_d5 = Conv(dim // 2, dim // 2, 3, d=5, act=act)
        
        self.conv_1x1 = Conv(dim * 2, dim, k=1, act=act)
        
    def forward(self, x):
        conv_3x3 = self.conv_3x3(x)
        x1, x2, x3 = self.conv_3x3_d1(conv_3x3), self.conv_3x3_d3(conv_3x3), self.conv_3x3_d5(conv_3x3)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out

class DWRC3(RepC3):
    def __init__(self, c1, c2, n=3, s=1, e=1, act='relu'):
        super().__init__(c1, c2, n, e)
        c_ = int(c2 * e)  # hidden channels
        if s == 2:
            self.conv_s2 = Conv(c1, c1, k=3, s=2, act=act)
        self.m = nn.Sequential(*[DWR(c_, act) for _ in range(n)])
    
    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        if hasattr(self, "conv_s2"):
            x = self.conv_s2(x)
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))

class C3_DWR(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DWR(c_) for _ in range(n)))

class C2f_DWR(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(DWR(self.c) for _ in range(n))
    
######################################## Dilation-wise Residual end ########################################

######################################## OrthoNets start ########################################

def gram_schmidt(input):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u
    output = []
    for x in input:
        for y in output:
            x = x - projection(y, x)
        x = x/x.norm(p=2)
        output.append(x)
    return torch.stack(output)

def initialize_orthogonal_filters(c, h, w):

    if h*w < c:
        n = c//(h*w)
        gram = []
        for i in range(n):
            gram.append(gram_schmidt(torch.rand([h * w, 1, h, w])))
        return torch.cat(gram, dim=0)
    else:
        return gram_schmidt(torch.rand([c, 1, h, w]))

class GramSchmidtTransform(torch.nn.Module):
    instance = {}
    constant_filter: torch.Tensor

    @staticmethod
    def build(c: int, h: int):
        if c not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        self.register_buffer("constant_filter", rand_ortho_filters.detach())
        
    def forward(self, x):
        _, _, h, w = x.shape
        _, H, W = self.constant_filter.shape
        if h != H or w != W: x = torch.nn.functional.adaptive_avg_pool2d(x, (H, W))
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)

class Attention_Ortho(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, FWT: GramSchmidtTransform, input: torch.Tensor):
        #happens once in case of BigFilter
        while input[0].size(-1) > 1:
            input = FWT(input)
        b = input.size(0)
        return input.view(b, -1)

class BasicBlock_Ortho(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', height=64, variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 
        
        self._excitation = nn.Sequential(
            nn.Linear(in_features=ch_out, out_features=round(ch_out / 16), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(ch_out / 16), out_features=ch_out, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention_Ortho()
        self.F_C_A = GramSchmidtTransform.build(ch_out, height)


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        
        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0),out.size(1)
        excitation = self._excitation(compressed).view(b, c, 1, 1)
        out = excitation * out 
        
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)

        return out

class BottleNeck_Ortho(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', height=64, variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)
        
        self._excitation = nn.Sequential(
            nn.Linear(in_features=ch_out * self.expansion, out_features=round(ch_out / 16 * self.expansion), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(ch_out / 16 * self.expansion), out_features=ch_out * self.expansion, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention_Ortho()
        self.F_C_A = GramSchmidtTransform.build(ch_out * self.expansion, height)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        compressed = self.OrthoAttention(self.F_C_A, out)
        b, c = out.size(0),out.size(1)
        excitation = self._excitation(compressed).view(b, c, 1, 1)
        out = excitation * out
        
        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

class Bottleneck_Ortho(Bottleneck):
	def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, height=16):
		super().__init__(c1, c2, shortcut, g, k, e)

		self._excitation = nn.Sequential(
			nn.Linear(in_features=c2, out_features=round(c2 / 16), bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=round(c2 / 16), out_features=c2, bias=False),
			nn.Sigmoid(),
		)
		self.OrthoAttention = Attention_Ortho()
		self.F_C_A = GramSchmidtTransform.build(c2, height)

	def forward(self, x):
		"""'forward()' applies the YOLO FPN to input data."""
		out = self.cv2(self.cv1(x))

		compressed = self.OrthoAttention(self.F_C_A, out)
		b, c = out.size(0),out.size(1)
		excitation = self._excitation(compressed).view(b, c, 1, 1)
		out = excitation * out
		return x + out if self.add else out

class C3_Ortho(C3):
    def __init__(self, c1, c2, n=1, height=16, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_Ortho(c_, c_, shortcut, g, k=(1, 3), e=1.0, height=height) for _ in range(n)))

class C2f_Ortho(C2f):
    def __init__(self, c1, c2, n=1, height=16, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_Ortho(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, height=height) for _ in range(n))

######################################## OrthoNets end ########################################

######################################## DCNV2 start ########################################

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCNV2(Bottleneck):
    """Standard bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNv2(c_, c2, k[1], 1)

class C3_DCNv2(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNV2(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_DCNv2(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DCNV2(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class BasicBlock_DCNv2(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = DCNv2(ch_out, ch_out, 3, act=None)

class BottleNeck_DCNv2(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = DCNv2(ch_out, ch_out, 3, stride=stride, act=None)

######################################## DCNV2 end ########################################

######################################## DCNV2_Dynamic start ########################################

class DCNv2_Offset_Attention(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, deformable_groups=1) -> None:
        super().__init__()
        
        padding = autopad(kernel_size, None, 1)
        self.out_channel = (deformable_groups * 3 * kernel_size * kernel_size)
        self.conv_offset_mask = nn.Conv2d(in_channels, self.out_channel, kernel_size, stride, padding, bias=True)
        self.attention = MPCA(self.out_channel)
        
    def forward(self, x):
        conv_offset_mask = self.conv_offset_mask(x)
        conv_offset_mask = self.attention(conv_offset_mask)
        return conv_offset_mask

class DCNv2_Dynamic(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2_Dynamic, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.conv_offset_mask = DCNv2_Offset_Attention(in_channels, kernel_size, stride, deformable_groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.conv_offset_mask.bias.data.zero_()

class Bottleneck_DCNV2_Dynamic(Bottleneck):
    """Standard bottleneck with DCNV2."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNv2_Dynamic(c_, c2, k[1], 1)

class C3_DCNv2_Dynamic(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNV2_Dynamic(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_DCNv2_Dynamic(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DCNV2_Dynamic(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class BasicBlock_DCNv2_Dynamic(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = DCNv2_Dynamic(ch_out, ch_out, 3, act=None)

class BottleNeck_DCNv2_Dynamic(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = DCNv2_Dynamic(ch_out, ch_out, 3, stride=stride, act=None)

######################################## DCNV2_Dynamic end ########################################

######################################## DCNV3 start ########################################

class DCNV3_YOLO(nn.Module):
    def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        
        if inc != ouc:
            self.stem_conv = Conv(inc, ouc, k=1)
        self.dcnv3 = DCNv3(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
    def forward(self, x):
        if hasattr(self, 'stem_conv'):
            x = self.stem_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.dcnv3(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(self.bn(x))
        return x

class Bottleneck_DCNV3(Bottleneck):
    """Standard bottleneck with DCNV3."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DCNV3_YOLO(c_, c2, k[1])

class C3_DCNv3(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DCNV3(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_DCNv3(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DCNV3(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

class BasicBlock_DCNv3(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = DCNV3_YOLO(ch_out, ch_out, 3, act=None)

class BottleNeck_DCNv3(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = DCNV3_YOLO(ch_out, ch_out, 3, s=stride, act=None)

######################################## DCNV3 end ########################################

######################################## iRMB and iRMB with CascadedGroupAttention start ########################################

class iRMB(nn.Module):
	def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
				 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
				 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
		super().__init__()
		self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
		self.act = Conv.default_act if act else nn.Identity()
		dim_mid = int(dim_in * exp_ratio)
		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
		self.attn_s = attn_s
		if self.attn_s:
			assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
			self.dim_head = dim_head
			self.window_size = window_size
			self.num_head = dim_in // dim_head
			self.scale = self.dim_head ** -0.5
			self.attn_pre = attn_pre
			self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
			self.v = nn.Sequential(
				nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
				self.act
			)
			self.attn_drop = nn.Dropout(attn_drop)
		else:
			if v_proj:
				self.v = nn.Sequential(
					nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
					self.act
				)
			else:
				self.v = nn.Identity()
		self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
		self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()
		
		self.proj_drop = nn.Dropout(drop)
		self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
		self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		B, C, H, W = x.shape
		if self.attn_s:
			# padding
			if self.window_size <= 0:
				window_size_W, window_size_H = W, H
			else:
				window_size_W, window_size_H = self.window_size, self.window_size
			pad_l, pad_t = 0, 0
			pad_r = (window_size_W - W % window_size_W) % window_size_W
			pad_b = (window_size_H - H % window_size_H) % window_size_H
			x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
			n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
			x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
			# attention
			b, c, h, w = x.shape
			qk = self.qk(x)
			qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
			q, k = qk[0], qk[1]
			attn_spa = (q @ k.transpose(-2, -1)) * self.scale
			attn_spa = attn_spa.softmax(dim=-1)
			attn_spa = self.attn_drop(attn_spa)
			if self.attn_pre:
				x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ x
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
				x_spa = self.v(x_spa)
			else:
				v = self.v(x)
				v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
				x_spa = attn_spa @ v
				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
			# unpadding
			x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
			if pad_r > 0 or pad_b > 0:
				x = x[:, :, :H, :W].contiguous()
		else:
			x = self.v(x)

		x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
		
		x = self.proj_drop(x)
		x = self.proj(x)
		
		x = (shortcut + self.drop_path(x)) if self.has_skip else x
		return x

class iRMB_Cascaded(nn.Module):
	def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
				 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, num_head=16, se_ratio=0.0,
				 attn_s=True, qkv_bias=False, drop=0., drop_path=0., v_group=False):
		super().__init__()
		self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
		self.act = Conv.default_act if act else nn.Identity()
		dim_mid = int(dim_in * exp_ratio)
		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
		self.attn_s = attn_s
		self.num_head = num_head
		if self.attn_s:
			self.attn = LocalWindowAttention(dim_mid)
		else:
			if v_proj:
				self.v = nn.Sequential(
					nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
					self.act
				)
			else:
				self.v = nn.Identity()
		self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
		self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()
		
		self.proj_drop = nn.Dropout(drop)
		self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
		self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.norm(x)
		B, C, H, W = x.shape
		if self.attn_s:
			x = self.attn(x)
		else:
			x = self.v(x)

		x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))
		
		x = self.proj_drop(x)
		x = self.proj(x)
		
		x = (shortcut + self.drop_path(x)) if self.has_skip else x
		return x

class C3_iRMB(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(iRMB(c_, c_) for _ in range(n)))

class C2f_iRMB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(iRMB(self.c, self.c) for _ in range(n))

class BasicBlock_iRMB(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = iRMB(ch_out, ch_out)

class BottleNeck_iRMB(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = iRMB(ch_out, ch_out, stride=stride)

class C3_iRMB_Cascaded(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(iRMB_Cascaded(c_, c_) for _ in range(n)))

class C2f_iRMB_Cascaded(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(iRMB_Cascaded(self.c, self.c) for _ in range(n))

class BasicBlock_iRMB_Cascaded(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = iRMB_Cascaded(ch_out, ch_out)

class BottleNeck_iRMB_Cascaded(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = iRMB_Cascaded(ch_out, ch_out, stride=stride)

######################################## iRMB and iRMB with CascadedGroupAttention end ########################################

######################################## resnet18 attention start ########################################

class BasicBlock_Attention(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)
        
        self.attention = CoordAtt(ch_out)
        # self.attention = BiLevelRoutingAttention_nchw(ch_out)
        # self.attention = SimAM()
        
    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.attention(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out

class BottleNeck_Attention(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)
        
        self.attention = CoordAtt(ch_out * self.expansion)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)
        out = self.attention(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out

class HGBlock_Attention(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2
        
        self.attention = CoordAtt(c1 + n * cm)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(self.attention(torch.cat(y, 1))))
        return y + x if self.add else y

class Bottleneck_Attention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.attention = CoordAtt(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))

class C2f_Attention(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
         super().__init__(c1, c2, n, shortcut, g, e)
         
         self.m = nn.ModuleList(Bottleneck_Attention(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

class C3_Attention(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
         super().__init__(c1, c2, n, shortcut, g, e)
         c_ = int(c2 * e)  # hidden channels
         self.m = nn.Sequential(*(Bottleneck_Attention(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
######################################## resnet18 attention end ########################################
