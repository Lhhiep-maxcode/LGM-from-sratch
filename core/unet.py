# core\model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention
from functools import partial
from typing import Tuple, Literal


class MVAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_frames: int = 4,
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = Attention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        BV, C, H, W = x.shape
        B = BV // self.num_frames

        res = x
        x = self.norm(x)

        # (BV, C, H, W) -> reshape: (B, V, C, H, W) -> permute: (B, V, H, W, C) -> reshape: (B, V * H * W, C)
        x = x.reshape(B, self.num_frames, C, H, W).permute(0, 1, 3, 4, 2).reshape(B, -1, C)
        # (B, V * H * W, C)
        x = self.attn(x)
        # (B, V * H * W, C) -> reshape: (B, V, H, W, C) -> permute: (B, V, C, H, W) -> reshape: (BV, C, H, W)
        x = x.reshape(B, self.num_frames, H, W, C).permute(0, 1, 4, 2, 3).reshape(BV, C, H, W)

        if self.residual:
            x = (x + res) * self.skip_scale
        return x
    

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    x = torch.randn((4, 32, 16, 16)).to(device)
    mvattn = MVAttention(32).to(device)
    resblock = ResnetBlock(32, 32, 'default').to(device)
    print(resblock(x).shape)
