# core\unet.py

import os
import warnings

from torch import Tensor
from torch import nn
from torch import randn, cuda

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // self.num_heads) ** -0.5    # scale for dot-product

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape   # N = H * W
        # (B, N, C) -> (3, B, heads, N, channel/head)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print("Expected: (B, N, 3, heads, channel/head) -> (3, B, heads, N, channel/head)")
        # print(x.shape, '-->', qkv.shape)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        # res = (B, heads, N, N)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = self.attn_drop(attn)
        # print("Expected: (B, heads, N, N)")
        # print(attn.shape)

        # (B, heads, N, channel/head) -> (B, N, heads, channel/head) -> (B, N, C)
        x1 = (attn @ v)
        x2 = x1.transpose(1, 2)
        x = x2.reshape(B, N, C)
        # print("Expected: (B, heads, N, channel/head) -> (B, N, heads, channel/head) -> (B, N, C)")
        # print(x1.shape, '-->', x2.shape, '-->', x.shape)

        # (B, N, C) -> (B, N, C)
        x1 = self.proj(x)
        x = self.proj_drop(x1)
        # print("Expected: (B, N, C) -> (B, N, C)")
        # print(x1.shape, '-->', x.shape)

        return x
    

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // self.num_heads) ** -0.5

        self.to_q = nn.Linear(dim_q, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim_k, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim_v, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        # q: [B, N, Cq]
        # k: [B, M, Ck]
        # v: [B, M, Cv]
        # return: [B, N, C]

        B, N, _ = q.shape
        _, M, _ = k.shape

        # [B, N, Cq] -> [B, N, C] -> [B, N, nh, C/nh] -> [B, nh, N, C/nh]
        q = self.scale * (self.to_q(q).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3))
        print("Expected: [B, nh, N, C/nh]")
        print(q.shape)
        # [B, nh, M, C/nh]
        k = (self.to_k(k).reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3))
        print("Expected: [B, nh, M, C/nh]")
        print(k.shape)
        # [B, nh, M, C/nh]
        v = (self.to_v(v).reshape(B, M, self.num_heads, -1).permute(0, 2, 1, 3))
        print("Expected: [B, nh, M, C/nh]")
        print(v.shape)

        # [B, nh, N, M]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 
        print("Expected [B, nh, N, M]")
        print(attn.shape)

        # [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        print("Expected [B, N, C]")
        print(x.shape)
        return x


if __name__ == "__main__":
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)
    attn = Attention(dim=32).to(device)
    cattn = CrossAttention(dim=32, dim_q=32, dim_k=32, dim_v=32).to(device)
    q = randn((16, 64, 32)).to(device)
    k = randn((16, 128, 32)).to(device)
    v = randn((16, 128, 32)).to(device)
    print(cattn(q, k, v).shape)
