# src/modules/quant_attn.py
"""量化版 Area Attention 模块。

基于 TurboQuant 的 PolarQuant 算法，实现特征图向量量化。
"""

import torch
import torch.nn as nn

from .polar_quant import PolarQuantTorch

__all__ = ["AAttnQuant", "ABlockQuant", "A2C2fQuant"]


# 检测 Flash Attention 可用性
USE_FLASH_ATTN = False
try:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
except Exception:
    pass


def _get_conv_module():
    """延迟导入 Conv 以避免循环依赖。"""
    from ultralytics.nn.modules.conv import Conv
    return Conv


def _get_c3k_module():
    """延迟导入 C3k 以避免循环依赖。"""
    from ultralytics.nn.modules.block import C3k
    return C3k


class AAttnQuant(nn.Module):
    """Area Attention with PolarQuant-based feature compression.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        area: Number of areas to divide the feature map (1 = no division).
        bit_width: Quantization bits per coordinate (2, 3, or 4 recommended).
        quantize_qk: Whether to quantize QK features (default: True).
        quantize_v: Whether to quantize V features (default: False).
    """

    def __init__(self, dim, num_heads, area=1, bit_width=4, quantize_qk=True, quantize_v=False):
        super().__init__()
        Conv = _get_conv_module()

        self.area = area
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        all_head_dim = self.head_dim * self.num_heads

        # Standard AAttn convolution layers
        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)

        # Quantization settings
        self.quantize_qk = quantize_qk
        self.quantize_v = quantize_v
        self.bit_width = bit_width

        # PolarQuant quantizers (per head_dim)
        if quantize_qk:
            self.pq_qk = PolarQuantTorch(d=self.head_dim, bit_width=bit_width, seed=42)
        if quantize_v:
            self.pq_v = PolarQuantTorch(d=self.head_dim, bit_width=bit_width, seed=43)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape

        q, k = qk.split([C, C], dim=2)

        # Optional: Quantize QK features
        if self.quantize_qk and self.training:
            q, k = self._quantize_qk(q, k)

        # Optional: Quantize V features
        if self.quantize_v and self.training:
            v = self._quantize_v(v)

        # Attention computation
        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)

    def _quantize_qk(self, q, k):
        B, N, C = q.shape
        q_reshaped = q.view(B * N, self.num_heads, self.head_dim)
        k_reshaped = k.view(B * N, self.num_heads, self.head_dim)

        q_quant = self.pq_qk.quantize_dequantize(q_reshaped.reshape(-1, self.head_dim))
        k_quant = self.pq_qk.quantize_dequantize(k_reshaped.reshape(-1, self.head_dim))

        return q_quant.view(B, N, C), k_quant.view(B, N, C)

    def _quantize_v(self, v):
        B, N, C = v.shape
        v_reshaped = v.view(B * N, self.num_heads, self.head_dim)
        v_quant = self.pq_v.quantize_dequantize(v_reshaped.reshape(-1, self.head_dim))
        return v_quant.view(B, N, C)


class ABlockQuant(nn.Module):
    """Area Attention Block with PolarQuant compression.

    Args:
        dim: Number of hidden channels.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio (default: 1.2).
        area: Number of areas for feature map division (default: 1).
        bit_width: Quantization bits (default: 4).
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1, bit_width=4):
        super().__init__()
        Conv = _get_conv_module()

        self.attn = AAttnQuant(dim, num_heads=num_heads, area=area, bit_width=bit_width)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2fQuant(nn.Module):
    """R-ELAN with PolarQuant-compressed Area Attention.

    Args:
        c1: Number of input channels.
        c2: Number of output channels.
        n: Number of 2xABlockQuant modules to stack (default: 1).
        a2: Whether to use area-attention (default: True).
        area: Number of areas for feature map division (default: 1).
        residual: Whether to use residual with layer scale (default: False).
        mlp_ratio: MLP expansion ratio (default: 2.0).
        bit_width: Quantization bits (default: 4).
        e: Expansion ratio for hidden channels (default: 0.5).
        g: Number of groups for grouped convolution (default: 1).
        shortcut: Whether to use shortcut connection (default: True).
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, bit_width=4, e=0.5, g=1, shortcut=True):
        super().__init__()
        Conv = _get_conv_module()
        C3k = _get_c3k_module()

        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        init_values = 0.01
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlockQuant(c_, num_heads, mlp_ratio, area, bit_width) for _ in range(2)))
            if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))