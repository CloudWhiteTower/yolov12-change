"""大豆豆荚分割改进模块：EMA、CoordAttention、A2C2fEMA、A2C2fEMASpatial、ECALayer、GRN、A2C2fECA、A2C2fEMAECA、A2C2fEMAGRN、TripletAttention、MSCALite、A2C2fTriplet、A2C2fMSCA。"""

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import A2C2f
from ultralytics.nn.modules.conv import SpatialAttention


class EMA(nn.Module):
    """Efficient Multi-Scale Attention (ICASSP 2023 / FEI-YOLO Agronomy 2024).

    通过分组通道 + 双分支(1x1/3x3) + 跨空间学习实现多尺度注意力。
    对大豆豆荚检测 mAP@0.5 提升约 0.8%。

    Args:
        c1: 输入通道数
        c2: 输出通道数（必须等于 c1）
        groups: 注意力分组数，默认 4
    """

    def __init__(self, c1: int, c2: int, groups: int = 4):
        super().__init__()
        assert c1 == c2, "EMA requires c1 == c2"
        self.groups = groups
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        gc = c1 // groups
        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gc = c // self.groups
        # 分组
        group_x = x.reshape(b * self.groups, gc, h, w)
        # 1×1 分支 → 全局 H 方向编码
        x_h = self.avg_pool_h(group_x)  # (b*g, gc, h, 1)
        x_w = self.avg_pool_w(group_x)  # (b*g, gc, 1, w)
        # 双分支处理
        x_cat = self.conv1x1(group_x)
        x_s = self.conv3x3(group_x)
        # 跨空间学习
        hw = self.sigmoid(x_h * x_w)
        out = x_cat * hw + x_s * hw
        return out.reshape(b, c, h, w)


class CoordAttention(nn.Module):
    """Coordinate Attention (CVPR 2021).

    将通道注意力分解为 X/Y 两个 1D 特征编码过程,
    捕获方向感知的长距离依赖, 保留精确位置信息。
    尤其适合于细长目标（如大豆豆荚）。

    Args:
        c1: 输入通道数
        c2: 输出通道数（必须等于 c1）
        reduction: 中间层通道压缩比，默认 32
    """

    def __init__(self, c1: int, c2: int, reduction: int = 32):
        super().__init__()
        assert c1 == c2, "CoordAttention requires c1 == c2"
        mid = max(8, c1 // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c1, mid, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid, c1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape
        # X/Y 方向全局池化
        x_h = self.pool_h(x)  # (b, c, h, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b, c, w, 1) → (b, c, w, 1)

        # 拼接 → 共享编码
        y = torch.cat([x_h, x_w], dim=2)  # (b, c, h+w, 1)
        y = self.act(self.bn1(self.conv1(y)))

        # 分割 → 独立映射
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复形状

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w


class A2C2fEMA(A2C2f):
    """A2C2f + EMA 融合模块：在 R-ELAN 输出后施加多尺度注意力。

    通过类继承保持层索引不变，实现预训练权重 ~98% 加载率。
    仅 EMA 子模块（~16K params per instance）随机初始化。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_ema = EMA(c2, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return self.post_ema(out)


class SimAM(nn.Module):
    """SimAM: Simple, Parameter-Free Attention Module (ICML 2021).

    基于计算神经科学的能量最小化原则，零可训练参数。
    对每个神经元计算其与同通道其他神经元的线性可分性:
        e_t = (x_t - μ)² / (4(σ² + λ))
        weight = sigmoid(1 / e_t)

    Args:
        lam: 正则化系数，默认 1e-4
    """

    def __init__(self, lam: float = 1e-4):
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = ((x - mu) ** 2).mean(dim=[2, 3], keepdim=True)
        # 能量函数: e_t = (x_t - μ)² / (4(σ² + λ))
        e_t = (x - mu) ** 2 / (4 * (var + self.lam)) + 0.5
        # 权重: sigmoid(1/e_t) — 偏离均值越大 → 能量越高 → 权重越大
        return x * torch.sigmoid(1.0 / e_t)


class A2C2fSimAM(A2C2f):
    """A2C2f + SimAM 融合模块：在 R-ELAN 输出后施加零参数 3D 注意力。

    通过类继承保持层索引不变，预训练权重 100% 加载率。
    SimAM 无可训练参数，不存在过拟合风险。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_simam = SimAM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return self.post_simam(out)


class A2C2fEMASpatial(A2C2f):
    """A2C2f + EMA + SpatialAttention 融合模块（R5 实验）。

    在 R-ELAN 输出后依次施加:
      1. EMA（通道维度多尺度注意力）
      2. SpatialAttention（空间维度注意力，CBAM 空间分支）

    设计依据：
      - EMA 在 R1 中证明有效（+0.73% mAP50），负责通道维度加权
      - SpatialAttention 负责空间维度加权，与 EMA 维度正交
      - 二者级联形成"通道→空间"递进注意力
      - 参数增量极低：EMA ~16K + SpatialAttention ~50 参数

    论文来源：
      - EMA: ICASSP 2023 / FEI-YOLO (Agronomy 2024)
      - SpatialAttention: CBAM (ECCV 2018)

    Args:
        继承 A2C2f 全部参数。
        kernel_size: SpatialAttention 卷积核大小，默认 7。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_ema = EMA(c2, c2)
        self.post_spatial = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        out = self.post_ema(out)
        return self.post_spatial(out)


class ECALayer(nn.Module):
    """ECA-Net: Efficient Channel Attention (CVPR 2020).

    通过 1D 自适应卷积捕获局部通道交互，避免 FC 层的降维信息损失。
    纯通道注意力，与 EMA 的空间门控完全正交。

    论文: Wang Q, Wu B, Zhu P, et al.
          "ECA-Net: Efficient Channel Attention for Deep CNNs", CVPR 2020.

    Args:
        channels: 输入通道数
        gamma: 核大小自适应参数，默认 2
        b: 核大小自适应偏移，默认 1
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs(math.log2(channels) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2, CVPR 2023).

    通过全局 L2 范数归一化促进通道多样性。零 sigmoid，零级联风险。
    gamma/beta 初始化为 0 → 训练初期等价于恒等映射。

    论文: Woo S, Debnath S, Hu R, et al.
          "ConvNeXt V2: Co-designing and Scaling ConvNets with MAE", CVPR 2023.

    Args:
        channels: 输入通道数
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class A2C2fECA(A2C2f):
    """A2C2f + ECA 融合模块：在 R-ELAN 输出后施加纯通道注意力。

    ECA 不含空间操作，与 A2C2f 内部的 Area Attention 正交。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_eca = ECALayer(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return self.post_eca(out)


class A2C2fEMAECA(A2C2f):
    """A2C2f + EMA + ECA 融合模块：分组空间注意力 + 全局通道注意力。

    EMA 负责空间维度门控（per-group），ECA 负责通道维度标定（global）。
    ECA 的 sigmoid 作用于通道标量 (B,C,1,1)，不与 EMA 的空间 sigmoid 级联。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_ema = EMA(c2, c2)
        self.post_eca = ECALayer(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        out = self.post_ema(out)
        return self.post_eca(out)


class A2C2fEMAGRN(A2C2f):
    """A2C2f + EMA + GRN 融合模块：分组空间注意力 + 全局通道多样性正则化。

    EMA 负责空间维度门控，GRN 负责促进通道多样性。
    GRN 完全不含 sigmoid/tanh（纯线性+残差），零级联风险。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_ema = EMA(c2, c2)
        self.post_grn = GRN(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        out = self.post_ema(out)
        return self.post_grn(out)


# ── Triplet Attention (WACV 2021) ────────────────────────────────────────


class _ZPool(nn.Module):
    """Z-Pool: 沿通道维度拼接 max 和 avg，输出 2 通道。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.max(dim=1, keepdim=True).values,
                          x.mean(dim=1, keepdim=True)], dim=1)


class _AttentionGate(nn.Module):
    """单个注意力分支：Z-Pool → 7×7 Conv → BN → Sigmoid。"""

    def __init__(self) -> None:
        super().__init__()
        self.compress = _ZPool()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.bn(self.conv(self.compress(x))))


class TripletAttention(nn.Module):
    """Triplet Attention: 三分支跨维度注意力（WACV 2021）。

    通过维度旋转(permute)让 7×7 卷积分别建模 (C×H)、(C×W)、(H×W) 交互，
    三分支取均值融合。无通道数相关参数，总参数量恒定 ~300。

    论文: Misra D, Nalamada T, Arasanipalai A U, Hou Q.
          "Rotate to Attend: Convolutional Triplet Attention Module", WACV 2021.
    代码: https://github.com/LandskapeAI/triplet-attention
    """

    def __init__(self) -> None:
        super().__init__()
        self.ch_h = _AttentionGate()  # C×H 分支
        self.ch_w = _AttentionGate()  # C×W 分支
        self.hw = _AttentionGate()    # H×W 分支（空间注意力）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分支 1: (B,C,H,W) → 旋转为 (B,H,C,W) → 注意力 → 转回
        x_perm1 = x.permute(0, 2, 1, 3)
        out1 = self.ch_h(x_perm1).permute(0, 2, 1, 3)

        # 分支 2: (B,C,H,W) → 旋转为 (B,W,H,C) → 注意力 → 转回
        x_perm2 = x.permute(0, 3, 2, 1)
        out2 = self.ch_w(x_perm2).permute(0, 3, 2, 1)

        # 分支 3: (B,C,H,W) 直接做空间注意力 (H×W)
        out3 = self.hw(x)

        return (out1 + out2 + out3) / 3.0


class A2C2fTriplet(A2C2f):
    """A2C2f + Triplet Attention 融合模块。

    在 R-ELAN 输出后施加跨维度三分支注意力，
    同时捕获通道-空间交互。参数增量恒定 ~300。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_triplet = TripletAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return self.post_triplet(out)


# ── MSCALite (SegNeXt NeurIPS 2022 简化版) ───────────────────────────────


class MSCALite(nn.Module):
    """轻量多尺度条形卷积注意力（SegNeXt 简化版）。

    DW 5×5 局部上下文 → 三组并行非对称 DW strip conv 捕获方向性长距离依赖 →
    相加 → 1×1 Conv 通道混合 → sigmoid 门控。

    条形卷积天然匹配细长豆荚的长轴/短轴方向。
    DW-only ~14K @128ch, ~56K @512ch。

    论文: Guo M-H, Lu C-Z, Hou Q, et al.
          "SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation",
          NeurIPS 2022.
    代码: https://github.com/Visual-Attention-Network/SegNeXt

    Args:
        channels: 输入/输出通道数
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # 局部上下文: DW 5×5
        self.dw5x5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2,
                                groups=channels, bias=False)

        # 三组并行非对称 strip DW conv（捕获 3 种尺度方向性依赖）
        # Strip 1: 1×7 + 7×1 (小尺度)
        self.strip1_h = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3),
                                   groups=channels, bias=False)
        self.strip1_v = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0),
                                   groups=channels, bias=False)
        # Strip 2: 1×11 + 11×1 (中尺度)
        self.strip2_h = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5),
                                   groups=channels, bias=False)
        self.strip2_v = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0),
                                   groups=channels, bias=False)
        # Strip 3: 1×21 + 21×1 (大尺度)
        self.strip3_h = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10),
                                   groups=channels, bias=False)
        self.strip3_v = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0),
                                   groups=channels, bias=False)

        # 通道混合: 1×1 Conv
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 局部上下文
        attn = self.dw5x5(x)

        # 三组 strip conv 并行 → 相加
        s1 = self.strip1_v(self.strip1_h(attn))
        s2 = self.strip2_v(self.strip2_h(attn))
        s3 = self.strip3_v(self.strip3_h(attn))
        attn = s1 + s2 + s3

        # 通道混合 → sigmoid 门控
        attn = self.sigmoid(self.proj(attn))
        return x * attn


class A2C2fMSCA(A2C2f):
    """A2C2f + MSCALite 融合模块。

    在 R-ELAN 输出后施加多尺度条形卷积注意力，
    通过方向性非对称 DW conv 捕获细长目标的长轴/短轴依赖。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_msca = MSCALite(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return self.post_msca(out)
