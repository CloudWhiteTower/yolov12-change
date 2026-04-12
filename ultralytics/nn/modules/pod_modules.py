"""大豆豆荚分割改进模块：EMA、CoordAttention、A2C2fEMA、A2C2fEMASpatial、ECALayer、GRN、A2C2fECA、A2C2fEMAECA、A2C2fEMAGRN、TripletAttention、MSCALite、A2C2fTriplet、A2C2fMSCA、A2C2fTripletMSCA、A2C2fMSCATriplet、A2C2fMSCATripletParallel。"""

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


# ── R8 融合模块：MSCA × Triplet ─────────────────────────────────────


class A2C2fTripletMSCA(A2C2f):
    """A2C2f → TripletAttention → MSCALite 串行融合。

    Triplet 在前：三分支加权均值（无 sigmoid 截断）保留完整幅度，
    MSCA 在后：单次 sigmoid 门控，避免双 sigmoid 级联衰减。
    符合 CBAM (ECCV 2018) 粗到细注意力级联范式。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_triplet = TripletAttention()
        self.post_msca = MSCALite(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        out = self.post_triplet(out)
        return self.post_msca(out)


class A2C2fMSCATriplet(A2C2f):
    """A2C2f → MSCALite → TripletAttention 串行融合。

    MSCA 在前提供方向性空间增强 + sigmoid 门控，
    Triplet 在后做跨维度交互（三分支均值）。
    存在 MSCA sigmoid 截断后 Triplet 幅度收窄风险，作为对照实验。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_msca = MSCALite(c2)
        self.post_triplet = TripletAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        out = self.post_msca(out)
        return self.post_triplet(out)


class A2C2fMSCATripletParallel(A2C2f):
    """A2C2f → 并行门控 MSCA + Triplet 融合。

    可学习标量 α（初始 0.5），经 sigmoid 映射后加权：
      out = σ(α) · MSCA(feat) + (1-σ(α)) · Triplet(feat)
    允许网络自适应发现两路最佳融合比例。参数增量仅 +1 scalar。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_msca = MSCALite(c2)
        self.post_triplet = TripletAttention()
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        w = torch.sigmoid(self.alpha)
        return w * self.post_msca(out) + (1.0 - w) * self.post_triplet(out)


# ── R9 新模块 ────────────────────────────────────────────────────────


class DropBlock2D(nn.Module):
    """DropBlock: structured dropout for convolutional features (NeurIPS 2018).

    在特征图上 drop 连续 block_size × block_size 区域，
    强制网络学习冗余表征，缓解过拟合。

    论文: Ghiasi et al., "DropBlock: A regularization method for
          convolutional networks", NeurIPS 2018.

    Args:
        block_size: drop 区域边长，默认 3
        keep_prob: 保留概率，默认 0.9（训练时生效，推理时恒等）
    """

    def __init__(self, block_size: int = 3, keep_prob: float = 0.9):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.keep_prob >= 1.0:
            return x
        _, _, h, w = x.shape
        # gamma: 每个位置被选为 block 中心的概率
        valid_h = max(h - self.block_size + 1, 1)
        valid_w = max(w - self.block_size + 1, 1)
        gamma = (1.0 - self.keep_prob) / (self.block_size ** 2)
        gamma *= (h * w) / (valid_h * valid_w)
        # 采样 mask（单通道广播到全部通道）
        mask = (torch.rand(x.shape[0], 1, h, w, device=x.device) < gamma).float()
        # 扩展为 block 区域
        pad = self.block_size // 2
        mask = torch.nn.functional.max_pool2d(
            mask, kernel_size=self.block_size, stride=1, padding=pad,
        )
        mask = 1.0 - mask
        # 归一化保持期望不变
        count = mask.numel()
        count_ones = mask.sum()
        if count_ones == 0:
            return x * 0.0
        return x * mask * (count / count_ones)


class FcaNetLite(nn.Module):
    """轻量 FcaNet: 多频谱通道注意力 (ICCV 2021 简化版).

    将 SE-Net 的 GAP（DC 分量）推广为多频 DCT 分量池化，
    不同通道组使用不同频率基，捕获更丰富的频域信息。
    DCT 基固定不可学习（论文验证优于可学习版本）。

    论文: Qin Z, Zhang P, Wu F, Li X.
          "FcaNet: Frequency Channel Attention Networks", ICCV 2021.
    代码: https://github.com/cfzd/FcaNet

    Args:
        channels: 输入通道数
        reduction: FC 层通道压缩比，默认 16
        num_freq: 使用的 DCT 频率数量，默认 4
    """

    def __init__(self, channels: int, reduction: int = 16, num_freq: int = 4):
        super().__init__()
        self.channels = channels
        self.num_freq = min(num_freq, channels)
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # 多频谱池化：将通道均分为 num_freq 组，各组使用不同 DCT 基加权
        group_size = c // self.num_freq
        pooled_parts = []
        for i in range(self.num_freq):
            start = i * group_size
            end = start + group_size if i < self.num_freq - 1 else c
            xi = x[:, start:end, :, :]
            if i == 0:
                # DC 分量（等价于 GAP）
                pi = xi.mean(dim=[2, 3])
            else:
                # 第 i 号频率：cos(π·i·u/H) 加权后求均值
                freq_h = torch.cos(
                    math.pi * i * torch.arange(h, device=x.device, dtype=x.dtype) / h
                ).view(1, 1, h, 1)
                freq_w = torch.cos(
                    math.pi * i * torch.arange(w, device=x.device, dtype=x.dtype) / w
                ).view(1, 1, 1, w)
                pi = (xi * freq_h * freq_w).mean(dim=[2, 3])
            pooled_parts.append(pi)
        pooled = torch.cat(pooled_parts, dim=1)  # (B, C)
        attn = self.fc(pooled).view(b, c, 1, 1)
        return x * attn


class A2C2fMSCATripletParallelDB(A2C2f):
    """A2C2fMSCATripletParallel + DropBlock 正则化 (R9 实验).

    在 parallel 融合输出后施加 DropBlock，
    强制网络学习更鲁棒特征，缓解 best→final 衰减。

    论文: Ghiasi et al. NeurIPS 2018 + R8 parallel 架构.

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_msca = MSCALite(c2)
        self.post_triplet = TripletAttention()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.dropblock = DropBlock2D(block_size=3, keep_prob=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        w = torch.sigmoid(self.alpha)
        fused = w * self.post_msca(out) + (1.0 - w) * self.post_triplet(out)
        return self.dropblock(fused)


class A2C2fFcaNet(A2C2f):
    """A2C2f + FcaNetLite 频域通道注意力 (R9 实验).

    在 R-ELAN 输出后施加多频谱通道注意力，
    与 MSCA（空间方向性）、Triplet（跨维度旋转）完全正交。

    论文: Qin et al., "FcaNet: Frequency Channel Attention Networks",
          ICCV 2021.

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_fca = FcaNetLite(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return self.post_fca(out)


class A2C2fParallelFcaNet(A2C2f):
    """A2C2fMSCATripletParallel → FcaNetLite 串行 (R9 实验).

    先用 parallel 融合 MSCA+Triplet，再用 FcaNet 做频域通道重校准。
    空间注意力(parallel) + 频域通道注意力(FcaNet) 互补。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_msca = MSCALite(c2)
        self.post_triplet = TripletAttention()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.post_fca = FcaNetLite(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        w = torch.sigmoid(self.alpha)
        fused = w * self.post_msca(out) + (1.0 - w) * self.post_triplet(out)
        return self.post_fca(fused)


class A2C2fParallelDBFcaNet(A2C2f):
    """A2C2fMSCATripletParallel + DropBlock + FcaNetLite (R9 实验).

    三合一组合：parallel 融合 → DropBlock 正则化 → FcaNet 频域通道校准。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_msca = MSCALite(c2)
        self.post_triplet = TripletAttention()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.dropblock = DropBlock2D(block_size=3, keep_prob=0.9)
        self.post_fca = FcaNetLite(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        w = torch.sigmoid(self.alpha)
        fused = w * self.post_msca(out) + (1.0 - w) * self.post_triplet(out)
        fused = self.dropblock(fused)
        return self.post_fca(fused)


class A2C2fTriwayParallel(A2C2f):
    """MSCA + Triplet + FcaNet 三路 softmax 并行融合 (R9 实验).

    三个正交维度注意力并行：
      - MSCA: 空间方向性（条形卷积匹配细长目标）
      - Triplet: 跨维度旋转（C×H, C×W, H×W）
      - FcaNet: 频域通道（多频谱通道注意力）
    通过 softmax 归一化的三标量门控 [α, β, γ] 自适应融合。

    Args:
        继承 A2C2f 全部参数，无额外参数。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, a2: bool = True, area: int = 1,
                 residual: bool = False, mlp_ratio: float = 2.0, e: float = 0.5,
                 g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, a2, area, residual, mlp_ratio, e, g, shortcut)
        self.post_msca = MSCALite(c2)
        self.post_triplet = TripletAttention()
        self.post_fca = FcaNetLite(c2)
        self.gate = nn.Parameter(torch.zeros(3))  # [α, β, γ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        w = torch.softmax(self.gate, dim=0)
        return (w[0] * self.post_msca(out)
                + w[1] * self.post_triplet(out)
                + w[2] * self.post_fca(out))
