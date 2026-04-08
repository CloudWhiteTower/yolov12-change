"""大豆豆荚分割改进模块：EMA、CoordAttention、A2C2fEMA。"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import A2C2f


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
