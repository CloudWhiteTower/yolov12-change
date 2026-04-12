# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics modules.
"""

from .block import (
    C1, C2, C2PSA, C3, C3TR, CIB, DFL, ELAN1, PSA, SPP, SPPELAN, SPPF,
    AConv, ADown, Attention, BNContrastiveHead, Bottleneck, BottleneckCSP,
    C2f, C2fAttn, C2fCIB, C2fPSA, C3Ghost, C3k2, C3x, CBFuse, CBLinear,
    ContrastiveHead, GhostBottleneck, HGBlock, HGStem, ImagePoolingAttn,
    Proto, RepC3, RepNCSPELAN4, RepVGGDW, ResNetLayer, SCDown, TorchVision,
    A2C2f,
)
from .conv import (
    CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv,
    DWConvTranspose2d, Focus, GhostConv, Index, LightConv, RepConv, SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect, v10Detect
from .transformer import (
    AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer,
    LayerNorm2d, MLPBlock, MSDeformAttn, TransformerBlock,
    TransformerEncoderLayer, TransformerLayer,
)
# 自定义量化模块
from .polar_quant import PolarQuantTorch
from .quant_attn import AAttnQuant, ABlockQuant, A2C2fQuant
# 大豆豆荚分割改进模块
from .pod_modules import EMA, CoordAttention, A2C2fEMA, SimAM, A2C2fSimAM, A2C2fEMASpatial, ECALayer, GRN, A2C2fECA, A2C2fEMAECA, A2C2fEMAGRN, TripletAttention, MSCALite, A2C2fTriplet, A2C2fMSCA, A2C2fTripletMSCA, A2C2fMSCATriplet, A2C2fMSCATripletParallel

__all__ = (
    "Conv", "Conv2", "LightConv", "RepConv", "DWConv", "DWConvTranspose2d",
    "ConvTranspose", "Focus", "GhostConv", "ChannelAttention", "SpatialAttention",
    "CBAM", "Concat", "TransformerLayer", "TransformerBlock", "MLPBlock",
    "LayerNorm2d", "DFL", "HGBlock", "HGStem", "SPP", "SPPF", "C1", "C2", "C3",
    "C2f", "C3k2", "SCDown", "C2fPSA", "C2PSA", "C2fAttn", "C3x", "C3TR", "C3Ghost",
    "GhostBottleneck", "Bottleneck", "BottleneckCSP", "Proto", "Detect", "Segment",
    "Pose", "Classify", "TransformerEncoderLayer", "RepC3", "RTDETRDecoder", "AIFI",
    "DeformableTransformerDecoder", "DeformableTransformerDecoderLayer", "MSDeformAttn",
    "MLP", "ResNetLayer", "OBB", "WorldDetect", "v10Detect", "ImagePoolingAttn",
    "ContrastiveHead", "BNContrastiveHead", "RepNCSPELAN4", "ADown", "SPPELAN",
    "CBFuse", "CBLinear", "AConv", "ELAN1", "RepVGGDW", "CIB", "C2fCIB",
    "Attention", "PSA", "TorchVision", "Index", "A2C2f",
    # 自定义量化模块
    "PolarQuantTorch", "AAttnQuant", "ABlockQuant", "A2C2fQuant",
    # 大豆豆荚分割改进模块
    "EMA", "CoordAttention", "A2C2fEMA", "SimAM", "A2C2fSimAM",
    "ECALayer", "GRN", "A2C2fECA", "A2C2fEMAECA", "A2C2fEMAGRN",
    "TripletAttention", "MSCALite", "A2C2fTriplet", "A2C2fMSCA",
    "A2C2fTripletMSCA", "A2C2fMSCATriplet", "A2C2fMSCATripletParallel",
)
