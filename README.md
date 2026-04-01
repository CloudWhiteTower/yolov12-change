# yolov12-change

YOLOv12 with PolarQuant feature compression for memory-efficient training.

## Installation

```bash
pip install git+https://github.com/CloudWhiteTower/yolov12-change.git
```

## Features

- **PolarQuantTorch**: Vector quantization based on TurboQuant algorithm
- **A2C2fQuant**: Quantized R-ELAN module for YOLOv12
- **AAttnQuant**: Area Attention with feature compression

## Usage

```python
from ultralytics import YOLO

# Load quantized model
model = YOLO('yolov12-quant.yaml')
model.train(data='dataset.yaml', epochs=100)
```

## Compression Ratio

| bit_width | Compression | Accuracy Loss |
|-----------|-------------|---------------|
| 4-bit | 3.8x | Minimal |
| 3-bit | 4.6x | Moderate |
| 2-bit | 6.4x | Higher |

## License

AGPL-3.0