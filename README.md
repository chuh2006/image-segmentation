# 图像和视频分割脚本

## 文件结构
- `src`
  - `img-seg.py`
  - `video-seg.py`

## img-seg.py

### 功能
`img-seg.py` 脚本使用 YOLO 模型对输入图像进行分割，并将分割结果显示和保存。该脚本可以处理单张图像，并将分割后的图像以网格形式显示。
`video-seg.py` 脚本使用 YOLO 模型对输入视频进行分割，并将分割结果保存为新的视频文件。该脚本可以处理视频文件，并在每一帧中进行分割操作。

### 使用方法
1. 确保已安装必要的依赖库，如 `opencv`, `pytorch`, `ultralytics`, `numpy` 和 `matplotlib`。
2. 文件中的视频或图片路径以及模型路径。

其中`opencv`, `numpy`, `matplotlib`可用以下方法安装:
```sh
pip install {name of lib}
```
`pytorch`需要前往[PyTorch官网](https://pytorch.org/get-started/locally/)根据设备安装。
