# AdaIN 风格迁移项目

本项目基于 PyTorch 实现了 [AdaIN (Adaptive Instance Normalization)](https://arxiv.org/abs/1703.06868) 算法，用于进行快速、高质量的艺术风格迁移。

## 项目简介

该项目可以：
1.  使用自定义的内容图像和风格图像训练新的风格迁移模型。
2.  使用已训练好的模型，对任意图片进行风格化处理。
3.  支持调整风格化程度（alpha 值），以控制内容与风格的融合比例。

---

## 环境设置

建议使用 Conda 创建独立的 Python 环境。

1.  **创建 Conda 环境**
    ```bash
    conda create -n adain python=3.8
    conda activate adain
    ```

2.  **安装依赖**
    本项目主要依赖于 `torch`, `torchvision`, `Pillow`, `tqdm` 和 `tensorboardX`。您可以通过 pip 安装它们：
    ```bash
    pip install torch torchvision
    pip install Pillow tqdm tensorboardX
    ```
    *注意：请根据您的硬件情况（是否支持 CUDA）安装相应版本的 PyTorch。*

---

## 如何使用

### 1. 训练新的风格模型

我们提供了一个自动化脚本 `run_training.py` 来简化训练流程，该脚本可以针对多种风格进行训练。

**步骤：**
1.  将您的内容图片放入 `pytorch-AdaIN/input/content/` 目录。
2.  将您想学习的风格图片放入 `pytorch-AdaIN/input/style/` 目录。
3.  修改 `run_training.py` 脚本，在 `STYLE_IMAGES` 列表中指定您想要训练的风格图片路径。
4.  运行训练脚本：
    ```bash
    python run_training.py
    ```

训练好的模型（`.pth` 文件）将根据风格图片自动命名，并保存在 `pytorch-AdaIN/experiments/` 目录下。训练过程的日志保存在 `pytorch-AdaIN/logs/` 目录下，可使用 TensorBoard 查看。

### 2. 进行风格迁移

`style_transfer_tasks.py` 脚本用于使用已训练好的模型执行一系列风格迁移任务。

**步骤：**
1.  确保您需要的解码器模型（`.pth` 文件）已存在于 `pytorch-AdaIN/experiments/` 目录下。
2.  将您想进行风格化的内容图片放入 `pytorch-AdaIN/myimage/` 目录。
3.  根据您的需求，修改 `style_transfer_tasks.py` 脚本中的文件路径和参数（例如 `alpha` 值）。
4.  运行脚本：
    ```bash
    python style_transfer_tasks.py
    ```

所有生成的风格化图片将保存在 `output/` 目录下的不同子文件夹中（`task2`, `task3`, `task4`）。

---

## 结果展示

以下是使用本项目的模型生成的一些风格迁移效果。

**内容图像:** `pytorch-AdaIN/myimage/1.jpg`
**风格图像:** `pytorch-AdaIN/input/style/la_muse.jpg`

**生成结果:**

![Stylized Result](pytorch-AdaIN/output/task3/1_stylized_la_muse.jpg)

---

## 主要文件说明

-   `train_task2.py`: 核心训练脚本，负责训练解码器。
-   `run_training.py`: 自动化训练的辅助脚本，可批量处理多种风格。
-   `style_transfer_tasks.py`: 用于执行所有风格迁移任务的推理脚本。
-   `net.py`: 定义了 VGG 编码器和解码器的网络结构。
-   `pytorch-AdaIN/experiments/`: 存放训练好的解码器模型。
-   `output/`: 存放最终生成的风格化图像。

