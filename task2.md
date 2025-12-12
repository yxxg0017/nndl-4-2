根据你提供的实验指导书文件，以下是**任务二：图像风格迁移**的详细实施步骤与对应的算法原理。

此任务的核心是复现并应用基于 **AdaIN (Adaptive Instance Normalization)** 的任意风格迁移模型。

---

### 一、 核心算法原理 (Principles)

[cite_start]本实验基于 Huang 等人提出的 AdaIN 算法 [cite: 189]，其核心思想认为图像的**风格（Style）**主要体现在特征空间中的统计特性（均值和方差）上，而**内容（Content）**则由特征的空间结构决定。

#### 1. 网络架构
[cite_start]整个网络由三部分组成 [cite: 112, 170]：
* **编码器 (Encoder, $f$)**: 使用预训练好的 VGG-19 网络（通常取前几层），参数固定不更新。它负责提取图像的特征。
* **AdaIN 层**: 位于编码器和解码器之间，用于将内容图像的特征统计特性对齐到风格图像的统计特性。
* **解码器 (Decoder, $g$)**: 一个镜像于编码器的网络，负责将经过 AdaIN 处理后的特征映射回图像空间，生成最终的风格化图片 $T(c,s)$。

#### 2. AdaIN 公式
[cite_start]AdaIN 模块不需要学习参数，它通过简单的统计变换实现风格融合 [cite: 168, 169]：
$$AdaIN(x,y)=\sigma(y)\left(\frac{x-\mu(x)}{\sigma(x)}\right) + \mu(y)$$
* $x$: 内容图像的特征 (Content feature)。
* $y$: 风格图像的特征 (Style feature)。
* $\mu(\cdot), \sigma(\cdot)$: 分别表示通道维度上的均值和标准差。
* **原理**: 先将内容特征 $x$ 标准化（减均值除方差），消除其原有的风格信息；然后应用风格特征 $y$ 的均值和方差，注入新的风格信息。

#### 3. 损失函数
[cite_start]训练仅更新 Decoder 的参数，损失函数 $L$ 由两部分加权组成 [cite: 179]：
$$L = L_c + \lambda L_s$$

* [cite_start]**内容损失 ($L_c$)**: 保证生成图像 $g(t)$ 的特征与 AdaIN 输出的特征 $t$ 一致 [cite: 180]。
    $$L_c = ||f(g(t)) - t||_2$$
* [cite_start]**风格损失 ($L_s$)**: 保证生成图像的统计特性（均值和方差）与风格图像一致。通常在 VGG 的多个层（如 relu1_1, relu2_1 等）上计算 [cite: 181]。
    $$L_s = \sum_{i=1}^{L} ||\mu(\phi_i(g(t))) - \mu(\phi_i(s))||_2 + \sum_{i=1}^{L} ||\sigma(\phi_i(g(t))) - \sigma(\phi_i(s))||_2$$

---

### 二、 实施步骤 (Implementation Steps)

根据实验书要求，你需要完成以下四个具体任务：

#### 步骤 1：环境搭建与模型复现 (对应任务 1)
1.  **准备数据集**:
    * [cite_start]**Content**: MS-COCO 数据集 (train2014) [cite: 109, 184]。
    * [cite_start]**Style**: WikiArt 数据集 [cite: 111, 184]。
2.  **获取代码**:
    * [cite_start]克隆指定的开源项目：`https://github.com/naoto0804/pytorch-AdaIN.git` [cite: 114, 185]。
3.  **配置环境**: 确保安装 PyTorch, TensorBoard 等依赖库。
4.  **开始训练**:
    * 运行训练脚本（通常是 `train.py` 或类似文件）。
    * [cite_start]**关键点**: 必须开启 TensorBoard 记录 Loss 变化，后续需截图放入报告 [cite: 114]。
    * **保存模型**: 代码通常会定期保存 `.pth` 或 `.model` 文件，你需要保留不同迭代次数的模型文件以备后用。

#### 步骤 2：训练监控与阶段性结果生成 (对应任务 2)
1.  [cite_start]**训练时长要求**: 总迭代次数 (Iterations) 不得少于 **10,000 次** [cite: 115, 186]。
2.  **特定节点采样**: 找出迭代次数分别达到总次数 **10%, 50%, 80%, 100%** 时的模型权重文件。
3.  **生成对比图**:
    * **Content图片**: 使用项目自带的 `input/content/cornell.jpg`。
    * **Style图片**: 使用项目自带的 `input/style/woman_with_hat_matisse.jpg`。
    * [cite_start]分别加载上述 4 个阶段的模型权重，运行测试脚本（如 `test.py`），生成 4 张风格迁移结果图并展示在报告中 [cite: 115, 186]。

#### 步骤 3：北邮校园特色风格迁移 (对应任务 3)
1.  **拍摄素材**:
    * 选取北京邮电大学 (BUPT) 的 **2个特色景点**。
    * [cite_start]拍摄 **自己与景点的合照**，得到 2 张 Content 图像 [cite: 126, 187]。
2.  **选择风格**: 选取任意一张你喜欢的艺术风格图片 (Style)。
3.  **推理生成**:
    * 使用训练好的最终模型（100% 进度的模型）。
    * 将你的合照作为 Input Content，艺术图作为 Input Style，生成 2 张结果图。
4.  **展示**: 报告中需包含：原图（人+景）、风格图、生成的迁移图。

#### 步骤 4：风格权重调节 (对应任务 4)
此步骤研究内容与风格的平衡。AdaIN 的输出特征 $t$ 与原始内容特征 $f(c)$ 可以进行加权插值：
$$t_{new} = \alpha \cdot AdaIN(f(c), f(s)) + (1-\alpha) \cdot f(c)$$
* $\alpha$ (Alpha): 风格权重。$\alpha=1$ 为完全风格化，$\alpha=0$ 为完全保留原始内容。

1.  [cite_start]**选择素材**: 从步骤 3 的合照中任选一张，并选择一张**新**的风格图片 [cite: 127, 188]。
2.  [cite_start]**设置 $\alpha$ 值 (根据学号)** [cite: 188]：
    * **学号尾号为偶数**: $\alpha$ 分别设置为 **0.3, 0.6, 0.9**。
    * **学号尾号为奇数**: $\alpha$ 分别设置为 **0.2, 0.5, 0.8**。
3.  **修改/运行代码**:
    * 在测试脚本中找到控制 alpha (或 content-style trade-off) 的参数。
    * 输入上述三个不同的数值，生成三张结果图。
4.  **观察结果**: 随着 $\alpha$ 增大，图像应越来越像风格图；随着 $\alpha$ 减小，图像应越来越像原照片。

---

### 三、 实训平台操作提示 (Tips)
* **后台运行**: 由于训练时间较长，建议使用 `nohup` 命令将训练挂在后台，防止网络中断导致训练停止。
    * [cite_start]命令示例: `nohup python -u train.py ... > train.log 2>&1 &` [cite: 131]。
* **查看日志**: 使用 `tail -f train.log` 查看实时训练进度。
* [cite_start]**会话保持**: 可以使用 `tmux` 工具来管理会话，避免误关窗口杀掉进程 [cite: 133, 134]。

Would you like me to explain specifically which files in the `pytorch-AdaIN` repository usually correspond to the encoder/decoder definitions to help you read the code?