import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# 假设你已经克隆了仓库，这里导入仓库中的模型定义
# 如果报错，请确保你在 pytorch-AdalN 根目录下运行，或者调整 import 路径
import net

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transfer(net, vgg, decoder, content, style, alpha=1.0):
    # 此函数模拟 AdaIN 的前向传播过程并支持 alpha 插值
    assert (0.0 <= alpha <= 1.0)
    
    # 1. Encode
    content_f = vgg(content)
    style_f = vgg(style)
    
    # 2. AdaIN with Alpha (题目核心要求 [cite: 49])
    # feat = AdaIN(content_f, style_f)
    # feat = alpha * feat + (1 - alpha) * content_f
    # 由于我们需要调用模型内部的 AdaIN，这里根据 naoto0804 的实现逻辑：
    # 该仓库通常在 decoder 或者 net forward 中实现。
    # 这里我们手动调用 net.adaptive_instance_normalization
    
    # 计算 AdaIN 特征
    feat = net.adaptive_instance_normalization(content_f, style_f)
    
    # 加权融合 [cite: 49]
    feat = feat * alpha + content_f * (1 - alpha)
    
    # 3. Decode
    return decoder(feat)

def run_bupt_experiment(content_path, style_path, output_path, model_path, alphas):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义网络并加载模型
    # 假设 VGG 权重路径是 'models/vgg_normalised.pth'
    # 如果您的 VGG 权重在其他位置，请修改此路径
    vgg_path = 'models/vgg_normalised.pth'
    
    # 实例化网络
    # 这里的 Net 类来自于 from model import Net
    net_model = Net(vgg_path)
    net_model.decoder.load_state_dict(torch.load(model_path))
    net_model.to(device)
    net_model.eval()

    decoder = net_model.decoder
    vgg = net_model.vgg
    
    vgg.to(device)
    decoder.to(device)

    # 处理图像
    content_tf = test_transform(512, False)
    style_tf = test_transform(512, False)

    content = content_tf(Image.open(content_path)).unsqueeze(0).to(device)
    style = style_tf(Image.open(style_path)).unsqueeze(0).to(device)

    # 循环不同的 alpha 值
    for alpha in alphas:
        with torch.no_grad():
            output = style_transfer(net_model, vgg, decoder, content, style, alpha)
        
        out_name = Path(output_path).stem + f"_alpha_{alpha}.jpg"
        save_image(output, out_name)
        print(f"Saved: {out_name} with alpha={alpha}")

if __name__ == '__main__':
    # 示例配置
    # 请修改为你的实际路径
    # 你的北邮照片路径
    CONTENT_IMG = "bupt_photos/bupt_spot1.jpg" 
    # 你的风格图片路径
    STYLE_IMG = "input/style/woman_with_hat_matisse.jpg" 
    # 训练好的 Decoder 模型路径
    MODEL_PATH = "decoder_iter_10000.pth" 
    

    alphas = [0.2, 0.5, 0.8]
    
    run_bupt_experiment(CONTENT_IMG, STYLE_IMG, "output_result.jpg", MODEL_PATH, alphas)