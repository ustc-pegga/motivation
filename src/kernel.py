import torch
import torch.nn as nn
import torchvision.models as models

# 创建原始MobileNet模型
original_model = models.mobilenet_v2(pretrained=True)

# 创建目标模型（5x5核大小）
target_model = models.mobilenet_v2(pretrained=False)
target_model.features[1][0] = nn.Conv2d(
    in_channels=32,  # 假设原始模型输入通道数为32
    out_channels=32, 
    kernel_size=5, 
    stride=1, 
    padding=2, 
    groups=32, 
    bias=False
)

# 迁移权重
target_model.features[1][0].weight.data[:, :32//target_model.features[1][0].groups, :, :] = original_model.features[1][0].weight.data

# 打印目标模型
print(target_model)
