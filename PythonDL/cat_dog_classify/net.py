import torch.nn as nn
from torchvision import models  # torchvision.models：PyTorch 视觉库，内置大量预训练模型（如 ResNet、VGG、ViT 等）
from torchvision.models import ResNet101_Weights  # 专门加载 ResNet101 官方预训练权重
'''
这是一个标准的迁移学习分类网络，核心流程：
1.加载在 ImageNet 上预训练好的 ResNet101，复用其强大的通用图像特征提取能力
2.替换最后一层全连接层，适配自定义分类任务的类别数
3.开启全参数微调，让模型在自定义数据集上进一步优化
4.前向传播直接复用 ResNet 的逻辑，快速完成分类任务
'''
class ResNet101Classifier(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        # 加载 torchvision 预训练的 ResNet101 模型，权重是在 ImageNet（1400万+图像、1000类）上训练好的
        self.net = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Linear(2048, class_num)  # 替换最后一层全连接层（迁移学习核心, 最后一层名fc）
        '''
        原理：预训练 ResNet101 的最后一层 fc 是 nn.Linear(2048, 1000)，适配 ImageNet 的 1000 类分类任务
        操作：将其替换为 nn.Linear(2048, class_num)，2048 是 ResNet101 最后一个卷积层的输出特征维度，class_num 是自定义任务的类别数，实现 “预训练特征提取 + 自定义分类” 的迁移学习
        作用：让模型适配你的具体任务，同时保留预训练的通用特征提取能力
        '''
        for param in self.net.parameters():  # 设置所有参数可训练（全参数微调）
            param.requires_grad = True

    def forward(self, x):
        output = self.net(x)
        return output  # 形状: [batch_size, class_num]（每个样本对应 class_num 个类别的概率 logits）