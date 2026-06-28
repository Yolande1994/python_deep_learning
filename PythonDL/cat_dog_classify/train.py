import os
os.environ['TORCH_HOME'] = './pretrain_weights'
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision import datasets

from net import ResNet101Classifier
from config import args

def train_and_val(dataloader, model, epoch, learning_rate):
    train_dataloader, val_dataloader = dataloader
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    # 训练阶段
    model.train()  # 切换到训练模式
    train_loss = 0.0
    train_acc = 0.0
    for batch_x, batch_y in train_dataloader:  # 训练集
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        optimizer.zero_grad()
        logits = model(batch_x)  # 前向传播 (batch_size, class_num)
        loss = loss_func(logits, batch_y)  # 计算损失
        loss.backward()   # 反向传播
        optimizer.step()  # 更新参数

        train_loss += loss.item()         # 累计损失
        pred = torch.argmax(logits, dim=-1)
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item() # 累计正确样本数
    # 计算并打印训练指标
    avg_train_loss = train_loss / len(train_dataloader)        # 平均损失：总损失 / batch 数量
    avg_train_acc = train_acc / len(train_dataloader.dataset)  # 平均准确率：总正确数 / 总样本数
    print(f'Epoch {epoch} | Train Loss: {avg_train_loss:.6f}, Train Acc: {avg_train_acc:.6f}')

    # 验证阶段
    model.eval()  # 切换到评估模式（关闭 dropout/bn 训练行为）
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():  # 关闭梯度计算，节省显存
        for batch_x, batch_y in val_dataloader:  # 验证集
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            logits = model(batch_x)
            loss = loss_func(logits, batch_y)

            val_loss += loss.item()
            pred = torch.argmax(logits, dim=-1)
            val_correct = (pred == batch_y).sum()
            val_acc += val_correct.item()
    # 计算并打印验证指标
    avg_val_loss = val_loss / len(val_dataloader)
    avg_val_acc = val_acc / len(val_dataloader.dataset)
    print(f'Epoch {epoch} | Val Loss: {avg_val_loss:.6f}, Val Acc: {avg_val_acc:.6f}')

    return avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc


if __name__ == '__main__':
    class_num = 2
    model = ResNet101Classifier(class_num).cuda()
    batch_size = args.per_batch_size  # 从 config.py 导入的 args 中，读取批次大小配置

    # 训练集 transform  （transforms.Compose 是 torchvision 提供的数据预处理流水线工具，按顺序执行列表中的操作）
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),       # 将输入图像等比例/强制缩放到 224×224 分辨率 (ResNet 系列模型的默认输入尺寸是 224×224)
        transforms.RandomHorizontalFlip(),  # 以 50% 的概率随机水平翻转图像, 增加训练数据的多样性，防止模型过拟合，提升模型泛化能力
        transforms.ToTensor(),  # 1.将 PIL图像/NumPy数组 转换为 PyTorch 张量， 2.同时将像素值从 [0, 255] 归一化到 [0, 1]， 3.并调整维度从 (H, W, C) 到 (C, H, W)（张量标准格式）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对张量进行标准化（[0, 1] 标准化到 [-1, 1]）（预处理，进网络之前）
    ])
    '''
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    作用：对张量进行标准化，公式为：output[channel] = (input[channel] - mean[channel]) / std[channel]
    (.5, .5, .5)：RGB 三通道的均值（mean）
    (.5, .5, .5)：RGB 三通道的标准差（std）
    效果：将像素值从 [0, 1] 映射到 [-1, 1]，加速模型收敛，提升训练稳定性。
    补充：若使用 ImageNet 预训练权重，主流标准归一化是 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])，更适配预训练模型的特征分布。
    '''
    # 验证集 transform （无数据增强，仅基础处理）
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 加载数据集
    train_datasets = datasets.ImageFolder(args.train_dir, transform = train_transform )
    val_datasets = datasets.ImageFolder(args.val_dir, transform = val_transform)
    # 封装DataLoader
    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False)
    # 打包成一个元组
    dataloader = (train_dataloader, val_dataloader)

    for epoch in range(1, args.max_epoch+1):
        print('epoch {} -----------------------'.format(epoch))
        # 学习率衰减逻辑
        for step in args.lr_steps:
            if epoch == step:
                args.lr *= args.decay_coeff
                print('当前学习率:', args.lr)
                break
        # 执行单 epoch 训练+验证
        train_and_val(dataloader, model, epoch, args.lr)

    # 模型保存前的目录检查
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)  # 如果目录不存在，自动创建该目录，避免后续 torch.save 因路径不存在而报错
    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(args.save_dir, "model_state_dict.pth"))