import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
'''
torchvision  →  PyTorch 视觉工具箱,专门做：图像分类、目标检测、分割、GAN 等 CV 任务
它包含 4 大核心功能
1.datasets：  下载 + 加载经典数据集（MNIST、CIFAR10、ImageNet 等）
2.models：    直接用现成训练好的模型（VGG、ResNet、MobileNet 等）
3.transforms：图片预处理、数据增强（裁剪、翻转、归一化）
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),     # 随机翻转
    transforms.ToTensor()                  # 转张量+归一化
4.utils：     画图、保存图片、制作网格图


torchvision.datasets.CIFARAR10()  →  自动下载+加载 CIFAR-10 图像分类数据集:
torchvision.datasets.数据集名(
    root='数据存放路径',   # 数据集保存的文件夹路径,没有会自动创建,第二次运行会自动检测，不再重复下载
    train=True,         # True → 加载训练集（5万张图片）   False → 加载测试集（1万张图片）
    download=False,     # True → 本地没有就自动下载       False → 不下载，只从本地读取
    transform=预处理方法  # 对图片做预处理  最常用：transform=torchvision.transforms.ToTensor()  功能：把图片→转为张量, 把像素值0~255→归一化到0~1, 把形状H×W×C→转为C×H×W（通道在前）
)
'''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_data(batch_size):
    # -------------------------- 训练集：添加数据增强 --------------------------
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),  # 随机裁剪(32x32, 填充4像素)，增强鲁棒性
        torchvision.transforms.RandomHorizontalFlip(),     # 随机水平翻转，模拟真实场景视角变化
        torchvision.transforms.ToTensor(),                 # 转张量 + 归一化到[0,1]
        # CIFAR-10 官方标准化参数（提升训练稳定性，加速收敛）
        torchvision.transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),  # RGB三通道均值
            std=(0.2023, 0.1994, 0.2010)    # RGB三通道标准差
)])
    # -------------------------- 测试集：仅标准化，不做增强 --------------------------
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010))])
    # 加载数据集（对应上方的数据预处理）
    train_dataset = torchvision.datasets.CIFAR10('./data/', train=True, download=False, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10('./data/', train=False, download=False, transform=transform_test)

    # 数据集下载或加载，转换为 PyTorch 可处理的张量格式（原始版）
    train_dataset = torchvision.datasets.CIFAR10('./data/', train=True, download=False, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10('./data/', train=False, download=False, transform=torchvision.transforms.ToTensor())
    # 用 DataLoader 准备数据集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 训练必须打乱，防止过拟合
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集不打乱，保证结果可复现
    print(f"批次长度: {len(train_loader)}",f"\n训练集总长度: {len(train_loader.dataset)}")
    print(train_loader.dataset,test_loader.dataset)
    return train_loader, test_loader


def show(dataloader):
    # 数据集可视化：从DataLoader取一批数据，以5×5网格展示25张图片并标注类别
    datas, labels = next(iter(dataloader))  # iter()：转为迭代器  next()：取出迭代器第一个批次数据
    for i in range(25):
        plt.subplot(5, 5, i+1)     # 创建5行5列的子图，定位到第i+1个位置
        data = datas[i].numpy()    # 张量转numpy数组，适配matplotlib
        data = data.transpose(1, 2, 0)  # 通道转换：PyTorch (C, H, W) → matplotlib (H, W, C)
        plt.imshow(data)
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10类别映射
        plt.title(f"{classes[labels[i]]}")  # 标注图片对应的类别标签
        plt.xticks([])  # 隐藏x轴刻度
        plt.yticks([])  # 隐藏y轴刻度
    plt.tight_layout()  # 统一调整子图间距，避免标签重叠
    plt.show()


# 定义网络结构： 轻量级 2层卷积 + 2层全连接的 CNN 模型，用于对 32×32 的 CIFAR-10 彩色图片进行 10 分类
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义卷积与池化部分 (提取图片视觉特征（边缘、纹理、颜色等），每个卷积核学习一种不同的特征)
        self.conv = nn.Sequential(  # 输入：[batch_size,3,32,32]（单样本维度：[3,32,32]）
            nn.Conv2d(3,8,3,1,1),   # 输入通道 3（RGB），输出通道 8，卷积核 3×3，步长 1，填充 1     尺寸变化: 3,32,32 → 8,32,32（尺寸不变，通道自定）
            nn.ReLU(),
            nn.MaxPool2d(2),        # 最大池化，池化核 2×2，步长默认 2（同窗口大小），下采样         尺寸变化: 8,32,32 → 8,16,16（尺寸减半，通道不变）
            nn.Conv2d(8,16,3,1,1),  # 输入通道 8（前层），输出通道 16，卷积核 3×3，步长 1，填充 1    尺寸变化: 8,16,16 → 16,16,16
            nn.ReLU(),
            nn.MaxPool2d(2)         #                                                      尺寸变化: 16,16,16 → 16,8,8（最终特征图尺寸）
        )                           # 3,32,32  →  16,8,8 （卷积层始终保持[C,H,W]）
        '''
        卷积尺寸计算
        公式：输出尺寸 = ⌊(n+2p−f)/s⌋+1  向下取整
        规律：3×3卷积核 + 步长1 + 填充1 = 形状不变  |  5×5卷积 + 步长1 + 填充2 = 形状不变  |  7×7卷积 + 步长1 + 填充3 = 形状不变
        规律：2×2池化核 + 默认步长2 = 形状一定变成原来的一半
        
        第一次卷积：n=32, p=1, f=3, s=1 → (32+2-3)/1 +1 = 32，尺寸不变
        第一次池化：n=32, f=2, s=2 → 32/2 = 16，尺寸减半
        第二次卷积：n=16, p=1, f=3, s=1 → (16+2-3)/1 +1 = 16，尺寸不变
        第二次池化：n=16, f=2, s=2 → 16/2 = 8，最终尺寸 8×8
        '''
        # 定义全连接部分  (将特征通过加权和映射为10个类别的分类结果，卷积层负责看懂图片，全连接层负责根据看懂的特征做类别判断)
        self.dense = nn.Sequential(
            nn.Linear(1024,256),    # 将 16×8×8=1024 的卷积层最终特征图 的展平后的1维特征向量，映射为 256 维隐藏层
            nn.ReLU(),
            nn.Dropout(0.5),       # 防止过拟合，可选
            nn.Linear(256,10)       # 将 256 维特征映射为 10 维输出，对应 10 个类别
        )                           # 1024个特征，对应 10组权重（每组1024个w）和 10个b，共同算出 10 个总分 → 取最大 → 得到 1 个类别

    def forward(self,x):
        x = self.conv(x)     # 实现卷积计算:   将输入图片送入卷积序列，完成特征提取，输出形状[batch_size, 16,8,8]
        x = x.flatten(1,-1)  # 展开特征图:    展平[batch_size, 16,8,8] → [batch_size, 1024]，适配全连接层输入
        x = self.dense(x)    # 实现全连接计算: 将展平的特征送入全连接序列，完成分类计算，输出[batch_size, 10]的 logits（每个类别的打分，越高概率越大）
        return x
        return F.softmax(x,dim=1)  # 【使用注意】：若使用CrossEntropyLoss损失函数，无需手动加F.softmax()，损失函数内部已包含softmax计算，加了会导致重复计算，冗余. 主流写法是直接返回x


def train(net, epochs, lr, batch_size):
    train_loader, test_loader = create_data(batch_size)  # 生成数据
    criterion = nn.CrossEntropyLoss()          # 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=lr)  # 定义优化器（Adam）
    #optimizer  = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # 记录每轮损失/准确率，用于后续画图
    total_train_losses = []
    total_test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        print(f'第 {epoch + 1} 轮训练 ----------------------------------')
        net.train()  # 切换到训练模式
        train_loss = 0.0  # 初始化当前 epoch 总损失
        train_correct = 0 # 初始化当前 epoch 正确预测样本数

        for inputs, labels in train_loader:  # 遍历训练集每个 batch
            inputs, labels = inputs.to(device), labels.to(device)  # 转移到GPU
            optimizer .zero_grad()      # 梯度清零
            outputs = net(inputs)  # 图片送入网络，前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer .step()     # 梯度更新

            train_loss += loss.item()  # item()：把张量类型的损失转为 Python 数值，避免内存泄漏
            pred = torch.argmax(outputs, dim=-1)  # 取最大值对应的索引
            train_correct += (pred == labels).sum().item()  # 统计正确预测数
            '''
            pred.eq(...)： 对比预测值和真实标签，相等则为True（1），不等为False（0）
            view_as(pred)：把标签形状转为和预测值一致，避免维度不匹配
            sum()：累加所有正确的样本数，累加到train_correct
            '''
        avg_train_loss = train_loss / len(train_loader.dataset)  # 一轮总损失除以总样本数（5万）= 平均损失
        acc = 100.0 * train_correct / len(train_loader.dataset)
        total_train_losses.append(avg_train_loss)
        train_accs.append(acc)
        '''
        100.0 * train_correct/总样本数：把正确数转为百分比  格式：45000/50000=90.000%
        cpu()：把 GPU 张量转到 CPU，方便存储（如果用 CPU 训练，这行可以删掉）
        '''
        print(f"[训练集] 平均损失：{avg_train_loss:.4f}   准确率：{acc:.3f}%")

        # 进行测试
        net.eval()  # 切换到评估模式（关闭dropout、batchnorm等训练专用层）
        test_loss = 0.0
        test_correct = 0
        with torch.no_grad():  # 关闭自动求导，大幅节省显存
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)  # 数据送入网络，前向计算
                loss = criterion(outputs, labels)  # 计算损失
                test_loss += loss.item()
                pred = torch.argmax(outputs, dim=-1)
                test_correct += (pred == labels).sum().item()

            avg_test_loss = test_loss / len(test_loader.dataset)
            test_acc = 100.0 * test_correct / len(test_loader.dataset)
            total_test_losses.append(avg_test_loss)
            test_accs.append(test_acc)
            print(f"[测试集] 平均损失：{avg_test_loss:.4f}   准确率：{test_acc:.3f}%")

    plt.subplot(2, 1, 1)
    plt.plot(range(epochs), train_accs, label='训练集')
    plt.plot(range(epochs), test_accs, label='测试集')
    plt.legend(loc='best')
    plt.xlabel('训练轮数')
    plt.ylabel('准确率')

    plt.subplot(2, 1, 2)
    plt.plot(range(epochs), total_train_losses, label='训练集')
    plt.plot(range(epochs), total_test_losses, label='测试集')
    plt.legend(loc='best')
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    #plt.savefig('结果图.jpg')
    plt.show()


if __name__ == '__main__':
    # 定义超参数
    lr = 0.002
    epochs = 20
    batch_size = 128
    # 自动判断用GPU还是CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = CNN().to(device)
    train(net, epochs, lr, batch_size)