# coding:gbk
# import cv2
# import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
'''
1.这是一个PyTorch 入门级 demo，用两层全连接神经网络解决 XOR 异或问题
2.完整实现了深度学习标准流程：定义网络 → 准备数据 → 训练 → 验证
3.核心验证了：带隐藏层 + 非线性激活函数的神经网络，可以解决线性不可分问题。
'''

# 定义网络结构
class XOR(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层：输入2维 → 隐藏层4维 → 输出2维
        self.dense = nn.Sequential(nn.Linear(2,4), nn.ReLU(), nn.Linear(4,2))  # Sequential(): 层的有序打包器，自动前向传播

    def forward(self,x):
        x = self.dense(x)   # 实现全连接计算
        return x

def train(net, epochs, lr):
    criterion = nn.CrossEntropyLoss()          # 定义损失函数
    optim = opt.Adam(net.parameters(), lr=lr)  # 定义优化器
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print('Epoch:{}----------------------------'.format(epoch))
        optim.zero_grad()      # 梯度清零
        outputs = net(inputs)  # 数据送入网络，进行前向计算
        loss = criterion(outputs, labels)  # 计算损失  (CrossEntropyLoss要求标签是整数类型,必须是LongTensor（int64类型）,否则报错)
        loss.backward()        # 反向传播，计算梯度
        optim.step()           # 梯度更新
    # 训练结束后进行测试，验证是否解决XOR问题
    net.eval()  # 切换到测试模式
    outputs = net(inputs)  # 数据送入网络，进行前向计算
    print(outputs.argmax(dim=1))  # 获取最大值下标，期望输出是[0, 0, 1, 1]
    print(labels)    #标签，[0, 0, 1, 1]

if __name__ == '__main__':
    # 自动判断用GPU还是CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 定义数据
    inputs = torch.FloatTensor([[0,0],[1,1],[0,1],[1,0]])  # 4组XOR输入   特征用 Float（浮点）
    labels = torch.LongTensor([0, 0, 1, 1])                # 对应正确标签   标签用 Long（整数）
    inputs, labels = inputs.cuda(), labels.cuda()          # 核心铁律：模型在哪，数据就必须在哪
    # 定义超参数
    lr = 0.005
    epochs = 1000
    # 初始化模型
    net = XOR().to(device)
    # 启动训练
    for _ in range(5):
        train(net, epochs, lr)


'''
nn.Sequential () = 层的有序打包器, 放入什么层，就按顺序执行什么层, 不用手动写前向传播的调用顺序, 让网络代码更简洁、更清晰

不用 Sequential 的写法（麻烦）：
class XOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4,2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


用 Sequential 的写法（简洁）：
self.dense = nn.Sequential(
    nn.Linear(2,4), 
    nn.ReLU(), 
    nn.Linear(4,2)
)

def forward(self, x):
    return self.dense(x)  # 一行搞定
'''

'''
nn.ReLU/nn.Linear → 类（class）： 要先创建对象，再调用，或放在 Sequential 里
F.relu /F.linear  → 函数（function）： 直接调用，像数学函数一样用

self.relu = nn.ReLU()    # 先创建
x = self.relu(x)         # 再调用

x = F.relu(x)            # 直接用，一步到位
'''

'''
net.parameters() 是模型自动生成的
只要你在网络里写了 nn.Linear() 这种层，PyTorch 就会自动帮你创建权重 w 和偏置 b，并全部打包进 parameters() 里

层 = 自动生成参数
parameters () = 自动收集所有参数
优化器 = 自动更新这些参数
你完全不需要手动创建、管理、命名任何权重！
'''

'''
模型在哪，数据就必须在哪:

# 这一行自动判断用CPU还是GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 模型搬过去
net = XOR().to(device)
# 数据也用同样方式搬过去（最标准）
inputs = inputs.to(device)
labels = labels.to(device)

这样写：有 GPU 用 GPU，没 GPU 自动用 CPU，永远不报错！
'''

'''
net.eval() → 把模型切换到 评估/测试 模式（关闭训练特有的层）
神经网络里有一些层，训练时和测试时行为不一样！
比如：
Dropout：训练时随机丢神经元，测试时不能丢
BatchNorm：训练时用批次统计，测试时用固定统计值
eval() 就是告诉模型：现在不训练了，别乱改结构，老老实实正常计算！

训练模式 train()：模型会做一些 “帮助训练” 的特殊操作
测试模式 eval()：模型完全稳定、完全正常、完全不随机地输出结果
你现在这个 XOR 网络没有 Dropout / BatchNorm，所以加不加结果一样. 但！写了是规范！是好习惯！必须写！

类似的：
with torch.no_grad(): → 停止计算梯度（省内存、提速）
测试时不需要更新权重，不需要反向传播，不计算梯度  →  省显存、省内存、速度更快
一句话：告诉 PyTorch“只算结果，别存梯度，我不训练”

测试/预测的时候 → 两个必须一起用！
'''