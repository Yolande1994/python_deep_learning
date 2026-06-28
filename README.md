# python_deep_learning

## 项目简介
本仓库为深度学习算法学习与实战项目合集，覆盖从经典机器学习分类器底层原理实现，到卷积神经网络工程化训练、迁移学习落地的完整学习路径。所有算法均包含原理级手动实现与可运行的完整工程代码，用于深度学习方向知识沉淀与求职作品集展示。

## 技术栈
- 编程语言：Python 3.x
- 深度学习框架：PyTorch、TorchVision
- 科学计算：NumPy
- 可视化：Matplotlib
- 交互演示：Jupyter Notebook

## 目录结构
项目主代码位于 `PythonDL/` 目录下，按职责划分为六大模块：

### 1. cat_dog_classify 猫狗二分类迁移学习项目
基于ResNet101预训练模型的工业级图像分类完整工程，实现全参数微调迁移学习：
- `net.py`：封装ResNet101网络结构，替换分类头适配二分类任务，支持全参数微调
- `train.py`：完整训练+验证流程，包含损失统计、准确率计算、学习率衰减、模型保存
- `config.py`：统一管理超参数、数据集路径、训练配置
- `data_split.py`：自动按比例划分训练集与验证集

### 2. classifiers 经典分类器底层实现
基于NumPy从零手写实现经典机器学习算法，不依赖高层封装，完整覆盖损失函数推导与梯度反向传播：
- `k_nearest_neighbor.py`：K近邻分类器，支持多版本距离计算实现
- `linear_svm.py`：线性SVM分类器，合页损失与梯度的向量化实现
- `softmax.py` / `softmax_classifier.py`：Softmax分类器，交叉熵损失与梯度推导
- `neural_net.py`：两层全连接神经网络，包含ReLU激活、L2正则化与完整反向传播

### 3. code 卷积神经网络工程实现
基于CIFAR-10数据集的CNN图像分类完整工程：
- `cifar_cnn.py`：轻量级卷积神经网络，包含数据增强、训练验证、结果可视化全流程
- `data/`：数据集存放目录

### 4. datasets 数据集工具
数据集下载、加载与预处理相关工具：
- `get_datasets.sh`：CIFAR-10数据集自动下载脚本
- `data_utils.py`：数据集加载、批处理、数据预处理工具函数

### 5. tutorials 入门练习与算法演示
深度学习基础语法练习与算法原理分步演示Notebook：
- 基础语法：PyTorch张量操作、线性回归、感知机、异或问题实现
- 算法演示：KNN、SVM、Softmax、两层神经网络的分步验证Notebook，包含调参与结果可视化

### 6. utils 通用工具集
跨模块复用的通用工具函数：
- `gradient_check.py`：数值梯度校验工具，验证反向传播梯度正确性
- `vis_utils.py`：图像网格可视化工具
- `distance_matrix_vis.py`：距离矩阵可视化脚本

### 环境依赖
pip install torch torchvision numpy matplotlib jupyter  

## 项目亮点  
- 1.底层原理扎实：经典分类器与损失函数全部基于 NumPy 从零实现，包含完整梯度推导，深入理解算法本质
- 2.工程化闭环：从数据划分、网络搭建、训练验证到模型保存，完整复现工业级项目开发流程
- 3.迁移学习落地：基于 ImageNet 预训练模型实现全参数微调，适配自定义二分类数据集
- 4.模块化目录设计：按职责拆分算法源码、工程项目、工具集、教程演示，结构清晰易维护
- 5.配套演示文档：Jupyter Notebook 分步验证算法效果，包含数据处理、参数调试与结果可视化
