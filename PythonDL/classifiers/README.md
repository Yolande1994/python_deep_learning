# 经典机器学习分类器

基于NumPy从零实现的经典分类算法集合，包含完整的损失函数推导与梯度反向传播实现，用于理解算法底层原理。

## 包含算法
- `k_nearest_neighbor.py`：K近邻分类器，支持双循环、单循环、纯向量化三种L2距离计算方式
- `linear_svm.py`：线性SVM，合页损失函数与梯度的循环版+向量化版实现
- `softmax.py`：Softmax分类器，交叉熵损失函数与梯度的循环版+向量化版实现
- `linear_classifier.py`：线性分类器基类，封装通用训练、预测流程
- `softmax_classifier.py`：Softmax分类器完整实现，继承线性分类器基类
- `neural_net.py`：两层全连接神经网络，手动实现前向传播与反向传播