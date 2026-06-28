from __future__ import print_function
import numpy as np
from classifiers.linear_svm import *
from classifiers.softmax import *

#线性分类器
class LinearClassifier(object):

    def __init__(self):
        self.W = None  # 权重 W 会在 train 里初始化
    """
    使用随机梯度下降训练这个线性分类器。
    输入：      
    -X：训练数据，numpy array，形状为（N，D），有N个样本，每个样本有D个特征   
    -Y：训练标签，numpy array，形状为（N,），  y[i]＝c 表示 x[i]属于c分类
    - learning_rate: 优化学习率(float) 
    - reg:           正则化强度(float) 
    - num_iters:     优化时的步数(integer) 
    - batch_size:    每步使用的样本数(integer) 
    - verbose:       是否打印优化进度 (boolean) 
    输出:
    一个list，包含了每轮训练的损失值( 可用于画图观察训练过程:plt.plot(loss_history) )
    """
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        N, D = X.shape
        C = np.max(y) + 1  # 标签从0开始，最大标签+1 = 类别总数

        # 懒初始化权重（如果还没初始化）
        if self.W is None:
          self.W = 0.001 * np.random.randn(D, C)  # 用小随机数初始化权重：避免梯度爆炸，同时打破对称性

        loss_history = []  # 准备存储训练过程中的损失值

        for it in range(num_iters):
            # 1. 随机采样一批数据（小批量 SGD）
            batch_indices = np.random.choice(N, size=batch_size, replace=True)  # 从 0~N-1 中随机选 batch_size 个索引（可重复采样，替换采样）
            # 根据索引取出对应的样本和标签
            X_batch = X[batch_indices]  # 形状：(batch_size, D)
            y_batch = y[batch_indices]  # 形状：(batch_size,)

            # 2. 计算当前 batch 的损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)  # 调用子类实现的 loss 函数
            loss_history.append(loss)  # 把当前步的损失存入历史列表，方便后续画图看收敛情况

            # 3. SGD 更新权重
            self.W -= learning_rate * grad  # 含义：沿着梯度反方向更新权重，让损失变小

            # 4. 可选：打印训练进度（verbose=True 时）
            if verbose and it % 100 == 0:
                print(f'迭代 {it} / {num_iters}: 损失 {loss:.4f}')

        return loss_history


    """
    使用这个分类器训练得到的权重预测数据点的标签
    输入：      
    -X：训练数据，numpy array，形状为（N，D），有N个样本，每个样本有D个特征   
    输出:
    - y_pred: 对输入数据X预测得到的标签，y_pred 是一维数组，长度为N，每个元素是一个整数，代表预测的类型
    """
    def predict(self, X):
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)  # np.argmax(..., axis=1)：按行取最大值的索引
        return y_pred


    def loss(self, X_batch, y_batch, reg):
        """
        计算损失函数及其导数.
        将在子类中实现.
        输入：
        -X_batch：numpy array，形状为（N，D），有N个数据点，每个点有D个维度
        -y_batch：numpy array,（N,）。y[i]＝c ，包含了小批次数据的标签
        - reg: (float) 正则化强度.
        返回一个元组tuple：
        - 损失，single float
        - 相对于权重self.W的梯度；与W相同的形状,array
        """
        pass


# 子类：继承基类 LinearClassifier，实现具体的 loss 方法
# SVM 完整展示
class LinearSVMdemo(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        # 这里直接把之前的完整 SVM 代码贴进来
        loss = 0.0
        num_train = X_batch.shape[0]
        delta = 1.0
        # 1. 计算得分
        scores = X_batch.dot(self.W)
        correct_class_scores = scores[np.arange(num_train), y_batch]
        margin = scores - correct_class_scores[:, np.newaxis] + delta
        margin[np.arange(num_train), y_batch] = 0.0
        # 2. 计算损失
        loss = np.sum(margin[margin > 0])
        loss /= num_train
        loss += reg * np.sum(self.W * self.W)
        # 3. 计算梯度
        mask = (margin > 0).astype(int)
        count_per_sample = np.sum(mask, axis=1)
        dW_contrib = np.zeros_like(scores)
        dW_contrib[np.arange(num_train), y_batch] = -count_per_sample
        dW_contrib += mask
        dW = X_batch.T.dot(dW_contrib)
        dW /= num_train
        dW += 2 * reg * self.W
        return loss, dW

# SVM 直接调用
class LinearSVM(LinearClassifier):
  """ 使用 多分类 SVM 损失函数计算损失 """
  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

# 基类 LinearClassifier 提供 train/predict. 子类只实现 loss
class Softmax(LinearClassifier):
  """ 使用 Softmax + 交叉熵 损失函数计算损失 """
  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

