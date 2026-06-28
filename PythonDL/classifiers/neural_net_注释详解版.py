from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

"""
一个两层完全连接的神经网络。网络的输入维度为N，隐层维度H，并在C类上执行分类。

用softmax损失函数和对权重矩阵进行L2正则化来对网络进行训练。
网络在第一次全连接层后使用Relu非线性。

就是说，该网络具有以下架构：
输入 - 全连接层 - Relu - 全连接层 - softmax

第二个全连接层的输出是每个类的得分。
"""
class TwoLayerNet(object):
    """
    初始化模型。 权重被初始化为很小（std）的随机值。 偏置被初始化为零。
    权重和偏置被存储在变量self.params里面，它是一个具有以下键的字典：
    W1: 第一层的权重 weights;  shape为 (D, H)
    b1: 第一层的偏置 biases;  shape 为 (H,)
    W2: 第二层的权重 weights; shape 为(H, C)
    b2: 第二层的偏置 biases;  shape 为(C,)

    输入:
    - input_size:  输入数据的维度 D
    - hidden_size: 隐藏层的神经元数量 H
    - output_size: 分类数 C
    """
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):  # 设: input_size = 4（样本特征数） hidden_size = 10（隐藏层神经元数） output_size = 3（分类数）
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)   # 形状 (D, H)  (4, 10)
        self.params['b1'] = np.zeros(hidden_size)                            # 形状 (H,)    (10,)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)  # 形状 (H, C)  (10, 3)
        self.params['b2'] = np.zeros(output_size)                            # 形状 (C,)    (3,)

    """
    为这个两层的全连接网络计算损失及梯度
    输入:
    - X: 输入数据 shape为(N, D). 每个 X[i] 为一个训练样本
    - y: 训练标签 shape为(N, ).  y[i] 是 X[i] 的标签
      y 是可选输入。如果没有输入y，只需要返回得分。如果输入了y，则返回损失及梯度
    - reg: 正则化强度λ.
    
    返回:
    如果 y 为 None, 返回得分矩阵，shape(N, C)，  scores[i, c] 是输入 X[i] 在 c 分类上的得分
    如果 y 非 None, 则返回一个 tuple:
    - loss:  本批次数据的损失 (数据损失+正则化损失) 
    - grads: 字典，保存各参数的梯度. (参数名称：损失函数在这些参数上的梯度；与self.params相同的键)
    """
    def loss(self, X, y=None, reg=0.0):
        # 从 params 字典取出参数
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        '''
        1. 输入层（数据入口）
        2. 第一层全连接（W1, b1）
        3. ReLU 激活层（独立层）
        4. 第二层全连接（W2, b2）
        5. 输出层（Softmax + 交叉熵）
        '''
        # ====================== 前向过程计算 ======================
        # 一.全连接第一层： X(N,D) × W1(D,H) + b1 → 输出(N,H)     全连接层 = 加权求和（X・W）+ 加每个神经元的偏置（b）    h1 = (x1w1 + x2w2 + x3w3 + ... + x10w10) + b1  （h1：第1个神经元的输出，只加第1个神经元的偏置b1，只加1次）
        H1 = np.dot(X, W1) + b1  # shape: (N, H)  # 没经过 ReLU 的“原始得分”
        #print("第一层输出 shape:", H1.shape,'\n +自定义b 的样子:\n',X.dot(W1) + [1,2,3,4,5,6,7,8,9,10])

        # 二.ReLU 激活层： 负数变0，正数保留
        H1_relu = np.maximum(0, H1)  # shape: (N, H)       神经网络的工作 = 把D个原始特征 → 变成H个高级特征
        #print("H_relu的样子:\n",H1_relu)  # 经过ReLU过滤后的得分

        # 三.全连接第二层： 输出最终分类得分  h_r(N,H) × W2(H,C) + b2 → 输出(N,C)     H1_relu是 “学到的H特征”，第二层W2是 “H特征的重要程度”，相乘相加 = 给每个类别打分
        scores = np.dot(H1_relu, W2) + b2  # shape: (N, C)
        '''
        全连接层的本质
        第一层：特征提取器（D → H）      把D个原始特征 → 提炼成H个高级抽象特征
        第二层：分类评判器（H → C）      给H个高级特征打分 → 计算每个类别的最终得分
        '''
        if y is None:
          return scores  # 无标签时直接返回得分（用于预测）
        # 四.输出层：Softmax + 交叉熵
        # 数值稳定处理：减去每行最大值，防止指数爆炸（exp(大数)会溢出）
        scores_stable = scores - np.max(scores, axis=1, keepdims=True)
        # 1. 计算 Softmax 概率
        exp_scores = np.exp(scores_stable)   # 分子：exp(稳定后得分)                  形状: (N, C)
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)  # 分母：每行所有类别 exp 之和（keepdims保持维度方便广播）  形状: (N, 1)
        probs = exp_scores / sum_exp_scores  # Softmax 概率：每个样本在每个类别上的概率   形状: (N, C)
        # 2. 计算 交叉熵损失
        correct_probs = probs[np.arange(N), y]           # 先取出每个样本对应真实标签 y 的概率值
        data_loss = np.sum(-np.log(correct_probs)) / N   # 对每个样本的正确概率取负对数，然后求平均（一个标量）

        # L2 正则化：对权重矩阵 W1、W2 做平方和，再乘以 0.5*reg （标量）
        reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))  # （乘 0.5 是为了求导后系数变成 1,方便计算梯度）

        # 总损失 = 数据损失 + 正则化损失
        loss = data_loss + reg_loss

        '''
        1. 损失函数本身 （∂L/∂L = 1）
        2. 输出层 （Softmax + 交叉熵）（∂L/∂scores = y_p-y_true / N）
        3. 第二层全连接 （∂L/∂X2 = ∂L/∂Y2 × W2.T   ∂L/∂W2 = X2.T × ∂L/∂Y2 + reg×W2   ∂L/∂B2 = 对 ∂L/∂Y2 按样本求和）
        4. ReLU 激活层 （上游梯度 ∂L/∂X：输入>0时原样传递，<=0时梯度置0）
        5. 第一层全连接 （∂L/∂X1 = ∂L/∂Y1 × W1.T   ∂L/∂W1 = X1.T × ∂L/∂Y1 + reg×W1   ∂L/∂B1 = 对 ∂L/∂Y1 按样本求和）
        '''
        # ====================== 反向传播：计算梯度 ======================
        # → 起点: 最终损失loss本身的导数   ∂L/∂L = 1 (损失loss是一个标量, 对它自己求导=1)
        grads = {}

        # 一. 计算输出层最终分类得分梯度：dscores
        # Softmax+交叉熵层：∂L/∂scores = y_p - y_true / N
        dscores = probs.copy()         # 先复制预测概率
        dscores[np.arange(N), y] -= 1  # probs 只在'正确类别'的位置减 1（仅独热情况）
        dscores /= N                   # 除以样本数 N! 平均损失对应平均梯度
        '''
        Softmax + 交叉熵 联合求导的梯度公式：
          dscores = (预测概率 - 真实概率) / N

        - 当真实标签是独热编码：正确类别为 1，其余为 0
        - 对正确类别：梯度 = (预测概率 - 1) / N                  正确类梯度 = 正确类概率 - 1
        - 对错误类别：梯度 = (预测概率 - 0) / N = 预测概率 / N     错误类梯度 = 错误类概率 - 0（对于错误类别，梯度初始值就是它的预测概率，但是错误类别也参与了梯度计算）

        直观作用：
          正确类别位置为负梯度 → 梯度下降时，对应得分会增加，让正确类概率更接近 1
          错误类别位置为正梯度 → 梯度下降时，对应得分会减少，让错误类概率更接近 0
        '''

        # 二. 计算第二层参数梯度：dW2, db2 与 特征梯度：dX2
        # 权重梯度：∂L/∂W = X.T × ∂L/∂Y            dW梯度 = 输入转置（该层的X.T）× 输出梯度（该层Y的梯度=上层传来的）
        grads['W2'] = np.dot(H1_relu.T, dscores) # dW2 = H1_relu.T @ dscores
        grads['W2'] += reg * W2                  # 加上L2正则化的梯度：∂(reg_loss)/∂W2 = reg * W2
        # 偏置梯度：∂L/∂B = 对 ∂L/∂Y 按样本求和
        grads['b2'] = np.sum(dscores, axis=0)    # 单样本时：∂L/∂B = ∂L/∂Y
        # 特征梯度：∂L/∂X = ∂L/∂Y × W.T
        dH1_relu = np.dot(dscores, W2.T)         # dH1_relu = dscores @ W2.T

        # 三. 反向传播通过 ReLU 层
        # ReLU之前的得分的导数：上游梯度 ∂L/∂X：输入>0时原样传递，<=0时梯度置0
        dH1 = dH1_relu.copy()  # 形状(N, H) = H1
        dH1[H1 <= 0] = 0       # H1的梯度 = 上游梯度 dH1_relu，在 H1<=0 的位置置0

        # 四. 计算第一层参数梯度：dW1, db1
        grads['W1'] = np.dot(X.T, dH1)     # dW1 = X.T @ dH1
        grads['W1'] += reg * W1            # 加上 L2 正则化的梯度：d(reg_loss)/dW1 = reg * W1
        grads['b1'] = np.sum(dH1, axis=0)  # db1 = 对 dH1 按样本求和
        # ============================================================================

        return loss, grads  # 如果有输入y，返回损失和梯度


    """
    使用随机梯度下降法(SGD)训练神经网络
    输入:
    - X: 训练数据 (N, D)
    - y: 训练标签 (N,)
    - X_val: 验证数据 (N_val, D)
    - y_val: 验证标签 (N_val,)
    - learning_rate: 学习率
    - learning_rate_decay: 学习率衰减率，标量，每个周期(epoch)后的学习率下降的量（比率）
    - reg: 正则化强度
    - num_iters: 总迭代次数
    - batch_size: 每次迭代用的样本数（batch大小）
    - verbose: 是否打印训练进度
    返回:
    - stats: 字典，记录训练过程中的损失、训练/验证准确率
    """
    def train(self, X, y, X_val=None, y_val=None, learning_rate=1e-3, learning_rate_decay=0.95, reg=5e-6, num_iters=100, batch_size=200, verbose=False, replace=True):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)  # 每个epoch的迭代次数

        # 记录训练过程
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # 从训练集中随机选batch_size个样本（有放回采样）
            batch_indices = np.random.choice(num_train, batch_size, replace=replace)
            X_batch = X[batch_indices]  # 批量数据 (batch_size, D)
            y_batch = y[batch_indices]  # 批量标签 (batch_size,)

            # 计算当前batch的损失和梯度
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)  # 记录损失

            # SGD更新公式: 参数 = 参数 - 学习率 × 梯度
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            # 打印进度（可选）
            if verbose and it % 100 == 0:
                print(f'迭代 {it}/{num_iters}: 损失 = {loss:.4f}')

            if X_val is not None and y_val is not None:
                # 每个epoch检查准确率 + 衰减学习率
                if it % iterations_per_epoch == 0:
                    # 计算当前训练集和验证集的准确率
                    train_acc = (self.predict(X_batch) == y_batch).mean()  # 用当前batch近似训练准确率
                    val_acc = (self.predict(X_val) == y_val).mean()        # 验证集准确率
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)
                    # 学习率衰减（让后期步长更小，收敛更稳定）
                    learning_rate *= learning_rate_decay

        # 返回训练记录
        if X_val is not None and y_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history
            }
        else:
            return {
                'loss_history': loss_history
            }


    def predict(self, X):
        # 前向传播得到分类得分（y=None时，loss直接返回scores）
        scores = self.loss(X)
        # 取每个样本得分最大的类别作为预测结果
        y_pred = np.argmax(scores, axis=1)  # axis=1表示按行取最大值的索引
        return y_pred


# ===================== 手写固定测试数据 + main 入口 =====================
if __name__ == '__main__':
    print("===== 开始测试前向传播 =====")
    # 1.准备数据：3个样本，每样本4个特征，3分类
    X = np.array([[1, 2, 3, 4],
                  [2, 3, 4, 5],
                  [3, 4, 5, 6]])
    y = np.array([0, 1, 2])
    # y = np.array([0, 1, 2])  对线性递增的输入（4→5→6）, 绝对学不到：小→大→小（0→1→0）,所以损失必然降不到0, 它能做到的最好结果是：样本1预测成0.5概率0,样本2预测成0.5概率1,样本3预测成0.5概率0. 这样损失最小,这个折中方案的损失≈0.63

    # 2.创建网络（严格匹配 __init__ 参数）
    input_size = 4
    hidden_size = 10
    output_size = 3
    net = TwoLayerNet(input_size, hidden_size, output_size)

    # 3.运行前向和反向传播一次
    scores = net.loss(X)  # 无 y
    loss, grads = net.loss(X, y, reg=0.1)

    print("\n===== 训练之前 =====")
    print("输入 X      shape:", X.shape)
    print("输出 scores shape:", scores.shape)
    print('三分类正常初始损失 -ln(1/3) ≈ 1.09861228',"\n初始损失 loss =", loss)

    # 4.训练模型
    stats = net.train(X,y,learning_rate_decay=None,learning_rate=0.1,reg=0,batch_size=3,num_iters=2000,replace=False)  # 接收返回的字典
    loss_history = stats['loss_history']  # 从字典里取出损失列表
    print("\n===== 训练之后 =====")
    print("训练后 loss =",loss_history[-1])

    # 5.绘图看损失
    plt.plot(loss_history)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()

    # 训练完成后，预测训练集
    y_pred = net.predict(X)
    print("\n训练集预测结果:", y_pred)
    print("真实标签:", y)
    print("预测准确率:", (y_pred == y).mean())