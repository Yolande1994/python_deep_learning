import numpy as np

"""
  Softmax 损失函数, 原始实现 (用循环)
  输入的数据维度为D, 有C个类型, 小批量数据中有N个样本
  输入:
  - W: 形状 (D, C) 权重.
  - X: 形状 (N, D) 一个批量的数据.
  - y: 形状 (N,)   标签; y[i] = c 表示 X[i] 属于分类 c, 0 <= c < C.
  - reg: (float) 正则化强度
  返回一个元组tuple:
  - loss: float 类型，当前批次的损失值
  - dW:   损失对 W 的梯度，与 W 形状相同
"""
def softmax_loss_naive(W, X, y, reg):
    # 初始化损失和梯度为0
    loss = 0.0
    dW = np.zeros_like(W)  # 梯度矩阵形状和 W 完全一致
    N, D = X.shape         # N: 样本数，D: 特征维度
    C = W.shape[1]         # C: 类别数
    # ------------------- 1. 遍历每个样本，计算损失和梯度 -------------------
    for i in range(N):
        # 步骤1：计算当前样本的分数 s = X[i] @ W
        s = X[i].dot(W)  # 形状 (C,)（分数越高越像这个类）
        # 步骤2：数值稳定技巧：每个分数减去当前样本的最大分数，防止指数爆炸 (数学上等价，数值上安全)
        s -= np.max(s)   # 最大分数位置变成0, 其他都≤0, 这样e^s永远不会变成无穷大
        # 步骤3：计算 Softmax 概率 p = e^s / sum(e^s)
        exp_s = np.exp(s)
        p = exp_s / np.sum(exp_s)  # 形状 (C,)
        # 步骤4：计算当前样本的交叉熵损失，累加到总损失 (只看真实类别的概率，给“真实答案”的概率打个负对数，真实答案的概率越小，损失就越大（惩罚越重）)
        correct_class = y[i]
        loss += -np.log(p[correct_class])  # 交叉熵损失公式：Li = - log(p_yi)
        '''
        L = -log(p):
        当概率 p 接近 1 时，-log(1) = 0  → 损失为 0 （预测完全正确，完美）       假设概率 0.9 → 损失 = -log(0.9) ≈ 0.105（很小，预测准）
        当概率 p 接近 0 时，-log(0) = +∞ → 损失无穷大（预测完全错误，惩罚最重）    假设概率 0.1 → 损失 = -log(0.1) ≈ 2.302（很大，预测烂）
        '''
        # 步骤5：计算当前样本对梯度的贡献 (dL_i就是损失对分数s的梯度  Softmax梯度的核心公式实现，把复杂的求导结果直接用“复制概率 + 真实类减1”的方式表达出来，既简洁又高效，是数学推导出来的固定公式，直接背下来都可以)
        dL_i = p.copy()           # 先复制概率向量      错误类的梯度：梯度 = 概率        例: dL_i = [0.1, 0.2, 0.7] (概率向量)
        dL_i[correct_class] -= 1  # 真实类别位置减 1    正确类的梯度：梯度 = 概率 - 1        dL_i = [0.1, 0.2, 0.7-1] = [0.1, 0.2, -0.3]
        # 步骤6：累加梯度：dW += X[i].T * dL_i    根据链式法则，把对分数s的梯度 dL_i 传递到对权重W的梯度 dW
        dW += X[i].reshape(-1, 1) @ dL_i.reshape(1, -1)  # 在numpy里，一维向量的外积也可以直接写：dW += np.outer(X[i], dL_i)
        '''
        图片特征 × 梯度 = 权重的梯度
        X[i] 形状：(D,)    X[i].reshape(-1, 1) → 变成 (D, 1)（列向量）
        dL_i 形状：(C,)    dL_i.reshape(1, -1) → 变成 (1, C)（行向量）
        矩阵乘法 @（外积）   (D,1) @ (1,C) → (D,C)     这个结果正好和 dW 的形状匹配，代表第 i 个样本对权重梯度的贡献
        累加梯度 dW +=     每个样本都会产生一个梯度贡献, 循环结束后，dW 里就累积了整个批次所有样本的梯度总和
        '''
    # ------------------- 2. 平均损失 + 加入正则项 -------------------
    loss /= N                          # 损失平均：除以样本数 N
    loss += 0.5 * reg * np.sum(W * W)  # 加入 L2 正则项：0.5 * reg * ||W||²
    # ------------------- 3. 平均梯度 + 加入正则项梯度 -------------------
    dW /= N         # 梯度平均：除以样本数 N
    dW += reg * W   # L2 正则项的梯度：reg * W

    return loss, dW


# 向量版本, 输入输出同循环版.
def softmax_loss_vectorized(W, X, y, reg):
    #初始化损失和梯度为0
    loss = 0.0
    dW = np.zeros_like(W)
    N, D = X.shape
    C = W.shape[1]
    # ------------------- 1. 计算所有样本的分数（矩阵乘法） -------------------
    # 循环版：for i in range(N): s = X[i].dot(W)
    # 向量化版：一次算完所有样本 → (N, C)
    scores = X.dot(W)  # 形状 (N, C)
    # ------------------- 2. 数值稳定：每行减去该行最大值 -------------------
    # 循环版：s -= np.max(s)
    # 向量化版：对每行求 max，keepdims保持维度方便广播    形状 (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)  # 形状 (N, C)
    '''
    axis=1 表示按行操作：对每一行（每个样本）单独求 max、sum
    keepdims=True 保持结果为 (N,1)，方便和 (N,C) 的 scores 做广播减法
    '''
    # ------------------- 3. 计算 Softmax 概率 -------------------
    # 循环版：exp_s = np.exp(s);  p = exp_s / np.sum(exp_s)
    exp_scores = np.exp(scores)  # 形状 (N, C)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # 形状 (N, C)
    # ------------------- 4. 计算损失 -------------------
    # 循环版：loss += -np.log(p[correct_class])
    # 向量化版：用高级索引取出所有样本的正确类概率，再求平均
    '''
    高级索引 probs[np.arange(N), y]
    X[2,3] = 普通索引，取第2行、第3列的 1 个值
    X[[行列表], [列列表]] = 高级索引，一一对应取多个值
    np.arange(N): 生成 [0,1,2,...,N-1]，对应每个样本
    np.arange(N): 行索引 → 第几个样本
    y :           列索引 → 这个样本的正确类别
    组合起来就是:   每个样本对应的正确类别的概率值，形状 (N,)
    '''
    correct_probs = probs[np.arange(N), y]  # 形状 (N,)
    loss = np.sum(-np.log(correct_probs)) / N
    loss += 0.5 * reg * np.sum(W * W)  # 加正则项
    # ------------------- 5. 计算梯度 -------------------
    # 循环版：dL_i = p.copy();  dL_i[correct_class] -= 1
    # 向量化版：先复制概率，再对每个样本的正确类位置减 1
    dL = probs.copy()  # 形状 (N, C)
    dL[np.arange(N), y] -= 1  # 对每个样本的真实类别位置减 1
    # 循环版：dW += X[i].reshape(-1,1) @ dL_i.reshape(1,-1)
    # 向量化版：矩阵乘法一步算完所有样本的梯度贡献 → (D, C)
    dW = X.T.dot(dL) / N  # 先平均
    dW += reg * W  # 加正则项梯度

    return loss, dW


# 简单测试：随机数据跑一遍
np.random.seed(42)
D = 5    # 特征维度
C = 3    # 类别数
N = 4    # 样本数
W = np.random.randn(D, C) * 0.01  # 小随机权重
X = np.random.randn(N, D)
y = np.random.randint(0, C, size=N)
reg = 0.1

# 循环版
loss_naive, grad_naive = softmax_loss_naive(W, X, y, reg)
# 向量化版
loss_vec, grad_vec = softmax_loss_vectorized(W, X, y, reg)

print("循环版损失:", loss_naive)
print("向量化版损失:", loss_vec)
print("损失是否一致:", np.allclose(loss_naive, loss_vec))
print("梯度是否一致:", np.allclose(grad_naive, grad_vec))

# randn()：正态分布，平均值=0，大部分数在-1~1之间，有正有负（权重初始化专用）
# rand()： 均匀分布，0~1之间，只有正数（不适合权重）
# randint()：从指定范围里，随机挑整数（最小值,最大值,生成数量）