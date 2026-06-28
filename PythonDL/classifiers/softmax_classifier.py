import numpy as np


class Softmax:
    def __init__(self):
        # 初始化 Softmax 分类器,权重 W 会在 train 里初始化
        self.W = None  # 权重矩阵，形状 (D, C)

    # 训练 Softmax 模型（使用 SGD 随机梯度下降）
    def train(self, X, y, learning_rate=1e-7, reg=5e4, num_iters=1000, batch_size=200, verbose=False):
        N, D = X.shape  # N 样本数，D 特征维度
        C = np.max(y) + 1  # 类别数

        # 初始化权重 W
        if self.W is None:
            self.W = 0.001 * np.random.randn(D, C)

        loss_history = []

        # 随机梯度下降迭代
        for it in range(num_iters):
            # 1. 随机采样一批数据（小批量 SGD）
            batch_indices = np.random.choice(N, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # 2. 计算损失和梯度（调用向量化版）
            loss, grad = self.softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)

            # 3. SGD 更新权重
            self.W -= learning_rate * grad

            # 打印进度
            if verbose and it % 100 == 0:
                print(f"迭代 {it}/{num_iters}, 损失: {loss:.4f}")

        return loss_history


    def predict(self, X):
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)  # 取分数最高的类别
        return y_pred


    # ===================== 向量化版 损失+梯度 =====================
    def softmax_loss_vectorized(self, W, X, y, reg):
        loss = 0.0
        dW = np.zeros_like(W)
        N, D = X.shape
        C = W.shape[1]

        # 1. 计算分数
        scores = X.dot(W)

        # 2. 数值稳定
        scores -= np.max(scores, axis=1, keepdims=True)

        # 3. Softmax 概率
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 4. 计算损失
        correct_probs = probs[np.arange(N), y]
        loss = np.sum(-np.log(correct_probs)) / N
        loss += 0.5 * reg * np.sum(W * W)

        # 5. 计算梯度
        dL = probs.copy()
        dL[np.arange(N), y] -= 1
        dW = X.T.dot(dL) / N
        dW += reg * W

        return loss, dW