from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # 前向
        H1 = np.dot(X, W1) + b1
        H1_relu = np.maximum(0, H1)
        scores = np.dot(H1_relu, W2) + b2

        if y is None:
            return scores

        # Softmax
        scores_stable = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_stable)
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
        probs = exp_scores / sum_exp_scores

        # 损失
        correct_probs = probs[np.arange(N), y]
        data_loss = np.sum(-np.log(correct_probs)) / N
        reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss

        # 反向
        grads = {}
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        # 第二层
        grads['W2'] = np.dot(H1_relu.T, dscores) + reg * W2
        grads['b2'] = np.sum(dscores, axis=0)
        dH1_relu = np.dot(dscores, W2.T)

        # ReLU
        dH1 = dH1_relu.copy()
        dH1[H1 <= 0] = 0

        # 第一层
        grads['W1'] = np.dot(X.T, dH1) + reg * W1
        grads['b1'] = np.sum(dH1, axis=0)

        return loss, grads

    def train(self, X, y, X_val=None, y_val=None,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100, batch_size=200,
              verbose=False, replace=True):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)  # 整数除法

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            # 采样
            batch_indices = np.random.choice(num_train, batch_size, replace=replace)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # 参数更新
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print(f'迭代 {it}/{num_iters}: 损失 = {loss:.4f}')

            # 学习率衰减（只在 epoch 节点执行，不会每轮都衰减）
            if X_val is not None and y_val is not None:
                if it % iterations_per_epoch == 0:
                    train_acc = (self.predict(X_batch) == y_batch).mean()
                    val_acc = (self.predict(X_val) == y_val).mean()
                    train_acc_history.append(train_acc)
                    val_acc_history.append(val_acc)
                    if learning_rate_decay is not None:
                        learning_rate *= learning_rate_decay

        if X_val is not None and y_val is not None:
            return {'loss_history': loss_history, 'train_acc_history': train_acc_history, 'val_acc_history': val_acc_history}
        else:
            return {'loss_history': loss_history}

    def predict(self, X):
        scores = self.loss(X)
        return np.argmax(scores, axis=1)


# ===================== 测试运行 =====================
if __name__ == '__main__':
    print("===== 开始测试前向传播 =====")
    X = np.array([[1, 2, 3, 4],
                  [2, 3, 4, 5],
                  [3, 4, 5, 6]])
    y = np.array([0, 1, 0])

    input_size = 4
    hidden_size = 10
    output_size = 3

    net = TwoLayerNet(input_size, hidden_size, output_size)

    scores = net.loss(X)
    loss, grads = net.loss(X, y, reg=0.1)

    print("\n===== 训练之前 =====")
    print("输入 X      shape:", X.shape)
    print("输出 scores shape:", scores.shape)
    print("初始损失 loss =", loss)

    # 这是能降到 0 的超参数组合
    stats = net.train(
        X, y,
        learning_rate=0.1,
        learning_rate_decay=None,
        reg=0,
        batch_size=3,
        num_iters=2000,
        replace=False,
        verbose=True
    )

    loss_history = stats['loss_history']
    print("\n===== 训练之后 =====")
    print("最终损失 loss =", loss_history[-1])

    plt.plot(loss_history)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss should go to near 0')
    plt.show()

    y_pred = net.predict(X)
    print("\n训练集预测结果:", y_pred)
    print("真实标签:", y)
    print("预测准确率:", (y_pred == y).mean())