import numpy as np
from random import shuffle

#  SVM 训练流程: 初始化W → 算损失 → 算梯度 → 用梯度更新W → 重复直到损失最小

def svm_loss_naive(W, X, y, reg):
  """
  结构化SVM合页损失函数，循环实现版本
  参数:
  - W: 权重矩阵，形状 (D, C)，D为特征维度，C为类别数
  - X: 批量样本特征，形状 (N, D)，N为样本数量
  - y: 样本标签，形状 (N,)
  - reg: L2正则化强度
  返回:
  - loss: 标量，总损失（合页损失 + L2正则化损失）
  - dW: 权重梯度矩阵，形状与W一致
  """
  # ===================== 初始化（训练流程前置准备）=====================
  dW = np.zeros(W.shape)    # 初始化梯度矩阵为0，形状和权重W一致（D×C）
  num_classes = W.shape[1]  # 类别数C（比如CIFAR-10的 10类）
  num_train = X.shape[0]    # 样本数N（比如开发集的500个样本）
  loss = 0.0                # 损失初始化为0
  delta = 1                 # SVM的间隔阈值（固定为1，是合页损失的超参数）
  # ===================== 二.计算损失（训练流程第二步）=====================
  for i in range(num_train):
    # 1. 计算当前样本的所有类别得分：
    #    单样本特征(3072,) × 权重矩阵(3072,10) = 得分向量(10,)
    #    得分向量每个元素对应样本在该类别的匹配度（无物理意义，仅用于比较）
    scores = X[i].dot(W)
    # 2. 取出当前样本的正确类别得分：
    #    y[i]是样本i的真实类别编号，scores[y[i]]即该样本在真实类别上的得分
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:  # 跳过正确类别（只计算错误类的损失）
        continue
      # 3. 计算间隔（合页损失核心）：
      #    margin = 错误类得分 - 正确类得分 + 安全间隔delta
      #    含义：衡量错误类对正确类的"威胁程度"，>0表示威胁存在，需要惩罚. 负数说明差距够大，不罚
      margin = scores[j] - correct_class_score + delta
      # 4. 合页损失计算：只有间隔 > 0（不满足安全间隔），才产生损失 （ max(0, margin) ）
      if margin > 0:
        loss += margin  # 累计错误类的惩罚损失
        # ===================== 三.计算梯度（训练流程第三步）=====================
        # 仅当margin>0时更新梯度（无威胁时无需调整权重）
        # 梯度更新逻辑：
        # - 正确类y[i]权重 -= X[i] → 让正确类得分更高
        # - 错误类j权重 += X[i]    → 让错误类得分更低
        # 最终目标：拉开正确类和错误类的得分差，满足安全间隔
        dW[:, y[i]] -= X[i]  # 正确类权重梯度更新
        dW[:, j] += X[i]     # 错误类权重梯度更新
  # ===================== 损失后处理（正则化+平均）=====================
  loss /= num_train            # 所有样本的损失求平均
  loss += reg * np.sum(W * W)  # 加上L2正则化损失（防止过拟合）
  # ===================== 梯度后处理（正则化+平均）=====================
  dW /= num_train    # 损失是平均过的，梯度也要对应平均（保持梯度和损失的尺度一致）
  dW += 2 * reg * W  # 加上L2正则化的梯度：d(reg*W²)/dW = 2*reg*W.  含义：惩罚权重过大，让梯度向"权重减小"的方向偏移

  return loss, dW
# ===================== 补充：训练流程第四步（权重更新示例）=====================
'''
# 完整训练流程示例（伪代码）
learning_rate = 0.01  # 学习率：梯度更新的步长
max_epochs = 100      # 最大训练轮数

# 1. 初始化W（训练流程第一步）
D = 3072  # 特征维度
C = 10    # 类别数
W = np.random.randn(D, C) * 0.001  # 随机初始化小权重

# 训练循环（重复更新直到损失最小）
for epoch in range(max_epochs):
    # 2. 算损失 + 3. 算梯度（调用上述函数）
    loss, dW = svm_loss_naive(W, X_train, y_train, reg=0.1)

    # 4. 用梯度更新W（训练流程第四步）
    # W = W - 学习率 * 梯度 → 梯度下降：向损失减小的方向更新
    W -= learning_rate * dW

    # 监控训练过程
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
'''


# 向量化版和循环版的数学公式完全一致，只是用矩阵运算替代了 Python 循环，结果应该几乎相同
def svm_loss_vectorized(W, X, y, reg):
  """
  结构化 SVM 损失函数，向量化实现（无 Python 循环，速度极快）
  输入输出与 svm_loss_naive 完全一致：
  - W: (D, C) 权重矩阵
  - X: (N, D) 批量特征数据
  - y: (N,) 标签数组
  - reg: L2 正则化强度
  返回：
  - loss: 标量，总损失（合页损失 + L2 正则）
  - dW: (D, C)，权重梯度
  """
  loss = 0.0
  dW = np.zeros(W.shape)  # 初始化梯度为0,形状和W完全一致: (3072, 10)（3072维特征对应10个类别）
  num_train = X.shape[0]  # 样本数 N
  num_classes = W.shape[1]  # 类别数 C
  delta = 1.0  # 合页损失的间隔阈值

  # ==============================================
  # 一、向量化计算损失（对比循环版：不用两层 for 循环）
  # ==============================================
  # 1. 计算所有样本在所有类别上的得分
  # 循环版：scores = X[i].dot(W) （逐样本循环,每次算 1 个样本的得分，结果是 (1,10)（一维数组））
  # 向量版：直接矩阵乘法，一次算出所有样本得分
  scores = X.dot(W)            #  X（500,3072） & W（3072,10） = （500,10）（二维矩阵）
  # 2. 取出每个样本的【正确类别得分】
  # 循环版：correct_class_score = scores[y[i]] （逐样本取）
  # 向量版：用高级索引一次性取所有样本的正确类得分
  correct_class_scores = scores[np.arange(num_train), y]  # shape: (N,)
  '''
  # np.arange(num_train): 生成 0~499 的数组（样本索引：0,1,2,...,num_train-1）
  scores[行索引数组, 列索引数组] 的规则：
  行索引数组: np.arange(num_train) → 选第 0 行、第 1 行、第 2 行... 第 N-1 行（所有样本）
  列索引数组: y                    → 对第 0 行选第 y[0] 列、 第 1 行选第 y[1] 列 ... 第 N-1 行选第 y[N-1] 列 （指定行 + 对应列）
  最终： 把这些选中的元素按顺序拼成一个一维数组 → correct_class_scores
  scores[i, j]	取第 i 行第 j 列的单个元素
  scores[i, :]	取第 i 行所有列（整行）
  scores[:, j]	取所有行第 j 列（整列）
  scores[布尔矩阵]	取布尔值为 True 的元素
  scores[行数组, 列数组]	按数组一一对应取元素  例:scores[[0,1], [3,1]], 取 第0行第3列(0,3), 第1行第1列(1,1) 的元素
  '''
  # 3. 计算所有样本-类别对的 margin（合页损失核心）
  # 循环版：margin = scores[j] - correct_class_score + delta （逐样本逐错误类循环）
  # 向量版：广播机制，所有样本的所有类别得分 - 对应样本的正确类别得分 + 间隔阈值
  # np.newaxis：给数组增加一个长度为1的维度（(N,)→(N,1)），目的是触发NumPy广播
  margin = scores - correct_class_scores[:, np.newaxis] + delta  # shape: (N, C)
  # 正确类别 j == y[i] 时 margin 置 0（避免正确类别参与损失计算，完全对应循环版里的 if j == y[i]: continue）
  margin[np.arange(num_train), y] = 0.0  # 精准定位到「每个样本的真实类别位置」，将这些位置的 margin 赋值为 0.0
  # 4. 合页损失：max(0, margin)，只累加 >0 的部分
  # 循环版：if margin > 0: loss += margin
  # 向量版：用布尔索引筛选 >0 的 margin，直接求和
  loss = np.sum(margin[margin > 0])
  # 5. 损失平均 + L2 正则化（和循环版完全一致）
  loss /= num_train
  loss += reg * np.sum(W * W)

  # ==============================================
  # 二、向量化计算梯度（对比循环版：梯度更新逻辑完全等价）
  # ==============================================
  # 1. 构造 mask：只有 margin > 0 的位置才需要更新梯度
  # 循环版：if margin > 0: 才更新 dW
  # 向量版：用布尔矩阵标记需要更新的位置
  mask = (margin > 0).astype(int)  # shape: (N, C)，>0 为 1，否则为 0
  # 2. 统计每个样本有多少个“危险错误类”（用于正确类梯度计算）
  # 循环版：每个危险错误类让正确类梯度 -= X[i]
  # 向量版：对每个样本 i，正确类 y[i] 的梯度 = -X[i] × 危险错误类数量
  count_per_sample = np.sum(mask, axis=1)  # shape: (N,)，每个样本的危险错误类个数
  # 3. 构造梯度贡献矩阵
  # 循环版：
  #   dW[:, y[i]] -= X[i]  （正确类）
  #   dW[:, j] += X[i]     （错误类 j）
  # 向量版：用矩阵运算批量完成
  dW_contrib = np.zeros_like(scores)  # shape: (N, C)
  # 步骤1：先给正确类赋值 -count_per_sample（不会被覆盖）
  dW_contrib[np.arange(num_train), y] = -count_per_sample
  # 步骤2：再给错误类赋值 1（正确类已提前赋值，不会被覆盖）
  dW_contrib += mask  # 等价于 dW_contrib[mask] = 1，但避免覆盖
  '''
  dW_contrib[mask] = 1  # 危险错误类贡献 +1
  # 对每个样本 i，正确类 y[i] 的贡献 = -危险错误类数量
  dW_contrib[np.arange(num_train), y] = -count_per_sample
  '''
  # 4. 用矩阵乘法计算梯度（X.T × dW_contrib）
  # 循环版：逐样本逐错误类更新 dW
  # 向量版：X.T (D,N) × dW_contrib (N,C) = (D,C)，直接得到总梯度
  dW = X.T.dot(dW_contrib)
  # 5. 梯度平均 + L2 正则化梯度（和循环版完全一致）
  dW /= num_train
  dW += 2 * reg * W

  return loss, dW