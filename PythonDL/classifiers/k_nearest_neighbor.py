import numpy as np
"""
kNN分类器实现
基于L2欧氏距离，支持双循环、单循环、纯向量化三种距离计算方式，包含多数投票预测逻辑
"""
class KNearestNeighbor(object):

  def __init__(self):
    pass  # kNN为懒惰学习算法，无训练参数，初始化无需额外操作


  def train(self, X, y):
    """
    存储训练样本与标签
    kNN无显式训练过程，仅记忆数据集，属于懒惰学习算法
    参数:
    - X: 训练样本，形状 (num_train, D)
    - y: 训练标签，形状 (num_train,)
    """
    self.X_train = X
    self.y_train = y


  def predict(self, X, k=1, num_loops=0):
    """
    对测试样本执行分类预测
    先计算距离矩阵，再基于k近邻多数投票输出预测标签
    参数:
    - X: 测试样本，形状 (num_test, D)
    - k: 近邻数量，默认值为1
    - num_loops: 距离计算模式，可选值：
        0: 纯向量化无循环（性能最优）
        1: 单循环向量化
        2: 双循环逐元素计算
    返回:
    - y_pred: 预测标签数组，形状 (num_test,)
    """
    # 根据 num_loops 的值选择不同的距离计算方法
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('循环数错误 %d' % num_loops)
    # 有了距离矩阵后，根据最近的 k 个邻居预测标签
    return self.predict_labels(dists, k=k)


  def compute_distances_two_loops(self, X):
    """
    用【双循环】计算测试样本与所有训练样本的 L2 距离
    优点：逻辑最直观，容易理解； 缺点：速度最慢，不适合大数据集。
    输入:
    - X: 测试数据，形状 (num_test, D)
    输出:
    - dists: 距离矩阵，形状 (num_test, num_train)
             dists[i, j] 是第 i 个测试样本与第 j 个训练样本的欧氏距离
    """
    num_test = X.shape[0]    # 测试样本数
    num_train = self.X_train.shape[0]  # 训练样本数
    dists = np.zeros((num_test, num_train))  # 初始化距离数组
    # 外层循环：遍历每个测试样本
    for i in range(num_test):
      # 内层循环：遍历每个训练样本
      for j in range(num_train):
        # 计算第 i 个测试点与第 j 个训练点之间的 L2 距离，保存到 dists[i, j]
        # 公式：L2距离 = sqrt( sum( (test[i] - train[j]) ^2 ) )
        dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
    return dists

  def compute_distances_one_loop(self, X):
    """
    用【单循环】计算距离
    对每个测试样本，一次性计算它与所有训练样本的距离，利用 NumPy 向量化广播加速。
    输入输出: 同上
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    # 只循环测试样本，对每个测试样本 i，一次性算它和所有训练样本的距离
    for i in range(num_test):
      # 计算第 i 个测试点与所有训练点之间的 L2 距离，保存到 dists[i, :]
      # 双循环是 “逐列赋值”（dists[i,j]）， 单循环是 “整行赋值”（dists[i,:] = dists[i]）
      # 核心是利用 NumPy 的向量化广播，一次性算完一行的值 ( X[i] 与 所有训练样本的距离（一维数组） )，不用再循环列
      dists[i, :] = np.sqrt(np.sum((X[i] - self.X_train) ** 2, axis=1))
    return dists

  def compute_distances_no_loops(self, X):
    """
    用【无循环】（纯向量化）计算距离
    这是最高效的实现，完全利用矩阵运算，避免 Python 循环。
    参数: 同之前
    提示：L2 距离公式可以展开为：
    ∑ (x - y)² = ∑ x² + ∑ y² - 2·x·yᵀ
    其中：
    ∑ x² ： 测试样本 x 自身的平方和（每个样本一个数）
    ∑ y² ： 训练样本 y 自身的平方和（每个样本一个数）
    x·yᵀ ： 测试样本与训练样本的矩阵乘法（得到交叉项）
    这样就可以用矩阵乘法和广播来计算，而不用循环。
    """
    # 用向量化方式计算所有测试点与训练点之间的 L2 距离，不用循环，结果保存到dists
    # 步骤:
    # 1. 计算 X_test² 的行和             (num_test, 1)
    # 2. 计算 X_train² 的行和            (num_train, 1)
    # 3. 计算交叉项：X_test @ X_train.T  (num_test, num_train)
    # 4. 代入公式：dists = sqrt( X_test² + X_train².T - 2 * X_test @ X_train.T )

    # 计算测试样本 X_test 每个样本元素的平方和      若没有keepdims=True,则形状: (num_test,)（一维数组）
    X_test_sq = np.sum(X**2, axis=1, keepdims=True)              # (num_test, 1)（二维数组）
    # 计算训练样本 X_train 每个样本元素的平方和
    X_train_sq = np.sum(self.X_train**2, axis=1, keepdims=True)  # (num_train, 1)
    # 计算矩阵乘法（得到交叉项: 每一行对应 1 个测试样本，每一列对应 1 个训练样本，交叉点的值是两者的点积）
    cross_term = X @ self.X_train.T                              # (num_test, num_train)
    # 代入公式
    dists = np.sqrt(X_test_sq + X_train_sq.T - 2 * cross_term)   # (num_test, num_train)
    '''
    广播机制生效：
    X_test_sq      (500,1) → 广播为 (500, 5000)（每行重复 5000 次）
    X_train_sq.T  (1,5000) → 广播为 (500, 5000)（每列重复 500 次）
    两者相加：正好得到每个测试样本与每个训练样本的'平方和'之和，形状与 cross_term 完全一致
    '''
    return dists


  # 根据距离矩阵，为每个测试样本预测标签
  def predict_labels(self, dists, k=1):
    """
    对每个测试样本：
    1. 找到距离最近的 k个训练样本
    2. 看这 k个样本的标签，哪个出现次数最多，就预测为哪个标签
    输入:
    - dists: 距离矩阵，形状为 (num_test, num_train)
    - k:     最近邻居的数量
    输出:
    - y_pred: 预测标签，形状为 (num_test,)
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)  # 初始化预测结果
    for i in range(num_test):
      # 用 np.argsort 对距离排序，取前 k 个，然后从 self.y_train 中取出标签
      sorted_indexes = np.argsort(dists[i])         # 对第 i 行的距离从小到大排序，得到索引
      closest_y = self.y_train[sorted_indexes[:k]]  # 取前 k 个索引对应的标签
      # 用 np.bincount 统计次数，再用 np.argmax 找到最大值的索引
      bincount = np.bincount(closest_y)             # np.bincount()核心：统计非负整数的出现次数，索引 = 数字，值 = 次数
      y_pred[i] = np.argmax(bincount)
    return y_pred  # (num_test,)一维数组