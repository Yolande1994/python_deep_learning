# 这段代码用 Python 手动实现了一个「感知器（Perceptron）」，并用它从零训练、学习 AND 逻辑运算（与运算），最终让机器学会了 1+1=1、0+0=0 这种逻辑规则
import numpy as np

class Perceptron(object):
    # 初始化感知器，设置输入参数的个数，激活函数。  激活函数的类型为阶跃函数：输入 > 0 输出 1，否则 0
    def __init__(self, input_num, activator,learning_rate=0.1):
        self.activator = activator
        self.weights = np.zeros(input_num) # 权重向量初始化为0
        self.bias = 0.0                    # 偏置初始化为0
        self.learning_rate = learning_rate

    # 打印学习到的权重、偏置项 （ __str__ 是 Python 中自定义类打印输出的核心方法，只有定义了它，print(类实例) 才会输出你想要的内容，否则只会打印默认的内存地址）
    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    # 输入向量，输出感知器的计算结果
    def predict(self, input_vec):
        # 计算向量 input_vec[x1,x2,x3...] 和 weights[w1,w2,w3,...] 的内积, 然后加上bias
        return self.activator(np.dot(input_vec, self.weights) + self.bias)  # 激活函数：self.activator(z) → 把线性输出映射为 0/1 的分类结果

    # 输入训练数据：一组向量、与每个向量对应的label；以及训练轮数
    def train(self, input_vecs, labels, iteration):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels)

    # 一次迭代，把所有的训练数据过一遍
    def _one_iteration(self, input_vecs, labels):
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        for (input_vec, label) in zip(input_vecs, labels):
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label)

    # 按照感知器规则更新权重
    def _update_weights(self, input_vec, output, label):
        # 首先计算本次更新的delta (损失)
        delta = label - output
        # wi = wi + rate * (t-y) * xi
        self.weights += input_vec * self.learning_rate * delta
        # 更新bias
        self.bias += self.learning_rate * delta

# 定义激活函数f
def f(x):
    return 1 if x > 0 else 0

# 构建训练数据集
def get_training_dataset():
    # 输入向量列表
    input_vecs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    # 期望的输出列表，注意要与输入一一对应
    labels = np.array([1, 0, 0, 0]) # # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    return input_vecs, labels

# 使用and真值表训练感知器 (未用到)
def train_and_perceptron():
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10)
    # 返回训练好的感知器
    return p


if __name__ == '__main__':
    # 训练and感知器
    And_perception = Perceptron(2, f)
    # 打印训练获得的权重
    print('训练之前: 初始权重与数据集')
    print(And_perception)  # 已定义 __str__(self)
    print('训练之前: 原始数据集与预测结果')
    print('1 and 1 = %d' % And_perception.predict([1, 1]))
    print('0 and 0 = %d' % And_perception.predict([0, 0]))
    print('1 and 0 = %d' % And_perception.predict([1, 0]))
    print('0 and 1 = %d' % And_perception.predict([0, 1]))
    print('\n')
    input_vecs, labels = get_training_dataset()
    epochs = 10
    And_perception.train(input_vecs, labels, epochs)  # 开始训练

    print('训练之后........')
    print(And_perception)
    # 测试
    print('1 and 1 = %d' % And_perception.predict([1, 1]))
    print('0 and 0 = %d' % And_perception.predict([0, 0]))
    print('1 and 0 = %d' % And_perception.predict([1, 0]))
    print('0 and 1 = %d' % And_perception.predict([0, 1]))