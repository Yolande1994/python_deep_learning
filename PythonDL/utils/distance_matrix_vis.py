# 导入绘图与数值计算库
import matplotlib.pyplot as plt
import numpy as np

# 配置绘图中文显示、负号渲染
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 构造距离矩阵：5行=测试样本，10列=训练样本，数值代表两两样本间L2距离
distance_matrix = np.array([
    [37, 95, 73, 60, 16, 16, 6, 87, 60, 71],
    [1, 99, 83, 21, 18, 18, 30, 52, 43, 29],
    [61, 14, 29, 37, 46, 79, 20, 51, 59, 5],
    [61, 17, 7, 95, 97, 81, 30, 10, 68, 44],
    [12, 50, 3, 91, 26, 66, 31, 52, 55, 18]
])

# 灰度热力图绘制距离矩阵，无插值保留原始数值色块
plt.imshow(distance_matrix, cmap='gray', interpolation='none')
# 右侧色条映射距离数值大小
plt.colorbar(label='距离值')
# 图表标题：标注矩阵行列对应样本含义
plt.title('小型距离矩阵可视化 (5测试样本 × 10训练样本)')
plt.xlabel('训练样本索引')
plt.ylabel('测试样本索引')
plt.show()