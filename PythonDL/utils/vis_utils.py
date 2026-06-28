from math import sqrt, ceil
import numpy as np


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
  将4维图像张量拼接为网格图，便于批量可视化
  参数:
  - Xs: 输入图像数据，形状 (N, H, W, C)
  - ubound: 输出网格的像素值上限，将数据归一化到 [0, ubound] 范围
  - padding: 网格中相邻图像之间的空白像素间距
  返回:
  - grid: 拼接完成的网格图像数组
  """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


def vis_grid(Xs):
    """
  将多张图像拼接为网格图，自动计算行列数
  参数:
  - Xs: 输入图像数据，形状 (N, H, W, C)
  返回:
  - G: 归一化到 [0,1] 范围的网格图像数组
  """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y * H + y:(y + 1) * H + y, x * W + x:(x + 1) * W + x, :] = Xs[n, :, :, :]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_nn(rows):
    """
  按二维行列结构拼接多组图像
  参数:
  - rows: 二维图像列表，每行对应一组图像
  返回:
  - G: 归一化到 [0,1] 范围的拼接网格图像
  """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y * H + y:(y + 1) * H + y, x * W + x:(x + 1) * W + x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G