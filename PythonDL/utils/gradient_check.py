from __future__ import print_function

import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
  中心差分法计算数值梯度
  参数:
  - f: 单输入参数的损失函数
  - x: 待求梯度的numpy数组（如权重矩阵）
  - verbose: 是否打印每个维度的梯度值
  - h: 差分步长
  返回:
  - grad: 与x同形状的数值梯度矩阵
  """
    # 计算原点函数值
    fx = f(x)
    grad = np.zeros_like(x)
    # 多维迭代器遍历所有元素
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        # 计算x+h处的函数值
        x[ix] = oldval + h
        fxph = f(x)
        # 计算x-h处的函数值
        x[ix] = oldval - h
        fxmh = f(x)
        # 还原原始值
        x[ix] = oldval

        # 中心差分公式计算偏导数
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
  针对数组输出型函数计算数值梯度
  参数:
  - f: 输入输出均为numpy数组的函数
  - x: 输入数组
  - df: 上游梯度
  - h: 差分步长
  返回:
  - grad: 数值梯度数组
  """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
  针对blob格式输入输出计算数值梯度
  函数约定接收多个输入blob参数，最后一个参数为输出blob，用于写入计算结果。
  调用示例: f(x, w, out)
  其中x、w为输入Blob，计算结果写入out。

  参数:
  - f: 计算函数
  - inputs: 输入blob元组
  - output: 输出blob
  - h: 差分步长
  返回:
  - numeric_diffs: 各输入blob对应的数值梯度列表
  """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=['multi_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args: net.forward(),
                                         inputs, output, h=h)


# 稀疏梯度校验
def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
  随机采样部分维度，对比数值梯度与解析梯度的差异
  参数:
  - f: 损失函数
  - x: 待校验变量
  - analytic_grad: 解析梯度结果
  - num_checks: 采样校验的维度数
  - h: 差分步长
  """
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('数值梯度: %f 解析梯度: %f, 差异: %e' % (grad_numerical, grad_analytic, rel_error))