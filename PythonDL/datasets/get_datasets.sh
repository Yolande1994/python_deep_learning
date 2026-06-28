#!/bin/bash
# 自动下载并解压CIFAR-10数据集
# 依赖wget工具，适用于Linux/macOS及Windows Git Bash/WSL环境

# 从多伦多大学官网下载CIFAR-10 Python版本压缩包
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# 解压数据集
tar -xzvf cifar-10-python.tar.gz
# 删除压缩包
rm cifar-10-python.tar.gz