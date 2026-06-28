import argparse  # 核心作用：让程序可以「外部传参」（不传就用默认，传了就覆盖）

# 这段代码使用 Python 标准库 argparse 定义了一个命令行参数解析函数 parse_args，核心功能是为深度学习训练脚本配置超参数、路径等运行选项，并提供默认值
def parse_args():
    # 1. 创建解析器
    parser = argparse.ArgumentParser()
    # 2. 添加参数
    parser.add_argument('--per_batch_size', type = int, default = 16)  # 每批次样本数量
    parser.add_argument('--max_epoch', type = int, default = 20)       # 最大训练轮次
    parser.add_argument('--lr', type = float, default = 0.0002, help = 'learning rate')  # 初始学习率
    parser.add_argument('--decay_coeff', type = int, default = 0.5, help = 'decay coefficient for learning rate')  # 学习率衰减系数
    parser.add_argument('--lr_steps', type=int, nargs='+', default = [10,15], help = 'epoch when to decay learning rate')  # 学习率衰减的 epoch 节点（第 30、45 轮时降低学习率）
    parser.add_argument('--MULTI_GPU', type = bool , default = False, help = 'whether to use multi gpu or not')     # 是否启用多 GPU 训练
    parser.add_argument('--train_dir', type = str, default = "D:\\Python Study\\PythonDL\\cat_dog_classify\\datasets\\train", help = 'dir of train image files')  # 本地路径, 训练集存放目录
    parser.add_argument('--val_dir', type = str, default = "D:\\Python Study\\PythonDL\\cat_dog_classify\\datasets\\val", help = 'dir of val image files')        # 本地路径, 验证集存放目录
    parser.add_argument('--save_dir', type = str, default = "D:\\Python Study\\PythonDL\\cat_dog_classify", help = 'dir to save model')                           # 本地路径, 训练好的模型保存目录
    # 3. 解析参数
    args = parser.parse_args()  # 解析命令行传入的参数（若未传入，则自动使用默认值）
    return args

args = parse_args()  # 调用函数，获取解析后的参数，供脚本全局使用（如训练逻辑中通过 args.lr 读取学习率）

# 4. 使用参数
print(f"max_epoch：{args.max_epoch}，lr：{args.lr}")

