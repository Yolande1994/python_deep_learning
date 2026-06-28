from __future__ import print_function    # 兼容Python2和Python3的print函数语法，确保跨版本运行
from six.moves import cPickle as pickle  # 导入兼容版的pickle（序列化库），用于读取CIFAR-10的二进制数据文件
# pickle 是Python内置的序列化/反序列化模块，可以把Python里的任意对象（如列表、字典、数组、自定义类实例）转换成「二进制字节流」（序列化），也能把这个字节流还原成原来的对象（反序列化）
import numpy as np
import os   # 导入os库，处理文件路径、目录操作
from matplotlib.pyplot import imread     # 导入matplotlib的图片读取函数（替代废弃的scipy.misc.imread）
import platform                          # 导入platform库，获取Python版本信息，做版本兼容

# 兼容 Python2/3的 pickle 文件读取函数 (处理 CIFAR-10 用 pickle 是因为 CIFAR-10 数据集本身是 pickle 序列化的二进制文件，必须用它反序列化才能读取数据)
def load_pickle(f):
    version = platform.python_version_tuple()     # 获取Python版本
    if version[0] == '2':                         # Python2直接读取
        return pickle.load(f)                     # pickle.load(): 反序列化功能，把二进制文件还原成 Python 对象
    elif version[0] == '3':                       # Python3需指定latin1编码，兼容旧文件，避免中文/特殊字符乱码
        return pickle.load(f, encoding='latin1')
    raise ValueError("无效Python版本: {}".format(version)) # 非2/3版本抛出错误

# 读取单个 CIFAR-10 批次文件
def load_CIFAR_batch(filename):
  """
  读取单个 CIFAR-10 批次文件
  作用：解析单个 CIFAR 批次的二进制文件，返回图片数据和标签
  参数：filename → CIFAR 批次文件路径（如 data_batch_1）
  返回：X - 该批次的图片数据，形状(10000, 32, 32, 3)，float类型
      Y - 该批次的标签数据，形状(10000,)，int类型（0-9对应10个类别）
  """
  with open(filename, 'rb') as f: # 以二进制读模式打开文件 ('r':文本只读模式（默认）, 'rb':二进制只读模式, 'w':文本写入模式（会清空原有内容）, 'wb':二进制写入模式)
    datadict = load_pickle(f)  # 调用兼容版pickle读取文件内容
    X = datadict['data']       # 取出图片数据：原始形状(10000, 3072)，3072=3*32*32（3通道、32x32像素）
    Y = datadict['labels']     # 取出标签数据：列表形式，长度10000，每个元素是0-9的数字
    '''
    数据格式转换（核心步骤）：
    1. reshape：  把原始的一维向量恢复成图片的维度结构 (10000, 3, 32, 32)（样本数, 通道数, 高度, 宽度）
    2. transpose：转置CIFAR-10原始数据维度 (10000, 3, 32, 32) → (10000, 32, 32, 3) 即: (样本数, 通道数, 高度, 宽度) → (样本数, 高度, 宽度, 通道数)，符合视觉习惯
    3. astype：   转为float类型，方便后续数值计算
    原始数据是「通道优先」（通道在高、宽前面），适合模型计算，但不符合人的视觉习惯；
    转置后是 「通道在后」（高、宽、通道），比如取第0张图片：X[0] 是 (32,32,3)，可以直接用 matplotlib 显示（因为 plt.imshow() 要求输入 (高，宽，通道)）。
    '''
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)  # 将标签列表转为numpy数组，方便后续处理   （长度10000列表 → (10000,)形状np数组）
    return X, Y

# 读取完整的 CIFAR-10 数据集
def load_CIFAR10(ROOT):
  """
  读取完整的 CIFAR-10 数据集
  作用：合并 5个训练批次，读取 1个测试批次，返回完整的 训练/测试 数据
  参数：ROOT - CIFAR-10 数据集根目录路径
  返回：X_train - 训练集图片，形状(50000, 32, 32, 3)
       Y_train - 训练集标签，形状(50000,)
       X_test - 测试集图片，形状(10000, 32, 32, 3)
       Y_test - 测试集标签，形状(10000,)

  CIFAR-10 有 6万张 32*32图, 5万张训练图被分成了 5个独立的批次文件（data_batch_1 到 data_batch_5），每个批次 1万张.  1万张测试图单独放在 test_batch 文件里
  os.path.join() 是 Python 标准库 os 提供的函数，专门用来拼接文件路径.  它会自动根据操作系统（Windows/Linux/macOS）选择正确的路径分隔符（如 Windows 用 \，Linux/macOS 用 /）
  'data_batch_%d' % (b, )：生成批次文件名.   %d 是占位符，表示这里要插入一个整数.   % (b, ) 表示把变量 b 的值插入到 %d 的位置
  """
  # 初始化空列表，存储各批次的图片和标签
  XS = []  # 存训练集图片
  YS = []  # 存训练集标签
  # 遍历5个训练批次（data_batch_1到data_batch_5）
  for b in range(1,6):
    # 拼接批次文件路径（输入：数据集根目录ROOT和当前批次号b    输出：当前批次文件的完整路径字符串，赋值给变量f    (ROOT, data_batch_1) → (ROOT/data_batch_1) ）
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)  # 读取当前批次的图片和标签 X.shape = (10000,32,32,3), Y.shape = (10000,)
    XS.append(X)
    YS.append(Y)
  # 合并所有训练批次：将5个(10000, 32, 32, 3)数组合并为(50000, 32, 32, 3)      np.concatenate(): NumPy 中拼接多个数组的核心函数
  X_train = np.concatenate(XS)  # X_train.shape = (50000, 32, 32, 3)
  Y_train = np.concatenate(YS)  # Y_train.shape = (50000,)
  X_test, Y_test = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))  # 读取测试批次文件（test_batch）   shape: (10000, 32, 32, 3),  (10000,)
  return X_train, Y_train, X_test, Y_test  # 返回完整的 训练/测试 数据

# 加载并预处理 CIFAR-10 数据（核心函数，给模型直接使用）
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True):
    """
    加载并预处理 CIFAR-10 数据（核心函数，给模型直接使用）
    作用：  1. 加载原始 CIFAR-10 数据
          2. 拆分训练/验证/测试集（减少数据量，加快训练）
          3. 数据归一化（减去均值图像）
          4. 调整维度顺序（适配 PyTorch等框架）
    参数： num_training - 训练集样本数，默认49000
         num_validation - 验证集样本数，默认1000
         num_test - 测试集样本数，默认1000
         subtract_mean - 是否减去均值图像（归一化），默认True
    返回：字典，包含预处理后的 训练/验证/测试 集：
        { 'X_train': 训练集图片, 'y_train': 训练集标签,
          'X_val': 验证集图片, 'y_val': 验证集标签,
          'X_test': 测试集图片, 'y_test': 测试集标签   }
    """
    # 1. 加载原始 CIFAR-10 数据（需确保数据集路径正确）
    cifar10_dir = 'cifar-10-batches-py' # CIFAR-10数据集的解压目录,可改成自己存放CIFAR-10数据集的实际路径， (CIFAR-10:包含 data_batch_1、test_batch 等二进制文件)
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # 2. 子集采样（拆分训练/验证集，减少数据量）
    # 验证集：从训练集的第49000个样本开始，取1000个（49000-49999）
    mask = list(range(num_training, num_training + num_validation))  # 生成索引列表 [49000, 49001, 49002, ..., 49999]
    X_val = X_train[mask]  # 验证集图片
    y_val = y_train[mask]  # 验证集标签
    # 训练集：取前49000个样本
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    # 测试集：取前1000个样本（减少数据量，加快测试）
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 3. 数据归一化：减去训练集的均值图像（提升模型精度）
    '''
    一. 什么是归一化（这里是 “去均值”）？
    在这段代码里，归一化特指减去均值图像（Mean Subtraction）：
      先计算训练集所有图片在每个像素位置上的平均值，得到一张 “均值图像”（形状和单张图片一样：(32, 32, 3)）；
      再把训练集、验证集、测试集中的每一张图片，都减去这张均值图像。
    直观理解：
      原始图片：每个像素值在 0~255 之间；
      去均值后：每个像素值围绕 0 上下浮动（有正有负），整体分布更 “中心化”。
    
    二. 为什么要做归一化？
    核心目的是提升模型训练的稳定性和精度，具体有三点：
    1.加速梯度下降收敛
      图像原始像素值范围大（0~255），会导致梯度更新时步长不稳定，模型收敛慢；
      去均值后，数据分布更接近 0，梯度下降更容易找到最优解，训练更快。
    2.提升模型精度
      图像的 “均值部分” 是所有图片的共性（比如 CIFAR-10 整体偏暗），减去均值后，模型更关注 “差异部分”（每张图的独特特征），而不是共性，分类更准。
    3.避免数据泄露
      代码里只用训练集的均值去归一化验证集和测试集，而不是用全部数据的均值，这样可以避免 验证/测试 集的信息泄露到训练过程中，保证模型泛化能力。
    '''
    if subtract_mean:
      mean_image = np.mean(X_train, axis=0)  # 计算训练集所有图片的均值（按像素位置求平均）.  X_train形状：(49000, 32, 32, 3) → mean_image形状：(32, 32, 3)
      # 可以把 X_train 想象成 49000 张叠在一起的 32×32×3 的图片，mean_image 是一张 “模板图”。广播就是把这张 “模板图” 复制 49000 次，然后和每张图逐像素相减，最后得到的还是 49000 张图片，只是每张图都 “减去了共性”
      X_train -= mean_image  # 训练集去均值
      X_val -= mean_image    # 验证集去均值（用训练集的均值，避免数据泄露）
      X_test -= mean_image   # 测试集去均值（同上）

    # 4. 调整维度顺序：从(样本数, 高, 宽, 通道) → (样本数, 通道, 高, 宽)
    # 适配PyTorch/TensorFlow等框架的输入格式（通道优先）
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # 5. 打包成字典返回，方便调用
    return { 'X_train': X_train, 'y_train': y_train,
             'X_val': X_val, 'y_val': y_val,
             'X_test': X_test, 'y_test': y_test,      }


# 两段代码分别实现了 CIFAR-10 和 TinyImageNet 两个数据集的加载与预处理逻辑，两个函数 get_CIFAR10_data / load_tiny_imagenet 相互独立，分别服务于不同的数据集


# 加载 TinyImageNet 数据集（64×64像素的小尺寸ImageNet）
def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
  """
  加载 TinyImageNet 数据集（64×64像素的小尺寸ImageNet）
  适用：TinyImageNet-100-A/B、TinyImageNet-200（目录结构相同）
  参数：
      path - 数据集根目录路径
      dtype - 数据类型，默认float32（适合模型计算）
      subtract_mean - 是否减去训练集均值图像，默认True
  返回：
      字典，包含：
      - class_names: 类别名称列表（每个类别对应WordNet名称）
      - X_train: 训练集图片，形状(N_tr, 3, 64, 64)
      - y_train: 训练集标签，形状(N_tr,)
      - X_val: 验证集图片，形状(N_val, 3, 64, 64)
      - y_val: 验证集标签，形状(N_val,)
      - X_test: 测试集图片，形状(N_test, 3, 64, 64)
      - y_test: 测试集标签（若文件不存在则为None）
      - mean_image: 训练集均值图像，形状(3, 64, 64)
  """
  # 1. 读取类别ID列表（wnids.txt存储所有类别唯一标识）
  with open(os.path.join(path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]  # 去除每行首尾空格，得到wnid列表

  # 2. 建立wnid到数字标签的映射（如'n01443537' → 0）
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # 3. 读取类别名称（words.txt存储wnid对应的自然语言名称）
  with open(os.path.join(path, 'words.txt'), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)  # 按制表符分割，构建wnid到名称的映射
    # 处理每个类别的名称（分割逗号、去除空格）
    for wnid, words in wnid_to_words.items():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
  class_names = [wnid_to_words[wnid] for wnid in wnids]  # 生成类别名称列表（按标签顺序）

  # 4. 加载训练集数据
  X_train = []  # 存训练集图片
  y_train = []  # 存训练集标签
  for i, wnid in enumerate(wnids):  # 遍历每个类别
    if (i + 1) % 20 == 0:  # 每加载20个类别打印进度（方便查看加载状态）
      print('正在加载训练集类别 %d / %d' % (i + 1, len(wnids)))
    # 读取该类别的图片文件名列表（boxes.txt存储文件名和标注框）
    boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      filenames = [x.split('\t')[0] for x in f]  # 取每行第一个字段（文件名）
    num_images = len(filenames)  # 该类别图片数量
    
    X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)             # 初始化当前类别图片数组：(图片数, 3, 64, 64)
    y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)  # 初始化当前类别标签：所有图片对应同一个数字标签
    # 遍历该类别所有图片文件
    for j, img_file in enumerate(filenames):
      img_file = os.path.join(path, 'train', wnid, 'images', img_file)  # 拼接图片文件路径
      img = imread(img_file)  # 读取图片（返回形状(64, 64, 3)或(64,64)（灰度图））
      if img.ndim == 2:  # 处理灰度图：增加通道维度（64,64）→ (64,64,1)
        img.shape = (64, 64, 1)
      X_train_block[j] = img.transpose(2, 0, 1)  # 调整维度顺序：(高, 宽, 通道) → (通道, 高, 宽)
    # 将当前类别数据加入训练集列表
    X_train.append(X_train_block)
    y_train.append(y_train_block)
      
  # 合并所有类别的训练数据
  X_train = np.concatenate(X_train, axis=0)
  y_train = np.concatenate(y_train, axis=0)
  
  # 5. 加载验证集数据
  with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
    img_files = []  # 验证集图片文件名
    val_wnids = []  # 验证集类别wnid
    for line in f:
      img_file, wnid = line.split('\t')[:2]  # 取每行前两个字段（文件名、wnid）
      img_files.append(img_file)
      val_wnids.append(wnid)
    num_val = len(img_files)  # 验证集图片数量
    y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])  # 转换验证集标签为数字
    X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)            # 初始化验证集图片数组
    # 遍历验证集图片
    for i, img_file in enumerate(img_files):
      img_file = os.path.join(path, 'val', 'images', img_file)
      img = imread(img_file)
      if img.ndim == 2:
        img.shape = (64, 64, 1)
      X_val[i] = img.transpose(2, 0, 1)

  # 6. 加载测试集数据（学生版通常无测试标签）
  # 读取测试集图片文件名列表
  img_files = os.listdir(os.path.join(path, 'test', 'images'))
  X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)  # 初始化测试集图片数组
  # 遍历测试集图片
  for i, img_file in enumerate(img_files):
    img_file = os.path.join(path, 'test', 'images', img_file)
    img = imread(img_file)
    if img.ndim == 2:
      img.shape = (64, 64, 1)
    X_test[i] = img.transpose(2, 0, 1)

  y_test = None  # 初始化测试集标签为None（默认无测试标签）
  y_test_file = os.path.join(path, 'test', 'test_annotations.txt')  # 检查是否存在测试标签文件
  if os.path.isfile(y_test_file):
    with open(y_test_file, 'r') as f:
      img_file_to_wnid = {}
      for line in f:
        line = line.split('\t')
        img_file_to_wnid[line[0]] = line[1]
    # 转换测试集标签为数字
    y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
    y_test = np.array(y_test)

  # 7. 计算训练集均值图像并归一化
  mean_image = X_train.mean(axis=0)
  if subtract_mean:  # 增加维度：(3,64,64) → (1,3,64,64)，方便广播运算
    X_train -= mean_image[None]
    X_val -= mean_image[None]
    X_test -= mean_image[None]
  # 返回处理后的数据集字典
  return {
    'class_names': class_names,
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,
    'mean_image': mean_image,
  }


# 加载磁盘中保存的模型文件
def load_models(models_dir):
  """
  加载磁盘中保存的模型文件
  作用：批量读取目录下的pickle模型文件，跳过无法解析的文件（如README）
  参数：models_dir - 模型文件所在目录路径
  返回：字典，键为模型文件名，值为模型对象
  """
  models = {}  # 存储加载的模型
  for model_file in os.listdir(models_dir):
    with open(os.path.join(models_dir, model_file), 'rb') as f:  # 拼接模型文件路径
      try:
        models[model_file] = load_pickle(f)['model']  # 读取模型（假设模型文件是pickle字典，包含'model'字段）
      except pickle.UnpicklingError:
        continue  # 跳过解析失败的文件（如非pickle文件）
  return models



if __name__ == "__main__":
    cifar10_dir = 'cifar-10-batches-py'
    # 调用函数
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    d = get_CIFAR10_data()
    # 打印结果
    print("训练数据 shape:", X_train.shape)
    print("训练标签 shape:", y_train.shape)
    print("测试数据 shape:", X_test.shape)
    print("测试标签 shape:", y_test.shape)
    print(d)