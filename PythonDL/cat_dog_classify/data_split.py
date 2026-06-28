import os
import random
import shutil

# ====================== 数据集划分配置 ======================
dataset_root = "./datasets"  # 原始数据集根目录，存放各分类子文件夹
output_root = "./datasets"   # 划分后训练集、验证集的输出根目录
split_ratio = 0.8            # 训练集占总数据的比例，剩余部分为验证集
categories = ["cat", "dog"]  # 类别文件夹名称列表
# ======================================================================

# 1. 创建目标目录结构
train_root = os.path.join(output_root, "train")
val_root = os.path.join(output_root, "val")

for cat in categories:
    # 创建 train/cat, train/dog
    os.makedirs(os.path.join(train_root, cat), exist_ok=True)
    # 创建 val/cat, val/dog
    os.makedirs(os.path.join(val_root, cat), exist_ok=True)

# 2. 遍历每个类别，划分图片
for cat in categories:
    # 拼接当前类别的源目录路径
    src_dir = os.path.join(dataset_root, cat)
    # 获取该文件夹下所有 jpg/png 图片
    all_images = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not all_images:
        print(f"警告：{src_dir} 中没有找到图片，请检查路径！")
        continue

    # 3. 随机打乱图片顺序（保证划分随机）
    random.shuffle(all_images)
    # 4. 按比例划分训练集/验证集
    split_idx = int(len(all_images) * split_ratio)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    print(f"类别 {cat}：总图片数 {len(all_images)}，训练集 {len(train_imgs)}，验证集 {len(val_imgs)}")

    # 5. 复制训练集图片
    for img_name in train_imgs:
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(train_root, cat, img_name)
        shutil.copy2(src_path, dst_path)  # copy2 保留文件元信息，比 copy 更稳妥

    # 6. 复制验证集图片
    for img_name in val_imgs:
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(val_root, cat, img_name)
        shutil.copy2(src_path, dst_path)

print("\n数据集划分完成")
print(f"训练集目录：{train_root}")
print(f"验证集目录：{val_root}")