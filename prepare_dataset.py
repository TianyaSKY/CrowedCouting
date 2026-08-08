import os
import glob
import cv2
import scipy.io as sio
import numpy as np
import shutil
import random
import yaml

def parse_mat_file(mat_path):
    """
    健壮地解析 ShanghaiTech 的 .mat 格式真值文件。
    """
    try:
        mat = sio.loadmat(mat_path)
        img_info = mat['image_info']
        # 取决于 scipy 解析嵌套 MATLAB 结构体的方式，
        # 坐标数据通常埋藏在深处：
        # 常见结构为: image_info -> 1x1 结构体 -> 'location' -> Nx2 数组
        if 'location' in img_info.dtype.names:
            locs = img_info['location'][0,0]
        else:
            # 兼容使用旧版本 scipy 解析的 ShanghaiTech 格式的备选方案
            locs = img_info[0, 0][0][0, 0][0]
        return locs
    except Exception as e:
        try:
            # 如果上面的索引提取失败，尝试其他的常规提取方式
            mat = sio.loadmat(mat_path)
            locs = mat['image_info'][0][0][0][0][0]
            return locs
        except Exception as e2:
            import h5py
            with h5py.File(mat_path, 'r') as f:
                locs = np.array(f['image_info']['location']).T
            return locs

def prepare_dataset(src_dir, dest_dir, val_ratio=0.1):
    os.makedirs(os.path.join(dest_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'labels', 'val'), exist_ok=True)

    # 获取所有训练图片
    train_images = glob.glob(os.path.join(src_dir, 'train_data', 'images', '*.jpg'))
    # 获取所有测试图片
    test_images = glob.glob(os.path.join(src_dir, 'test_data', 'images', '*.jpg'))
    
    # 我们可以从训练数据中切分出一部分作为验证集
    random.seed(42)
    random.shuffle(train_images)
    
    num_val = int(len(train_images) * val_ratio)
    val_list = train_images[:num_val]
    train_list = train_images[num_val:]
    
    # 将测试图片也作为验证集，以不占用训练数据为前提
    # ShanghaiTech 数据集本身自带测试集
    val_list = test_images
    train_list = train_images
    
    splits = {
        'train': train_list,
        'val': val_list
    }
    
    for split, img_list in splits.items():
        for i, img_path in enumerate(img_list):
            img_name = os.path.basename(img_path)
            # 图片保存路径
            dest_img_path = os.path.join(dest_dir, 'images', split, img_name)
            shutil.copy(img_path, dest_img_path)
            
            # 对应的标注文件路径
            # src_dir / {split}_data / ground_truth / GT_{img_name.replace('.jpg', '.mat')}
            if split == 'train':
                mat_path = os.path.join(src_dir, 'train_data', 'ground_truth', 'GT_' + img_name.replace('.jpg', '.mat'))
            else:
                mat_path = os.path.join(src_dir, 'test_data', 'ground_truth', 'GT_' + img_name.replace('.jpg', '.mat'))
                
            locs = parse_mat_file(mat_path)
            
            # 获取图片尺寸以用于归一化坐标
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            label_name = img_name.replace('.jpg', '.txt')
            dest_label_path = os.path.join(dest_dir, 'labels', split, label_name)
            
            with open(dest_label_path, 'w') as f:
                for pt in locs:
                    x, y = pt[0], pt[1]
                    # YOLO 格式要求坐标归一化到 [0, 1] 之间
                    # 点预测格式：类别ID cx cy
                    # 这里为了兼容 YOLO 的原生读取器，我们写入极小的 w 和 h 虚拟边界框
                    nx = x / w
                    ny = y / h
                    # 限制坐标在 [0, 1] 范围内，防止越界报错
                    nx = max(0, min(nx, 1.0))
                    ny = max(0, min(ny, 1.0))
                    f.write(f"0 {nx:.6f} {ny:.6f} 0.010000 0.010000\n")
                    
            if (i + 1) % 50 == 0:
                print(f"已处理 {i+1}/{len(img_list)} 张 {split} 图片")

    # 创建 dataset.yaml 配置文件
    yaml_path = os.path.join(dest_dir, 'dataset.yaml')
    dataset_dict = {
        'path': os.path.abspath(dest_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': {0: 'person'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_dict, f, default_flow_style=False)
        
    print(f"数据集准备完成。YAML 配置文件已保存至 {yaml_path}")

if __name__ == '__main__':
    src = 'data/part_B_final'
    dest = 'datasets/shanghaitech_B'
    print(f"正在从 {src} 准备数据集并保存至 {dest}")
    prepare_dataset(src, dest)
