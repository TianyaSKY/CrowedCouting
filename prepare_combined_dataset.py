import os
import glob
import cv2
import scipy.io as sio
import numpy as np
import shutil
import yaml

def parse_mat_file(mat_path):
    """解析 ShanghaiTech 的 .mat 格式真值文件。"""
    try:
        mat = sio.loadmat(mat_path)
        img_info = mat['image_info']
        if 'location' in img_info.dtype.names:
            locs = img_info['location'][0,0]
        else:
            locs = img_info[0, 0][0][0, 0][0]
        return locs
    except Exception as e:
        try:
            mat = sio.loadmat(mat_path)
            locs = mat['image_info'][0][0][0][0][0]
            return locs
        except Exception as e2:
            import h5py
            with h5py.File(mat_path, 'r') as f:
                locs = np.array(f['image_info']['location']).T
            return locs

def process_single_part(src_dir, dest_dir, prefix):
    """处理单个 Part 并在文件名前加上前缀，防止冲突。"""
    os.makedirs(os.path.join(dest_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, 'labels', 'val'), exist_ok=True)

    train_images = glob.glob(os.path.join(src_dir, 'train_data', 'images', '*.jpg'))
    test_images = glob.glob(os.path.join(src_dir, 'test_data', 'images', '*.jpg'))
    
    splits = {
        'train': train_images,
        'val': test_images
    }
    
    for split, img_list in splits.items():
        for i, img_path in enumerate(img_list):
            img_name = os.path.basename(img_path)
            new_img_name = f"{prefix}_{img_name}"
            
            dest_img_path = os.path.join(dest_dir, 'images', split, new_img_name)
            shutil.copy(img_path, dest_img_path)
            
            if split == 'train':
                mat_path = os.path.join(src_dir, 'train_data', 'ground_truth', 'GT_' + img_name.replace('.jpg', '.mat'))
            else:
                mat_path = os.path.join(src_dir, 'test_data', 'ground_truth', 'GT_' + img_name.replace('.jpg', '.mat'))
                
            locs = parse_mat_file(mat_path)
            
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            
            new_label_name = new_img_name.replace('.jpg', '.txt')
            dest_label_path = os.path.join(dest_dir, 'labels', split, new_label_name)
            
            with open(dest_label_path, 'w') as f:
                for pt in locs:
                    x, y = pt[0], pt[1]
                    nx = x / w
                    ny = y / h
                    nx = max(0, min(nx, 1.0))
                    ny = max(0, min(ny, 1.0))
                    f.write(f"0 {nx:.6f} {ny:.6f} 0.010000 0.010000\n")
                    
            if (i + 1) % 100 == 0:
                print(f"[{prefix}] 已处理 {i+1}/{len(img_list)} 张 {split} 图片")

def prepare_combined_dataset():
    dest = 'datasets/shanghaitech_AB'
    print(f"正在清理并准备合并后的数据集文件夹: {dest}")
    if os.path.exists(dest):
        shutil.rmtree(dest)
        
    process_single_part('data/part_A_final', dest, 'part_A')
    process_single_part('data/part_B_final', dest, 'part_B')
    
    yaml_path = os.path.join(dest, 'dataset.yaml')
    dataset_dict = {
        'path': os.path.abspath(dest),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': {0: 'person'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_dict, f, default_flow_style=False)
        
    print(f"合并数据集完成。YAML 配置文件已保存至 {yaml_path}")

if __name__ == '__main__':
    prepare_combined_dataset()
