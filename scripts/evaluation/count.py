import os
import glob
import torch
import numpy as np
from models.crowd_model import CrowdCountingModel
from ultralytics import YOLO

def evaluate_crowd_model(weights_path, val_images_dir, val_labels_dir):
    """
    加载自定义的 YOLO11 人群计数模型，测试不同置信度阈值下的 MAE 和 RMSE。
    """
    print(f"正在加载自定义模型...")
    model = CrowdCountingModel("models/yolo11-crowd.yaml", ch=3, nc=1, verbose=False)
    
    if os.path.exists(weights_path):
        print(f"从 {weights_path} 加载权重")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "model" in ckpt and ckpt["model"] is not None:
                state_dict = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt["model"]
            elif "ema" in ckpt and ckpt["ema"] is not None:
                state_dict = ckpt["ema"].state_dict() if hasattr(ckpt["ema"], "state_dict") else ckpt["ema"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
    else:
        print(f"错误: 未找到权重文件 {weights_path}")
        return
        
    model.eval()
    
    # 封装在 YOLO 对象中
    yolo = YOLO("yolo11n.yaml")
    yolo.model = model
    
    img_paths = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    print(f"找到 {len(img_paths)} 张验证图片。")
    
    # 获取所有图片的真实人数
    gt_counts = []
    valid_img_paths = []
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, img_name.replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                gt_count = len(f.readlines())
            gt_counts.append(gt_count)
            valid_img_paths.append(img_path)
    
    gt_counts = np.array(gt_counts)
    
    # 根据模型微调响应，搜索中高置信度阈值以找出最佳 MAE
    conf_thresholds = [0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50]
    
    print("\n开始在不同置信度阈值下评估...")
    best_mae = float('inf')
    best_conf = 0.0
    best_rmse = 0.0
    
    # 预存所有前向推理的原始检测框/置信度，以避免重复进行网络前向推理
    # YOLO.predict 会执行全套前向+后处理，我们可以直接运行预测并保存不同阈值下的结果
    # 为了速度，我们直接对每个 conf 进行完整测试
    for conf in conf_thresholds:
        pred_counts = []
        for img_path in valid_img_paths:
            results = yolo.predict(img_path, conf=conf, imgsz=1024, max_det=2000, verbose=False)
            pred_count = len(results[0].boxes)
            pred_counts.append(pred_count)
            
        pred_counts = np.array(pred_counts)
        mae = np.mean(np.abs(gt_counts - pred_counts))
        rmse = np.sqrt(np.mean((gt_counts - pred_counts) ** 2))
        
        print(f"Conf = {conf:.2f} -> 预测总数: {np.sum(pred_counts)}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        if mae < best_mae:
            best_mae = mae
            best_rmse = rmse
            best_conf = conf
            
    print("\n" + "="*40)
    print("最佳配置结果:")
    print(f"最佳置信度阈值 (Conf): {best_conf:.2f}")
    print(f"真实总人数: {np.sum(gt_counts)}")
    print(f"对应 MAE: {best_mae:.2f}")
    print(f"对应 RMSE: {best_rmse:.2f}")
    print("="*40)
    
if __name__ == "__main__":
    weights = "runs/detect/crowd_counting_v4_1024/weights/best.pt"
    images_dir = "datasets/shanghaitech_AB/images/val"
    labels_dir = "datasets/shanghaitech_AB/labels/val"
    
    evaluate_crowd_model(weights, images_dir, labels_dir)
