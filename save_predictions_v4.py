import os
import sys
import cv2
import torch
import numpy as np
sys.path.append("/home/tianya/PythonProjects/CrowedCouting")
from models.crowd_model import CrowdCountingModel
from ultralytics import YOLO

def visualize_samples(weights_path, val_images_dir, val_labels_dir, output_dir, samples_list, conf=0.42):
    print("正在加载模型...")
    model = CrowdCountingModel("models/yolo11-crowd.yaml", ch=3, nc=1, verbose=False)
    
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "model" in ckpt and ckpt["model"] is not None:
                state_dict = ckpt["model"].state_dict()
            elif "ema" in ckpt and ckpt["ema"] is not None:
                state_dict = ckpt["ema"].state_dict() if hasattr(ckpt["ema"], "state_dict") else ckpt["ema"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
        print(f"权重已加载: {weights_path}")
    else:
        print(f"错误: 未找到权重文件 {weights_path}")
        return
        
    model.eval()
    
    yolo = YOLO("yolo11n.yaml")
    yolo.model = model
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in samples_list:
        img_path = os.path.join(val_images_dir, img_name)
        label_path = os.path.join(val_labels_dir, img_name.replace(".jpg", ".txt"))
        
        if not os.path.exists(img_path):
            print(f"找不到图片: {img_path}")
            continue
            
        # 读取原图
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # 复制底图用于绘制
        draw_img = img.copy()
        
        # 1. 绘制真实点 (GT) - 红色 (Red)
        gt_count = 0
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        px = int(float(parts[1]) * w)
                        py = int(float(parts[2]) * h)
                        cv2.circle(draw_img, (px, py), radius=4, color=(0, 0, 255), thickness=-1)
                        gt_count += 1
                        
        # 2. 预测点 (Pred) - 绿色 (Green)
        results = yolo.predict(img_path, conf=conf, imgsz=1024, max_det=2000, verbose=False)
        boxes = results[0].boxes
        pred_count = len(boxes)
        
        for box in boxes:
            x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
            cv2.circle(draw_img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
            
        # 3. 绘制文字背景条和对比信息
        cv2.rectangle(draw_img, (0, 0), (w, 50), (30, 30, 30), -1)
        info_text = f"{img_name} | GT: {gt_count} | Pred: {pred_count} | Err: {abs(gt_count - pred_count)}"
        cv2.putText(draw_img, info_text, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 保存可视化结果
        out_path = os.path.join(output_dir, img_name.replace(".jpg", "_visualized.jpg"))
        cv2.imwrite(out_path, draw_img)
        print(f"已生成并保存图像: {out_path} (GT: {gt_count}, Pred: {pred_count}, Err: {abs(gt_count - pred_count)})")

if __name__ == "__main__":
    weights = "runs/detect/crowd_counting_v4_1024/weights/best.pt"
    images_dir = "datasets/shanghaitech_AB/images/val"
    labels_dir = "datasets/shanghaitech_AB/labels/val"
    output_dir = "runs/detect/crowd_counting_v4_1024/visualizations"
    
    samples = [
        "part_B_IMG_196.jpg", # 预测完全一致的低密度图
        "part_A_IMG_73.jpg",  # 预测优秀的密集图
        "part_B_IMG_108.jpg", # 一般偏差图
        "part_B_IMG_281.jpg", # 虚警偏差图
        "part_B_IMG_311.jpg", # 近处大人物图，用来验证近处识别改进
        "part_B_IMG_167.jpg"  # 高低密度过渡图
    ]
    
    visualize_samples(weights, images_dir, labels_dir, output_dir, samples, conf=0.42)
