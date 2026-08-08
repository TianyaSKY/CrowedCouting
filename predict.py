import os
import cv2
import torch
import numpy as np
from models.crowd_model import CrowdCountingModel
from ultralytics import YOLO

def predict_crowd(img_path, weights_path=None):
    """在单张图像上运行推理并绘制预测点。"""
    print(f"正在加载自定义模型...")
    model = CrowdCountingModel("models/yolo11-crowd.yaml", ch=3, nc=1, verbose=False)
    
    if weights_path and os.path.exists(weights_path):
        print(f"从 {weights_path} 加载权重")
        # 需要安全地加载状态字典 (state dict)
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"].state_dict() if "model" in ckpt else ckpt)
    else:
        print("警告: 未提供权重文件，模型将使用随机初始化参数进行推理。")
    
    model.eval()
    
    # 封装在 YOLO 对象中，以便更方便地加载图像和应用 NMS
    yolo = YOLO("yolo11n.yaml")
    yolo.model = model
    
    print(f"正在 {img_path} 上运行推理...")
    results = yolo.predict(img_path, conf=0.25, iou=0.45, imgsz=640)
    
    # 处理推理结果
    for result in results:
        img = result.orig_img
        boxes = result.boxes
        
        print(f"检测到 {len(boxes)} 个人。")
        
        # 绘制中心点
        for box in boxes:
            # 在 PointDetect 头中，边界框的宽和高被强制设为了1
            # 因此这里的 (x, y) 就是我们预测的目标中心点。
            x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
            conf = float(box.conf[0])
            
            # 为检测到的每个人绘制一个绿色的圆点
            cv2.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
            # 可选：绘制置信度得分
            # cv2.putText(img, f"{conf:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
            
        # 在左上角绘制总人数
        cv2.putText(img, f"Count: {len(boxes)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # 保存输出图像
        out_path = img_path.replace(".jpg", "_pred.jpg")
        cv2.imwrite(out_path, img)
        print(f"已将预测结果保存至 {out_path}")

if __name__ == "__main__":
    # 在一张验证集的测试图片上进行测试
    sample_img = "datasets/shanghaitech_AB/images/val/part_B_IMG_1.jpg"
    weights = "runs/detect/crowd_counting_yolo11/weights/best.pt"
    
    if os.path.exists(sample_img):
        predict_crowd(sample_img, weights)
    else:
        print(f"未找到测试图片 {sample_img}。请先运行数据集准备脚本。")
