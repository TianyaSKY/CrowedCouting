import os
import cv2
import numpy as np
from ultralytics import YOLO

def predict_and_compare_standard(img_path, label_path, weights_path, out_path, conf=0.15):
    """
    加载标准的 YOLO11n 微调权重，在单张图片上推理并在输出图上比对真实标注点与预测中心点。
    """
    if not os.path.exists(weights_path):
        print(f"错误: 未找到权重文件 {weights_path}")
        return

    # 1. 加载官方标准模型权重
    model = YOLO(weights_path)
    
    # 2. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片 {img_path}")
        return
    h, w = img.shape[:2]
    
    # 3. 读取并绘制真实人头点 (Ground Truth - 红色点)
    gt_pts = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    nx, ny = float(parts[1]), float(parts[2])
                    px = int(nx * w)
                    py = int(ny * h)
                    gt_pts.append((px, py))
                    # 绘制红色实心点
                    cv2.circle(img, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
                    
    # 4. 运行标准模型预测并提取框的中心点 (Predictions - 绿色圆圈)
    results = model.predict(img_path, conf=conf, imgsz=640, max_det=2000, verbose=False)
    boxes = results[0].boxes
    pred_pts = []
    for box in boxes:
        # 提取边界框中心坐标作为预测点
        x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
        pred_pts.append((x, y))
        # 绘制绿色圆圈
        cv2.circle(img, (x, y), radius=4, color=(0, 255, 0), thickness=1)

    # 5. 写文本信息
    text_gt = f"Ground Truth: {len(gt_pts)}"
    text_pred = f"Standard YOLO: {len(pred_pts)}"
    text_err = f"Error: {len(pred_pts) - len(gt_pts):+d}"
    
    cv2.putText(img, text_gt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(img, text_pred, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, text_err, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    cv2.putText(img, "Red solid: GT  |  Green circle: Std YOLO Pred", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 6. 保存输出
    cv2.imwrite(out_path, img)
    print(f"[{os.path.basename(img_path)}] 推理完成。")
    print(f"  真实点数: {len(gt_pts)} | 预测点数: {len(pred_pts)} (误差: {len(pred_pts) - len(gt_pts):+d})")
    print(f"  结果图像保存至 {out_path}\n")

if __name__ == "__main__":
    weights = "runs/detect/standard_yolo11n/weights/best.pt"
    
    # 依然选择相同的 3 张测试图片以便和之前的自定义模型作视觉对比
    samples = [
        "part_A_IMG_146.jpg",  # Part A 密集场景
        "part_B_IMG_203.jpg",  # Part B 中等街景
        "part_B_IMG_281.jpg"   # Part B 中等街景
    ]
    
    val_images_dir = "datasets/shanghaitech_AB/images/val"
    val_labels_dir = "datasets/shanghaitech_AB/labels/val"
    out_dir = "runs/detect/standard_yolo11n/test_samples"
    os.makedirs(out_dir, exist_ok=True)
    
    for s in samples:
        img_p = os.path.join(val_images_dir, s)
        lbl_p = os.path.join(val_labels_dir, s.replace(".jpg", ".txt"))
        out_p = os.path.join(out_dir, f"compare_std_{s}")
        
        if os.path.exists(img_p):
            predict_and_compare_standard(img_p, lbl_p, weights, out_p, conf=0.15)
        else:
            print(f"未找到测试样本 {img_p}")
