import os
import glob
import cv2
import torch
import numpy as np
from models.crowd_model import CrowdCountingModel
from ultralytics import YOLO

def match_points(gt_pts, pred_pts, max_dist=15):
    """
    使用快速最近邻距离匹配算法计算两组点集之间的 True Positives (TP), False Positives (FP), False Negatives (FN)。
    """
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)
        
    gt_pts = np.array(gt_pts)
    pred_pts = np.array(pred_pts)
    
    matched_pred = set()
    tp = 0
    
    # 依次为每一个 GT 点寻找最近的、且未被匹配的预测点
    for gt in gt_pts:
        if len(matched_pred) == len(pred_pts):
            break
        # 向量化计算当前 GT 点到所有预测点的距离
        dists = np.linalg.norm(pred_pts - gt, axis=1)
        
        # 屏蔽掉已经被占用的预测点
        if matched_pred:
            dists[list(matched_pred)] = float('inf')
            
        min_idx = np.argmin(dists)
        if dists[min_idx] <= max_dist:
            matched_pred.add(min_idx)
            tp += 1
            
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn

def evaluate_localization_metrics(weights_path, val_images_dir, val_labels_dir, conf=0.22, max_dist=15):
    """
    计算基于点匹配距离阈值的精确度 (Precision)、召回率 (Recall) 和 F1-Score。
    """
    print(f"正在加载模型以评估几何定位精度...")
    model = CrowdCountingModel("models/yolo11-crowd.yaml", ch=3, nc=1, verbose=False)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"].state_dict() if "model" in ckpt else ckpt)
    else:
        print(f"错误: 未找到权重文件 {weights_path}")
        return

    model.eval()
    
    yolo = YOLO("yolo11n.yaml")
    yolo.model = model
    
    img_paths = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    print(f"找到 {len(img_paths)} 张验证集图片，正以最大容忍偏差距离 {max_dist} 像素进行点对点配对评估...")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, img_name.replace(".jpg", ".txt"))
        
        # 读取 GT 点
        gt_pts = []
        if os.path.exists(label_path):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        px = int(float(parts[1]) * w)
                        py = int(float(parts[2]) * h)
                        gt_pts.append((px, py))
        else:
            continue
            
        # 运行预测
        results = yolo.predict(img_path, conf=conf, imgsz=640, max_det=2000, verbose=False)
        pred_pts = []
        for box in results[0].boxes:
            x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
            pred_pts.append((x, y))
            
        # 进行几何点对点匹配
        tp, fp, fn = match_points(gt_pts, pred_pts, max_dist=max_dist)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if (i + 1) % 100 == 0:
            print(f"已评估 {i+1}/{len(img_paths)} 张图片...")
            
    # 计算指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*50)
    print(f"几何定位配对评估结果 (匹配上限距离: {max_dist} 像素):")
    print(f"匹配成功人头数 (TP): {total_tp}")
    print(f"模型误报人数 (FP): {total_fp}")
    print(f"模型漏检人数 (FN): {total_fn}")
    print(f"精确率 (Precision): {precision*100:.2f}% (预测点里多少是真正人头)")
    print(f"召回率 (Recall): {recall*100:.2f}% (有多少真实人头被检测出来了)")
    print(f"点定位 F1-Score: {f1*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    weights = "runs/detect/crowd_counting_yolo11/weights/best.pt"
    images_dir = "datasets/shanghaitech_AB/images/val"
    labels_dir = "datasets/shanghaitech_AB/labels/val"
    # 以 15 像素为几何距离误差容忍度
    evaluate_localization_metrics(weights, images_dir, labels_dir, conf=0.22, max_dist=15)
