import os
import glob
import cv2
import torch
import numpy as np
from models.crowd_model import CrowdCountingModel
from ultralytics import YOLO

def match_points(gt_pts, pred_pts, max_dist=15):
    """
    使用快速最近邻匹配计算 TP, FP, FN
    """
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)
        
    gt_pts = np.array(gt_pts)
    pred_pts = np.array(pred_pts)
    
    matched_pred = set()
    tp = 0
    
    for gt in gt_pts:
        if len(matched_pred) == len(pred_pts):
            break
        dists = np.linalg.norm(pred_pts - gt, axis=1)
        if matched_pred:
            dists[list(matched_pred)] = float('inf')
            
        min_idx = np.argmin(dists)
        if dists[min_idx] <= max_dist:
            matched_pred.add(min_idx)
            tp += 1
            
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn

def get_predictions_custom(model, yolo, img_path, conf=0.22):
    results = yolo.predict(img_path, conf=conf, imgsz=640, max_det=2000, verbose=False)
    pred_pts = []
    for box in results[0].boxes:
        x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
        pred_pts.append((x, y))
    return pred_pts

def get_predictions_standard(model, img_path, conf=0.15):
    results = model.predict(img_path, conf=conf, imgsz=640, max_det=2000, verbose=False)
    pred_pts = []
    for box in results[0].boxes:
        x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
        pred_pts.append((x, y))
    return pred_pts

def main():
    val_images_dir = "datasets/shanghaitech_AB/images/val"
    val_labels_dir = "datasets/shanghaitech_AB/labels/val"
    
    custom_weights = "runs/detect/crowd_counting_yolo11/weights/best.pt"
    standard_weights = "runs/detect/standard_yolo11n/weights/best.pt"
    
    # 1. 加载自定义模型
    print("加载自定义点检测模型...")
    custom_model = CrowdCountingModel("models/yolo11-crowd.yaml", ch=3, nc=1, verbose=False)
    if os.path.exists(custom_weights):
        ckpt = torch.load(custom_weights, map_location="cpu", weights_only=False)
        custom_model.load_state_dict(ckpt["model"].state_dict() if "model" in ckpt else ckpt)
    custom_model.eval()
    yolo_custom = YOLO("yolo11n.yaml")
    yolo_custom.model = custom_model
    
    # 2. 加载标准模型
    print("加载标准 YOLO11n 模型...")
    standard_model = YOLO(standard_weights)
    
    # 3. 读取验证集数据
    img_paths = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    print(f"找到 {len(img_paths)} 张验证集图片。")
    
    dataset = []
    for i, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, img_name.replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_pts = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        px = int(float(parts[1]) * w)
                        py = int(float(parts[2]) * h)
                        gt_pts.append((px, py))
            dataset.append({
                "img_path": img_path,
                "gt_pts": gt_pts
            })
            
    thresholds = [5, 10, 15, 20, 25]
    results_custom = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    results_std = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    
    print("正在评估自定义点检测模型与标准微调模型...")
    for idx, data in enumerate(dataset):
        img_path = data["img_path"]
        gt_pts = data["gt_pts"]
        
        # 预测
        pred_custom = get_predictions_custom(custom_model, yolo_custom, img_path, conf=0.22)
        pred_std = get_predictions_standard(standard_model, img_path, conf=0.15)
        
        # 匹配不同阈值
        for t in thresholds:
            tp_c, fp_c, fn_c = match_points(gt_pts, pred_custom, max_dist=t)
            results_custom[t]["tp"] += tp_c
            results_custom[t]["fp"] += fp_c
            results_custom[t]["fn"] += fn_c
            
            tp_s, fp_s, fn_s = match_points(gt_pts, pred_std, max_dist=t)
            results_std[t]["tp"] += tp_s
            results_std[t]["fp"] += fp_s
            results_std[t]["fn"] += fn_s
            
        if (idx + 1) % 50 == 0:
            print(f"已评估 {idx+1}/{len(dataset)} 张图片...")
            
    print("\n" + "="*80)
    print(" 距离容忍度 (d) |  模型类型  | Precision | Recall | F1-Score | TP | FP | FN")
    print("-" * 80)
    
    for t in thresholds:
        # 自定义模型
        tp_c = results_custom[t]["tp"]
        fp_c = results_custom[t]["fp"]
        fn_c = results_custom[t]["fn"]
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        f_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0
        
        # 标准模型
        tp_s = results_std[t]["tp"]
        fp_s = results_std[t]["fp"]
        fn_s = results_std[t]["fn"]
        p_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
        r_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
        f_s = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) > 0 else 0
        
        print(f"    {t:2d} px     |  Custom    |  {p_c*100:6.2f}%  | {r_c*100:5.2f}% |  {f_c*100:6.2f}% | {tp_c} | {fp_c} | {fn_c}")
        print(f"    {t:2d} px     |  Standard  |  {p_s*100:6.2f}%  | {r_s*100:5.2f}% |  {f_s*100:6.2f}% | {tp_s} | {fp_s} | {fn_s}")
        print("-" * 80)
        
if __name__ == "__main__":
    main()
