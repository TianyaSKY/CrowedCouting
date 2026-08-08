import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def match_points(gt_pts, pred_pts, max_dist=15):
    """使用快速最近邻匹配计算 TP, FP, FN。"""
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

def evaluate_standard_model(weights_path, val_images_dir, val_labels_dir):
    """
    加载标准的 YOLO11n 微调权重，测试不同置信度阈值下的 MAE, RMSE 和点对点几何 F1-Score。
    """
    print(f"正在加载标准微调模型...")
    if not os.path.exists(weights_path):
        print(f"错误: 未找到权重文件 {weights_path}")
        return
        
    model = YOLO(weights_path)
    img_paths = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    print(f"找到 {len(img_paths)} 张验证图片。")
    
    # 真实数据读取
    all_gt_pts = []
    valid_img_paths = []
    for img_path in img_paths:
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
            all_gt_pts.append(gt_pts)
            valid_img_paths.append(img_path)

    # 网格搜索最优置信度阈值
    conf_thresholds = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    best_f1 = 0
    best_conf = 0
    best_metrics = {}

    for conf in conf_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        pred_counts = []
        gt_counts = [len(gt) for gt in all_gt_pts]
        
        for i, img_path in enumerate(valid_img_paths):
            results = model.predict(img_path, conf=conf, imgsz=640, max_det=2000, verbose=False)
            boxes = results[0].boxes
            pred_pts = []
            for box in boxes:
                # 标准检测框的中心点即为预测人头点
                x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
                pred_pts.append((x, y))
            
            pred_counts.append(len(pred_pts))
            
            # 计算几何匹配 (15 像素容忍误差)
            tp, fp, fn = match_points(all_gt_pts[i], pred_pts, max_dist=15)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        pred_counts = np.array(pred_counts)
        gt_counts = np.array(gt_counts)
        mae = np.mean(np.abs(gt_counts - pred_counts))
        rmse = np.sqrt(np.mean((gt_counts - pred_counts) ** 2))
        
        print(f"Conf = {conf:.2f} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, F1-Score: {f1*100:.2f}% (P={precision*100:.2f}%, R={recall*100:.2f}%)")
        
        if f1 > best_f1:
            best_f1 = f1
            best_conf = conf
            best_metrics = {
                "mae": mae,
                "rmse": rmse,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "gt_total": np.sum(gt_counts),
                "pred_total": np.sum(pred_counts)
            }
            
    print("\n" + "="*50)
    print("标准 YOLO11n 模型最佳配置结果:")
    print(f"最佳置信度阈值 (Conf): {best_conf:.2f}")
    print(f"真实总人数: {best_metrics['gt_total']}")
    print(f"预测总人数: {best_metrics['pred_total']}")
    print(f"对应 MAE: {best_metrics['mae']:.2f}")
    print(f"对应 RMSE: {best_metrics['rmse']:.2f}")
    print(f"对应精确率 (Precision): {best_metrics['precision']*100:.2f}%")
    print(f"对应召回率 (Recall): {best_metrics['recall']*100:.2f}%")
    print(f"对应最佳点定位 F1-Score: {best_metrics['f1']*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    weights = "runs/detect/standard_yolo11n/weights/best.pt"
    images_dir = "datasets/shanghaitech_AB/images/val"
    labels_dir = "datasets/shanghaitech_AB/labels/val"
    evaluate_standard_model(weights, images_dir, labels_dir)
