import os
import glob
import torch
import numpy as np
from models.crowd_model import CrowdCountingModel
from ultralytics import YOLO

def evaluate_detailed(weights_path, val_images_dir, val_labels_dir, conf=0.18):
    """
    在最佳置信度阈值 0.18 下，评估每张验证集图片的真实人数和预测人数，并打印分析报告。
    """
    print(f"正在加载模型...")
    model = CrowdCountingModel("models/yolo11-crowd.yaml", ch=3, nc=1, verbose=False)
    
    if os.path.exists(weights_path):
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
    
    yolo = YOLO("yolo11n.yaml")
    yolo.model = model
    
    img_paths = glob.glob(os.path.join(val_images_dir, "*.jpg"))
    
    results_list = []
    
    print(f"开始详细评估 {len(img_paths)} 张验证图片...")
    for i, img_path in enumerate(img_paths):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(val_labels_dir, img_name.replace(".jpg", ".txt"))
        
        # 获取真实人数
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                gt_count = len(f.readlines())
        else:
            continue
            
        # 预测人数
        results = yolo.predict(img_path, conf=conf, imgsz=1024, max_det=2000, verbose=False)
        pred_count = len(results[0].boxes)
        
        abs_err = abs(gt_count - pred_count)
        rel_err = (abs_err / gt_count) * 100 if gt_count > 0 else 0
        part = "Part A" if "part_A" in img_name else "Part B"
        
        results_list.append({
            "image": img_name,
            "part": part,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "abs_error": abs_err,
            "rel_error_pct": round(rel_err, 2)
        })
        
    # 保存详细记录至 CSV
    csv_out = os.path.join(os.path.dirname(weights_path), "..", "val_detailed_results.csv")
    csv_out = os.path.abspath(csv_out)
    
    with open(csv_out, "w") as f:
        f.write("image,part,gt_count,pred_count,abs_error,rel_error_pct\n")
        for r in results_list:
            f.write(f"{r['image']},{r['part']},{r['gt_count']},{r['pred_count']},{r['abs_error']},{r['rel_error_pct']}\n")
            
    # 统计数据
    abs_errors = [r["abs_error"] for r in results_list]
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(np.array(abs_errors) ** 2))
    
    print("\n" + "="*50)
    print(f"详细统计汇总:")
    print(f"评估图像总数: {len(results_list)} 张")
    print(f"真实总人数: {sum(r['gt_count'] for r in results_list)}")
    print(f"预测总人数: {sum(r['pred_count'] for r in results_list)}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"详细单图数据已写入: {csv_out}")
    print("="*50)
    
    # 分解看 Part A 和 Part B 各自的 MAE
    parts_data = {}
    for r in results_list:
        p = r["part"]
        if p not in parts_data:
            parts_data[p] = []
        parts_data[p].append(r)
        
    for part, items in parts_data.items():
        part_gt_sum = sum(x["gt_count"] for x in items)
        part_pred_sum = sum(x["pred_count"] for x in items)
        part_mae = np.mean([x["abs_error"] for x in items])
        print(f"[{part}] 图片数: {len(items)}, 真实总数: {part_gt_sum}, 预测总数: {part_pred_sum}, MAE: {part_mae:.2f}")
    print("="*50)

    # 排序结果以找到最好和最差的
    sorted_by_err = sorted(results_list, key=lambda x: x["abs_error"])
    
    # 找出预测最准的前 5 张图
    print("预测最准的 5 张图样例:")
    for row in sorted_by_err[:5]:
        print(f" 图片: {row['image']} | 真实人数: {row['gt_count']} | 预测人数: {row['pred_count']} | 误差: {row['abs_error']}")
        
    print("-"*50)
    # 找出误差最大的前 5 张图
    print("误差最大的 5 张图样例（用于Debug分析）:")
    for row in sorted_by_err[-5:][::-1]:
        print(f" 图片: {row['image']} | 真实人数: {row['gt_count']} | 预测人数: {row['pred_count']} | 误差: {row['abs_error']}")
    print("="*50)

if __name__ == "__main__":
    weights = "runs/detect/crowd_counting_v4_1024/weights/best.pt"
    images_dir = "datasets/shanghaitech_AB/images/val"
    labels_dir = "datasets/shanghaitech_AB/labels/val"
    evaluate_detailed(weights, images_dir, labels_dir, conf=0.42)
