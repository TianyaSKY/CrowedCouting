import os
import cv2
import torch
from models.crowd_model import CrowdCountingModel
from ultralytics import YOLO

def predict_and_compare(img_path, label_path, weights_path, out_path, conf=0.22):
    """
    在单张图片上运行推理，并在输出图像中比对真实标注点（红色）与模型预测点（绿色）。
    """
    # 1. 加载模型与权重
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
                    # 绘制红色实心圆点代表真实标记
                    cv2.circle(img, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
                    
    # 4. 运行模型预测点 (Predictions - 绿色圆圈)
    results = yolo.predict(img_path, conf=conf, imgsz=1024, max_det=2000, verbose=False)
    boxes = results[0].boxes
    pred_pts = []
    for box in boxes:
        x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
        pred_pts.append((x, y))
        # 绘制绿色圆圈代表预测结果 (用以套在红点外侧，方便观察重合度)
        cv2.circle(img, (x, y), radius=4, color=(0, 255, 0), thickness=1)

    # 5. 在图像上写上计数文本
    text_gt = f"Ground Truth: {len(gt_pts)}"
    text_pred = f"Prediction: {len(pred_pts)}"
    text_err = f"Error: {len(pred_pts) - len(gt_pts):+d}"
    
    cv2.putText(img, text_gt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(img, text_pred, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img, text_err, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    # 绘制图例
    cv2.putText(img, "Red solid: GT  |  Green circle: Pred", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 6. 保存输出
    cv2.imwrite(out_path, img)
    print(f"[{os.path.basename(img_path)}] 推理完成。")
    print(f"  真实点数: {len(gt_pts)} | 预测点数: {len(pred_pts)} (误差: {len(pred_pts) - len(gt_pts):+d})")
    print(f"  结果图像保存至 {out_path}\n")

if __name__ == "__main__":
    weights = "runs/detect/crowd_counting_v2-2/weights/best.pt"
    
    samples = [
        "part_A_IMG_146.jpg",  # Part A 密集场景
        "part_B_IMG_203.jpg",  # Part B 中等密度
        "part_B_IMG_281.jpg",  # Part B 稀疏/中等密度
        "part_B_IMG_1.jpg"     # Part B 低密度场景
    ]
    
    val_images_dir = "datasets/shanghaitech_AB/images/val"
    val_labels_dir = "datasets/shanghaitech_AB/labels/val"
    out_dir = "runs/detect/crowd_counting_v2-2/test_samples"
    os.makedirs(out_dir, exist_ok=True)
    
    for s in samples:
        img_p = os.path.join(val_images_dir, s)
        lbl_p = os.path.join(val_labels_dir, s.replace(".jpg", ".txt"))
        out_p = os.path.join(out_dir, f"compare_{s}")
        
        if os.path.exists(img_p):
            predict_and_compare(img_p, lbl_p, weights, out_p, conf=0.46)
        else:
            print(f"未找到测试样本 {img_p}")
