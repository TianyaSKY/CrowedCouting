import os
from ultralytics import YOLO

def train_standard_yolo():
    """
    使用标准的 YOLO11n 模型（带预训练权重）在合并后的数据增强数据集上进行微调，
    作为一个对照基准（Baseline）。
    """
    print("正在初始化官方标准的 YOLO11n 模型...")
    # 加载带有官方预训练权重的标准 YOLO11n 模型
    model = YOLO("yolo11n.pt")
    
    args = dict(
        data="datasets/shanghaitech_AB/dataset.yaml",
        epochs=20,             # 训练 20 轮作为对比
        imgsz=640,
        batch=8,
        name="standard_yolo11n",
        device="0",            # GPU
        save=True,
        workers=2              # 降低 worker 数量以防止 dataloader 产生死锁
    )
    
    print("开始训练标准 YOLO11n...")
    model.train(**args)
    print("训练结束。")

if __name__ == "__main__":
    train_standard_yolo()
