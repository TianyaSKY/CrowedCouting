import os
from copy import copy
from models.crowd_model import CrowdCountingModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator

class CrowdTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CrowdCountingModel("models/yolo11-crowd.yaml", ch=3, nc=1, verbose=verbose)
        if weights and weights not in ["yolo11n.pt", "yolo11n.yaml"]:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "cls_loss", "off_loss", "placeholder"
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

def resume_crowd_model():
    print("准备从上一次的断点恢复训练，并将总轮数调整为 200 Epochs...")
    
    # 指向 v3 最后的断点权重
    last_weights = "runs/detect/crowd_counting_v3_1024/weights/last.pt"
    if not os.path.exists(last_weights):
        print(f"错误: 未找到权重文件 {last_weights}")
        return
        
    args = dict(
        model=last_weights,
        data="datasets/shanghaitech_AB/dataset.yaml",
        epochs=200,      # 修改为 200 轮（约 6.4 小时，适合挂机一整晚）
        imgsz=1024,
        batch=8,
        resume=True,     # 开启恢复模式
        device="0",
        save=True,
        workers=4,
        # 保持之前的增强设置
        mosaic=0.0,
        erasing=0.0,
        close_mosaic=0,
        max_det=2000,
        flipud=0.5,
        scale=0.3,
    )
    
    trainer = CrowdTrainer(overrides=args)
    trainer.train()
    print("断点恢复训练结束。")

if __name__ == "__main__":
    resume_crowd_model()
