import os
from copy import copy
from models.crowd_model import CrowdCountingModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator

class CrowdTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        """重写 get_model，返回我们的自定义人群计数模型并加载预训练的 Backbone 权重。"""
        model = CrowdCountingModel("models/yolo11-crowd.yaml", ch=3, nc=1, verbose=verbose)
        
        # 尝试加载本地预训练 Backbone 权重 yolo26n.pt (YOLO11n 原生权重)
        pretrained_backbone = "yolo26n.pt"
        if os.path.exists(pretrained_backbone):
            model.load_from_pretrained(pretrained_backbone)
            
        if weights and weights not in ["yolo11n.pt", "yolo11n.yaml"]:
            model.load(weights)
        return model

    def get_validator(self):
        """重写以更新 loss_names 并返回标准验证器。"""
        self.loss_names = "cls_loss", "off_loss", "placeholder"
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

def train_crowd_model():
    """训练 YOLO11 人群计数模型。"""
    print("开始初始化自定义 CrowdTrainer 并进行训练...")
    
    args = dict(
        model="models/yolo11-crowd.yaml",
        data="datasets/shanghaitech_AB/dataset.yaml",
        epochs=300,
        imgsz=1024,
        batch=8,
        name="crowd_counting_v3_1024",
        device="0",
        save=True,
        workers=4,
        pretrained=False,
        # === 数据增强修复 ===
        mosaic=0.0,       # 关闭 Mosaic：密集人群拼贴会导致单图标注数爆炸，超出 max_det
        erasing=0.0,      # 关闭 Random Erasing：会擦除图像但不移除对应标注，产生虚假正样本
        close_mosaic=0,   # 不需要 close_mosaic（已全程关闭）
        max_det=2000,     # 提高最大检测数以适应密集场景（Part A 单图可达 1500 人）
        flipud=0.5,       # 开启上下翻转作为替代增强
        scale=0.3,        # 缩放增强适度降低
    )
    
    trainer = CrowdTrainer(overrides=args)
    trainer.train()
    
    print("训练结束。")

if __name__ == "__main__":
    train_crowd_model()
