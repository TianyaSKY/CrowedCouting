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

def train_crowd_model_v4():
    """训练 v4 模型：使用自适应匹配阈值修复近处大目标漏检问题。"""
    print("=" * 60)
    print("开始训练 v4 模型（自适应匹配阈值修复版）")
    print("改动说明: loss.py 中正样本匹配阈值从固定 16px 改为 stride*1.5")
    print("=" * 60)
    
    args = dict(
        model="models/yolo11-crowd.yaml",
        data="datasets/shanghaitech_AB/dataset.yaml",
        epochs=200,
        imgsz=1024,
        batch=8,
        name="crowd_counting_v4_1024",
        device="0",
        save=True,
        workers=4,
        pretrained=False,
        patience=50,       # 早停耐心值：50轮不改善则停止（原生 fitness 仍为 0，但作为保险）
        # === 数据增强设置 ===
        mosaic=0.0,         # 关闭 Mosaic
        erasing=0.0,        # 关闭 Random Erasing
        close_mosaic=0,
        max_det=2000,       # 密集场景最大检测数
        flipud=0.5,         # 上下翻转
        scale=0.3,          # 缩放增强
    )
    
    trainer = CrowdTrainer(overrides=args)
    trainer.train()
    
    print("训练结束。")

if __name__ == "__main__":
    train_crowd_model_v4()
