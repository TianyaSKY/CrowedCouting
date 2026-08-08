import torch
from ultralytics.nn.tasks import DetectionModel, parse_model
from ultralytics.utils.loss import v8DetectionLoss

from models.modules import PointDetect
from models.loss import CrowdPointLoss
import models.modules as custom_modules

# 使用补丁方式使 ultralytics 的 parse_model 能够识别自定义的 PointDetect 头
# parse_model 内部在 ultralytics.nn.tasks 中使用了 eval()，因此我们需将自定义模块注入其命名空间
import ultralytics.nn.tasks as tasks
tasks.PointDetect = PointDetect

class CrowdCountingModel(DetectionModel):
    """
    基于 YOLO11 的人群计数模型，预测人体目标中心点，而不预测边界框。
    """
    def __init__(self, cfg="models/yolo11-crowd.yaml", ch=3, nc=1, verbose=True):
        # 覆盖标准解析器，使其调用我们自定义的 DetectionModel 初始化逻辑
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        
        # 将原生的 Detect 头替换为我们的 PointDetect 头
        m = self.model[-1]
        if type(m).__name__ == "Detect":
            # 提取各个检测层的输入通道数
            ch_list = [x[0].conv.in_channels for x in m.cv2]
            point_head = PointDetect(nc=nc, ch=ch_list)
            # 继承计算好的 stride
            point_head.stride = m.stride
            point_head.bias_init()
            
            # 继承必须的架构解析元数据
            point_head.f = getattr(m, 'f', -1)
            point_head.i = getattr(m, 'i', len(self.model) - 1)
            point_head.type = getattr(m, 'type', 'Detect')
            
            self.model[-1] = point_head
        
    def init_criterion(self):
        """初始化用于 PointDetect 的损失函数标准。"""
        return CrowdPointLoss(self)

    def load_from_pretrained(self, weights_path):
        """
        从预训练的 YOLO11 权重中加载 Backbone 的层（层 0 至 10）。
        """
        import os
        if not os.path.exists(weights_path):
            print(f"未找到预训练权重 {weights_path}")
            return
        
        print(f"正在从 {weights_path} 加载预训练 Backbone 权重...")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        pretrained_dict = ckpt["model"].state_dict() if "model" in ckpt else ckpt
        model_dict = self.state_dict()
        
        loaded_keys = []
        for k, v in pretrained_dict.items():
            parts = k.split('.')
            if len(parts) >= 2 and parts[0] == 'model' and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx <= 10:  # 属于 Backbone
                    if k in model_dict and model_dict[k].shape == v.shape:
                        model_dict[k].copy_(v)
                        loaded_keys.append(k)
                        
        self.load_state_dict(model_dict)
        print(f"成功加载了 {len(loaded_keys)} 个 Backbone 参数张量！")
