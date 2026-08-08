import math
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import DFL

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """从特征图中生成锚点(anchors)。"""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

class PointDetect(nn.Module):
    """用于人群计数的YOLO点检测头。"""
    dynamic = False  # 强制网格重建
    export = False  # 导出模式
    shape = None
    anchors = torch.empty(0)  # 初始化锚点
    strides = torch.empty(0)  # 初始化步长

    def __init__(self, nc=1, ch=()):
        """初始化 PointDetect 层。"""
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数
        self.no = nc + 2  # 每个锚点的输出数量：nc个类别 + 2个偏移量(dx, dy)
        self.stride = torch.zeros(self.nl)  # 步长将在构建模型时计算
        
        # 分类分支
        c2 = max(16, ch[0] // 4, self.nc)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.nc, 1)) for x in ch
        )
        
        # 点偏移回归分支 (dx, dy)
        c3 = max(16, ch[0] // 4, 2)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, 2, 1)) for x in ch
        )

    def forward(self, x):
        """拼接并返回预测的点偏移和类别概率。"""
        shape = x[0].shape  # 批次、通道、高、宽 (BCHW)
        for i in range(self.nl):
            # 类别得分
            cls_out = self.cv2[i](x[i])
            # 点偏移 (dx, dy)
            off_out = self.cv3[i](x[i])
            # 拼接
            x[i] = torch.cat((cls_out, off_out), 1)

        if self.training:
            return x

        # 推理路径
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # 将所有层展平并拼接
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        # 将结果重新分离为类别得分和偏移量
        cls = x_cat[:, :self.nc, :]
        off = x_cat[:, self.nc:, :]
        
        cls = cls.sigmoid()
        
        # 解码点坐标：(锚点坐标 + 偏移量) * 步长
        # off 维度: (B, 2, N), anchors 维度: (2, N), strides 维度: (1, N)
        anchors = self.anchors.unsqueeze(0)  # (1, 2, N)
        strides = self.strides.unsqueeze(0)  # (1, 1, N)
        
        pred_points = (anchors + off) * strides
        
        # 为了兼容原生 NMS，将点结果与分类结果重新拼接
        # 注意: Ultralytics 的 NMS 默认接收 (B, 4+nc, N) 格式。
        # 因此，我们将预测的 (x,y) 点位转化为一个小号虚拟边界框 (x, y, 1, 1) 从而兼容默认逻辑。
        
        bboxes = torch.cat((
            pred_points,
            torch.ones_like(pred_points)  # 设置虚拟宽高 w=1, h=1
        ), dim=1) # (B, 4, N)
        
        out = torch.cat((bboxes, cls), dim=1)
        
        return out if self.export else (out, x)

    def bias_init(self):
        """初始化偏置，加速网络收敛。"""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            # 分类分支偏置初始化
            a[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)
            # 偏移分支偏置初始化
            b[-1].bias.data[:] = 0.0
