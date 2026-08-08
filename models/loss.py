import torch
import torch.nn as nn
import torch.nn.functional as F

class CrowdPointLoss(nn.Module):
    """
    用于使用 PointDetect 头进行人群计数的损失函数。
    计算分类损失 (Focal Loss) 和点偏移损失 (L1)。
    
    改进点：
    1. 移除了全局辅助计数损失（它在训练早期产生过大梯度，干扰分类分支学习）
    2. 采用 Top-K (K=3) 多正样本匹配，解决密集人群中锚点竞争导致 GT 点静默丢失的问题
    3. 提高偏移损失权重至 5.0，强化定位精度
    """
    def __init__(self, model):
        super().__init__()
        device = next(model.parameters()).device
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")
        
        m = model.model[-1]  # PointDetect
        self.stride = m.stride
        self.nc = m.nc
        self.topk = 3  # 每个 GT 点匹配的锚点数量

    def forward(self, preds, batch):
        """
        计算点检测损失。
        preds: 由特征张量组成的列表 [P2, P3, P4, P5]，每个张量维度为 (B, nc+2, H, W)
        batch: 字典类型数据，其中 "bboxes" -> (N, 4)，此时我们仅提取中心点。
        """
        loss = torch.zeros(3, device=self.device)  # 分别为：分类损失，偏移损失，占位(保持兼容)
        
        # 兼容验证阶段：在 eval 模式下，网络前向传播会返回 (out, x) 格式
        if isinstance(preds, tuple):
            preds = preds[1]
            
        # 将所有预测结果展平并拼接为一个张量: (B, nc+2, N_anchors)
        pred_cat = torch.cat([p.view(p.shape[0], self.nc + 2, -1) for p in preds], dim=2)
        pred_cls = pred_cat[:, :self.nc, :]  # 分类得分: (B, nc, N_anchors)
        pred_off = pred_cat[:, self.nc:, :]  # 偏移量: (B, 2, N_anchors)
        
        bs = pred_cat.shape[0]
        
        # 构建锚点坐标池
        anchor_points = []
        stride_tensor = []
        for i, p in enumerate(preds):
            _, _, h, w = p.shape
            sx = torch.arange(w, device=self.device).float() + 0.5
            sy = torch.arange(h, device=self.device).float() + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchors = torch.stack((sx, sy), dim=-1).view(-1, 2)
            anchor_points.append(anchors)
            strides = torch.full((h * w, 1), self.stride[i], device=self.device)
            stride_tensor.append(strides)
            
        anchor_points = torch.cat(anchor_points)  # 总的锚点数: (N_anchors, 2)
        stride_tensor = torch.cat(stride_tensor)  # 对应步长: (N_anchors, 1)
        
        # 获取真值点 (Target points)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        
        # 获得用于还原归一化坐标的特征图尺寸
        imgsz = torch.tensor(preds[0].shape[2:], device=self.device, dtype=torch.float32) * self.stride[0]
        
        # 定义各类真值的掩码矩阵 (维度为 BS, N_anchors)
        n_anchors = anchor_points.shape[0]
        target_cls = torch.zeros((bs, n_anchors, self.nc), device=self.device)
        target_off = torch.zeros((bs, n_anchors, 2), device=self.device)
        fg_mask = torch.zeros((bs, n_anchors), dtype=torch.bool, device=self.device)
        # 用于记录每个锚点被匹配到的 GT 偏移量的累加（处理多 GT 竞争同一锚点的情况）
        anchor_match_count = torch.zeros((bs, n_anchors), device=self.device)
        
        for i in range(bs):
            idx = targets[:, 0] == i
            if not idx.any():
                continue
                
            gt_pts = targets[idx, 2:4] * imgsz[[1, 0]]  # (N_gt, 2) 真值点坐标放大到图像绝对尺寸
            n_gt = gt_pts.shape[0]
            
            # 锚点在图像空间中的绝对坐标
            anchor_img_coords = anchor_points * stride_tensor
            
            # 计算欧氏距离矩阵: (N_gt, N_anchors)
            dist = torch.cdist(gt_pts, anchor_img_coords)
            
            # ========== Top-K 多正样本匹配 ==========
            # 为每个 GT 点找到最近的 K 个锚点（而不是只找 1 个）
            k = min(self.topk, n_anchors)  # 防止锚点总数不足 K
            topk_dist, topk_idx = dist.topk(k, dim=1, largest=False)  # (N_gt, K)
            
            # 距离阈值过滤：按锚点所在特征层的 stride 自适应缩放
            # P2(stride=4)→6px, P3(stride=8)→12px, P4(stride=16)→24px, P5(stride=32)→48px
            # 保证每层特征图上至少覆盖 1.5 个网格步长，解决 P5 上大目标无法匹配正样本的问题
            topk_strides = stride_tensor[topk_idx].squeeze(-1)  # (N_gt, K)
            adaptive_threshold = topk_strides * 1.5  # (N_gt, K)
            valid_pairs = topk_dist < adaptive_threshold  # (N_gt, K) 布尔掩码
            
            # 展开所有有效的 (GT_index, Anchor_index) 配对
            gt_indices = torch.arange(n_gt, device=self.device).unsqueeze(1).expand_as(topk_idx)
            
            valid_anchor_idx = topk_idx[valid_pairs]     # 有效的锚点索引
            valid_gt_idx = gt_indices[valid_pairs]        # 对应的 GT 索引
            valid_gt_pts_matched = gt_pts[valid_gt_idx]   # 对应的 GT 坐标
            
            if valid_anchor_idx.numel() == 0:
                continue
            
            # 设置正样本掩码
            fg_mask[i, valid_anchor_idx] = True
            
            # 分类目标设定 (所有正样本设为 1.0)
            target_cls[i, valid_anchor_idx, 0] = 1.0
            
            # 偏移目标设定: (GT_coord / stride) - anchor_point
            valid_strides = stride_tensor[valid_anchor_idx]
            valid_anchors = anchor_points[valid_anchor_idx]
            
            gt_feat_pts = valid_gt_pts_matched / valid_strides
            offsets = gt_feat_pts - valid_anchors
            
            # 处理多个 GT 匹配到同一锚点的情况：
            # 使用 scatter 取平均偏移（而非简单覆盖）
            target_off_accum = torch.zeros((n_anchors, 2), device=self.device)
            match_count = torch.zeros(n_anchors, device=self.device)
            
            target_off_accum.scatter_add_(0, valid_anchor_idx.unsqueeze(1).expand(-1, 2), offsets)
            match_count.scatter_add_(0, valid_anchor_idx, torch.ones_like(valid_anchor_idx, dtype=torch.float32))
            
            # 对匹配到多个 GT 的锚点取平均偏移
            matched_mask = match_count > 0
            target_off_accum[matched_mask] /= match_count[matched_mask].unsqueeze(1)
            
            target_off[i] = target_off_accum
            anchor_match_count[i] = match_count
        
        # 1. 分类损失计算 (BCE + Focal Loss)
        pred_cls_trans = pred_cls.transpose(1, 2)
        bce_loss = self.bce(pred_cls_trans, target_cls)
        p = torch.sigmoid(pred_cls_trans)
        
        # Focal Loss 权重计算
        gamma = 2.0
        alpha = 0.25
        focal_weight = target_cls * alpha * ((1.0 - p) ** gamma) + (1.0 - target_cls) * (1.0 - alpha) * (p ** gamma)
        cls_loss = (focal_weight * bce_loss).sum()
        
        # 用分类正样本数进行归一化并适当放大权重
        num_pos = max(fg_mask.sum(), 1.0)
        cls_loss = (cls_loss / num_pos) * 20.0
        
        # 2. 偏移量损失计算 (L1)
        if fg_mask.sum() > 0:
            off_loss = self.l1(pred_off.transpose(1, 2)[fg_mask], target_off[fg_mask]).mean()
        else:
            off_loss = torch.tensor(0.0, device=self.device)
            
        # 3. 全局辅助计数损失已移除
        # 原因：训练早期 34000 个网格概率求和值远大于真实人数，
        # 产生的巨大梯度干扰分类分支学习，且梯度不区分正负样本。
        
        loss[0] = cls_loss * 1.0     # 分类损失权重
        loss[1] = off_loss * 5.0     # 偏移损失权重 (从 1.0 提升至 5.0，强化定位精度)
        loss[2] = 0.0                # 占位，保持与训练器的 3-slot 格式兼容
        
        total_loss = loss.sum()
        
        return total_loss * bs, loss.detach()

    def __call__(self, preds, batch):
        """Ultralytics 训练引擎 API 兼容封装接口。"""
        return self.forward(preds, batch)
