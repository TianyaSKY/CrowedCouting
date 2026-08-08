import torch

def make_anchors(strides, img_size=1024):
    anchor_points, stride_tensor = [], []
    for stride in strides:
        h = w = img_size // stride
        sx = torch.arange(end=w, dtype=torch.float32) + 0.5
        sy = torch.arange(end=h, dtype=torch.float32) + 0.5
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

strides = [4, 8, 16, 32]
anchor_points, stride_tensor = make_anchors(strides, img_size=1024)

# 随机生成 1000 个目标点（模拟一张图里的1000个人头）
torch.manual_seed(42)
gt_pts = torch.rand(1000, 2) * 1024

# 还原 V4 loss.py 中的匹配逻辑
anchor_img_coords = anchor_points * stride_tensor
dist = torch.cdist(gt_pts, anchor_img_coords)

# 为每个真实点寻找最近的 3 个锚点
k = 3
topk_dist, topk_idx = dist.topk(k, dim=1, largest=False)

# 获取这 3 个锚点所对应的网络层 stride
matched_strides = stride_tensor[topk_idx].squeeze(-1)

# 统计 P2(stride=4), P3(stride=8), P4(stride=16), P5(stride=32) 被命中的总次数
unique, counts = torch.unique(matched_strides, return_counts=True)
print("对于 1000 个随机目标点（总计分配 3000 个正样本锚点）：")
for u, c in zip(unique, counts):
    level = {4: "P2", 8: "P3", 16: "P4", 32: "P5"}.get(int(u.item()), "Unknown")
    print(f"分配给 {level} 层的正样本数: {c.item()} 个")

