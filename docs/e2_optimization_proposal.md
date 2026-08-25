# E2 专家大尺寸目标漏检优化设计方案

## 1. 方案背景与问题定义

在当前的“原生多尺度点检测模型（Native Multiscale Point Model）”中，E2 专家负责在 YOLO11 的深层特征图 $P_5$（Stride 32，输入 $640 \times 640$ 时特征图分辨率为 $20 \times 20$）上进行点预测。

### 1.1 现行架构参数与瓶颈
* **特征分辨率**：$20 \times 20$（单像素物理覆盖原图 $32 \times 32$ 区域）
* **参考点配置**：每格 $K=16$ 个参考点（$4 \times 4$ 均匀子网格，总候选点 6400）
* **输出头结构**：`nn.Conv2d(256, 16 * 3, kernel_size=1)`（48 通道输出）

### 1.2 实测痛点
在包含大量稀疏、大尺寸目标的测试集（如 ShanghaiTech Part B）中，E2 的单专家独立预测性能严重失常：
* **Ground Truth 总数**：5379
* **E0 预测总数**：5315.6（MAE: 13.670，表现优异）
* **E1 预测总数**：4568.5（MAE: 25.762）
* **E2 预测总数**：**2653.1（MAE: 68.147，预测数量腰斩，严重漏检）**

### 1.3 核心诱因总结
1. **单特征 16 点强行复用**：16 个空间上相隔甚远的子点共享同一个 256 维特征向量 $\mathbf{h}_{i, j}$；
2. **15:1 负梯度淹没**：大目标单点正监督下，同一网格内 15 个负样本的梯度反向叠加，将大目标的特征强烈抑制；
3. **空间盲目性**：$1\times 1$ 通道投影无法感知亚像素级别的局部图像轮廓。

---

## 2. 方案技术选型与对比矩阵

为彻底解决上述问题，我们设计了三套递进式的技术优化方案：

| 评估维度 | 方案一：PixelShuffle 亚像素空间展开（首选） | 方案二：Deformable 可形变空间采样 | 方案三：高斯软标签平滑（辅助） |
| :--- | :--- | :--- | :--- |
| **技术机制** | 将 $P_5$ 通过通道重排在空间上放大 4 倍至 $80 \times 80$，每点 $K=1$ | 在 16 个点物理坐标处做可学习双线性插值采样 | 将硬标签 $y \in \{0,1\}$ 改为基于中心距离的高斯软标签 |
| **解决负梯度内耗** | **彻底消除**（每个点拥有独立像素特征） | **彻底消除**（正负样本特征物理隔离） | **大幅缓解**（邻域变为弱正样本） |
| **空间去歧义能力** | **极强**（重回标准二维空间卷积） | **极强**（自适应连续坐标特征抓取） | **一般**（特征仍未解耦） |
| **计算与内存开销** | **极低**（仅内存重排，无浮点计算激增） | **中等**（需多次 `grid_sample` 插值） | **零开销**（仅修改 Loss 计算） |
| **工程改造难度** | **极低**（只需改动 E2 专家头代码） | **中等**（需重构前向特征采样逻辑） | **极低**（仅改动损失函数模块） |
| **推荐指数** | ⭐⭐⭐⭐⭐ **（首选主方案）** | ⭐⭐⭐⭐ **（高阶备选方案）** | ⭐⭐⭐⭐ **（协同增益方案）** |

---

## 3. 方案一深度设计：PixelShuffle 亚像素空间展开方案（首选）

### 3.1 数学原理与张量维度变换
PixelShuffle（亚像素卷积）通过将高维通道维度的信息重排至空间几何维度，实现低算力、高保真的特征图上采样。

```mermaid
flowchart LR
    A["P5 Feature (B, 256, 20, 20)"] --> B["Expand Conv (B, 64×16, 20, 20)"]
    B --> C["PixelShuffle(upscale=4) (B, 64, 80, 80)"]
    C --> D["ConvBlock (B, 64, 80, 80)"]
    D --> E["Prediction 1x1 Conv (B, 3, 80, 80)"]
    E --> F["Flatten to 6400 Candidates (B, 6400)"]
```

* **输入**：$P_5$ 特征 $\mathbf{F}_5 \in \mathbb{R}^{B \times 256 \times 20 \times 20}$
* **通道升维**：$\text{Conv}_{3\times 3}(\mathbf{F}_5) \to \mathbf{F}_{\text{exp}} \in \mathbb{R}^{B \times (C_{\text{mid}} \cdot r^2) \times 20 \times 20}$，其中 $r = 4$，$C_{\text{mid}} = 64$；
* **空间重排**：$\text{PixelShuffle}(r=4) \to \mathbf{F}_{\text{spatial}} \in \mathbb{R}^{B \times 64 \times 80 \times 80}$；
* **点属性预测**：$\text{Conv}_{1\times 1}(\mathbf{F}_{\text{spatial}}) \to \mathbf{P} \in \mathbb{R}^{B \times 3 \times 80 \times 80}$，输出每个空间位置的 $[z, dx, dy]$。

### 3.2 为什么能从数学上根治大目标漏检？
1. **特征 1:1 独立占有**：
   展开后的 $80 \times 80$ 特征图上，每个特征像素直接对齐原图 $8 \times 8$ 区域，与 E0 完全一致。
2. **零负梯度内耗**：
   在大目标区域内，被匈牙利匹配命中的中心像素点受到正向梯度激励，其周围的邻近像素受到各自独立的负向抑制，**各像素点之间在反向传播中梯度完全独立，互不抵消**。
3. **完美继承大感受野语义**：
   特征仍旧来源于 $P_5$ 顶层，具备识别大目标的宏观上下文语义，同时获得了与底层相同的空间分辨率。

### 3.3 模块接口与代码设计（Blueprint）

```python
class PixelShufflePointExpert(nn.Module):
    """基于 PixelShuffle 空间展开的 E2 专家头。
    
    输入: P5 特征 (B, C, 20, 20)
    输出: 展开为 80x80 空间网格，总候选点仍为 6400，但每个候选点独占特征像素。
    """
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        upscale_factor: int = 4,
        prior_probability: float = 0.01,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.hidden_channels = hidden_channels
        
        # 1. 升维卷积: 将通道扩展至 hidden_channels * (4^2) = hidden_channels * 16
        self.expand_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels * (upscale_factor ** 2),
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_group_norm(hidden_channels * (upscale_factor ** 2)),
            nn.SiLU(inplace=True),
        )
        
        # 2. 亚像素重排: (B, hidden_channels * 16, 20, 20) -> (B, hidden_channels, 80, 80)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
        # 3. 空间特征平滑与预测头 (输出通道仅为 3: 1 个 confidence, 2 个 offset)
        self.body = ConvBlock(hidden_channels, hidden_channels)
        self.prediction = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        
        # 4. Focal loss 偏置初始化
        prior_bias = math.log(prior_probability / (1.0 - prior_probability))
        nn.init.normal_(self.prediction.weight, std=0.01)
        with torch.no_grad():
            self.prediction.bias[0].fill_(prior_bias)
            self.prediction.bias[1:].zero_()

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # feature: (B, 256, 20, 20)
        x = self.pixel_shuffle(self.expand_conv(feature))  # (B, 64, 80, 80)
        x = self.body(x)                                    # (B, 64, 80, 80)
        output = self.prediction(x)                        # (B, 3, 80, 80)
        
        confidence_logits = output[:, 0:1]                 # (B, 1, 80, 80)
        offsets = output[:, 1:3].unsqueeze(1)              # (B, 1, 2, 80, 80)
        return confidence_logits, offsets
```

### 3.4 架构向后兼容性说明
* **总候选点数保持 6400**：$80 \times 80 \times 1 = 6400$，与旧版 $20 \times 20 \times 16 = 6400$ 严格一致；
* **有效参考间距保持 8 px**：$640 / 80 = 8\text{ px}$；
* **匈牙利匹配与损失函数完全零修改兼容**：上层候选池 Concat 形状、`expert_indices` 索引分布完全一致。

---

## 4. 方案二深度设计：Deformable Point Sampling 方案（高阶）

### 4.1 核心工作流程
1. 网络首先基于 $P_5$ 初始网格特征，预测 16 个参考点的自适应空间偏移场 $\Delta \mathbf{p}_{i, j, k}$；
2. 利用连续双线性插值采样算子 $\mathcal{S}(\mathbf{F}, \mathbf{p})$ 从 $P_5$ 特征图上精准采样 16 个点对应的独立特征向量：
   $$\mathbf{f}_{i, j, k} = \text{BilinearSample}\left(\mathbf{F}_5, \mathbf{b}_{i, j, k} + \Delta \mathbf{p}_{i, j, k}\right)$$
3. 将采样的独立特征传入轻量共享 MLP 输出分类置信度与最终细调量。

### 4.2 优缺点权衡
* **优势**：具备连续亚像素几何自适应能力，对非刚体形变人群、俯仰大视角目标更鲁棒；
* **代价**：引入额外的特征网格采样（`grid_sample`）算子，前向推理速度略有下降。

---

## 5. 方案三深度设计：Gaussian Soft Label 损失平滑（训练协同）

### 5.1 数学公式推导
针对大尺寸目标，在计算分类损失时不再使用离散硬二值标签 $y \in \{0, 1\}$，而是构建连续空间高斯软标签分布：
$$y_k = \exp\left( - \frac{\|\mathbf{p}_k - \mathbf{g}\|^2}{2 \sigma_{\text{target}}^2} \right)$$
其中 $\sigma_{\text{target}}$ 与目标尺度成正比。

### 5.2 软标签 Focal Loss 计算
$$L_{\text{cls}}(z, y) = - y^\alpha (1 - \sigma(z))^\gamma \log \sigma(z) - (1 - y)^\alpha \sigma(z)^\gamma \log (1 - \sigma(z))$$
* **作用机制**：将大目标周围的负样本从“强行置零”转变为“按距离平滑衰减的正监督”，避免大目标区域的特征在反向传播中被负梯度直接抹杀。

---

## 6. 实施路线与消融验证方案

建议按以下三步安全迭代：

```
Step 1: E2 专家头 PixelShuffle 改造 (models/moe_point_head.py)
   │
   ▼
Step 2: 单专家消融验证 (ablation_ft_E2)
   │   - 在 ShanghaiTech Part B 上重新微调
   │   - 验证目标: 预测总数从 2653 恢复到 4800~5300 区间, MAE 从 68 降至 20 左右
   │
   ▼
Step 3: 联合多专家全量微调 (Joint Native Multiscale Training)
       - 观察全局匈牙利匹配中 E2 的 GT Winner 比例是否从 0% 正常回升至 20%~35%
```

---

## 7. 审阅要点反馈与决策项

在开始实施代码改造前，请审阅以下决策点：
1. **是否采用方案一（PixelShuffle）作为主改造路线？**（推荐：是，收益最直接且零破坏性）；
2. **中间通道数配置**：PixelShuffle 的中间隐藏通道建议设为 `64`（计算量极小且特征表达充足），是否维持该默认值？
3. **单专家先行还是直接联合训练**：建议先跑 `ablation_ft_E2` 快速验证指标是否跃升，确认后再启动全局联合微调。
