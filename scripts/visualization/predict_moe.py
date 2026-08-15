import argparse
import os

import cv2
import numpy as np
import torch

from models.yolo11_moe_point import YOLO11MoEPoint

# 三个专家的可视化颜色（BGR）：route 0=P3 专家、1=P4 专家、2=P5 专家
EXPERT_COLORS = [
    (0, 0, 255),    # P3 局部细节 - 红色
    (0, 255, 0),    # P4 中层上下文 - 绿色
    (255, 0, 0),    # P5 大范围上下文 - 蓝色
]


def load_model(weights_path, checkpoint_path, device):
    """构建模型并加载训练好的权重。"""
    model = YOLO11MoEPoint(weights=weights_path).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"从 {checkpoint_path} 加载权重")
        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = (
            ckpt["model"] if "model" in ckpt else ckpt
        )
        print(
            f"checkpoint: epoch={ckpt.get('epoch')} "
            f"best_mae={ckpt.get('best_mae')}"
        )
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as error:
            print(
                f"警告: 旧版 checkpoint 缺少新参数({error})，"
                "缺失部分使用初始化值"
            )
            model.load_state_dict(
                state_dict, strict=False
            )
    else:
        print("警告: 未提供权重文件，模型使用随机初始化参数进行推理。")

    model.eval()
    return model


def predict_image(
    model,
    image_bgr: np.ndarray,
    device: str,
    imgsz: int = 384,
    conf_threshold: float = 0.5,
):
    """对单张图像推理，返回 (像素坐标点, 路由索引, 置信度)。

    输入图像会缩放/放大到 imgsz x imgsz（训练时使用的尺度），
    输出的点坐标会映射回原始图像尺寸。
    """
    original_height, original_width = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR
    )

    tensor = (
        torch.from_numpy(
            image_rgb.astype(np.float32) / 255.0
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        predictions = model(
            tensor,
            temperature=0.5,
            hard_route=True,
        )

    scores = predictions["logits"].sigmoid()[0]
    points = predictions["points"][0]
    routes = predictions["gates"].argmax(dim=-1)[0]

    keep = scores > conf_threshold

    selected_points = points[keep].cpu().numpy()
    selected_routes = routes[keep].cpu().numpy()
    selected_scores = scores[keep].cpu().numpy()

    # 映射回原始图像尺寸
    scale_x = original_width / imgsz
    scale_y = original_height / imgsz
    selected_points[:, 0] *= scale_x
    selected_points[:, 1] *= scale_y

    return selected_points, selected_routes, selected_scores


def draw_predictions(
    image_bgr: np.ndarray,
    points: np.ndarray,
    routes: np.ndarray,
) -> np.ndarray:
    """绘制预测点，三个专家使用不同颜色。"""
    result = image_bgr.copy()

    for point, route in zip(points, routes):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        color = EXPERT_COLORS[int(route) % len(EXPERT_COLORS)]
        cv2.circle(result, (x, y), radius=3, color=color, thickness=-1)

    count = len(points)
    cv2.putText(
        result,
        f"Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    return result


def predict_main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    model = load_model(args.weights, args.checkpoint, device)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"无法读取图片 {args.image}")

    points, routes, scores = predict_image(
        model,
        image,
        device,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
    )

    print(f"检测到 {len(points)} 个人。")
    for route in range(3):
        print(
            f"  专家 {route}: {int((routes == route).sum())} 个点"
        )

    result = draw_predictions(image, points, routes)
    out_path = args.output or args.image.replace(
        ".jpg", "_moe_pred.jpg"
    )
    cv2.imwrite(out_path, result)
    print(f"已将预测结果保存至 {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11 + 点级 Scale-MoE Head 推理与可视化"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="输入图片路径"
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11n.pt",
        help="YOLO11 预训练权重（用于构建 Backbone+Neck）"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="runs/moe_point/best.pt",
        help="训练好的 MoE 模型权重"
    )
    parser.add_argument(
        "--imgsz", type=int, default=384,
        help="推理输入尺寸"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="置信度阈值"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出图片路径"
    )
    return parser.parse_args()


if __name__ == "__main__":
    predict_main(parse_args())
