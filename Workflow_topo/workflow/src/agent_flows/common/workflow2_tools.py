"""workflow2工具：模拟摄像头采集与图像处理"""

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:  # 可选GPU支持
    import torch
    import torch.nn.functional as F
except Exception:  # torch 不可用时保持CPU流程
    torch = None
    F = None

CAMERA_IMAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "camera_images"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_camera_images() -> List[Path]:
    images = [p for p in CAMERA_IMAGE_DIR.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]
    return images


def capture_camera_image(camera_id: str, scene: str, secure_environment: bool = False) -> Dict[str, Any]:
    """模拟摄像头采集图像，需要安全执行环境"""
    if not secure_environment:
        raise PermissionError("摄像头采集需要安全执行环境")

    images = _list_camera_images()
    if not images:
        raise FileNotFoundError("data/camera_images 下没有可用示例图像")

    chosen = random.choice(images)
    image = cv2.imread(str(chosen))
    if image is None:
        raise ValueError(f"无法读取图像文件: {chosen}")

    return {
        "camera_id": camera_id,
        "scene": scene,
        "path": str(chosen),
        "image": image,
    }


def _gpu_smooth(image: np.ndarray) -> np.ndarray:
    """在GPU上执行简单3x3平滑卷积，返回uint8数组"""
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape[:2]
    if h < 3 or w < 3:
        return image

    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).cuda() / 255.0
    kernel = torch.ones((1, 1, 3, 3), device=tensor.device) / 9.0
    smoothed = F.conv2d(tensor, kernel, padding=1)
    smoothed = (smoothed.clamp(0, 1) * 255.0).squeeze().byte().cpu().numpy()
    return smoothed


def preprocess_image(image: np.ndarray, operations: Dict[str, bool] | None = None) -> Tuple[np.ndarray, List[str]]:
    """对图像进行预处理，返回处理后的图像与步骤"""
    ops = {
        "grayscale": True,
        "gaussian_blur": True,
        "denoise": False,
        "gpu_accel": True,
    }
    if operations:
        ops.update(operations)

    processed = image.copy()
    steps: List[str] = []

    if ops.get("grayscale", True) and len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        steps.append("grayscale")

    if ops.get("gaussian_blur", True):
        processed = cv2.GaussianBlur(processed, (5, 5), sigmaX=1.0)
        steps.append("gaussian_blur")

    if ops.get("denoise", False):
        processed = cv2.fastNlMeansDenoising(processed, None, h=10, templateWindowSize=7, searchWindowSize=21)
        steps.append("denoise")

    if ops.get("gpu_accel", True):
        gpu_reason = None
        if torch is None:
            gpu_reason = "gpu_skip_no_torch"
        elif not torch.cuda.is_available():
            gpu_reason = "gpu_skip_no_cuda"
        else:
            try:
                processed = _gpu_smooth(processed)
                steps.append("gpu_smooth3x3")
            except Exception:
                gpu_reason = "gpu_skip_error"

        if gpu_reason:
            steps.append(gpu_reason)

    return processed, steps


def extract_image_features(image: np.ndarray) -> Dict[str, Any]:
    """提取简单的图像特征（亮度、边缘、直方图等）"""
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist_sum = float(hist.sum()) if hist.sum() else 1.0
    hist_norm = (hist / hist_sum).round(4).tolist()

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean() / 255.0)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    )
    occupancy_ratio = float(adaptive.mean() / 255.0)

    mean_intensity = float(gray.mean())
    std_intensity = float(gray.std())

    return {
        "shape": {
            "height": int(gray.shape[0]),
            "width": int(gray.shape[1]),
            "channels": 1 if len(image.shape) == 2 else int(image.shape[2]),
        },
        "mean_intensity": round(mean_intensity, 2),
        "std_intensity": round(std_intensity, 2),
        "edge_density": round(edge_density, 4),
        "laplacian_var": round(lap_var, 2),
        "occupancy_ratio": round(occupancy_ratio, 4),
        "histogram_32": hist_norm,
    }


def rate_noise_level(features: Dict[str, Any]) -> Dict[str, Any]:
    """基于 Laplacian 方差粗略评估清晰度/噪声水平"""
    lap_var = float(features.get("laplacian_var") or 0.0)
    if lap_var < 200:
        level = "偏平滑/可能轻微模糊"
        note = "纹理较弱，细节可能不足"
    elif lap_var < 600:
        level = "正常"
        note = "纹理适中，可正常分析"
    else:
        level = "纹理/噪声略多"
        note = "细节丰富但可能带少量噪声"
    return {"laplacian_var": round(lap_var, 1), "level": level, "note": note}


def estimate_congestion_level(features: Dict[str, Any]) -> Dict[str, Any]:
    """根据边缘密度与占用度估计路口拥堵程度，返回等级和理由"""
    edge_density = float(features.get("edge_density") or 0.0)
    occupancy = float(features.get("occupancy_ratio") or 0.0)
    mean_intensity = float(features.get("mean_intensity") or 0.0)
    std_intensity = float(features.get("std_intensity") or 0.0)

    # 加权得分：边缘密度偏向车辆轮廓，占用度偏向画面填充度
    score = 0.55 * edge_density + 0.45 * occupancy

    # 规则调优：提高对真正高占用/高边缘的敏感度，同时仍压制“占用高但边缘极低”的误判
    if occupancy >= 0.45 or edge_density >= 0.26 or score >= 0.36:
        level = "拥堵"
        reason = f"占用{occupancy:.2f}/边缘{edge_density:.2f}或综合得分{score:.2f}偏高，车流密集"
    elif edge_density >= 0.21 and occupancy >= 0.30:
        level = "正常偏高"
        reason = f"边缘{edge_density:.2f}与占用{occupancy:.2f}处于中高位，接近拥堵但尚可通行"
    elif edge_density < 0.15 and occupancy >= 0.32:
        level = "正常"
        reason = f"边缘{edge_density:.2f}较低但占用{occupancy:.2f}偏高，可能是光照/背景导致，不判为拥堵"
    elif edge_density >= 0.15 or occupancy >= 0.22:
        level = "正常"
        reason = f"边缘{edge_density:.2f}/占用{occupancy:.2f}中位，车流正常"
    else:
        level = "稀疏"
        reason = f"边缘{edge_density:.2f}/占用{occupancy:.2f}偏低，车流稀疏"

    note = f"亮度{mean_intensity:.1f}，亮度波动{std_intensity:.1f}"

    return {
        "level": level,
        "edge_density": round(edge_density, 3),
        "occupancy_ratio": round(occupancy, 3),
        "score": round(score, 3),
        "reason": reason,
        "note": note,
    }


def build_congestion_narrative(features: Dict[str, Any]) -> str:
    congestion = estimate_congestion_level(features)
    noise = rate_noise_level(features)
    parts = [
        f"拥堵判定：{congestion['level']} ({congestion['reason']})",
        f"边缘密度 {congestion['edge_density']}, 占用度 {congestion['occupancy_ratio']}, 综合得分 {congestion.get('score')}",
        f"画质：{noise['level']} ({noise['note']}，Laplacian方差 {noise['laplacian_var']})",
    ]
    return "；".join(parts)


def summarize_scene_from_features(features: Dict[str, Any]) -> str:
    """基于特征生成简单描述，用于LLM提示"""
    shape = features.get("shape", {})
    width = shape.get("width")
    height = shape.get("height")
    channels = shape.get("channels")
    mean_intensity = features.get("mean_intensity")
    edge_density = features.get("edge_density")
    std_intensity = features.get("std_intensity")

    desc = [
        f"分辨率约为 {width}x{height}" if width and height else "分辨率未知",
        f"通道数 {channels}" if channels else "通道未知",
        f"平均亮度 {mean_intensity}" if mean_intensity is not None else "亮度未知",
        f"边缘密度 {edge_density}" if edge_density is not None else "边缘密度未知",
        f"亮度标准差 {std_intensity}" if std_intensity is not None else "亮度标准差未知",
    ]
    return "，".join(desc)
